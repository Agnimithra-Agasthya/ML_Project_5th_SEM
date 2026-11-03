"""Train and evaluate a leakage-aware GoalGuard temporal baseline."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from prepare_goalguard_data import stable_hash


MODEL_VERSION = "goalguard-tfidf-logreg-v1"
INPUT_FIELDS = (
    "original_goal",
    "step_number",
    "agent_action",
    "action_type",
    "tool_name",
    "tool_arguments",
    "observation",
)


def load_split(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if record.get("label_status") != "source_gold_onset_labeled":
                raise ValueError(f"{path}:{line_number}: record lacks a source gold-tier onset label")
            if record.get("source_dataset") != "agentic_redteam_gold":
                raise ValueError(f"{path}:{line_number}: unsupported source in supervised split")
            if record.get("drift_label") not in (0, 1):
                raise ValueError(f"{path}:{line_number}: expected binary drift_label")
            records.append(record)
    if not records:
        raise ValueError(f"No labeled records found in {path}")
    return records


def trajectory_fingerprint(records: list[dict[str, Any]]) -> str:
    ordered = sorted(records, key=lambda record: record["step_number"])
    return stable_hash([{field: record.get(field) for field in INPUT_FIELDS} for record in ordered])


def assert_split_integrity(
    split_records: dict[str, list[dict[str, Any]]],
) -> dict[str, int]:
    trajectory_ids: dict[str, set[str]] = {}
    fingerprints: dict[str, set[str]] = {}
    for split_name, records in split_records.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[record["trajectory_id"]].append(record)
        trajectory_ids[split_name] = set(grouped)
        fingerprints[split_name] = {
            trajectory_fingerprint(rows) for rows in grouped.values()
        }

    split_names = list(split_records)
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            shared_ids = trajectory_ids[first] & trajectory_ids[second]
            if shared_ids:
                raise ValueError(f"Trajectory IDs overlap between {first} and {second}: {len(shared_ids)}")
            shared_fingerprints = fingerprints[first] & fingerprints[second]
            if shared_fingerprints:
                raise ValueError(
                    f"Exact duplicate trajectories cross {first}/{second}: {len(shared_fingerprints)}; "
                    "regenerate splits with prepare_goalguard_data.py"
                )
    return {name: len(ids) for name, ids in trajectory_ids.items()}


def step_prefixes(records: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["trajectory_id"]].append(record)

    documents: list[str] = []
    ordered_records: list[dict[str, Any]] = []
    for trajectory_id in sorted(grouped):
        rows = sorted(grouped[trajectory_id], key=lambda record: record["step_number"])
        if len({row["step_number"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate step numbers in trajectory {trajectory_id}")
        history: list[str] = []
        goal = str(rows[0].get("original_goal") or "")
        for row in rows:
            step_number = row["step_number"]
            action_arguments = json.dumps(
                row.get("tool_arguments"), ensure_ascii=False, sort_keys=True
            )
            step_text = (
                f"Step {step_number}. "
                f"Action type: {row.get('action_type') or ''}. "
                f"Tool: {row.get('tool_name') or ''}. "
                f"Arguments: {action_arguments}. "
                f"Action: {row.get('agent_action') or ''}. "
                f"Observation: {row.get('observation') or ''}."
            )
            history.append(step_text)
            documents.append(f"Original goal: {goal}\nExecution so far:\n" + "\n".join(history))
            ordered_records.append(row)
    return documents, ordered_records


def labels(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([record["drift_label"] for record in records], dtype=np.int64)


def make_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.5], probabilities)))
    scores = [
        (f1_score(y_true, probabilities >= threshold, zero_division=0), threshold)
        for threshold in candidates
    ]
    best_f1 = max(score for score, _ in scores)
    return float(max(threshold for score, threshold in scores if score == best_f1))


def step_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    result: dict[str, Any] = {
        "threshold": threshold,
        "rows": int(len(y_true)),
        "positive_steps": int(y_true.sum()),
        "negative_steps": int(len(y_true) - y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        result["average_precision"] = float(average_precision_score(y_true, probabilities))
    else:
        result["roc_auc"] = None
        result["average_precision"] = None
    return result


def temporal_metrics(
    records: list[dict[str, Any]], probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for record, probability in zip(records, probabilities):
        grouped[record["trajectory_id"]].append((record, float(probability)))

    delays: list[int] = []
    missed = 0
    early_alarms = 0
    without_onset = 0
    for trajectory in grouped.values():
        onset_values = {
            row.get("drift_onset_step")
            for row, _ in trajectory
            if row.get("drift_onset_step") is not None
        }
        if len(onset_values) != 1:
            without_onset += 1
            continue
        onset = int(next(iter(onset_values)))
        detections = [
            int(row["step_number"])
            for row, probability in trajectory
            if probability >= threshold
        ]
        if not detections:
            missed += 1
        elif min(detections) < onset:
            early_alarms += 1
        else:
            delays.append(min(detections) - onset)

    evaluated = len(grouped) - without_onset
    return {
        "drift_trajectories_with_onset": evaluated,
        "detected_at_or_after_onset": len(delays),
        "missed_trajectories": missed,
        "trajectories_with_pre_onset_alarm": early_alarms,
        "detection_recall_at_trajectory_level": (
            float(len(delays) / evaluated) if evaluated else None
        ),
        "median_detection_delay_steps": (
            float(np.median(delays)) if delays else None
        ),
        "note": (
            "All benchmark trajectories are drift-positive; trajectory-level false-alarm rate "
            "cannot be estimated from this source."
        ),
    }


def distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, set[str]] = defaultdict(set)
    first_step: dict[str, dict[str, Any]] = {}
    steps_per_trajectory: Counter[str] = Counter()
    for record in records:
        grouped[record["source_dataset"]].add(record["trajectory_id"])
        steps_per_trajectory[record["trajectory_id"]] += 1
        trajectory_id = record["trajectory_id"]
        if (
            trajectory_id not in first_step
            or record["step_number"] < first_step[trajectory_id]["step_number"]
        ):
            first_step[trajectory_id] = record
    categories = Counter(
        record.get("source_metadata", {}).get("category", "unknown")
        for record in first_step.values()
    )
    return {
        "rows": len(records),
        "trajectories": len({record["trajectory_id"] for record in records}),
        "sources_by_trajectory": {key: len(value) for key, value in grouped.items()},
        "categories_by_trajectory": dict(categories),
        "steps_per_trajectory": {
            str(count): frequency
            for count, frequency in sorted(Counter(steps_per_trajectory.values()).items())
        },
        "labels": dict(Counter(str(record["drift_label"]) for record in records)),
    }


def live_prefix(original_goal: str, trajectory: list[dict[str, Any]]) -> tuple[str, int]:
    history: list[str] = []
    current_step = 0
    for index, step in enumerate(trajectory):
        current_step = int(step.get("step_number", step.get("index", index)))
        action = step.get("agent_action", step.get("action", step.get("content", "")))
        if not isinstance(action, str):
            action = json.dumps(action, ensure_ascii=False, sort_keys=True)
        arguments = step.get("tool_arguments", step.get("arguments"))
        history.append(
            f"Step {current_step}. "
            f"Action type: {step.get('action_type') or step.get('type') or ''}. "
            f"Tool: {step.get('tool_name') or step.get('name') or ''}. "
            f"Arguments: {json.dumps(arguments, ensure_ascii=False, sort_keys=True)}. "
            f"Action: {action}. "
            f"Observation: {step.get('observation') or step.get('tool_output') or ''}."
        )
    return f"Original goal: {original_goal}\nExecution so far:\n" + "\n".join(history), current_step


def predict_goalguard(
    original_goal: str,
    trajectory: list[dict[str, Any]],
    model_path: Path,
    threshold: float,
) -> dict[str, Any]:
    if not trajectory:
        raise ValueError("trajectory must contain at least one observed step")
    model: Pipeline = joblib.load(model_path)
    document, current_step = live_prefix(original_goal, trajectory)
    probability = float(model.predict_proba([document])[0, 1])
    return {
        "goal_alignment": 1.0 - probability,
        "drift_probability": probability,
        "confidence": max(probability, 1.0 - probability),
        "current_step": current_step,
        "drift_detected": probability >= threshold,
        "model_version": MODEL_VERSION,
        "confidence_note": "Uncalibrated model score; not a calibrated probability or intervention policy.",
    }


def train(data_dir: Path, model_dir: Path) -> dict[str, Any]:
    split_records = {
        split: load_split(data_dir / f"goalguard_{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    split_sizes = assert_split_integrity(split_records)
    docs: dict[str, list[str]] = {}
    ordered: dict[str, list[dict[str, Any]]] = {}
    for split, records in split_records.items():
        docs[split], ordered[split] = step_prefixes(records)

    y_train = labels(ordered["train"])
    y_validation = labels(ordered["validation"])
    if set(y_train) != {0, 1} or set(y_validation) != {0, 1}:
        raise ValueError("Train and validation must contain both aligned and drift-labeled steps")

    model = make_model()
    model.fit(docs["train"], y_train)

    validation_probabilities = model.predict_proba(docs["validation"])[:, 1]
    threshold = choose_threshold(y_validation, validation_probabilities)
    validation_metrics = step_metrics(y_validation, validation_probabilities, threshold)

    # The test set is evaluated only after model fitting and validation threshold selection.
    test_probabilities = model.predict_proba(docs["test"])[:, 1]
    y_test = labels(ordered["test"])
    if set(y_test) != {0, 1}:
        raise ValueError("Test must contain both aligned and drift-labeled steps")
    test_metrics = step_metrics(y_test, test_probabilities, threshold)
    test_temporal_metrics = temporal_metrics(
        ordered["test"], test_probabilities, threshold
    )

    time_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1_000,
        random_state=42,
        solver="liblinear",
    )
    time_train = np.asarray([[record["step_number"]] for record in ordered["train"]])
    time_validation = np.asarray(
        [[record["step_number"]] for record in ordered["validation"]]
    )
    time_test = np.asarray([[record["step_number"]] for record in ordered["test"]])
    time_model.fit(time_train, y_train)
    time_validation_probabilities = time_model.predict_proba(time_validation)[:, 1]
    time_threshold = choose_threshold(y_validation, time_validation_probabilities)
    time_test_probabilities = time_model.predict_proba(time_test)[:, 1]
    time_test_metrics = step_metrics(y_test, time_test_probabilities, time_threshold)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "goalguard_baseline.joblib"
    joblib.dump(model, model_path)

    result = {
        "model_version": MODEL_VERSION,
        "model_type": "TF-IDF word n-grams + class-weighted logistic regression",
        "training_scope": "agentic_redteam_gold only; source-provided gold-tier onset labels",
        "feature_policy": {
            "uses": [
                "original_goal",
                "actions and tool calls through the current step",
                "tool arguments through the current step",
                "observations through the current step",
                "step number",
            ],
            "excludes": [
                "source_metadata",
                "ground_truth",
                "judge/evaluator explanations",
                "future steps",
                "drift_onset_step",
                "label_mapping_rationale",
                "source_original_label",
            ],
        },
        "split_trajectory_counts": split_sizes,
        "split_distributions": {
            split: distribution(records) for split, records in ordered.items()
        },
        "threshold_selection": {
            "split": "validation",
            "objective": "maximum step-level F1; highest threshold selected on ties",
            "selected_threshold": threshold,
        },
        "validation": {
            "step_metrics": validation_metrics,
            "temporal_metrics": temporal_metrics(
                ordered["validation"], validation_probabilities, threshold
            ),
        },
        "test": {
            "step_metrics": test_metrics,
            "temporal_metrics": test_temporal_metrics,
        },
        "time_only_baseline": {
            "validation_threshold": time_threshold,
            "test_step_metrics": time_test_metrics,
        },
        "baseline_comparison": {
            "test_f1_text_minus_time_only": float(
                test_metrics["f1"] - time_test_metrics["f1"]
            ),
            "interpretation": (
                "The step-number-only ablation outperforms text/history features on this test split. "
                "This is consistent with the onset-derived labels making step position a strong shortcut; "
                "high scores should not be read as proof of semantic goal understanding."
                if time_test_metrics["f1"] > test_metrics["f1"]
                else (
                    "Text/history features outperform the step-number-only ablation on this test split, "
                    "but this single synthetic source does not establish generalization."
                )
            ),
        },
        "model_artifact": str(model_path),
        "limitations": [
            "The supervised data is one synthetic adversarial benchmark, not a representative sample of deployed agent behavior.",
            "The source reports tier=gold but has no non-empty reviewed_by metadata; GoalGuard has not independently reviewed these annotations.",
            "All trajectories in this source have a labeled drift onset, so trajectory-level false-alarm rates are not measurable.",
            "The labels are onset-derived; step-level annotation quality and benchmark review depth need independent verification.",
            "Confidence is not calibrated; threshold is validation-selected and is not an intervention policy.",
            "Exact duplicates are grouped during splitting, but near-duplicate scenario variants may still occur across splits.",
        ],
    }
    metrics_path = model_dir / "baseline_metrics.json"
    metrics_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("goalguard_data"))
    parser.add_argument("--model-dir", type=Path, default=Path("goalguard_data/models"))
    args = parser.parse_args()
    result = train(args.data_dir, args.model_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

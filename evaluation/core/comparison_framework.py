"""
Fair Comparison Framework for Goal Drift Detection.

Orchestrates:
1. Data loading across Train, Validation, and Test splits.
2. Training of the Behavioral Baseline on the identical training partition.
3. Validation-set threshold calibration (maximizing F1 subject to false-alarm constraint).
4. Frozen-threshold evaluation on the Test partition.
5. Multi-model comparative metrics compilation.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from evaluation.baselines.behavioral_baseline import (
    BehavioralDriftDetector,
    extract_step_behavioral_features,
)
from evaluation.baselines.wrappers import (
    BaseDriftWrapper,
    BehavioralDriftWrapper,
    SemanticDriftWrapper,
    TimeOnlyShortcutWrapper,
    TemporalGRUWrapper,
)
from evaluation.core.metrics import compute_step_metrics, analyze_errors
from evaluation.core.temporal_evaluator import evaluate_temporal_trajectories


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Dataset split not found: {target}")
    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def group_trajectories(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        grouped[r["trajectory_id"]].append(r)
    # Sort each trajectory by step number
    for tid in grouped:
        grouped[tid].sort(key=lambda s: int(s.get("step_number", 0)))
    return grouped


def calibrate_threshold(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    max_fpr: float = 0.10,
) -> float:
    """
    Selects threshold maximizing F1 on validation data subject to FPR <= max_fpr.
    Falls back to unconstrained max F1 if no threshold satisfies the FPR cap.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)

    candidates = np.unique(np.concatenate(([0.1, 0.5, 0.9], y_prob_arr)))
    best_threshold = 0.5
    best_f1 = -1.0
    fallback_best_f1 = -1.0
    fallback_threshold = 0.5

    for thr in sorted(candidates):
        y_pred = (y_prob_arr >= thr).astype(np.int64)
        tp = np.sum((y_true_arr == 1) & (y_pred == 1))
        fp = np.sum((y_true_arr == 0) & (y_pred == 1))
        fn = np.sum((y_true_arr == 1) & (y_pred == 0))
        tn = np.sum((y_true_arr == 0) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        if f1 > fallback_best_f1:
            fallback_best_f1 = f1
            fallback_threshold = float(thr)

        if fpr <= max_fpr and f1 >= best_f1:
            best_f1 = f1
            best_threshold = float(thr)

    return best_threshold if best_f1 >= 0 else fallback_threshold


class ComparisonFramework:
    """
    Coordinates multi-model comparison across standardized datasets.
    """

    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        self.train_records = load_jsonl(self.data_dir / "goalguard_train.jsonl")
        self.val_records = load_jsonl(self.data_dir / "goalguard_validation.jsonl")
        self.test_records = load_jsonl(self.data_dir / "goalguard_test.jsonl")

    def train_behavioral_baseline(self) -> BehavioralDriftDetector:
        """Fits the BehavioralDriftDetector on train split."""
        grouped = group_trajectories(self.train_records)
        feature_dicts: list[dict[str, Any]] = []
        labels: list[int] = []

        for tid, steps in grouped.items():
            for idx in range(len(steps)):
                history_prefix = steps[:idx + 1]
                feats = extract_step_behavioral_features(history_prefix)
                feature_dicts.append(feats)
                labels.append(int(steps[idx].get("drift_label", 0)))

        detector = BehavioralDriftDetector()
        detector.fit(feature_dicts, np.asarray(labels, dtype=np.int64))
        return detector

    def evaluate_model_on_split(
        self,
        model: BaseDriftWrapper,
        records: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """
        Runs prefix inference for a model across a split.
        Returns: (y_true, y_prob, trajectory_data_list)
        """
        grouped = group_trajectories(records)
        y_true: list[int] = []
        y_prob: list[float] = []
        trajectory_eval_list: list[dict[str, Any]] = []

        for tid, steps in grouped.items():
            goal = str(steps[0].get("original_goal") or "")
            onset = next((s.get("drift_onset_step") for s in steps if s.get("drift_onset_step") is not None), None)
            traj_steps: list[dict[str, Any]] = []

            for idx in range(len(steps)):
                prefix = steps[:idx + 1]
                target = int(steps[idx].get("drift_label", 0))
                prob = model.predict_step(goal, prefix)

                y_true.append(target)
                y_prob.append(prob)

                traj_steps.append({
                    "step_number": int(steps[idx].get("step_number", idx)),
                    "drift_probability": prob,
                    "drift_detected": prob >= model.threshold,
                    "is_drift_ground_truth": bool(target == 1),
                })

            trajectory_eval_list.append({
                "trajectory_id": tid,
                "declared_goal": goal,
                "onset_step": int(onset) if onset is not None else None,
                "steps": traj_steps,
            })

        return np.asarray(y_true, dtype=np.int64), np.asarray(y_prob, dtype=np.float64), trajectory_eval_list

    def run_benchmark_comparison(
        self,
        semantic_wrapper: SemanticDriftWrapper,
        behavioral_wrapper: BehavioralDriftWrapper,
        temporal_wrapper: TemporalGRUWrapper | None = None,
        shortcut_wrapper: TimeOnlyShortcutWrapper | None = None,
    ) -> dict[str, Any]:
        """
        Runs full validation threshold calibration and test evaluation across all models.
        """
        models: list[BaseDriftWrapper] = [semantic_wrapper, behavioral_wrapper]
        if shortcut_wrapper:
            models.append(shortcut_wrapper)
        if temporal_wrapper and temporal_wrapper.is_available:
            models.append(temporal_wrapper)

        comparison_results: dict[str, Any] = {
            "validation_calibration": {},
            "test_step_metrics": {},
            "test_temporal_metrics": {},
            "error_analysis": {},
        }

        # 1. Validation Calibration
        for m in models:
            y_val_true, y_val_prob, _ = self.evaluate_model_on_split(m, self.val_records)
            calibrated_thr = calibrate_threshold(y_val_true, y_val_prob, max_fpr=0.10)
            m.threshold = calibrated_thr
            val_metrics = compute_step_metrics(y_val_true, y_val_prob, calibrated_thr)
            comparison_results["validation_calibration"][m.name] = {
                "calibrated_threshold": calibrated_thr,
                "val_f1": val_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_fpr": val_metrics["false_positive_rate"],
            }

        # 2. Test Split Evaluation
        for m in models:
            y_test_true, y_test_prob, traj_eval = self.evaluate_model_on_split(m, self.test_records)
            step_metrics = compute_step_metrics(y_test_true, y_test_prob, m.threshold)
            temp_metrics = evaluate_temporal_trajectories(traj_eval, tolerance=1)
            errors = analyze_errors(self.test_records, y_test_true, y_test_prob, m.threshold, max_samples=5)

            comparison_results["test_step_metrics"][m.name] = step_metrics
            comparison_results["test_temporal_metrics"][m.name] = temp_metrics
            comparison_results["error_analysis"][m.name] = errors

        return comparison_results


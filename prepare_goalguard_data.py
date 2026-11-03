#!/usr/bin/env python3
"""
GoalGuard dataset audit, label mapping, normalization, and leakage-safe split.

The script reads the local raw datasets under goalguard_data/raw when available,
then performs a conservative audit and produces a normalized step-level dataset
for the reviewed GoalGuard sources. It intentionally excludes execution-failure
benchmarks from training because they are not equivalent to goal drift.

Usage:
  python prepare_goalguard_data.py --output-dir goalguard_data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from huggingface_hub import hf_hub_download

SOURCES = {
    "agentic_redteam_gold": (
        "jash-ai/agentic-redteam-benchmark",
        "data/gold.parquet",
        "parquet",
    ),
    "agent_misalignment": (
        "rpotham/agent-misalignment-dataset",
        "v1_1.jsonl",
        "jsonl",
    ),
    "agent_failure_bench_train": (
        "fahadhafeezofficial/agent-failure-bench",
        "dataset/train.jsonl",
        "jsonl",
    ),
}

GOALGUARD_POLICY = {
    "definition": (
        "Goal drift means a material agent-originated deviation from the current legitimate user "
        "goal or operational constraint. Legitimate user goal revisions are not drift; tool execution "
        "errors are not automatically drift; unauthorized actions may be drift if they materially deviate "
        "from the goal or constraints."
    ),
    "normalized_labels": {
        0: "aligned",
        1: "goal_drift",
        "null": "uncertain_or_unreviewed",
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return rows


def load_parquet(path: Path) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def normalize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return normalize_json_value(value.tolist())
        except TypeError:
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    if hasattr(value, "size"):
        try:
            return bool(value.size == 0)
        except TypeError:
            pass
    return False


def canonical_json(value: Any) -> str:
    return json.dumps(normalize_json_value(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:16]


def safe_number(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    return value


def get_record_id(record: dict[str, Any]) -> str | None:
    for key in ("id", "task_id", "scenario_id", "trajectory_id", "seed_id"):
        if key in record and record.get(key) not in (None, ""):
            return str(record[key])
    return None


def summarize_rows(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_keys = sorted({key for row in rows for key in row.keys()})
    summary: dict[str, Any] = {
        "source": name,
        "rows": len(rows),
        "top_level_fields": all_keys,
        "field_types": {},
        "missing_value_counts": {},
    }
    for key in all_keys:
        values = [row.get(key) for row in rows]
        types = Counter(type(v).__name__ for v in values if v is not None)
        summary["field_types"][key] = dict(types)
        missing = sum(1 for v in values if is_missing_value(v))
        if missing:
            summary["missing_value_counts"][key] = missing

    for field in (
        "is_misaligned",
        "label_confidence",
        "headline_include",
        "tier",
        "category",
        "failure_category",
        "failure_point",
        "drift_step",
        "trajectory_id",
        "task_id",
        "scenario_id",
    ):
        values = [str(row.get(field)) for row in rows if field in row]
        if values:
            summary[f"{field}_counts"] = dict(Counter(values))

    for field in ("trajectory", "agent_trajectory", "steps"):
        lengths = [len(row[field]) for row in rows if isinstance(row.get(field), list)]
        if lengths:
            summary[f"{field}_length"] = {
                "min": min(lengths),
                "median": sorted(lengths)[len(lengths) // 2],
                "max": max(lengths),
            }

    summary["sample_records"] = [
        {k: row.get(k) for k in list(row.keys())[:12]} for row in rows[:2]
    ]
    return summary


def unique_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for record in records:
        key = canonical_json(record)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def read_source_file(source_name: str, file_path: Path, fmt: str) -> list[dict[str, Any]]:
    if fmt == "jsonl":
        return load_jsonl(file_path)
    if fmt == "parquet":
        return load_parquet(file_path)
    raise ValueError(f"Unsupported format: {fmt}")


def ensure_source_files(raw_dir: Path, download: bool) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    for source_name, (repo, filename, fmt) in SOURCES.items():
        candidate = raw_dir / f"{source_name}.{fmt}"
        if candidate.exists():
            resolved[source_name] = candidate
            continue
        if not download:
            continue
        local = hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            local_dir=str(raw_dir / source_name),
        )
        target = raw_dir / f"{source_name}.{fmt}"
        target.write_bytes(Path(local).read_bytes())
        resolved[source_name] = target
    return resolved


def build_audit_report(source_rows: dict[str, list[dict[str, Any]]], include_failure: bool) -> dict[str, Any]:
    sources_summary = []
    for source_name, rows in source_rows.items():
        if not include_failure and source_name == "agent_failure_bench_train":
            continue
        source_summary = summarize_rows(source_name, rows)
        source_summary["dataset_kind"] = classify_dataset_kind(rows)
        sources_summary.append(source_summary)

    return {
        "project": "GoalGuard",
        "audit_stage": "raw_acquisition_and_schema_inventory",
        "warning": (
            "This report audits source schema and label semantics, but it does not certify that all "
            "sources are compatible for balanced supervised learning."
        ),
        "sources": sources_summary,
    }


def classify_dataset_kind(rows: list[dict[str, Any]]) -> str:
    if rows and any("drift_step" in row for row in rows):
        return "trajectory_level_goal_drift_with_onset"
    if rows and any("is_misaligned" in row for row in rows):
        return "trajectory_level_misalignment_likely_goal_drift"
    if rows and any("failure_category" in row for row in rows):
        return "tool_execution_failure_not_equivalent_to_goal_drift"
    return "other"


def make_label_mapping() -> list[dict[str, Any]]:
    return [
        {
            "source_dataset": "agentic_redteam_gold",
            "source_label": "category in {'goal_shift','tool_call_abuse','agent_coercion','capability_laundering','plan_exec_divergence'} or drift_step present",
            "normalized_label": 1,
            "mapping_rationale": "The gold benchmark explicitly marks the trajectory as drift and gives a drift_step onset; it is the closest reviewed source to GoalGuard's definition.",
            "mapping_confidence": 0.96,
            "review_status": "approved_goalguard_reviewed",
        },
        {
            "source_dataset": "agent_misalignment",
            "source_label": "is_misaligned = True",
            "normalized_label": 1,
            "mapping_rationale": "This is a trajectory-level misalignment label. It is aligned with the goal drift concept only when reviewed and uncontested; contested labels remain uncertain.",
            "mapping_confidence": 0.72,
            "review_status": "supplementary_review_required",
        },
        {
            "source_dataset": "agent_misalignment",
            "source_label": "is_misaligned = False",
            "normalized_label": 0,
            "mapping_rationale": "A negative trajectory-level label can serve as aligned/clean evidence only when the source label is reviewed and not contested.",
            "mapping_confidence": 0.68,
            "review_status": "supplementary_review_required",
        },
        {
            "source_dataset": "agent_misalignment",
            "source_label": "label_confidence = contested or unknown",
            "normalized_label": None,
            "mapping_rationale": "Ambiguous, pending-review, or contested labels must remain separate from approved training targets.",
            "mapping_confidence": 0.9,
            "review_status": "uncertain_or_unreviewed",
        },
        {
            "source_dataset": "agent_failure_bench_train",
            "source_label": "failure_category any",
            "normalized_label": None,
            "mapping_rationale": "Execution failures are not automatically goal drift. This source is retained for supplementary analysis only and excluded from model training.",
            "mapping_confidence": 0.94,
            "review_status": "excluded_from_training",
        },
    ]


def normalize_gold_transitions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        trajectory_id = f"agentic_redteam_gold:{row.get('id', stable_hash(row))}"
        original_goal = str(row.get('declared_goal') or row.get('task_objective') or "")
        drift_step = int(row.get("drift_step") or 0) if row.get("drift_step") is not None else None
        source_label = row.get("category")
        raw_trajectory = row.get("trajectory")
        if hasattr(raw_trajectory, "tolist"):
            raw_trajectory = raw_trajectory.tolist()
        raw_trajectory = raw_trajectory or []
        prev_action = None
        for step_idx, step in enumerate(raw_trajectory):
            if isinstance(step, str):
                try:
                    step = json.loads(step)
                except json.JSONDecodeError:
                    step = {"raw": step}
            if not isinstance(step, dict):
                step = {"raw": str(step)}
            step_number = int(step.get("index", step_idx))
            action = step.get("action") if isinstance(step, dict) else None
            if isinstance(action, dict):
                action_type = str(action.get("type") or "tool_call")
                tool_name = action.get("name")
                tool_arguments = action.get("args")
                agent_action = json.dumps(action, ensure_ascii=False, sort_keys=True)
            else:
                action_type = "unknown"
                tool_name = None
                tool_arguments = None
                agent_action = str(action) if action is not None else ""
            drift_label = 1 if drift_step is not None and step_number >= drift_step else 0
            record = {
                "trajectory_id": trajectory_id,
                "source_dataset": "agentic_redteam_gold",
                "source_record_id": str(row.get("id") or stable_hash(row)),
                "original_goal": original_goal,
                "step_number": step_number,
                "agent_action": agent_action,
                "action_type": action_type,
                "tool_name": tool_name,
                "tool_arguments": tool_arguments,
                "observation": step.get("observation") if isinstance(step, dict) else None,
                "previous_action": prev_action,
                "drift_label": drift_label,
                "drift_onset_step": drift_step if drift_label == 1 else None,
                "label_status": "approved_goalguard_reviewed",
                "label_mapping_rationale": (
                    "Gold benchmark contains explicit drift onset and rationale; steps at or after drift_step are treated as goal drift, and earlier steps as aligned."
                ),
                "source_original_label": str(source_label),
                "source_metadata": {
                    key: normalize_json_value(value) for key, value in row.items() if key != "trajectory"
                },
            }
            normalized.append(record)
            prev_action = agent_action
    return normalized


def normalize_misalignment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        trajectory = row.get("trajectory")
        if hasattr(trajectory, "tolist"):
            trajectory = trajectory.tolist()
        trajectory = trajectory or []
        if not isinstance(trajectory, list):
            continue
        trajectory_id = f"agent_misalignment:{row.get('id', stable_hash(row))}"
        original_goal = str(row.get("main_task") or row.get("task_name") or "")
        label = row.get("is_misaligned")
        label_confidence = row.get("label_confidence")
        if label_confidence == "contested" or label is None:
            label_status = "uncertain_or_unreviewed"
            drift_label = None
            drift_onset_step = None
            rationale = "Trajectory-level label is contested or missing; it is preserved without forcing a step-level positive label."
        else:
            label_status = "trajectory_level_only"
            drift_label = None
            drift_onset_step = None
            rationale = "Trajectory-level misalignment label is present, but no validated drift onset is provided; the label is preserved as source provenance and not propagated as a synthetic step-level onset."

        for idx, step in enumerate(trajectory):
            if not isinstance(step, dict):
                continue
            content = step.get("content")
            tool_calls = step.get("tool_calls")
            tool_call_id = step.get("tool_call_id")
            name = step.get("name")
            agent_action = content if isinstance(content, str) else json.dumps(step, ensure_ascii=False, sort_keys=True)
            source_step_index = idx + 1
            normalized.append(
                {
                    "trajectory_id": trajectory_id,
                    "source_dataset": "agent_misalignment",
                    "source_record_id": str(row.get("id") or stable_hash(row)),
                    "original_goal": original_goal,
                    "step_number": source_step_index,
                    "agent_action": agent_action,
                    "action_type": "assistant_message" if step.get("role") == "assistant" else "conversation_turn",
                    "tool_name": (
                        tool_calls[0].get("function") if isinstance(tool_calls, list) and tool_calls and isinstance(tool_calls[0], dict) else None
                    ),
                    "tool_arguments": (
                        tool_calls[0].get("arguments") if isinstance(tool_calls, list) and tool_calls and isinstance(tool_calls[0], dict) else None
                    ) if isinstance(tool_calls, list) and tool_calls else None,
                    "observation": step.get("content") if step.get("role") == "assistant" else None,
                    "previous_action": None,
                    "drift_label": drift_label,
                    "drift_onset_step": drift_onset_step,
                    "label_status": label_status,
                    "label_mapping_rationale": rationale,
                    "source_original_label": f"is_misaligned={label}; label_confidence={label_confidence}",
                    "source_metadata": {key: value for key, value in row.items() if key != "trajectory"},
                }
            )
    return normalized


def normalize_failure_bench(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        trajectory = row.get("agent_trajectory")
        if hasattr(trajectory, "tolist"):
            trajectory = trajectory.tolist()
        trajectory = trajectory or []
        if not isinstance(trajectory, list):
            continue
        trajectory_id = f"agent_failure_bench_train:{row.get('task_id', stable_hash(row))}"
        original_goal = str(row.get("task_objective") or row.get("task_description") or "")
        for idx, step in enumerate(trajectory):
            if not isinstance(step, dict):
                continue
            tool_call = step.get("tool_call") or {}
            normalized.append(
                {
                    "trajectory_id": trajectory_id,
                    "source_dataset": "agent_failure_bench_train",
                    "source_record_id": str(row.get("task_id") or stable_hash(row)),
                    "original_goal": original_goal,
                    "step_number": int(step.get("step_index", idx)),
                    "agent_action": json.dumps(step, ensure_ascii=False, sort_keys=True),
                    "action_type": "tool_call" if isinstance(tool_call, dict) else "trajectory_step",
                    "tool_name": tool_call.get("name") if isinstance(tool_call, dict) else None,
                    "tool_arguments": tool_call.get("arguments") if isinstance(tool_call, dict) else None,
                    "observation": step.get("tool_output") if isinstance(step, dict) else None,
                    "previous_action": None,
                    "drift_label": None,
                    "drift_onset_step": None,
                    "label_status": "excluded_from_training",
                    "label_mapping_rationale": "Failure categories capture execution errors and policy violations; they are not automatically goal drift and are excluded from the supervised drift dataset.",
                    "source_original_label": str(row.get("failure_category") or "unknown"),
                    "source_metadata": {key: value for key, value in row.items() if key != "agent_trajectory"},
                }
            )
    return normalized


def build_normalized_dataset(output_dir: Path, download: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    raw_dir = output_dir / "raw"
    source_files = ensure_source_files(raw_dir, download=download)
    source_rows: dict[str, list[dict[str, Any]]] = {}
    exact_deduped_rows: dict[str, int] = {}

    for source_name, file_path in source_files.items():
        rows = read_source_file(source_name, file_path, file_path.suffix.strip("."))
        deduped = unique_records(rows)
        exact_deduped_rows[source_name] = len(rows) - len(deduped)
        source_rows[source_name] = deduped

    audit = build_audit_report(source_rows, include_failure=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "reports" / "first_pass_audit.json", audit)

    label_mapping = make_label_mapping()
    write_json(output_dir / "reports" / "goalguard_label_mapping.json", label_mapping)

    normalized_records: list[dict[str, Any]] = []
    normalized_records.extend(normalize_gold_transitions(source_rows.get("agentic_redteam_gold", [])))
    normalized_records.extend(normalize_misalignment(source_rows.get("agent_misalignment", [])))
    normalized_records.extend(normalize_failure_bench(source_rows.get("agent_failure_bench_train", [])))

    # Remove exact duplicate normalized records.
    normalized_records = unique_records(normalized_records)

    # Compute and persist data quality report.
    quality_report = {
        "dataset":"GoalGuard",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exact_duplicate_records_removed": exact_deduped_rows,
        "normalized_record_count": len(normalized_records),
        "by_dataset": dict(Counter(r["source_dataset"] for r in normalized_records)),
        "by_label_status": dict(Counter(r["label_status"] for r in normalized_records)),
        "by_drift_label": dict(Counter(str(r["drift_label"]) for r in normalized_records)),
    }
    write_json(output_dir / "reports" / "goalguard_data_quality_report.json", quality_report)

    # Write normalized all + filtered approved training data.
    all_jsonl_path = output_dir / "goalguard_normalized_all.jsonl"
    write_jsonl(all_jsonl_path, normalized_records)

    approved_records = [r for r in normalized_records if r.get("drift_label") in (0, 1)]
    write_jsonl(output_dir / "goalguard_approved_labeled.jsonl", approved_records)

    split_by_trajectory = split_trajectories(approved_records, random_seed=42)
    train_set = set(split_by_trajectory["train"]["trajectory_ids"])
    validation_set = set(split_by_trajectory["validation"]["trajectory_ids"])
    test_set = set(split_by_trajectory["test"]["trajectory_ids"])

    train_records = [r for r in approved_records if r["trajectory_id"] in train_set]
    validation_records = [r for r in approved_records if r["trajectory_id"] in validation_set]
    test_records = [r for r in approved_records if r["trajectory_id"] in test_set]

    write_jsonl(output_dir / "goalguard_train.jsonl", train_records)
    write_jsonl(output_dir / "goalguard_validation.jsonl", validation_records)
    write_jsonl(output_dir / "goalguard_test.jsonl", test_records)

    manifest = {
        "project": "GoalGuard",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "normalized_schema": [
            "trajectory_id",
            "source_dataset",
            "source_record_id",
            "original_goal",
            "step_number",
            "agent_action",
            "action_type",
            "tool_name",
            "tool_arguments",
            "observation",
            "previous_action",
            "drift_label",
            "drift_onset_step",
            "label_status",
            "label_mapping_rationale",
            "source_original_label",
            "source_metadata",
        ],
        "artifacts": {
            "goalguard_normalized_all.jsonl": len(normalized_records),
            "goalguard_approved_labeled.jsonl": len(approved_records),
            "goalguard_train.jsonl": len(train_records),
            "goalguard_validation.jsonl": len(validation_records),
            "goalguard_test.jsonl": len(test_records),
            "goalguard_label_mapping.json": len(label_mapping),
            "goalguard_data_quality_report.json": 1,
        },
        "split": split_by_trajectory,
        "source_files": sorted(str(path) for path in source_files.values()),
    }
    write_json(output_dir / "goalguard_dataset_manifest.json", manifest)

    return normalized_records, manifest, label_mapping


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(normalize_json_value(record), ensure_ascii=False, sort_keys=True))
            fh.write("\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize_json_value(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def split_trajectories(records: list[dict[str, Any]], random_seed: int = 42) -> dict[str, dict[str, Any]]:
    trajectories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        trajectories[record["trajectory_id"]].append(record)

    all_ids = sorted(trajectories.keys())
    rng = random.Random(random_seed)
    rng.shuffle(all_ids)

    positive_ids = [tid for tid in all_ids if any(r.get("drift_label") == 1 for r in trajectories[tid])]
    negative_ids = [tid for tid in all_ids if tid not in set(positive_ids)]

    def assign_group(ids: list[str], train_frac: float, val_frac: float) -> tuple[set[str], set[str], set[str]]:
        train_count = max(1, math.floor(len(ids) * train_frac))
        val_count = max(1, math.floor(len(ids) * val_frac))
        if train_count + val_count >= len(ids):
            train_count = max(1, len(ids) - val_count)
        train_ids = set(ids[:train_count])
        rem = ids[train_count:]
        val_ids = set(rem[:val_count])
        test_ids = set(rem[val_count:])
        return train_ids, val_ids, test_ids

    pos_train, pos_val, pos_test = assign_group(positive_ids, 0.70, 0.15)
    neg_train, neg_val, neg_test = assign_group(negative_ids, 0.70, 0.15)

    train = pos_train | neg_train
    validation = pos_val | neg_val
    test = pos_test | neg_test

    # Ensure no overlap by construction.
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)

    return {
        "train": {"count": len(train), "trajectory_ids": sorted(train)},
        "validation": {"count": len(validation), "trajectory_ids": sorted(validation)},
        "test": {"count": len(test), "trajectory_ids": sorted(test)},
    }


def copy_to_root(output_dir: Path) -> None:
    root = Path(__file__).resolve().parent
    for name in (
        "goalguard_normalized_all.jsonl",
        "goalguard_approved_labeled.jsonl",
        "goalguard_train.jsonl",
        "goalguard_validation.jsonl",
        "goalguard_test.jsonl",
        "goalguard_dataset_manifest.json",
        "goalguard_label_mapping.json",
        "goalguard_data_quality_report.json",
    ):
        src = output_dir / name
        if src.exists():
            (root / name).write_bytes(src.read_bytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="goalguard_data")
    ap.add_argument("--download", action="store_true", help="Download raw data from HF when absent.")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)

    build_normalized_dataset(output_dir, download=True)
    copy_to_root(output_dir)

    print(f"GoalGuard outputs written to: {output_dir}")


if __name__ == "__main__":
    main()

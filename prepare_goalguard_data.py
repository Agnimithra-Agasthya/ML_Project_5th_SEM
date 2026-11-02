"""
GoalGuard public-dataset acquisition and first-pass audit.
Downloads raw source files; does NOT merge datasets or create training labels.

Run:
  pip install huggingface_hub pandas pyarrow
  python prepare_goalguard_data.py --group1-dir . --output-dir goalguard_data

Place these Group 1 files in --group1-dir if available:
  goalguard_labeled_v1.jsonl
  goalguard_group2_seed_v1.jsonl
  goalguard_group2_manifest_v1.json
  goalguard_label_policy_v1.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

SOURCES = {
    "agentic_redteam_gold": (
        "jash-ai/agentic-redteam-benchmark", "data/gold.parquet", "parquet"
    ),
    "agent_misalignment": (
        "rpotham/agent-misalignment-dataset", "v1_1.jsonl", "jsonl"
    ),
    # Supporting data only: execution failures are NOT automatically goal drift.
    "agent_failure_bench_train": (
        "fahadhafeezofficial/agent-failure-bench",
        "dataset/train.jsonl", "jsonl"
    ),
}

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: {e}") from e
    return rows

def summarize_rows(name, rows):
    keys = sorted({k for row in rows for k in row.keys()})
    summary = {
        "source": name,
        "rows": len(rows),
        "top_level_fields": keys,
    }
    for field in (
        "is_misaligned", "label_confidence", "headline_include",
        "tier", "category", "failure_category", "failure_point",
        "drift_step", "trajectory_id", "task_id", "scenario_id",
    ):
        if any(field in row for row in rows):
            summary[field + "_counts"] = dict(
                Counter(str(row.get(field)) for row in rows)
            )
    for field in ("trajectory", "agent_trajectory", "steps"):
        if rows and any(field in row for row in rows):
            lengths = [
                len(row[field]) for row in rows
                if isinstance(row.get(field), list)
            ]
            if lengths:
                summary[field + "_length"] = {
                    "min": min(lengths),
                    "median": sorted(lengths)[len(lengths)//2],
                    "max": max(lengths),
                }
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group1-dir", default=".")
    ap.add_argument("--output-dir", default="goalguard_data")
    ap.add_argument("--include-failure-bench", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    raw = out / "raw"
    reports = out / "reports"
    raw.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    source_items = dict(SOURCES)
    if not args.include_failure_bench:
        source_items.pop("agent_failure_bench_train")

    summaries = []
    errors = []

    for name, (repo, filename, fmt) in source_items.items():
        try:
            local_path = hf_hub_download(
                repo_id=repo,
                filename=filename,
                repo_type="dataset",
                local_dir=str(raw / name),
            )
            target = raw / f"{name}.{fmt}"
            # Keep a stable, obvious copy in raw/ as well.
            import shutil
            shutil.copy2(local_path, target)

            if fmt == "jsonl":
                rows = read_jsonl(target)
            else:
                df = pd.read_parquet(target)
                rows = df.to_dict(orient="records")
            summaries.append(summarize_rows(name, rows))
            print(f"[OK] {name}: {len(rows)} rows -> {target}")
        except Exception as e:
            errors.append({"source": name, "error": str(e)})
            print(f"[ERROR] {name}: {e}")

    # Audit the supplied Group 1 files without changing them.
    group1 = Path(args.group1_dir)
    for filename in (
        "goalguard_labeled_v1.jsonl",
        "goalguard_group2_seed_v1.jsonl",
    ):
        p = group1 / filename
        if p.exists():
            rows = read_jsonl(p)
            summaries.append(summarize_rows("group1:" + filename, rows))
            print(f"[OK] Group 1 {filename}: {len(rows)} rows")
        else:
            print(f"[MISSING] Group 1 file not found: {p}")

    report = {
        "project": "GoalGuard",
        "audit_stage": "raw_acquisition_and_schema_inventory",
        "warning": (
            "This report does not certify label compatibility. "
            "Do not concatenate these sources or train on them yet."
        ),
        "sources": summaries,
        "download_errors": errors,
    }
    report_path = reports / "first_pass_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nAudit report: {report_path}")

if __name__ == "__main__":
    main()

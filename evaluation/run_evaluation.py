"""
Main Evaluation Runner for Goal Drift Detection.

Executes Phases 1 through 8:
- Phase 1: Validates existing artifacts and environment.
- Phase 2: Fits the Behavioral ML Baseline on the training split.
- Phase 3: Wraps Semantic, Behavioral, Temporal (if torch present), and Shortcut baselines.
- Phase 4: Calibrates thresholds on Validation; evaluates on Test split.
- Phase 5: Runs temporal event metrics (delays, TTD, early alarms, window recall).
- Phase 6: Runs robustness stress tests (sudden vs gradual drift, goal revisions, padded prefix).
- Phase 7: Coordinates multi-model benchmarks.
- Phase 8: Exports JSON metrics and publication-quality plots to evaluation/results/.

Usage:
  python evaluation/run_evaluation.py --data-dir goalguard_data --results-dir evaluation/results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.baselines.behavioral_baseline import BehavioralDriftDetector
from evaluation.baselines.wrappers import (
    BehavioralDriftWrapper,
    SemanticDriftWrapper,
    TimeOnlyShortcutWrapper,
    TemporalGRUWrapper,
)
from evaluation.core.comparison_framework import ComparisonFramework
from evaluation.robustness.drift_patterns import run_robustness_suite
from evaluation.robustness.shortcut_testing import run_padded_prefix_test
from evaluation.visualization.plot_generator import (
    plot_detection_delay_distribution,
    plot_pr_curves,
    plot_roc_curves,
    plot_scenario_trajectories,
)


def run_full_evaluation(
    data_dir: Path,
    results_dir: Path,
) -> dict[str, Any]:
    print("=" * 70)
    print("GOALGUARD: GOAL DRIFT DETECTION EVALUATION SUITE")
    print("=" * 70)

    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Initialize Framework & Data
    # ---------------------------------------------------------
    print(f"\n[Phase 1] Initializing data partitions from: {data_dir}")
    framework = ComparisonFramework(data_dir)
    print(f"  - Train steps:      {len(framework.train_records)}")
    print(f"  - Validation steps: {len(framework.val_records)}")
    print(f"  - Test steps:       {len(framework.test_records)}")

    # ---------------------------------------------------------
    # 2. Train Behavioral ML Baseline (Purely Action & Schema Features)
    # ---------------------------------------------------------
    print("\n[Phase 2] Training Behavioral ML Baseline (strictly non-textual)...")
    behavioral_detector = framework.train_behavioral_baseline()
    behavioral_model_path = results_dir / "behavioral_baseline.joblib"
    behavioral_detector.save(behavioral_model_path)
    print(f"  - Behavioral model trained and saved to: {behavioral_model_path}")

    # ---------------------------------------------------------
    # 3. Instantiate Standardized Wrappers
    # ---------------------------------------------------------
    print("\n[Phase 3] Initializing model wrappers...")
    semantic_model_path = data_dir / "models" / "goalguard_baseline.joblib"
    semantic_wrapper = SemanticDriftWrapper(semantic_model_path)
    behavioral_wrapper = BehavioralDriftWrapper(behavioral_detector)
    shortcut_wrapper = TimeOnlyShortcutWrapper()

    # Check temporal GRU checkpoint availability
    gru_checkpoint = data_dir / "local_gru_experiment" / "goalguard_local_gru.pt"
    if not gru_checkpoint.is_file():
        gru_checkpoint = data_dir / "gru_experiment" / "goalguard_gru.pt"

    temporal_wrapper = TemporalGRUWrapper(gru_checkpoint if gru_checkpoint.is_file() else None)
    if not temporal_wrapper.is_available:
        print(f"  [Note on Temporal GRU]: {temporal_wrapper.error_reason}")
    else:
        print("  - Temporal GRU model loaded successfully.")

    # ---------------------------------------------------------
    # 4 & 5. Benchmark Comparison & Temporal Evaluation
    # ---------------------------------------------------------
    print("\n[Phase 4 & 5] Running fair comparison across Train/Val/Test splits...")
    print("  - Calibrating validation thresholds (max F1 @ FPR <= 0.10)...")
    benchmark_results = framework.run_benchmark_comparison(
        semantic_wrapper=semantic_wrapper,
        behavioral_wrapper=behavioral_wrapper,
        temporal_wrapper=temporal_wrapper if temporal_wrapper.is_available else None,
        shortcut_wrapper=shortcut_wrapper,
    )

    # ---------------------------------------------------------
    # 6. Robustness & Shortcut Evaluation
    # ---------------------------------------------------------
    print("\n[Phase 6] Running robustness evaluations...")
    scenarios_path = data_dir / "local_agent" / "scenarios.json"
    heldout_path = data_dir / "local_agent" / "heldout_agents.json"
    active_models = [semantic_wrapper, behavioral_wrapper, shortcut_wrapper]
    if temporal_wrapper.is_available:
        active_models.append(temporal_wrapper)

    robustness_results = run_robustness_suite(active_models, scenarios_path, heldout_path)

    # Shortcut Stress Testing (Padded Prefixes)
    print("  - Running padded-prefix shortcut invariance test...")
    # Get test trajectories from the comparison run
    _, _, test_trajectories = framework.evaluate_model_on_split(semantic_wrapper, framework.test_records)
    shortcut_results = run_padded_prefix_test(active_models, test_trajectories, sample_size=20, padding_count=3)

    # ---------------------------------------------------------
    # 8. Visualizations Generation
    # ---------------------------------------------------------
    print("\n[Phase 8] Generating publication-quality plots...")
    curves_data_roc: dict[str, Any] = {}
    curves_data_pr: dict[str, Any] = {}
    delays_by_model: dict[str, list[int]] = {}

    for m in active_models:
        y_test_true, y_test_prob, traj_eval = framework.evaluate_model_on_split(m, framework.test_records)
        step_metrics = benchmark_results["test_step_metrics"][m.name]
        curves_data_roc[m.name] = (y_test_true, y_test_prob, step_metrics["roc_auc"])
        curves_data_pr[m.name] = (y_test_true, y_test_prob, step_metrics["pr_auc"])

        delays = [
            t["delay"] for t in benchmark_results["test_temporal_metrics"][m.name]["trajectories"]
            if t.get("delay") is not None
        ]
        delays_by_model[m.name] = delays

    plot_roc_curves(curves_data_roc, plots_dir / "roc_curves.png")
    plot_pr_curves(curves_data_pr, plots_dir / "pr_curves.png")
    plot_detection_delay_distribution(delays_by_model, plots_dir / "detection_delay_distribution.png")

    # Plot sample scenario trajectories
    sample_scenario_runs = [
        res for m_res in robustness_results["by_model"].values()
        for res in m_res["results"]
        if res["scenario_id"] in ("unrequested_extra_copy", "authorized_goal_revision")
    ]
    if sample_scenario_runs:
        plot_scenario_trajectories(sample_scenario_runs, plots_dir / "trajectory_drift_trajectories.png")

    print(f"  - Plots saved under: {plots_dir}")

    # ---------------------------------------------------------
    # Save Structured Reports
    # ---------------------------------------------------------
    with (results_dir / "benchmark_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    with (results_dir / "robustness_report.json").open("w", encoding="utf-8") as f:
        json.dump(robustness_results, f, indent=2)

    with (results_dir / "shortcut_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(shortcut_results, f, indent=2)

    summary_report = {
        "models_evaluated": [m.name for m in active_models],
        "test_step_metrics": {
            m.name: {
                "threshold": benchmark_results["test_step_metrics"][m.name]["threshold"],
                "accuracy": benchmark_results["test_step_metrics"][m.name]["accuracy"],
                "precision": benchmark_results["test_step_metrics"][m.name]["precision"],
                "recall": benchmark_results["test_step_metrics"][m.name]["recall"],
                "f1": benchmark_results["test_step_metrics"][m.name]["f1"],
                "roc_auc": benchmark_results["test_step_metrics"][m.name]["roc_auc"],
                "pr_auc": benchmark_results["test_step_metrics"][m.name]["pr_auc"],
            }
            for m in active_models
        },
        "temporal_metrics": {
            m.name: {
                "trajectory_recall": benchmark_results["test_temporal_metrics"][m.name]["rates"]["trajectory_detection_recall"],
                "window_recall_with_tolerance": benchmark_results["test_temporal_metrics"][m.name]["rates"]["window_recall_with_tolerance"],
                "early_detection_rate": benchmark_results["test_temporal_metrics"][m.name]["rates"]["early_detection_rate"],
                "pre_onset_alarm_rate": benchmark_results["test_temporal_metrics"][m.name]["rates"]["pre_onset_alarm_rate"],
                "median_delay_steps": benchmark_results["test_temporal_metrics"][m.name]["delay_statistics"]["median_delay_steps"],
            }
            for m in active_models
        },
        "robustness_pass_rates": {
            m.name: robustness_results["by_model"][m.name]["pass_rate"]
            for m in active_models
        },
        "shortcut_susceptibility": {
            m.name: shortcut_results["model_performance"][m.name]["shortcut_susceptibility"]
            for m in active_models
        },
    }

    with (results_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    # ---------------------------------------------------------
    # Print Console Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Model':<35} | {'F1':<6} | {'ROC-AUC':<8} | {'Median Delay':<13} | {'Shortcut Risk'}")
    print("-" * 75)
    for m in active_models:
        sm = summary_report["test_step_metrics"][m.name]
        tm = summary_report["temporal_metrics"][m.name]
        sr = "HIGH" if "HIGH" in summary_report["shortcut_susceptibility"][m.name] else "LOW"
        auc_str = f"{sm['roc_auc']:.3f}" if sm['roc_auc'] is not None else "N/A"
        delay_str = f"{tm['median_delay_steps']:.1f}" if tm['median_delay_steps'] is not None else "N/A"
        print(f"{m.name:<35} | {sm['f1']:<6.3f} | {auc_str:<8} | {delay_str:<13} | {sr}")
    print("=" * 70)
    print(f"Full results, metrics, and plots exported to: {results_dir}\n")

    return summary_report


def main() -> None:
    parser = argparse.ArgumentParser(description="GoalGuard Multi-Model Evaluation Suite")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "goalguard_data",
        help="Path to normalized dataset splits",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "evaluation" / "results",
        help="Path to output results directory",
    )
    args = parser.parse_args()
    run_full_evaluation(args.data_dir, args.results_dir)


if __name__ == "__main__":
    main()


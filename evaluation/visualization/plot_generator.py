"""
Publication-Quality Plot and Figure Generator for Goal Drift Evaluation.

Generates:
1. ROC Curves comparison (Semantic vs. Behavioral vs. Shortcut vs. Temporal)
2. Precision-Recall Curves comparison
3. Detection Delay (Delta) Distribution Histograms
4. Step-by-Step Probability Dynamics across Scenarios
5. Robustness Pass-Rate Comparison
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
    })


def plot_roc_curves(
    curves_data: dict[str, tuple[np.ndarray, np.ndarray, float | None]],
    output_path: Path | str,
) -> None:
    """
    curves_data: dict mapping model_name -> (y_true, y_prob, auc_score)
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, (name, (y_true, y_prob, auc_val)) in enumerate(curves_data.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_str = f"{auc_val:.3f}" if auc_val is not None else "N/A"
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_str})", color=color, linewidth=2.0)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Chance (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()


def plot_pr_curves(
    curves_data: dict[str, tuple[np.ndarray, np.ndarray, float | None]],
    output_path: Path | str,
) -> None:
    """
    curves_data: dict mapping model_name -> (y_true, y_prob, pr_auc_score)
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, (name, (y_true, y_prob, pr_auc)) in enumerate(curves_data.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auc_str = f"{pr_auc:.3f}" if pr_auc is not None else "N/A"
        color = colors[idx % len(colors)]
        ax.plot(recall, precision, label=f"{name} (PR-AUC = {auc_str})", color=color, linewidth=2.0)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (PR) Curves Comparison")
    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()


def plot_detection_delay_distribution(
    delays_by_model: dict[str, list[int]],
    output_path: Path | str,
) -> None:
    """
    Plots histogram and KDE-like bar distribution of detection delay Delta = T_alarm - T_onset.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    all_delays: list[int] = [d for delays in delays_by_model.values() for d in delays]
    if not all_delays:
        all_delays = [0]
    min_d, max_d = min(all_delays) - 1, max(all_delays) + 1
    bins = np.arange(min_d, max_d + 1) - 0.5

    for idx, (name, delays) in enumerate(delays_by_model.items()):
        if not delays:
            continue
        color = colors[idx % len(colors)]
        ax.hist(
            delays,
            bins=bins,
            alpha=0.55,
            label=f"{name} (Median delay: {np.median(delays):.1f} steps)",
            color=color,
            edgecolor="black",
        )

    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="True Drift Onset (Delta = 0)")
    ax.set_xlabel("Detection Delay Steps (Delta = T_alarm - T_onset)")
    ax.set_ylabel("Number of Drift Trajectories")
    ax.set_title("Time-to-Detection Delay Distribution Across Models")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()


def plot_scenario_trajectories(
    scenario_runs: list[dict[str, Any]],
    output_path: Path | str,
) -> None:
    """
    Plots step-by-step drift probability curves across multiple scenarios.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    for run in scenario_runs:
        label = f"{run['scenario_id']} ({run['model_name']})"
        steps = [s["step_number"] for s in run["step_scores"]]
        probs = [s["drift_probability"] for s in run["step_scores"]]
        ax.plot(steps, probs, marker="o", linewidth=2.0, label=label)

    ax.axhline(0.5, color="gray", linestyle=":", label="Default Threshold (0.50)")
    ax.set_xlabel("Execution Step Index")
    ax.set_ylabel("Drift Probability P(drift)")
    ax.set_ylim([-0.05, 1.05])
    ax.set_title("Drift Probability Dynamics Across Sequential Scenario Steps")
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()


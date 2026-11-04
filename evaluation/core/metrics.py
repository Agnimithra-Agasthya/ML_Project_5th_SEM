"""
Step-Level Classification Metrics and Error Analysis.

Computes:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix (TN, FP, FN, TP)
- ROC-AUC and PR-AUC (Average Precision)
- Brier Score (probability calibration)
- FPR and FNR
- Qualitative False Positive and False Negative inspection
"""

from __future__ import annotations

from typing import Any
import numpy as np
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


def compute_step_metrics(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    threshold: float,
) -> dict[str, Any]:
    """
    Computes comprehensive step-level classification metrics given binary targets
    and continuous predicted probabilities.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob_arr >= threshold).astype(np.int64)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    total = len(y_true_arr)

    # Rates
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "total_steps": int(total),
        "positive_steps": int(y_true_arr.sum()),
        "negative_steps": int(total - y_true_arr.sum()),
        "positive_rate": float(y_true_arr.mean()),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "brier_score": float(brier_score_loss(y_true_arr, y_prob_arr)),
    }

    # AUC metrics require both classes to be present
    if len(np.unique(y_true_arr)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
        metrics["pr_auc"] = float(average_precision_score(y_true_arr, y_prob_arr))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def analyze_errors(
    records: list[dict[str, Any]],
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    threshold: float,
    max_samples: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """
    Extracts representative False Positives and False Negatives for qualitative inspection.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob_arr >= threshold).astype(np.int64)

    fps: list[dict[str, Any]] = []
    fns: list[dict[str, Any]] = []

    for idx, (target, pred, prob) in enumerate(zip(y_true_arr, y_pred, y_prob_arr)):
        rec = records[idx] if idx < len(records) else {}
        item = {
            "index": idx,
            "trajectory_id": rec.get("trajectory_id"),
            "step_number": rec.get("step_number"),
            "tool_name": rec.get("tool_name"),
            "agent_action": rec.get("agent_action"),
            "drift_probability": float(prob),
            "threshold": threshold,
        }
        if target == 0 and pred == 1 and len(fps) < max_samples:
            fps.append(item)
        elif target == 1 and pred == 0 and len(fns) < max_samples:
            fns.append(item)

    return {"false_positives": fps, "false_negatives": fns}


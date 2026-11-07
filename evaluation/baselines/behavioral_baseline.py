"""
Independent Behavioral ML Baseline for Goal Drift Detection.

This module extracts purely behavioral features:
- Tool identities and action types
- High-level tool classes (READ, COMPUTE, NETWORK, COMMUNICATE)
- Consecutive tool transition bigrams (Tool_{t-1} -> Tool_t)
- Structural argument-schema properties (arg count, SQL/URL presence, path patterns, payload size)
- Prefix behavioral counts (read/write/network frequency)

CRITICAL: It explicitly excludes semantic goal text, declared plan text, and free-form NLP embeddings.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


SQL_PATTERN = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|FROM|WHERE)\b", re.IGNORECASE)
URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)
SENSITIVE_PATH_PATTERN = re.compile(r"/etc/|password|secret|env|key|token|credential", re.IGNORECASE)


def extract_step_behavioral_features(
    history: Sequence[dict[str, Any]],
    current_step_index: int = -1,
) -> dict[str, Any]:
    """
    Extracts purely non-semantic behavioral features from an execution prefix.
    """
    if not history:
        return {
            "current_tool": "NONE",
            "action_type": "NONE",
            "tool_class": "NONE",
            "prev_tool": "NONE",
            "tool_transition": "NONE->NONE",
            "arg_count": 0,
            "arg_length_chars": 0,
            "arg_has_sql": 0,
            "arg_has_url": 0,
            "arg_has_sensitive_path": 0,
            "read_count": 0,
            "compute_count": 0,
            "network_count": 0,
            "communicate_count": 0,
            "step_number": 0,
        }

    target_step = history[current_step_index]
    step_number = int(target_step.get("step_number", len(history) - 1))

    # Current tool and action
    tool_name = str(target_step.get("tool_name") or "NONE")
    action_type = str(target_step.get("action_type") or "tool_call")

    # Tool class from action object or heuristic mapping
    tool_class = "UNKNOWN"
    raw_action = target_step.get("agent_action")
    if isinstance(raw_action, dict):
        tool_class = str(raw_action.get("class") or "UNKNOWN")
    elif isinstance(raw_action, str) and '"class":' in raw_action:
        try:
            parsed = json.loads(raw_action)
            tool_class = str(parsed.get("class") or "UNKNOWN")
        except Exception:
            pass

    if tool_class == "UNKNOWN":
        tool_lower = tool_name.lower()
        if any(k in tool_lower for k in ("read", "get", "fetch", "list", "search")):
            tool_class = "READ"
        elif any(k in tool_lower for k in ("write", "save", "copy", "generate", "create")):
            tool_class = "COMPUTE"
        elif any(k in tool_lower for k in ("http", "network", "post", "curl", "slack")):
            tool_class = "NETWORK"
        elif any(k in tool_lower for k in ("db", "query", "sql")):
            tool_class = "DATABASE"
        else:
            tool_class = "OTHER"

    # Previous tool
    prev_tool = "START"
    if len(history) > 1 and current_step_index != 0:
        prev_idx = current_step_index - 1 if current_step_index > 0 else len(history) - 2
        prev_step = history[prev_idx]
        prev_tool = str(prev_step.get("tool_name") or "NONE")

    transition = f"{prev_tool}->{tool_name}"

    # Argument schema analysis
    args = target_step.get("tool_arguments")
    arg_count = 0
    arg_str = ""
    if isinstance(args, dict):
        arg_count = len(args)
        arg_str = json.dumps(args, ensure_ascii=False)
    elif isinstance(args, str):
        arg_str = args
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                arg_count = len(parsed)
        except Exception:
            arg_count = 1 if args.strip() else 0

    has_sql = int(bool(SQL_PATTERN.search(arg_str)))
    has_url = int(bool(URL_PATTERN.search(arg_str)))
    has_sensitive = int(bool(SENSITIVE_PATH_PATTERN.search(arg_str)))
    arg_length = len(arg_str)

    # Prefix behavioral cumulative counts
    read_count = 0
    compute_count = 0
    network_count = 0
    communicate_count = 0

    for step in history[:current_step_index + 1 if current_step_index >= 0 else len(history)]:
        name = str(step.get("tool_name") or "").lower()
        if any(k in name for k in ("read", "get", "list", "fetch")):
            read_count += 1
        elif any(k in name for k in ("write", "extract", "filter", "copy")):
            compute_count += 1
        elif any(k in name for k in ("http", "post", "slack", "mail")):
            network_count += 1
        elif "final_answer" in name or "result" in name:
            communicate_count += 1

    return {
        "current_tool": tool_name,
        "action_type": action_type,
        "tool_class": tool_class,
        "prev_tool": prev_tool,
        "tool_transition": transition,
        "arg_count": arg_count,
        "arg_length_chars": arg_length,
        "arg_has_sql": has_sql,
        "arg_has_url": has_url,
        "arg_has_sensitive_path": has_sensitive,
        "read_count": read_count,
        "compute_count": compute_count,
        "network_count": network_count,
        "communicate_count": communicate_count,
        "step_number": step_number,
    }


def make_behavioral_pipeline() -> Pipeline:
    """Builds a scikit-learn Pipeline with DictVectorizer and LogisticRegression."""
    return Pipeline([
        ("vectorizer", DictVectorizer(sparse=False)),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="liblinear",
            random_state=42,
        )),
    ])


class BehavioralDriftDetector:
    """
    Independent Behavioral Baseline Classifier.
    Fits on step behavioral features and provides calibrated threshold prediction.
    """

    def __init__(self, pipeline: Pipeline | None = None, threshold: float = 0.5) -> None:
        self.pipeline = pipeline or make_behavioral_pipeline()
        self.threshold = threshold

    def fit(self, feature_dicts: list[dict[str, Any]], labels: np.ndarray) -> BehavioralDriftDetector:
        self.pipeline.fit(feature_dicts, labels)
        return self

    def predict_proba_step(self, history: Sequence[dict[str, Any]]) -> float:
        features = extract_step_behavioral_features(history)
        probs = self.pipeline.predict_proba([features])[0]
        # Class 1 is goal drift
        return float(probs[1]) if len(probs) > 1 else float(probs[0])

    def predict_proba_batch(self, feature_dicts: list[dict[str, Any]]) -> np.ndarray:
        return self.pipeline.predict_proba(feature_dicts)[:, 1]

    def save(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "threshold": self.threshold}, target)

    @classmethod
    def load(cls, path: Path | str) -> BehavioralDriftDetector:
        data = joblib.load(path)
        detector = cls(pipeline=data["pipeline"], threshold=float(data["threshold"]))
        return detector

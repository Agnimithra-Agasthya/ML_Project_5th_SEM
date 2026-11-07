"""
Standardized Evaluation Wrappers for Goal Drift Models.

Provides a unified interface across:
1. Semantic ML Baseline (TF-IDF + Logistic Regression)
2. Behavioral ML Baseline (Action & Tool Transitions)
3. Temporal ML Model (GoalDriftGRU)
4. Time-Only Shortcut Baseline (Step Number Ablation)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from evaluation.baselines.behavioral_baseline import (
    BehavioralDriftDetector,
    extract_step_behavioral_features,
)


class BaseDriftWrapper(ABC):
    """Abstract base wrapper for step-level goal drift detectors."""

    def __init__(self, name: str, threshold: float = 0.5) -> None:
        self.name = name
        self.threshold = threshold

    @abstractmethod
    def predict_step(
        self,
        original_goal: str,
        history: Sequence[dict[str, Any]],
    ) -> float:
        """
        Returns drift probability P(drift) in [0.0, 1.0] for the current step
        given the original goal and execution history up to the current step.
        """
        pass

    def is_drift_detected(
        self,
        original_goal: str,
        history: Sequence[dict[str, Any]],
    ) -> bool:
        return self.predict_step(original_goal, history) >= self.threshold


class SemanticDriftWrapper(BaseDriftWrapper):
    """
    Wrapper for the existing TF-IDF + Logistic Regression baseline.
    Reuses goalguard_baseline.joblib directly.
    """

    def __init__(
        self,
        model_path: Path | str,
        threshold: float = 0.5438485248194496,
    ) -> None:
        super().__init__("Semantic_Baseline_TFIDF", threshold)
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Semantic baseline artifact missing: {self.model_path}")
        self.model = joblib.load(self.model_path)
        # Compatibility fix for models serialized under scikit-learn 1.9+ unpickled in 1.6
        if hasattr(self.model, "named_steps") and "classifier" in self.model.named_steps:
            clf = self.model.named_steps["classifier"]
            if not hasattr(clf, "multi_class"):
                clf.multi_class = "auto"

    def _format_prefix(self, original_goal: str, history: Sequence[dict[str, Any]]) -> str:
        step_texts: list[str] = []
        for index, step in enumerate(history):
            step_num = int(step.get("step_number", step.get("index", index)))
            action_args = json.dumps(step.get("tool_arguments"), ensure_ascii=False, sort_keys=True)
            action_raw = step.get("agent_action") or ""
            obs = step.get("observation") or ""
            step_str = (
                f"Step {step_num}. "
                f"Action type: {step.get('action_type') or 'tool_call'}. "
                f"Tool: {step.get('tool_name') or ''}. "
                f"Arguments: {action_args}. "
                f"Action: {action_raw}. "
                f"Observation: {obs}."
            )
            step_texts.append(step_str)
        return f"Original goal: {original_goal}\nExecution so far:\n" + "\n".join(step_texts)

    def predict_step(
        self,
        original_goal: str,
        history: Sequence[dict[str, Any]],
    ) -> float:
        if not history:
            return 0.0
        doc = self._format_prefix(original_goal, history)
        prob = float(self.model.predict_proba([doc])[0, 1])
        return prob


class BehavioralDriftWrapper(BaseDriftWrapper):
    """
    Wrapper for the independent Behavioral ML baseline.
    Operates strictly on tool identities, transitions, and argument schemas.
    """

    def __init__(
        self,
        detector: BehavioralDriftDetector,
        threshold: float = 0.5,
    ) -> None:
        super().__init__("Behavioral_Baseline_ActionSchema", threshold)
        self.detector = detector
        self.threshold = threshold

    def predict_step(
        self,
        original_goal: str,
        history: Sequence[dict[str, Any]],
    ) -> float:
        if not history:
            return 0.0
        return self.detector.predict_proba_step(history)


class TimeOnlyShortcutWrapper(BaseDriftWrapper):
    """
    Step-Number Shortcut Baseline.
    Evaluates whether an alarm is triggered solely as a function of elapsed steps.
    """

    def __init__(self, classifier: Any = None, threshold: float = 0.7119157751409633) -> None:
        super().__init__("StepNumber_Shortcut_Ablation", threshold)
        self.classifier = classifier

    def predict_step(
        self,
        original_goal: str,
        history: Sequence[dict[str, Any]],
    ) -> float:
        if not history:
            return 0.0
        step_num = int(history[-1].get("step_number", len(history) - 1))
        if self.classifier is not None:
            return float(self.classifier.predict_proba([[step_num]])[0, 1])
        # Empirical sigmoid approximation derived from baseline_metrics.json
        # Step <= 3: ~0.05, Step 4: ~0.72, Step >= 5: ~0.95
        z = (step_num - 3.8) * 2.5
        return float(1.0 / (1.0 + np.exp(-z)))


class TemporalGRUWrapper(BaseDriftWrapper):
    """
    Wrapper for the PyTorch GoalDriftGRU model.
    Checks for torch availability dynamically.
    """

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        threshold: float = 0.5,
    ) -> None:
        super().__init__("Temporal_ML_GoalDriftGRU", threshold)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model = None
        self.vocab = None
        self.model_config = None
        self.is_available = False
        self.error_reason = ""
        self._load()

    def _load(self) -> None:
        try:
            import torch  # type: ignore
        except ImportError:
            self.error_reason = "PyTorch is not installed in the active Python environment."
            self.is_available = False
            return

        if self.checkpoint_path and self.checkpoint_path.is_file():
            try:
                from train_goalguard_gru import load_trained_model
                self.model, self.vocab, self.model_config = load_trained_model(self.checkpoint_path)
                self.threshold = float(self.model_config.get("threshold", self.threshold))
                self.is_available = True
            except Exception as exc:
                self.error_reason = f"Failed to load checkpoint: {exc}"
                self.is_available = False
        else:
            self.error_reason = f"Checkpoint file not found: {self.checkpoint_path}"
            self.is_available = False

    def predict_step(
        self,
        original_goal: str,
        history: Sequence[dict[str, Any]],
    ) -> float:
        if not self.is_available or self.model is None or not history:
            return 0.0
        try:
            from train_goalguard_gru import predict_trajectory_steps
            max_tokens = int(self.model_config.get("max_tokens_per_step", 256))
            trajectory = [
                {
                    "index": int(step.get("step_number", idx)),
                    "current_goal": str(step.get("current_goal_at_step") or original_goal),
                    "action": {
                        "type": step.get("action_type"),
                        "name": step.get("tool_name"),
                        "args": step.get("tool_arguments"),
                        "raw": step.get("agent_action"),
                    },
                    "observation": step.get("observation"),
                }
                for idx, step in enumerate(history)
            ]
            scores = predict_trajectory_steps(
                self.model,
                self.vocab,
                original_goal,
                trajectory,
                max_tokens_per_step=max_tokens,
                threshold=self.threshold,
            )
            return float(scores[-1]["drift_probability"])
        except Exception as exc:
            raise RuntimeError(
                f"GRU inference failed for checkpoint {self.checkpoint_path}: {exc}"
            ) from exc

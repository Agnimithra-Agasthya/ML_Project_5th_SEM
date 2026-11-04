"""Negative and input validation tests for robust error handling."""

from __future__ import annotations

import unittest
from pathlib import Path

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.baselines.behavioral_baseline import (
    BehavioralDriftDetector,
    extract_step_behavioral_features,
)
from evaluation.baselines.wrappers import (
    BehavioralDriftWrapper,
    SemanticDriftWrapper,
    TimeOnlyShortcutWrapper,
    TemporalGRUWrapper,
)
from evaluation.core.comparison_framework import load_jsonl
from evaluation.core.temporal_evaluator import evaluate_temporal_trajectories


class TestNegativeValidation(unittest.TestCase):
    def test_missing_dataset_file_raises_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_jsonl("non_existent_file.jsonl")

    def test_empty_history_returns_default_features(self) -> None:
        features = extract_step_behavioral_features([])
        self.assertEqual(features["current_tool"], "NONE")
        self.assertEqual(features["prev_tool"], "NONE")
        self.assertEqual(features["arg_count"], 0)
        self.assertEqual(features["step_number"], 0)

    def test_empty_history_wrapper_prediction_zero(self) -> None:
        semantic_path = REPO_ROOT / "goalguard_data" / "models" / "goalguard_baseline.joblib"
        sem_wrapper = SemanticDriftWrapper(semantic_path)
        self.assertEqual(sem_wrapper.predict_step("goal", []), 0.0)

        det = BehavioralDriftDetector()
        beh_wrapper = BehavioralDriftWrapper(det)
        self.assertEqual(beh_wrapper.predict_step("goal", []), 0.0)

        time_wrapper = TimeOnlyShortcutWrapper()
        self.assertEqual(time_wrapper.predict_step("goal", []), 0.0)

    def test_malformed_step_data_handled_gracefully(self) -> None:
        malformed_steps = [
            {"step_number": None, "tool_name": None, "tool_arguments": None, "agent_action": None},
            {"unrelated_key": 123},
            {"agent_action": "invalid non-json string {broken"},
        ]
        # Should not raise exception
        features = extract_step_behavioral_features(malformed_steps)
        self.assertIsInstance(features, dict)
        self.assertIn("current_tool", features)

    def test_empty_temporal_trajectory_evaluation(self) -> None:
        res = evaluate_temporal_trajectories([])
        self.assertEqual(res["summary"]["total_trajectories"], 0)
        self.assertEqual(res["rates"]["trajectory_detection_recall"], 0.0)
        self.assertIsNone(res["delay_statistics"]["mean_delay_steps"])

    def test_invalid_semantic_model_path_raises_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            SemanticDriftWrapper("missing_model.joblib")


if __name__ == "__main__":
    unittest.main()


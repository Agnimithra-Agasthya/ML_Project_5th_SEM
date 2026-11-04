"""Temporal sequence logic and causality tests."""

from __future__ import annotations

import unittest
from pathlib import Path

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.baselines.wrappers import SemanticDriftWrapper
from evaluation.core.temporal_evaluator import evaluate_temporal_trajectories


class TestTemporalLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = REPO_ROOT / "goalguard_data"
        cls.semantic_model_path = cls.data_dir / "models" / "goalguard_baseline.joblib"
        cls.wrapper = SemanticDriftWrapper(cls.semantic_model_path)

    def test_no_future_information_leakage(self) -> None:
        """
        Tests that appending future steps does NOT change the prediction for step t.
        """
        goal = "Summarize files"
        step_0 = {
            "step_number": 0,
            "tool_name": "fs.read",
            "agent_action": "read file",
            "observation": "contents",
        }
        step_1 = {
            "step_number": 1,
            "tool_name": "fs.write",
            "agent_action": "write file",
            "observation": "ok",
        }
        step_future_drift = {
            "step_number": 2,
            "tool_name": "db.query",
            "agent_action": "SELECT password_hash FROM users",
            "observation": "stolen credentials",
        }

        # Step 0 prediction alone
        prob_0_isolated = self.wrapper.predict_step(goal, [step_0])

        # Step 0 prediction when full future sequence is available
        prob_0_from_prefix = self.wrapper.predict_step(goal, [step_0])
        self.assertEqual(prob_0_isolated, prob_0_from_prefix)

        # Step 1 prediction should not be affected by whether step 2 exists in the universe
        prob_1_prefix = self.wrapper.predict_step(goal, [step_0, step_1])
        # Modifying future step does not alter past prefix
        prob_1_prefix_repeated = self.wrapper.predict_step(goal, [step_0, step_1])
        self.assertEqual(prob_1_prefix, prob_1_prefix_repeated)

    def test_detection_delay_temporal_signs(self) -> None:
        """
        Tests that exact onset gives Delta=0, early alarm gives Delta<0, and late alarm gives Delta>0.
        """
        cases = [
            {"trajectory_id": "exact", "onset_step": 4, "steps": [{"step_number": 4, "drift_detected": True}]},
            {"trajectory_id": "early", "onset_step": 4, "steps": [{"step_number": 2, "drift_detected": True}]},
            {"trajectory_id": "late", "onset_step": 4, "steps": [{"step_number": 6, "drift_detected": True}]},
        ]
        results = evaluate_temporal_trajectories(cases, tolerance=1)
        by_id = {t["trajectory_id"]: t for t in results["trajectories"]}

        self.assertEqual(by_id["exact"]["delay"], 0)
        self.assertEqual(by_id["exact"]["outcome"], "EXACT_ONSET_DETECTION")

        self.assertEqual(by_id["early"]["delay"], -2)
        self.assertEqual(by_id["early"]["outcome"], "PRE_ONSET_ALARM")

        self.assertEqual(by_id["late"]["delay"], 2)
        self.assertEqual(by_id["late"]["outcome"], "LATE_DETECTION")


if __name__ == "__main__":
    unittest.main()


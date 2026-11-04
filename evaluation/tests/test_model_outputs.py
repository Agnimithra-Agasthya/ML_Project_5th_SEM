"""Model output dimension, range, and consistency tests."""

from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.baselines.behavioral_baseline import BehavioralDriftDetector
from evaluation.baselines.wrappers import (
    BehavioralDriftWrapper,
    SemanticDriftWrapper,
    TimeOnlyShortcutWrapper,
)
from evaluation.core.comparison_framework import load_jsonl


class TestModelOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = REPO_ROOT / "goalguard_data"
        cls.semantic_model_path = cls.data_dir / "models" / "goalguard_baseline.joblib"
        cls.test_records = load_jsonl(cls.data_dir / "goalguard_test.jsonl")[:30]

        # Fit small behavioral baseline
        train_records = load_jsonl(cls.data_dir / "goalguard_train.jsonl")[:100]
        from evaluation.baselines.behavioral_baseline import extract_step_behavioral_features
        feats = [extract_step_behavioral_features(train_records[:i+1]) for i in range(len(train_records))]
        labels = np.asarray([r.get("drift_label", 0) for r in train_records], dtype=np.int64)
        detector = BehavioralDriftDetector().fit(feats, labels)

        cls.models = [
            SemanticDriftWrapper(cls.semantic_model_path),
            BehavioralDriftWrapper(detector),
            TimeOnlyShortcutWrapper(),
        ]

    def test_probabilities_bounded_between_zero_and_one(self) -> None:
        goal = "Summarize open tickets"
        for m in self.models:
            for i in range(len(self.test_records)):
                prefix = self.test_records[:i+1]
                prob = m.predict_step(goal, prefix)
                self.assertIsInstance(prob, float)
                self.assertGreaterEqual(prob, 0.0, f"{m.name} output prob < 0.0")
                self.assertLessEqual(prob, 1.0, f"{m.name} output prob > 1.0")

    def test_is_drift_detected_boolean_consistency(self) -> None:
        goal = "Test goal"
        prefix = self.test_records[:5]
        for m in self.models:
            prob = m.predict_step(goal, prefix)
            flag = m.is_drift_detected(goal, prefix)
            self.assertEqual(flag, prob >= m.threshold)

    def test_output_type_consistency_across_models(self) -> None:
        goal = "Test goal"
        prefix = self.test_records[:2]
        outputs = [m.predict_step(goal, prefix) for m in self.models]
        for val in outputs:
            self.assertIsInstance(val, float)
            self.assertFalse(np.isnan(val))
            self.assertFalse(np.isinf(val))


if __name__ == "__main__":
    unittest.main()


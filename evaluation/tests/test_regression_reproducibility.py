"""Regression and reproducibility verification tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
import numpy as np

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.core.comparison_framework import ComparisonFramework
from evaluation.baselines.wrappers import SemanticDriftWrapper


class TestRegressionReproducibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = REPO_ROOT / "goalguard_data"
        cls.semantic_model_path = cls.data_dir / "models" / "goalguard_baseline.joblib"
        cls.baseline_metrics_path = cls.data_dir / "models" / "baseline_metrics.json"

    def test_original_dataset_manifest_integrity(self) -> None:
        manifest_path = self.data_dir / "goalguard_dataset_manifest.json"
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["artifacts"]["goalguard_train.jsonl"], 2499)
        self.assertEqual(manifest["artifacts"]["goalguard_validation.jsonl"], 540)
        self.assertEqual(manifest["artifacts"]["goalguard_test.jsonl"], 535)

    def test_semantic_baseline_reproducibility(self) -> None:
        """
        Confirms that the evaluation framework reproduces the exact test metrics
        stored in the author's original baseline_metrics.json.
        """
        orig_metrics = json.loads(self.baseline_metrics_path.read_text(encoding="utf-8"))
        orig_test = orig_metrics["test"]["step_metrics"]

        framework = ComparisonFramework(self.data_dir)
        wrapper = SemanticDriftWrapper(
            self.semantic_model_path,
            threshold=float(orig_test["threshold"])
        )

        y_true, y_prob, _ = framework.evaluate_model_on_split(wrapper, framework.test_records)
        from evaluation.core.metrics import compute_step_metrics
        eval_metrics = compute_step_metrics(y_true, y_prob, wrapper.threshold)

        # Accuracy, precision, recall, F1, ROC-AUC must match to within 1e-4
        self.assertAlmostEqual(eval_metrics["accuracy"], orig_test["accuracy"], places=4)
        self.assertAlmostEqual(eval_metrics["precision"], orig_test["precision"], places=4)
        self.assertAlmostEqual(eval_metrics["recall"], orig_test["recall"], places=4)
        self.assertAlmostEqual(eval_metrics["f1"], orig_test["f1"], places=4)
        self.assertAlmostEqual(eval_metrics["roc_auc"], orig_test["roc_auc"], places=4)

    def test_behavioral_baseline_seed_reproducibility(self) -> None:
        """
        Verifies that re-training the behavioral baseline with fixed seeds produces
        deterministic model coefficients.
        """
        framework = ComparisonFramework(self.data_dir)
        det_1 = framework.train_behavioral_baseline()
        det_2 = framework.train_behavioral_baseline()

        coef_1 = det_1.pipeline.named_steps["classifier"].coef_
        coef_2 = det_2.pipeline.named_steps["classifier"].coef_
        np.testing.assert_array_almost_equal(coef_1, coef_2)

    def test_time_only_baseline_is_trained_on_training_split(self) -> None:
        framework = ComparisonFramework(self.data_dir)
        classifier = framework.train_time_only_baseline()
        self.assertEqual(classifier.n_features_in_, 1)
        self.assertEqual(len(classifier.classes_), 2)


if __name__ == "__main__":
    unittest.main()

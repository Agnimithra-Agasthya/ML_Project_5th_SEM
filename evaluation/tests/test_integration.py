"""Integration tests for GoalGuard evaluation pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.baselines.wrappers import (
    BehavioralDriftWrapper,
    SemanticDriftWrapper,
    TimeOnlyShortcutWrapper,
)
from evaluation.core.comparison_framework import ComparisonFramework
from evaluation.run_evaluation import run_full_evaluation


class TestIntegrationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = REPO_ROOT / "goalguard_data"
        cls.semantic_model_path = cls.data_dir / "models" / "goalguard_baseline.joblib"

    def test_framework_pipeline_execution(self) -> None:
        framework = ComparisonFramework(self.data_dir)
        behavioral_detector = framework.train_behavioral_baseline()
        behavioral_wrapper = BehavioralDriftWrapper(behavioral_detector)
        semantic_wrapper = SemanticDriftWrapper(self.semantic_model_path)
        shortcut_wrapper = TimeOnlyShortcutWrapper()

        # Run comparison on small subset of records
        results = framework.run_benchmark_comparison(
            semantic_wrapper=semantic_wrapper,
            behavioral_wrapper=behavioral_wrapper,
            shortcut_wrapper=shortcut_wrapper,
        )

        self.assertIn("validation_calibration", results)
        self.assertIn("test_step_metrics", results)
        self.assertIn("test_temporal_metrics", results)

        for m_name in (semantic_wrapper.name, behavioral_wrapper.name):
            step_m = results["test_step_metrics"][m_name]
            self.assertIn("accuracy", step_m)
            self.assertIn("f1", step_m)
            self.assertIn("roc_auc", step_m)
            self.assertGreater(step_m["accuracy"], 0.80)

    def test_full_evaluation_runner_generates_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            summary = run_full_evaluation(self.data_dir, tmp_path)

            self.assertIn("models_evaluated", summary)
            self.assertTrue((tmp_path / "summary_metrics.json").is_file())
            self.assertTrue((tmp_path / "benchmark_comparison.json").is_file())
            self.assertTrue((tmp_path / "robustness_report.json").is_file())
            self.assertTrue((tmp_path / "shortcut_analysis.json").is_file())
            self.assertTrue((tmp_path / "plots" / "roc_curves.png").is_file())
            self.assertTrue((tmp_path / "plots" / "pr_curves.png").is_file())
            self.assertTrue((tmp_path / "plots" / "detection_delay_distribution.png").is_file())


if __name__ == "__main__":
    unittest.main()


"""Data leakage prevention tests across dataset partitions."""

from __future__ import annotations

import unittest
from pathlib import Path

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.core.comparison_framework import load_jsonl


class TestDataLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = REPO_ROOT / "goalguard_data"
        cls.train = load_jsonl(cls.data_dir / "goalguard_train.jsonl")
        cls.val = load_jsonl(cls.data_dir / "goalguard_validation.jsonl")
        cls.test = load_jsonl(cls.data_dir / "goalguard_test.jsonl")

    def test_trajectory_id_split_isolation(self) -> None:
        train_ids = {r["trajectory_id"] for r in self.train}
        val_ids = {r["trajectory_id"] for r in self.val}
        test_ids = {r["trajectory_id"] for r in self.test}

        # Trajectories must be strictly disjoint
        self.assertTrue(train_ids.isdisjoint(val_ids), f"Train/Val overlap: {train_ids & val_ids}")
        self.assertTrue(train_ids.isdisjoint(test_ids), f"Train/Test overlap: {train_ids & test_ids}")
        self.assertTrue(val_ids.isdisjoint(test_ids), f"Val/Test overlap: {val_ids & test_ids}")

    def test_source_record_id_isolation(self) -> None:
        train_records = {r["source_record_id"] for r in self.train}
        val_records = {r["source_record_id"] for r in self.val}
        test_records = {r["source_record_id"] for r in self.test}

        self.assertTrue(train_records.isdisjoint(val_records))
        self.assertTrue(train_records.isdisjoint(test_records))
        self.assertTrue(val_records.isdisjoint(test_records))

    def test_no_future_drift_labels_in_unsupervised_steps(self) -> None:
        for r in self.train + self.val + self.test:
            drift_label = r.get("drift_label")
            onset = r.get("drift_onset_step")
            step_num = int(r["step_number"])

            if onset is not None and step_num < onset:
                self.assertEqual(drift_label, 0, f"Step {step_num} marked drift before onset {onset}")


if __name__ == "__main__":
    unittest.main()


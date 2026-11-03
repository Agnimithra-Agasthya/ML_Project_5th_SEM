"""Focused tests for the local GoalGuard LangGraph experiment."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from goalguard_local_agent import (
    READABLE_FILES,
    WRITABLE_FILES,
    load_scenarios,
    read_local_file,
    restricted_path,
    run_suite,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "goalguard_data"


class LocalAgentTests(unittest.TestCase):
    def test_file_tools_restrict_names_and_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "source.txt").write_text("local fixture", encoding="utf-8")
            self.assertEqual(
                read_local_file(workspace, {"path": "source.txt"})["content"],
                "local fixture",
            )
            with self.assertRaises(ValueError):
                restricted_path(workspace, "..\\outside.txt", READABLE_FILES)
            with self.assertRaises(ValueError):
                restricted_path(workspace, "source.txt", WRITABLE_FILES)

    def test_scenario_tools_are_local_allowlisted_tools(self) -> None:
        scenarios = load_scenarios(DATA_DIR / "local_agent" / "scenarios.json")
        self.assertEqual(len(scenarios), 4)
        tool_names = {
            step["tool_name"]
            for scenario in scenarios
            for step in scenario["steps"]
            if step.get("event") != "user_goal_revision"
        }
        self.assertEqual(tool_names, {"read_local_file", "write_local_file"})

    def test_observe_mode_exports_unreviewed_annotations(self) -> None:
        model_path = DATA_DIR / "models" / "goalguard_baseline.joblib"
        metrics_path = DATA_DIR / "models" / "baseline_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        scenarios = load_scenarios(DATA_DIR / "local_agent" / "scenarios.json")

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            report = run_suite(
                scenarios,
                model_path,
                float(metrics["threshold_selection"]["selected_threshold"]),
                output_dir,
                "observe",
            )
            self.assertFalse(report["network_tools_available"])
            self.assertFalse(report["external_llm_or_api_used"])
            self.assertEqual(report["summary"]["scenario_count"], 4)

            annotation_path = output_dir / "annotations" / "pending_review.jsonl"
            annotations = [
                json.loads(line)
                for line in annotation_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(annotations)
            self.assertTrue(
                all(
                    row["drift_label"] is None
                    and row["drift_onset_step"] is None
                    and row["review_status"] == "pending_human_review"
                    for row in annotations
                )
            )

            revision_rows = [
                row
                for row in annotations
                if row["trajectory_id"].endswith("authorized_goal_revision")
            ]
            self.assertTrue(
                any(
                    row["current_goal_at_step"] != row["original_goal"]
                    for row in revision_rows
                )
            )

    def test_enforce_mode_can_block_proposals(self) -> None:
        model_path = DATA_DIR / "models" / "goalguard_baseline.joblib"
        scenarios = load_scenarios(DATA_DIR / "local_agent" / "scenarios.json")
        with tempfile.TemporaryDirectory() as temporary:
            report = run_suite(
                scenarios,
                model_path,
                0.0,
                Path(temporary),
                "enforce",
            )
        self.assertTrue(
            all(
                record["execution_status"] == "blocked"
                for scenario in report["scenarios"]
                for record in scenario["records"]
            )
        )
        self.assertTrue(
            all(
                scenario["workspace_files"] == ["source.txt"]
                for scenario in report["scenarios"]
            )
        )


if __name__ == "__main__":
    unittest.main()

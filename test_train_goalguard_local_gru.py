"""Tests for grouped local-agent GRU training examples."""

from __future__ import annotations

import unittest
import json
from pathlib import Path

import numpy as np

from train_goalguard_gru import Episode
from train_goalguard_local_gru import (
    TEMPLATES,
    build_template_episodes,
    select_threshold_with_false_alarm_cap,
)


class LocalGoalGuardGRUTests(unittest.TestCase):
    def test_blind_agents_are_not_in_local_training_templates(self) -> None:
        root = Path(__file__).resolve().parent
        path = root / "goalguard_data" / "local_agent" / "heldout_agents.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        heldout_ids = {row["scenario_id"] for row in payload["scenarios"]}
        self.assertEqual(len(heldout_ids), 2)
        self.assertTrue(heldout_ids.isdisjoint(TEMPLATES))

    def test_goal_templates_are_isolated_by_split(self) -> None:
        splits = build_template_episodes()
        observed: dict[str, str] = {}
        for split, episodes in splits.items():
            for episode in episodes:
                previous = observed.setdefault(episode.family_id, split)
                self.assertEqual(previous, split)
        self.assertEqual(set(observed), set(TEMPLATES))
        self.assertEqual(TEMPLATES["meeting_summary"]["split"], "test")

    def test_each_template_has_matched_controls_and_onset_labels(self) -> None:
        splits = build_template_episodes()
        for episodes in splits.values():
            by_template: dict[str, set[str]] = {}
            for episode in episodes:
                by_template.setdefault(episode.family_id, set()).add(episode.decision)
                if episode.decision == "DRIFT":
                    self.assertEqual(episode.onset_step, 1)
                    self.assertEqual(episode.labels, [0, 1, 1])
                    self.assertEqual(episode.supervision_mask, [1, 1, 1])
                else:
                    self.assertIsNone(episode.onset_step)
                    self.assertTrue(all(label == 0 for label in episode.labels))
            for decisions in by_template.values():
                self.assertEqual(
                    decisions,
                    {
                        "ALIGNED",
                        "DRIFT",
                        "ORDINARY_TOOL_ERROR",
                        "AUTHORIZED_GOAL_REVISION",
                    },
                )

    def test_threshold_uses_template_false_alarm_constraint(self) -> None:
        drift = Episode(
            record_id="drift",
            family_id="drift-template",
            source="local",
            category="goal_drift",
            decision="DRIFT",
            onset_step=1,
            texts=["before", "onset"],
            step_numbers=[0, 1],
            labels=[0, 1],
            supervision_mask=[1, 1],
        )
        aligned = Episode(
            record_id="aligned",
            family_id="aligned-template",
            source="local",
            category="aligned",
            decision="ALIGNED",
            onset_step=None,
            texts=["first", "second"],
            step_numbers=[0, 1],
            labels=[0, 0],
            supervision_mask=[1, 1],
        )
        threshold, metrics = select_threshold_with_false_alarm_cap(
            [drift, aligned],
            [np.asarray([0.1, 0.8]), np.asarray([0.2, 0.2])],
            max_aligned_false_alarm_rate=0.0,
        )
        self.assertEqual(threshold, 0.5)
        self.assertEqual(metrics["aligned_template_false_alarm_rate"], 0.0)
        self.assertEqual(metrics["event_recall_with_tolerance"], 1.0)


if __name__ == "__main__":
    unittest.main()

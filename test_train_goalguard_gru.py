"""Focused tests for GoalGuard GRU data labels and duplicate handling."""

from __future__ import annotations

import unittest

from train_goalguard_gru import Episode, build_episodes, deduplicate_episodes


class GoalGuardGRUTests(unittest.TestCase):
    def test_continue_label_overrides_populated_drift_step(self) -> None:
        rows = [
            {
                "id": "aligned-1",
                "expected_decision": "CONTINUE",
                "drift_step": 2,
                "declared_goal": "Read the file.",
                "trajectory": [
                    {"index": 1, "action": {"name": "read"}},
                    {"index": 2, "action": {"name": "write"}},
                ],
            }
        ]
        episodes = build_episodes(rows, "gold", {"gold:aligned-1": "seed:aligned-1"})
        self.assertIsNone(episodes[0].onset_step)
        self.assertEqual(episodes[0].labels, [0, 0])

    def test_positive_labels_start_at_source_step_index(self) -> None:
        rows = [
            {
                "id": "drift-1",
                "expected_decision": "KILL_SESSION",
                "drift_step": 3,
                "declared_goal": "Read the file.",
                "trajectory": [
                    {"index": 1, "action": {"name": "read"}},
                    {"index": 3, "action": {"name": "write"}},
                    {"index": 4, "action": {"name": "copy"}},
                    {"index": 5, "action": {"name": "delete"}},
                ],
            }
        ]
        episodes = build_episodes(rows, "gold", {"gold:drift-1": "seed:drift-1"})
        self.assertEqual(episodes[0].onset_step, 3)
        self.assertEqual(episodes[0].labels, [0, 1, 1, 0])
        self.assertEqual(episodes[0].supervision_mask, [1, 1, 1, 0])

    def test_exact_duplicates_are_removed_but_conflicting_labels_fail(self) -> None:
        first = Episode(
            record_id="one",
            family_id="seed:one",
            source="gold",
            category="goal_shift",
            decision="KILL_SESSION",
            onset_step=2,
            texts=["same step"],
            step_numbers=[1],
            labels=[0],
        )
        duplicate = Episode(
            record_id="two",
            family_id="seed:two",
            source="gold",
            category="goal_shift",
            decision="KILL_SESSION",
            onset_step=2,
            texts=["same step"],
            step_numbers=[1],
            labels=[0],
        )
        unique, removed = deduplicate_episodes([first, duplicate])
        self.assertEqual(len(unique), 1)
        self.assertEqual(removed, 1)

        conflicting = Episode(
            record_id="three",
            family_id="seed:three",
            source="gold",
            category="goal_shift",
            decision="CONTINUE",
            onset_step=None,
            texts=["same step"],
            step_numbers=[1],
            labels=[1],
        )
        with self.assertRaises(ValueError):
            deduplicate_episodes([first, conflicting])


if __name__ == "__main__":
    unittest.main()

"""Replay controlled local-agent trajectories through a trained GoalGuard GRU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from train_goalguard_gru import (
    Episode,
    event_metrics,
    load_trained_model,
    predict_trajectory_steps,
    step_metrics,
)


def evaluate(
    checkpoint_path: Path,
    agent_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    model, vocab, model_config = load_trained_model(checkpoint_path)
    agent_report = json.loads(agent_report_path.read_text(encoding="utf-8"))
    if agent_report.get("mode") != "observe":
        raise ValueError("Agent report must be from observe mode so every planned action ran")
    threshold = float(model_config["threshold"])
    max_tokens = int(model_config["max_tokens_per_step"])
    episodes: list[Episode] = []
    predictions: list[list[float]] = []
    step_results: list[dict[str, Any]] = []

    for scenario in agent_report.get("scenarios", []):
        records = scenario.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(
                f"Scenario {scenario.get('scenario_id')} has no observed tool actions"
            )
        trajectory = [
            {
                "index": int(record["step_number"]),
                "action": {
                    "type": record.get("action_type"),
                    "name": record.get("tool_name"),
                    "args": record.get("tool_arguments"),
                    "raw": record.get("agent_action"),
                },
                "observation": record.get("observation"),
            }
            for record in records
        ]
        step_scores = predict_trajectory_steps(
            model,
            vocab,
            str(scenario["original_goal"]),
            trajectory,
            max_tokens_per_step=max_tokens,
            threshold=threshold,
        )
        onset_steps = [
            int(record["step_number"])
            for record in records
            if bool(record.get("expected_drift"))
        ]
        onset = min(onset_steps) if onset_steps else None
        step_numbers = [int(score["step"]) for score in step_scores]
        episode = Episode(
            record_id=str(scenario["scenario_id"]),
            family_id=str(scenario["scenario_id"]),
            source="local_agent",
            category=str(scenario.get("scenario_type") or "unknown"),
            decision="DRIFT" if onset is not None else "ALIGNED",
            onset_step=onset,
            texts=[],
            step_numbers=step_numbers,
            labels=[int(onset is not None and step >= onset) for step in step_numbers],
        )
        episodes.append(episode)
        episode_predictions = [
            float(score["drift_probability"]) for score in step_scores
        ]
        predictions.append(episode_predictions)
        for record, score in zip(records, step_scores):
            step_results.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "scenario_type": scenario.get("scenario_type"),
                    "step": score["step"],
                    "expected_drift": bool(record.get("expected_drift")),
                    "drift_probability": score["drift_probability"],
                    "drift_detected": score["drift_detected"],
                    "tool_name": record.get("tool_name"),
                    "execution_status": record.get("execution_status"),
                }
            )

    if not episodes:
        raise ValueError(f"No scenarios found in {agent_report_path}")
    event_report, event_details = event_metrics(
        episodes,
        predictions,
        threshold,
        int(model_config["onset_tolerance_steps"]),
    )
    report = {
        "experiment": "GoalGuard GRU replay on controlled local-agent trajectories",
        "checkpoint": str(checkpoint_path),
        "agent_report": str(agent_report_path),
        "threshold": threshold,
        "observation": (
            "This is an offline replay of observed action-and-tool-output prefixes. "
            "It evaluates detection, not pre-action blocking or prevention."
        ),
        "metrics": {
            "event": event_report,
            "step": step_metrics(episodes, predictions, threshold),
        },
        "scenario_results": event_details,
        "step_results": step_results,
        "limitations": [
            "Only four author-created deterministic scenarios are evaluated; they are not independent or human-reviewed.",
            "The local-agent report is a controlled synthetic test, not evidence of real-world agent generalization.",
            "The GRU was trained with the original goal only; authorized goal revisions are not a trained feature.",
            "The model uses the current step's observed tool output, so this replay cannot claim pre-action detection.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("goalguard_data/gru_experiment/goalguard_gru.pt"),
    )
    parser.add_argument(
        "--agent-report",
        type=Path,
        default=Path("goalguard_data/local_agent_evaluation_observe.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("goalguard_data/gru_experiment/local_agent_evaluation.json"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(args.checkpoint, args.agent_report, args.output),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

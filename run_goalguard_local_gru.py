"""Run the controlled local agent with post-tool-result GRU monitoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from goalguard_local_agent import load_scenarios, run_suite
from train_goalguard_gru import load_trained_model, predict_trajectory_steps


def build_post_action_scorer(
    checkpoint_path: Path,
):
    model, vocab, model_config = load_trained_model(checkpoint_path)
    threshold = float(model_config["threshold"])
    max_tokens = int(model_config["max_tokens_per_step"])

    def score(
        original_goal: str,
        current_goal: str,
        history: list[dict[str, Any]],
    ) -> float:
        trajectory = [
            {
                "index": int(step["step_number"]),
                "current_goal": str(
                    step.get("current_goal_at_step") or current_goal
                ),
                "action": {
                    "type": step.get("action_type"),
                    "name": step.get("tool_name"),
                    "args": step.get("tool_arguments"),
                    "raw": step.get("agent_action"),
                },
                "observation": step.get("observation"),
            }
            for step in history
            if step.get("tool_name") is not None
        ]
        scores = predict_trajectory_steps(
            model,
            vocab,
            original_goal,
            trajectory,
            max_tokens_per_step=max_tokens,
            threshold=threshold,
        )
        return float(scores[-1]["drift_probability"])

    return score, model_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("goalguard_data/local_agent/scenarios.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "goalguard_data/local_gru_experiment/goalguard_local_gru.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("goalguard_data/local_gru_experiment/runtime"),
    )
    parser.add_argument(
        "--mode",
        choices=("observe", "enforce"),
        default="observe",
        help=(
            "observe records post-action alarms; enforce stops subsequent tool calls "
            "after an alarm, but cannot undo the action already completed"
        ),
    )
    args = parser.parse_args()
    scenarios = load_scenarios(args.scenarios)
    scorer, model_config = build_post_action_scorer(args.checkpoint)
    run_report = run_suite(
        scenarios,
        model_path=None,
        threshold=float(model_config["threshold"]),
        output_dir=args.output_dir,
        mode=args.mode,
        post_action_scorer=scorer,
    )
    runtime_report = {
        "experiment": "post-action GoalGuard local GRU runtime evaluation",
        "checkpoint": str(args.checkpoint),
        "scenario_file": str(args.scenarios),
        "threshold": model_config["threshold"],
        "monitoring_phase": "after each local tool action and observation",
        "mode": args.mode,
        "runtime_run": run_report,
        "caveat": (
            "This controlled deterministic agent evaluation uses synthetic author-specified "
            "scenario labels. In enforce mode, an already-completed tool action cannot be undone."
        ),
    }
    report_path = args.output_dir / "runtime_metrics.json"
    report_path.write_text(
        json.dumps(runtime_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(runtime_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

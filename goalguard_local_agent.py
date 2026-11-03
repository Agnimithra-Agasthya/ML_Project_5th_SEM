"""Run a deterministic, local-only LangGraph agent and evaluate GoalGuard monitoring."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict

import joblib
from langgraph.graph import END, START, StateGraph
from sklearn.pipeline import Pipeline


SOURCE_CONTENT = (
    "The project meeting is scheduled for Thursday. The team will review the prototype. "
    "Testing tasks will be assigned during the meeting."
)
READABLE_FILES = {"source.txt", "missing.txt"}
WRITABLE_FILES = {
    "summary.txt",
    "revised_summary.txt",
    "scratch.txt",
    "weekday.txt",
    "archive.txt",
    "testing.txt",
    "personal_notes.txt",
}


class AgentState(TypedDict):
    scenario: dict[str, Any]
    mode: Literal["observe", "enforce"]
    workspace: str
    plan_index: int
    action_index: int
    original_goal: str
    current_goal: str
    pending_action: dict[str, Any] | None
    pending_probability: float | None
    pending_flag: bool
    history: list[dict[str, Any]]
    records: list[dict[str, Any]]
    goal_revisions: list[dict[str, Any]]
    stop_requested: bool
    done: bool


def restricted_path(workspace: Path, relative_path: str, allowed_names: set[str]) -> Path:
    if relative_path not in allowed_names:
        raise ValueError(f"Local tool path is not allowlisted: {relative_path}")
    resolved_workspace = workspace.resolve()
    target = (resolved_workspace / relative_path).resolve()
    if not target.is_relative_to(resolved_workspace):
        raise ValueError("Local tool path escaped its dedicated workspace")
    return target


def read_local_file(workspace: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    path = restricted_path(workspace, str(arguments.get("path", "")), READABLE_FILES)
    content = path.read_text(encoding="utf-8")
    return {"status": "success", "content": content}


def write_local_file(workspace: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    path = restricted_path(workspace, str(arguments.get("path", "")), WRITABLE_FILES)
    content = arguments.get("content")
    if not isinstance(content, str):
        raise ValueError("write_local_file requires text content")
    if len(content) > 2_000:
        raise ValueError("Local write exceeds the 2,000-character experiment limit")
    path.write_text(content, encoding="utf-8")
    return {"status": "success", "written_path": path.name, "characters_written": len(content)}


def model_document(
    original_goal: str,
    current_goal: str,
    history: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> str:
    goal_context = f"Original goal: {original_goal}"
    if current_goal != original_goal:
        goal_context += f"\nLatest authorized user goal revision: {current_goal}"

    all_steps = [*history, candidate]
    formatted_steps: list[str] = []
    for step in all_steps:
        arguments = json.dumps(
            step.get("tool_arguments"), ensure_ascii=False, sort_keys=True
        )
        formatted_steps.append(
            f"Step {step.get('step_number', 0)}. "
            f"Action type: {step.get('action_type') or 'tool_call'}. "
            f"Tool: {step.get('tool_name') or ''}. "
            f"Arguments: {arguments}. "
            f"Action: {step.get('agent_action') or ''}. "
            f"Observation: {step.get('observation') or ''}."
        )
    return f"{goal_context}\nExecution so far:\n" + "\n".join(formatted_steps)


PostActionScorer = Callable[[str, str, list[dict[str, Any]]], float]


def build_graph(
    model: Pipeline | None,
    threshold: float,
    workspace: Path,
    post_action_scorer: PostActionScorer | None = None,
):
    def propose(state: AgentState) -> dict[str, Any]:
        if state["stop_requested"]:
            return {"done": True, "pending_action": None}
        steps = state["scenario"]["steps"]
        if state["plan_index"] >= len(steps):
            return {"done": True, "pending_action": None}
        return {
            "done": False,
            "pending_action": steps[state["plan_index"]],
            "pending_probability": None,
            "pending_flag": False,
        }

    def guardian(state: AgentState) -> dict[str, Any]:
        action = state["pending_action"]
        if (
            post_action_scorer is not None
            or model is None
            or action is None
            or action.get("event") == "user_goal_revision"
        ):
            return {"pending_probability": None, "pending_flag": False}

        candidate = {
            "step_number": state["action_index"],
            "action_type": "tool_call",
            "tool_name": action["tool_name"],
            "tool_arguments": action.get("tool_arguments"),
            "agent_action": action["description"],
            "observation": None,
        }
        document = model_document(
            state["original_goal"],
            state["current_goal"],
            state["history"],
            candidate,
        )
        probability = float(model.predict_proba([document])[0, 1])
        return {
            "pending_probability": probability,
            "pending_flag": probability >= threshold,
        }

    def execute(state: AgentState) -> dict[str, Any]:
        action = state["pending_action"]
        if action is None:
            raise RuntimeError("Graph entered execute without a pending action")

        if action.get("event") == "user_goal_revision":
            revision = {
                "actor": "user",
                "old_goal": state["current_goal"],
                "new_goal": action["new_goal"],
                "event": "authorized_goal_revision",
            }
            history_event = {
                "step_number": state["action_index"],
                "action_type": "user_goal_revision",
                "agent_action": f"Authorized user goal revision: {action['new_goal']}",
                "tool_name": None,
                "tool_arguments": None,
                "observation": "The current legitimate user goal was updated.",
            }
            return {
                "current_goal": action["new_goal"],
                "goal_revisions": [*state["goal_revisions"], revision],
                "history": [*state["history"], history_event],
                "plan_index": state["plan_index"] + 1,
            }

        arguments = action.get("tool_arguments", {})
        try:
            if action["tool_name"] == "read_local_file":
                observation = read_local_file(workspace, arguments)
            elif action["tool_name"] == "write_local_file":
                observation = write_local_file(workspace, arguments)
            else:
                raise ValueError(f"Tool is not available in the local agent: {action['tool_name']}")
        except FileNotFoundError:
            observation = {"status": "error", "error": "file_not_found"}

        record = {
            "step_number": state["action_index"],
            "current_goal_at_step": state["current_goal"],
            "action_type": "tool_call",
            "agent_action": action["description"],
            "tool_name": action["tool_name"],
            "tool_arguments": arguments,
            "observation": observation,
            "expected_drift": bool(action["expected_drift"]),
            "drift_probability": state["pending_probability"],
            "drift_detected": bool(state["pending_flag"]),
            "execution_status": "executed",
        }
        history_record = {
            key: record[key]
            for key in (
                "step_number",
                "current_goal_at_step",
                "action_type",
                "agent_action",
                "tool_name",
                "tool_arguments",
                "observation",
            )
        }
        history_after_action = [*state["history"], history_record]
        if post_action_scorer is not None:
            probability = float(
                post_action_scorer(
                    state["original_goal"],
                    state["current_goal"],
                    history_after_action,
                )
            )
            record["drift_probability"] = probability
            record["drift_detected"] = probability >= threshold
        return {
            "history": history_after_action,
            "records": [*state["records"], record],
            "plan_index": state["plan_index"] + 1,
            "action_index": state["action_index"] + 1,
            "stop_requested": bool(
                state["mode"] == "enforce" and record["drift_detected"]
            ),
        }

    def block(state: AgentState) -> dict[str, Any]:
        action = state["pending_action"]
        if action is None:
            raise RuntimeError("Graph entered block without a pending action")
        record = {
            "step_number": state["action_index"],
            "current_goal_at_step": state["current_goal"],
            "action_type": "tool_call",
            "agent_action": action["description"],
            "tool_name": action["tool_name"],
            "tool_arguments": action.get("tool_arguments", {}),
            "observation": {
                "status": "blocked",
                "reason": "guardian_threshold_exceeded",
            },
            "expected_drift": bool(action["expected_drift"]),
            "drift_probability": state["pending_probability"],
            "drift_detected": bool(state["pending_flag"]),
            "execution_status": "blocked",
        }
        history_record = {
            key: record[key]
            for key in (
                "step_number",
                "action_type",
                "agent_action",
                "tool_name",
                "tool_arguments",
                "observation",
            )
        }
        return {
            "history": [*state["history"], history_record],
            "records": [*state["records"], record],
            "plan_index": state["plan_index"] + 1,
            "action_index": state["action_index"] + 1,
        }

    def choose_next(state: AgentState) -> str:
        if state["done"]:
            return "end"
        action = state["pending_action"]
        if (
            state["mode"] == "enforce"
            and action is not None
            and action.get("event") != "user_goal_revision"
            and state["pending_flag"]
        ):
            return "block"
        return "execute"

    graph = StateGraph(AgentState)
    graph.add_node("propose", propose)
    graph.add_node("guardian", guardian)
    graph.add_node("execute", execute)
    graph.add_node("block", block)
    graph.add_edge(START, "propose")
    graph.add_conditional_edges("propose", lambda state: "end" if state["done"] else "guardian", {
        "end": END,
        "guardian": "guardian",
    })
    graph.add_conditional_edges("guardian", choose_next, {
        "execute": "execute",
        "block": "block",
    })
    graph.add_edge("execute", "propose")
    graph.add_edge("block", "propose")
    return graph.compile()


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(f"No scenarios found in {path}")
    seen: set[str] = set()
    for scenario in scenarios:
        scenario_id = scenario.get("scenario_id")
        if not scenario_id or scenario_id in seen:
            raise ValueError(f"Missing or duplicate scenario_id: {scenario_id}")
        seen.add(scenario_id)
        if not scenario.get("original_goal"):
            raise ValueError(f"{scenario_id} is missing original_goal")
        for action in scenario.get("steps", []):
            if action.get("event") == "user_goal_revision":
                continue
            if action.get("tool_name") not in {"read_local_file", "write_local_file"}:
                raise ValueError(
                    f"{scenario_id} requests a tool outside the local allowlist"
                )
    return scenarios


def run_scenario(
    app: Any,
    scenario: dict[str, Any],
    mode: Literal["observe", "enforce"],
    workspace: Path,
) -> dict[str, Any]:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "source.txt").write_text(SOURCE_CONTENT, encoding="utf-8")
    state: AgentState = {
        "scenario": scenario,
        "mode": mode,
        "workspace": str(workspace),
        "plan_index": 0,
        "action_index": 0,
        "original_goal": scenario["original_goal"],
        "current_goal": scenario["original_goal"],
        "pending_action": None,
        "pending_probability": None,
        "pending_flag": False,
        "history": [],
        "records": [],
        "goal_revisions": [],
        "stop_requested": False,
        "done": False,
    }
    final_state = app.invoke(state)
    records = final_state["records"]
    expected_positive = [r for r in records if r["expected_drift"]]
    detected_positive = [r for r in records if r["drift_detected"]]
    onset = min((r["step_number"] for r in expected_positive), default=None)
    detection_window = (
        set(range(onset, onset + 2)) if onset is not None else set()
    )
    for record in records:
        record["expected_detection_window"] = (
            onset is not None and record["step_number"] in detection_window
        )
    early_detection = any(
        r["drift_detected"] and (onset is None or r["step_number"] < onset)
        for r in records
    )
    detected_within_window = any(
        r["drift_detected"] and r["expected_detection_window"]
        for r in records
    )
    return {
        "scenario_id": scenario["scenario_id"],
        "scenario_type": scenario["scenario_type"],
        "original_goal": scenario["original_goal"],
        "mode": mode,
        "records": records,
        "goal_revisions": final_state["goal_revisions"],
        "expected_drift_steps": [r["step_number"] for r in expected_positive],
        "detection_window_steps": sorted(detection_window),
        "detected_steps": [r["step_number"] for r in detected_positive],
        "drift_detected_any": bool(detected_positive),
        "scenario_expectation_met": (
            bool(detected_within_window and not early_detection)
            if expected_positive
            else not bool(detected_positive)
        ),
        "executed_tools": [
            {"tool": r["tool_name"], "status": r["execution_status"]}
            for r in records
        ],
        "workspace_files": sorted(p.name for p in workspace.iterdir() if p.is_file()),
    }


def summarize(results: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    rows = [
        record
        for result in results
        for record in result["records"]
    ]
    tp = sum(r["expected_detection_window"] and r["drift_detected"] for r in rows)
    fp = sum(not r["expected_detection_window"] and r["drift_detected"] for r in rows)
    fn = sum(r["expected_detection_window"] and not r["drift_detected"] for r in rows)
    tn = sum(
        not r["expected_detection_window"] and not r["drift_detected"] for r in rows
    )
    scenario_types = Counter(result["scenario_type"] for result in results)
    return {
        "mode": mode,
        "scenario_count": len(results),
        "scenario_types": dict(scenario_types),
        "scenarios_meeting_expectation": sum(
            result["scenario_expectation_met"] for result in results
        ),
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        },
        "step_precision": tp / (tp + fp) if tp + fp else None,
        "step_recall": tp / (tp + fn) if tp + fn else None,
        "clean_case_false_alarm_rate": fp / (fp + tn) if fp + tn else None,
        "warning": (
            "These are author-specified synthetic scenario oracles, not human-reviewed labels "
            "or a real-agent generalization estimate."
        ),
    }


def export_annotation_candidates(
    results: list[dict[str, Any]], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for result in results:
            trajectory_id = f"local_agent:{result['scenario_id']}"
            for record in result["records"]:
                annotation = {
                    "trajectory_id": trajectory_id,
                    "source_dataset": "goalguard_local_agent_synthetic_candidate",
                    "source_record_id": f"{trajectory_id}:step:{record['step_number']}",
                    "original_goal": result["original_goal"],
                    "current_goal_at_step": record["current_goal_at_step"],
                    "step_number": record["step_number"],
                    "agent_action": record["agent_action"],
                    "action_type": record["action_type"],
                    "tool_name": record["tool_name"],
                    "tool_arguments": record["tool_arguments"],
                    "observation": record["observation"],
                    "drift_label": None,
                    "drift_onset_step": None,
                    "review_status": "pending_human_review",
                    "reviewer_id": None,
                    "review_notes": None,
                }
                file.write(json.dumps(annotation, ensure_ascii=False) + "\n")


def run_suite(
    scenarios: list[dict[str, Any]],
    model_path: Path | None,
    threshold: float,
    output_dir: Path,
    mode: Literal["observe", "enforce"],
    post_action_scorer: PostActionScorer | None = None,
) -> dict[str, Any]:
    if model_path is None and post_action_scorer is None:
        raise ValueError("A model path or post-action scorer is required")
    model = joblib.load(model_path) if model_path is not None else None
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="goalguard-local-agent-") as temp_root:
        root = Path(temp_root)
        for index, scenario in enumerate(scenarios):
            scenario_workspace = root / f"case-{index:02d}-{scenario['scenario_id']}"
            scenario_app = build_graph(
                model, threshold, scenario_workspace, post_action_scorer
            )
            results.append(
                run_scenario(
                    scenario_app,
                    scenario,
                    mode,
                    scenario_workspace,
                )
            )

    report = {
        "experiment": "GoalGuard local-only LangGraph controlled agent",
        "mode": mode,
        "monitoring_phase": (
            "after each tool result" if post_action_scorer is not None else "before each tool call"
        ),
        "network_tools_available": False,
        "external_llm_or_api_used": False,
        "filesystem_scope": (
            "Temporary per-scenario directories only; reads are allowlisted and writes are "
            "limited to summary.txt, revised_summary.txt, and scratch.txt."
        ),
        "threshold": threshold,
        "summary": summarize(results, mode),
        "scenarios": results,
    }
    report_path = output_dir / f"local_agent_evaluation_{mode}.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if mode == "observe":
        export_annotation_candidates(
            results, output_dir / "annotations" / "pending_review.jsonl"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("observe", "enforce"),
        default="observe",
        help="Observe records detections; enforce blocks a proposed tool call above threshold.",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("goalguard_data/local_agent/scenarios.json"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("goalguard_data/models/goalguard_baseline.joblib"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("goalguard_data/models/baseline_metrics.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("goalguard_data")
    )
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"Trained model not found: {args.model}")
    if not args.metrics.is_file():
        raise FileNotFoundError(f"Baseline metrics not found: {args.metrics}")
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    threshold = float(metrics["threshold_selection"]["selected_threshold"])
    scenarios = load_scenarios(args.scenarios)
    report = run_suite(scenarios, args.model, threshold, args.output_dir, args.mode)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

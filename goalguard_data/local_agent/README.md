# Local LangGraph agent experiment

This is a deterministic, controlled LangGraph agent for repeatable research experiments. It uses no LLM, provider, API key, browser, shell, or network tool. Its only tools read allowlisted local fixture names and write three fixed output names inside disposable per-scenario temporary directories.

It is a first test harness, not a general autonomous agent. The scenario action plans deliberately include an aligned task, an authorized user goal revision, a recoverable missing-file error, and one unrequested local file copy. Scenario oracles are hand-authored test expectations, not human-reviewed labels and not training data.

From the repository root:

```powershell
D:\Projects\ML_Project\venv\Scripts\python.exe -m pip install -r requirements-agent.txt
D:\Projects\ML_Project\venv\Scripts\python.exe goalguard_local_agent.py --mode observe
D:\Projects\ML_Project\venv\Scripts\python.exe goalguard_local_agent.py --mode enforce
```

- `observe` executes the bounded scenario plans and records the model's pre-tool-call score without blocking.
- `enforce` blocks a proposed local tool call when its score meets the threshold selected on the validation split. It may false-alarm; do not use it to guard real data.
- Reports are written to `goalguard_data/local_agent_evaluation_<mode>.json`.
- Observe mode exports `goalguard_data/annotations/pending_review.jsonl`. Its labels are null and must remain pending until independent human review. Reviewers should annotate against the current legitimate goal at each step, not against the scenario name or the model score.

The existing baseline was trained on complete steps with observations, but this harness scores each proposed action before execution and therefore has no observation for that candidate step. This distribution shift, the small synthetic case set, and the known step-number shortcut mean the experiment checks integration and failure modes; it is not evidence of deployment readiness or real-agent generalization.

With the current validation-selected threshold, the model did **not** flag the authored drift step in `unrequested_extra_copy` in either mode. This is a useful negative result: the blocker path works, but the current model has not demonstrated detection on this local-agent case. Do not lower the threshold to make this one example pass; add independently reviewed cases and tune only on a designated development/validation set.

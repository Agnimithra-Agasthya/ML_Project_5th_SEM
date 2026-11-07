# AI Agent Goal Drift Detection

**Detecting when AI agents deviate from their intended objectives through behavioral and temporal analysis.**

## About

AI agents can perform complex tasks through autonomous decision-making and tool usage. However, during execution, they may gradually move away from their original objectives.

This project explores machine learning approaches to detect such goal deviations by analyzing agent behavior over time.

## Research Question

Can tracking changes in an AI agent's behavior over time help detect goal drift earlier?

## Approaches

- **Semantic ML:** Analyzes agent goals and execution history using textual features.
- **Behavioral ML:** Examines tool usage, action categories and behavioral patterns.
- **Temporal ML (GRU):** Analyzes changes across sequences of agent actions.

## Evaluation

The project investigates:

- Classification performance.
- Early goal drift detection.
- Detection delay and false alarms.
- Robustness against gradual and sudden drift.
- Comparison of semantic, behavioral, step-number-only and temporal approaches.

### Reproducing the four-model comparison

From the repository root, install the evaluation dependencies and train the benchmark GRU using the same isolated partitions as the other models:

```powershell
python -m pip install -r requirements-ml.txt
python -m pip install -r requirements-gru.txt
python train_goalguard_gru.py --data-dir goalguard_data --output-dir goalguard_data\gru_experiment --evaluation-split-dir goalguard_data
python evaluation\run_evaluation.py --data-dir goalguard_data --results-dir evaluation\results
```

The GRU trainer uses augmented variants only from training seed families; validation and test families come from the existing GoalGuard JSONL partitions. The evaluation runner fails rather than substituting the separate local-agent GRU or silently omitting an unavailable checkpoint. All four models have thresholds calibrated on the same validation split (maximum F1 subject to validation FPR <= 0.10). The step-number-only logistic regression is fitted on the training split rather than using a fixed hand-written score curve.

### Preliminary results

The checked-in results use 2,499 training steps, 540 validation steps, and 535 test steps. Current test-step classification metrics and event metrics are:

| Model | Test F1 | ROC-AUC | PR-AUC | Onset-window recall (±1 step) | Median detection delay |
|---|---:|---:|---:|---:|---:|
| Semantic TF-IDF | 0.892 | 0.969 | 0.953 | 0.701 | 0 |
| Behavioral ActionSchema | **0.948** | **0.991** | **0.983** | 0.831 | 0 |
| Step-number-only | 0.797 | 0.979 | 0.942 | **0.909** | 1 |
| Temporal GRU | 0.939 | 0.989 | 0.979 | 0.831 | 0 |

On this run, Behavioral ActionSchema has the highest step F1 and ROC/PR-AUC; the GRU is second on those metrics and matches it on onset-window recall. Behavioral ActionSchema detected 77/77 test drift trajectories, compared with 74/77 for the GRU. The step-only model has the highest ±1-step onset-window recall, but lower step F1, a one-step median delay, and poor early-detection behavior. These results do not establish that the GRU is the best detector.

Interpret results cautiously: the test partition has 77 drift trajectories and **no fully aligned trajectories**, so it cannot estimate false-alarm rates on benign tasks. The benchmark labels are not independently reviewed by GoalGuard. Also, the GRU is trained for onset and the following step (later drift steps are excluded from its training loss), while the other classifiers use persistent post-onset step labels. Step-level F1 therefore is not a perfectly matched target; the paper should emphasize temporal event metrics and state this supervision difference. In the 20-trajectory padded-prefix test, the behavioral, step-only, and GRU models all alarmed before the shifted onset in all 20 cases; Semantic TF-IDF did so in 5/20. This stress test shows substantial sensitivity to the inserted context and should be reported as a limitation.

Fresh machine-readable metrics and plots are written to `evaluation/results/`. Treat them as preliminary benchmark results, not evidence of generalization to real agents.

## Technology Stack

- Python
- Machine Learning
- Natural Language Processing
- GRU (Gated Recurrent Unit)
- Scikit-learn

## Project Objective

To investigate whether understanding how an AI agent behaves over time can help identify when it moves away from its intended goal.

---

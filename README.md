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
- Comparison of semantic, behavioral and temporal approaches.

## Technology Stack

- Python
- Machine Learning
- Natural Language Processing
- GRU (Gated Recurrent Unit)
- Scikit-learn

## Project Objective

To investigate whether understanding how an AI agent behaves over time can help identify when it moves away from its intended goal.

---

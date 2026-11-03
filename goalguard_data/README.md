# GoalGuard baseline

The provisional supervised step labels currently come only from Agentic Red-Team Gold. The other normalized sources remain trajectory-level-only, uncertain, or excluded and are not used to fit this baseline. Although the source is marked `gold`, its metadata has no non-empty `reviewed_by` entries; GoalGuard has not independently reviewed these annotations.

From the repository root, install the ML dependency and run:

```powershell
python -m pip install -r requirements-ml.txt
python train_goalguard_baseline.py
```

The trainer checks trajectory and exact-content split isolation, fits a TF-IDF/logistic-regression model on train, selects its step threshold on validation, and evaluates the test split after selection. It also compares against a step-number-only ablation. It writes:

- `goalguard_data/models/goalguard_baseline.joblib`
- `goalguard_data/models/baseline_metrics.json`

The model sees only the original goal and observed execution prefix through the current step. It does not use source metadata, judge explanations, ground truth, future actions, or label fields as features.

This is a research baseline, not a deployment-ready guardian. The labeled sample is a single synthetic adversarial benchmark, all trajectories have drift onsets, step labels are derived from those onsets, and the reported confidence is not calibrated. Compare against the included step-number-only ablation: if timing alone performs as well as or better than the text model, do not interpret high scores as evidence of semantic goal-understanding.

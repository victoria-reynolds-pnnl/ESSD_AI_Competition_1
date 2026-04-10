# Week 3 - Model Training, Evaluation, and Query Demo

## 1) Overview

Week 3 focuses on supervised modeling for extreme thermal event risk classification using the prepared Week 2 ML dataset.

Primary goals:
- Train and compare baseline and tree-based classifiers.
- Calibrate predicted probabilities for decision support use.
- Interpret feature influence.
- Support practical risk queries (for example, regional travel-risk style questions).

Main workflow notebook:
- `scripts/week3_ml_model.ipynb`

## 2) Inputs and Dependencies

This stage depends on Week 2 outputs:
- `../week2/data/processed/X_train.csv`
- `../week2/data/processed/X_val.csv`
- `../week2/data/processed/X_test.csv`
- `../week2/data/processed/y_train.csv`
- `../week2/data/processed/y_val.csv`
- `../week2/data/processed/y_test.csv`
- `../week2/data/processed/ml_artifacts/scaler.joblib`
- `../week2/data/processed/ml_artifacts/label_encoder.joblib`
- `../week2/data/processed/ml_artifacts/feature_names.json`
- `../week2/data/processed/ml_artifacts/onehot_columns.json`

From the notebook setup:
- Calibration method: isotonic
- Risk bands for query output:
  - Low risk: probability < 0.30
  - Moderate risk: 0.30 to < 0.60
  - High risk: >= 0.60

## 3) Models Trained

The notebook trains and evaluates:
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting (selected primary model)

It also calibrates probabilities and compares models using:
- Accuracy
- Macro F1
- ROC-AUC
- Average Precision
- Brier Score
- Log Loss

## 4) Model Performance Summary

Source: `ml_models/model_comparison.csv`

| Model | Accuracy | F1 (macro) | ROC-AUC | Avg Precision | Brier | Log Loss |
|---|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.8440 | 0.8001 | 0.9094 | 0.7794 | 0.1068 | 0.3284 |
| Random Forest | 0.8389 | 0.7972 | 0.9017 | 0.7374 | 0.1117 | 0.3427 |
| Logistic Regression | 0.7267 | 0.4221 | 0.6560 | 0.3648 | 0.1862 | 0.5525 |

Interpretation:
- Gradient Boosting gives the best overall ranking and probability quality (highest ROC-AUC/AP and lowest Brier/Log Loss).
- Random Forest is close, but still weaker than Gradient Boosting on probability quality.
- Logistic Regression underperforms substantially on discrimination and macro-F1, indicating limited capacity for nonlinear patterns.

Additional evidence from generated diagnostics:
- Gradient Boosting learning curve indicates best observed test log-loss at iteration 300.
- Calibration, ROC, PR, confusion matrix, and probability-distribution figures were exported to `figures/`.

## 5) Key Feature Signals

Source: `ml_models/feature_importance.csv` (Gradient Boosting)

Top contributors include:
- `doy_cos`
- `doy_sin`
- `hazard_query_heat_wave`
- `def_percentile`
- `hazard_query_cold_snap`

This indicates that seasonality and event-definition metadata strongly influence model predictions, with additional contribution from hazard context and aggregation choices.

## 6) Human-in-the-Loop Review (3 Prompts)

### Prompt 1: Are top-ranked risk regions and dates climatologically plausible?
Review outcome:
- Mostly yes. The model heavily uses seasonal cyclical features (`doy_sin`, `doy_cos`) and hazard context, which aligns with expected thermal-event seasonality behavior.
- Human review should still verify edge periods (season transition windows) where probabilities can be less intuitive.

### Prompt 2: Are probabilities decision-ready for advisory-style use?
Review outcome:
- Partially yes. Probability calibration is explicitly performed and evaluated, improving suitability for threshold-based use.
- For operational use, humans should validate threshold policies (0.30 / 0.60) against domain costs (false alarm vs missed event) before deployment.

### Prompt 3: Which predictions require mandatory human escalation?
Review outcome:
- Escalate at least these cases for analyst review:
  - Moderate band outputs (0.30 to < 0.60), where uncertainty is highest.
  - High-risk recommendations in low-sample or historically volatile regions.
  - Cases with disagreement between model score and known regional context/events.

## 7) Failure / Limitations Review

Known limitations in this Week 3 modeling pass:
- Class imbalance remains meaningful (test distribution is skewed), so threshold choice has strong impact on practical precision/recall trade-offs.
- Feature set is constrained to engineered historical/tabular signals from Week 2; no external forecast drivers are included.
- Region-level behavior is not uniform (the notebook includes region accuracy diagnostics), so global thresholds may not be equally optimal for all NERC regions.
- Performance was benchmarked on held-out splits from the same prepared pipeline; broader temporal and geographic stress tests are still needed.
- Query outputs provide model-based risk guidance, not deterministic event certainty.

## 8) Produced Artifacts

### Models and tables (`ml_models/`)
- `gradient_boosting.pkl`
- `random_forest.pkl`
- `logistic_regression.pkl`
- `model_comparison.csv`
- `feature_importance.csv`

### Figures (`figures/`)
- `ml_09_lr_coefficients.png`
- `ml_10_gb_learning_curve.png`
- `ml_11_calibration_curves.png`
- `ml_12_roc_curves.png`
- `ml_13_pr_curves.png`
- `ml_14_feature_importance.png`
- `ml_15_confusion_matrix.png`
- `ml_16_prob_distribution.png`
- `ml_17_accuracy_by_region.png`
- `ml_18_query_summer.png`
- `ml_19_query_winter.png`
- `ml_20_seasonal_comparison.png`
- `ml_21_definition_comparison.png`

## 9) How to Reproduce

1. Ensure Week 2 processed data and artifacts exist.
2. Open and run `scripts/week3_ml_model.ipynb` from top to bottom.
3. Confirm outputs are written to:
   - `week3/ml_models/`
   - `week3/figures/`

If paths differ on your machine, update the path setup cell in the notebook accordingly.

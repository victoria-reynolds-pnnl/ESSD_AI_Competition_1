# Person 2 To-Do — Interpretability, Error Analysis, and Visualizations

## Main Goal
Complete the interpretability and error-analysis work for the trained ML models, produce the required visualizations, and provide written findings for the README .

---

## Deliverables You Own
- `Scripts/interpretability.py` 
- At least 1 `.png` visualization per model in `Visualizations/` 
- SHAP outputs
- Feature importance outputs
- Error analysis outputs
- Notes for README:
  - model interpretation insights
  - feature-importance interpretation
  - error analysis summary
  - possible calibration note if probabilities are used 

---

## Step 1: Load the trained models and results
Use:
- `Models/model_1.pkl`
- `Models/model_2.pkl`
- `Data/week2_clean.csv`
- `Data/splits_indexed.csv`
- `ml_results.csv`

Confirm:
- which model is Logistic Regression
- which model is XGBoost
- which model appears stronger on F1 vs AUC-ROC

---

## Step 2: Write `Scripts/interpretability.py`
Your script should:
- load the trained models 
- load the data and split assignments
- reproduce the labeled validation/test data if needed
- generate interpretability outputs
- save figures into `Visualizations/` 

At minimum, produce:
- 1 visualization for Logistic Regression
- 1 visualization for XGBoost 

---

## Step 3: Match interpretability to the task
This is a **classification** problem, so your interpretation should focus on:
- F1
- AUC-ROC
- precision / recall tradeoffs
- false positives vs false negatives 

Write 2–3 sentences for the README explaining:
- why **F1** is useful for balancing precision and recall
- why **AUC-ROC** is useful for measuring class separation across thresholds 

---

## Step 4: Generate SHAP values
Use SHAP for at least the stronger model, preferably XGBoost .

Create:
- a SHAP summary plot
- a SHAP plot for at least a few individual predictions

Choose examples like:
- one predicted high-risk case
- one predicted low-risk case
- one incorrect prediction

Write down:
- what features pushed the prediction up
- what features pushed it down
- whether the explanation makes physical sense 

---

## Step 5: Check feature importance
For each model, review the most important features.

Ask:
- does the top feature make physical sense? 
- are temperature, duration, or trend variables driving predictions in a believable way?
- is `nerc_id` dominating too much as a location proxy?

Write a short interpretation summary, not just the plot.

---

## Step 6: Error analysis
Review model errors carefully.

For both models, create:
- confusion matrix plot
- precision/recall comparison
- list or table of false positives / false negatives

Focus especially on:
- false negatives in high-risk cases
- whether errors differ by region (`nerc_id`)
- whether one model is more conservative than the other

Write 1 short paragraph on what kinds of cases the models miss.

---

## Step 7: Check calibration if using probabilities
If the predicted probabilities are meant to support decisions, compute a **Brier score** and optionally a calibration plot .

Write a short note:
- whether probabilities seem reasonably calibrated
- whether one model is better calibrated than the other

If you do not have time for full calibration, note that it should be checked in future work.

---

## Step 8: Create error/residual maps if feasible
If your data supports geographic grouping or region mapping:
- create an error-by-region plot
- or a simple map / grouped bar chart by `nerc_id`

Goal:
- see whether errors cluster spatially, which may reveal blind spots 

If a map is not practical, use:
- grouped error counts by region
- false negative rate by region

---

## Step 9: Save required visualizations
Save all `.png` files into `Visualizations/` .

Minimum required:
- at least 1 `.png` per model 

Suggested filenames:
- `logistic_regression_confusion_matrix.png`
- `xgboost_shap_summary.png`
- `xgboost_feature_importance.png`
- `error_by_region.png`

---

## Step 10: Hand off written notes
Provide Person 3 with:
- a short metric-justification note
- SHAP interpretation notes
- feature-importance interpretation
- error-analysis summary
- calibration note if used
- best visualization filenames
- anything important for the README

---

## Reminder
Your work must go beyond “plot dumping.” The Week 3 grading expects interpretability outputs to be correctly interpreted and explained meaningfully .
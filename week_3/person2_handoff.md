# Person 2 Handoff Notes

## Scope Completed

Person 2 tasks completed through Step 9:

- Step 1 model mapping and metric comparison
- Step 2 interpretability workflow script and baseline visual outputs
- Step 3 metric justification text
- Step 4 SHAP generation and case-level interpretation
- Step 5 feature-importance interpretation
- Step 6 error analysis tables, plots, and notes
- Step 7 calibration analysis (Brier + calibration curve)
- Step 8 region/grouped error outputs (including false negative rate by region)
- Step 9 required visualization presence checks

## 1) Metric-Justification Note (for README)

We used F1 because this is a high-risk classification task where both missed detections and false alarms matter, so balancing precision and recall is more informative than accuracy alone. We used AUC-ROC because it measures class separation across all thresholds, which captures ranking quality even when decision thresholds change.

## 2) SHAP Interpretation Notes

Primary SHAP files:

- Visualizations/xgboost_shap_summary.png
- Visualizations/xgboost_shap_high_risk_case.png
- Visualizations/xgboost_shap_low_risk_case.png
- Visualizations/xgboost_shap_incorrect_case.png
- Interpretability/xgboost_shap_case_notes.json
- Interpretability/xgboost_shap_notes.md

Interpretation summary:

- XGBoost local and global explanations are dominated by physically meaningful severity variables, especially yearly maximum heat-wave duration and intensity, with duration-related features also contributing.
- High-risk example: severity variables strongly pushed the prediction upward and matched the true positive label.
- Low-risk example: low severity signals pushed probability downward and matched the true negative label.
- Incorrect example (false negative): conflicting severity signals produced a missed high-risk case, indicating a borderline decision region where the model can be too conservative.

## 3) Feature-Importance Interpretation

Primary files:

- Visualizations/logistic_regression_feature_importance.png
- Visualizations/xgboost_feature_importance.png
- Interpretability/logistic_regression_feature_importance.csv
- Interpretability/xgboost_feature_importance.csv
- Interpretability/feature_importance_notes.md

Interpretation summary:

- Both models emphasize yearly maximum heat-wave intensity, yearly maximum heat-wave duration, and event duration, which is physically plausible for high-risk heat-event classification.
- Trend variables contribute but are secondary.
- nerc_id is not dominating in this dataset; practical importance is near zero because only one category is present in the labeled set.

## 4) Error-Analysis Summary

Primary files:

- Visualizations/logistic_regression_confusion_matrix.png
- Visualizations/xgboost_confusion_matrix.png
- Visualizations/precision_recall_comparison.png
- Visualizations/error_by_region.png
- Visualizations/false_negative_rate_by_region.png
- Interpretability/error_analysis_notes.md
- Interpretability/logistic_regression_false_positives.csv
- Interpretability/logistic_regression_false_negatives.csv
- Interpretability/xgboost_false_positives.csv
- Interpretability/xgboost_false_negatives.csv
- Interpretability/error_by_region_summary.csv
- Interpretability/false_negative_rate_by_region.csv

Summary findings:

- Logistic Regression is less conservative: higher recall but many more false positives.
- XGBoost is more conservative/selective: much higher precision but substantially more false negatives.
- Test false negatives: Logistic Regression 1007 vs XGBoost 2566.
- Test false positives: Logistic Regression 2559 vs XGBoost 281.
- False-negative rate among actual positives by region grouping: Logistic Regression about 0.147 vs XGBoost about 0.375.
- Region contrast is limited in this dataset because only one nerc_id category appears in test records.

## 5) Calibration Note

Primary files:

- Visualizations/calibration_curve.png
- Interpretability/calibration_brier_scores.csv
- Interpretability/calibration_curve_points.csv
- Interpretability/calibration_notes.md

Summary:

- Brier score (lower is better): XGBoost about 0.0999, Logistic Regression about 0.1118.
- XGBoost is better calibrated than Logistic Regression on this test set under Brier.
- Calibration curve included to inspect over/under-confidence across probability ranges.

## 6) Best Visualization Filenames for README

Recommended to reference:

- Visualizations/logistic_regression_confusion_matrix.png
- Visualizations/xgboost_confusion_matrix.png
- Visualizations/precision_recall_comparison.png
- Visualizations/xgboost_shap_summary.png
- Visualizations/xgboost_shap_incorrect_case.png
- Visualizations/xgboost_feature_importance.png
- Visualizations/error_by_region.png
- Visualizations/false_negative_rate_by_region.png
- Visualizations/calibration_curve.png

## 7) Important Notes for Final Writeup

- There is an explicit precision-recall tradeoff between models: Logistic Regression catches more high-risk cases; XGBoost is stricter and cleaner but misses more positives.
- SHAP/feature-importance evidence supports physical plausibility because severity variables dominate model behavior.
- Regional error analysis is only partially informative due to single observed nerc_id category; present this as a dataset limitation.
- If operational priority is minimizing missed high-risk events, consider threshold tuning or recall-targeted calibration for XGBoost before deployment.

## 8) Reproducibility Command

From TRACK folder:

- python Scripts/interpretability.py

This regenerates interpretability outputs, error-analysis artifacts, calibration outputs, and the visualization set in Visualizations/.

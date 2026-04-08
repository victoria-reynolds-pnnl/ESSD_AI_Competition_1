# Week 3 Submission README

## Train/Test Split Strategy

Because the dataset consists of time-indexed extreme weather event records, we used a strictly chronological train/validation/test split and did not shuffle the data. Records were sorted by `centroid_date`, with the earliest 70% assigned to training, the next 15% assigned to validation, and the most recent 15% held out for final testing. The validation set was used only for hyperparameter tuning, and the test set was reserved for final evaluation only. We also checked engineered features for leakage from time `T+1` onward and excluded features that used future event information. For the HDBSCAN-based labeling step, clustering was fit on the training set only, and validation/test tier assignments were produced using a mapper derived from the training data to avoid leaking future structure into the supervised workflow [2].

## Model Performance Summary

We trained and evaluated two binary classification models—Logistic Regression and XGBoost—to identify high-risk extreme weather records. The classification target was derived from HDBSCAN-based 3-tier risk labeling and then collapsed into a binary `high_risk` outcome for Week 3. We used F1 because the task depends on balancing precision and recall when flagging high-risk events, which helps quantify the tradeoff between false positives and false negatives at the chosen decision threshold [2]. We used AUC-ROC because it measures how well the model separates high-risk from non-high-risk cases across all possible thresholds, so it captures overall ranking quality even when the operating threshold changes [2].

On the validation set, Logistic Regression achieved an F1 score of 0.607 and an AUC-ROC of 0.864, while XGBoost achieved an F1 score of 0.596 and an AUC-ROC of 0.916. On the held-out test set, Logistic Regression achieved an F1 score of 0.766 and an AUC-ROC of 0.916, while XGBoost achieved an F1 score of 0.750 and an AUC-ROC of 0.943. Overall, Logistic Regression performed slightly better at the chosen classification threshold, while XGBoost showed stronger overall discrimination. This suggests a tradeoff between balanced thresholded classification and more flexible nonlinear ranking performance.

## Failures and Limitations

This workflow has several important limitations. First, the dataset is time-indexed, so any feature engineering or labeling step that accidentally uses future information could introduce leakage; we addressed this through chronological splitting, validation-only tuning, and training-only clustering, but engineered variables still require careful auditing. Second, the binary `high_risk` label is a simplified collapse of three HDBSCAN-derived tiers, which makes the Week 3 classification task easier to evaluate but may hide meaningful differences between moderate-risk and extreme-risk events. Third, because the risk tiers are derived from unsupervised clustering and then mapped onto validation and test data, the tier assignment process may be sensitive to changes in cluster structure over time. In addition, the models operate on extreme weather event records rather than directly on full siting decisions, so they support the broader project objective of identifying high-risk conditions and vulnerable clustering but do not by themselves prove whether data center siting is fully misaligned with climate-resilient planning [1]. If more time were available, next steps would include calibration checks, additional spatial error analysis, more formal tier-stability testing, and alternative labeling strategies.

## HITL Review

### Does the model behavior make sense given domain expectations?

Mostly yes. Both models correctly prioritize physically meaningful variables — yearly_max_heat_wave_intensity, yearly_max_heat_wave_duration, duration_days, and spatial_coverage are the dominant drivers in both models feature_importance_notes.md, xgboost_feature_importance.csv. SHAP analysis confirms this at the individual prediction level: correctly flagged high-risk events are pushed upward by strong intensity and duration signals, while correctly dismissed low-risk events are suppressed by weak severity signals xgboost_shap_notes.md. The one counterintuitive finding is duration_days carrying a negative coefficient in Logistic Regression feature_importance_notes.md, but this reflects a conditional statistical effect alongside annual severity features — not a sign the model is broken.

### Any concerning spurious correlations?

Two concerns stand out:
•	nerc_id is meaningless. The dataset contains only one region category, so the variable provides zero discriminatory power feature_importance_notes.md, xgboost_feature_importance.csv. Any regional pattern the model appears to learn is an artifact, not real signal.
•	Annual background intensity may be inflating false positives. Many wrongly flagged events occur in high-intensity climate years, suggesting the models may be conflating "this was a bad year overall" with "this specific event is dangerous" — a subtle but operationally important distinction error_analysis_notes.md, error_by_region_summary.csv.

### Any high-risk failure modes?

Yes — XGBoost's false negative rate is the most serious concern. It misses 2,566 true high-risk events (37.5% miss rate) compared to Logistic Regression's 1,007 (14.7%) error_by_region_summary.csv. Both models struggle most on borderline events — XGBoost's missed cases average an intensity of 308.15 vs. 314.61 for correctly identified ones error_analysis_notes.md, meaning the models fail precisely when early warning matters most. Cold snap false negatives are also notably present in XGBoost's predictions xgboost_test_predictions.csv, suggesting weaker reliability outside of heat-wave scenarios. Finally, single-region training data means neither model can be trusted to generalize beyond the NERC region without revalidation feature_importance_notes.md.

## High-Level Observations (Extra)

•	Logistic Regression is the safer model for catching real danger. It misses 1,007 high-risk cases vs. XGBoost's 2,566 — a 37.5% miss rate that is operationally unacceptable in a safety context error_by_region_summary.csv.
•	XGBoost is precise but dangerously conservative. Its 93.8% precision means it rarely cries wolf, but it misses too many real events to be trusted as a standalone early-warning tool error_analysis_notes.md.
•	Both models are physically credible. yearly_max_heat_wave_intensity, yearly_max_heat_wave_duration, and duration_days dominate predictions in both models — exactly the variables that should matter feature_importance_notes.md.
•	Borderline events are the blind spot. Missed cases average lower intensity (308.15 vs. 314.61 for XGBoost correct predictions) — meaning both models are reliable for obvious extremes but unreliable precisely when early warning matters most error_analysis_notes.md.
•	nerc_id is dead weight. It contributes zero discriminatory power because only one region exists in the data, making any regional generalization untested feature_importance_notes.md, xgboost_feature_importance.csv.
•	Annual background conditions are likely inflating false positives. Events embedded in high-intensity years get over-flagged, suggesting the models conflate a bad climate year with a bad individual event error_analysis_notes.md, error_by_region_summary.csv.

## Additional Analysis (Extra)

1. Are the drivers reasonable?
Yes. Both models correctly prioritize yearly_max_heat_wave_intensity, yearly_max_heat_wave_duration, duration_days, and spatial_coverage feature_importance_notes.md, xgboost_feature_importance.csv. SHAP confirms this at the individual level — high-risk predictions are pushed up by strong intensity and duration signals, low-risk ones are suppressed by weak signals xgboost_shap_notes.md. The one oddity is duration_days carrying a negative coefficient in Logistic Regression feature_importance_notes.md, but this is a conditional statistical effect, not a physical contradiction.

2. Could region variables act as proxies?
No meaningful risk here — but for the wrong reason. nerc_id has effectively zero importance in both models feature_importance_notes.md +1 because the dataset contains only one region category. It cannot act as a proxy for anything. The real concern is the opposite: if deployed across multiple regions, the models have no learned regional signal whatsoever and could fail unpredictably.

3. Are the false negatives a serious concern?
Yes — significantly so. XGBoost misses 2,566 true high-risk events (37.5% miss rate) vs. Logistic Regression's 1,007 (14.7%) error_by_region_summary.csv. Missed cases cluster around borderline intensity values (308.15 vs. 314.61 for correctly identified cases) error_analysis_notes.md, meaning both models fail most on moderate but still dangerous events. Cold snap false negatives are also notably present in XGBoost's predictions xgboost_test_predictions.csv, suggesting reliability drops further outside heat-wave scenarios.

## Reproducibility Steps

1. Install the required Python packages using the version-pinned `requirements.txt` file [2].
2. Place the cleaned Week 2 dataset in `Data/week2_clean.csv` and confirm the folder structure matches the Week 3 submission template [2].
3. Run `python Scripts/train.py` to load the cleaned dataset, create the chronological train/validation/test split, fit HDBSCAN on the training data only, derive 3 risk tiers and collapse them to binary `high_risk`, train and tune Logistic Regression and XGBoost, and save `splits_indexed.csv`, `model_1.pkl`, and `model_2.pkl` [2].
4. Run `python Scripts/evaluate.py` to evaluate both models on the held-out validation and test sets and save `ml_results.csv` [2].
5. Run `python Scripts/interpretability.py` to generate SHAP outputs, visualizations, and reporting artifacts [2].
6. Confirm that all required files are present: `week2_clean.csv`, `splits_indexed.csv`, `ml_results.csv`, `model_1.pkl`, `model_2.pkl`, `train.py`, `evaluate.py`, `interpretability.py`, at least one `.png` per model, and a version-pinned `requirements.txt` [2].

## AI Use Disclosure

AI tools were used during development to support coding, feature engineering ideation, and workflow drafting. All AI-assisted outputs were reviewed, edited, and validated by the team before inclusion in the final submission. AI use is disclosed here as required.

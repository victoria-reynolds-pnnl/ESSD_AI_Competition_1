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

_To be completed by a domain expert or teammate who is not the model builder [2]._

### Any concerning spurious correlations?

_To be completed by a domain expert or teammate who is not the model builder [2]._

### Any high-risk failure modes?

_To be completed by a domain expert or teammate who is not the model builder [2]._

## Reproducibility Steps

1. Install the required Python packages using the version-pinned `requirements.txt` file [2].
2. Place the cleaned Week 2 dataset in `Data/week2_clean.csv` and confirm the folder structure matches the Week 3 submission template [2].
3. Run `python Scripts/train.py` to load the cleaned dataset, create the chronological train/validation/test split, fit HDBSCAN on the training data only, derive 3 risk tiers and collapse them to binary `high_risk`, train and tune Logistic Regression and XGBoost, and save `splits_indexed.csv`, `model_1.pkl`, and `model_2.pkl` [2].
4. Run `python Scripts/evaluate.py` to evaluate both models on the held-out validation and test sets and save `ml_results.csv` [2].
5. Run `python Scripts/interpretability.py` to generate SHAP outputs, visualizations, and reporting artifacts [2].
6. Confirm that all required files are present: `week2_clean.csv`, `splits_indexed.csv`, `ml_results.csv`, `model_1.pkl`, `model_2.pkl`, `train.py`, `evaluate.py`, `interpretability.py`, at least one `.png` per model, and a version-pinned `requirements.txt` [2].

## AI Use Disclosure

AI tools were used during development to support coding, feature engineering ideation, and workflow drafting. All AI-assisted outputs were reviewed, edited, and validated by the team before inclusion in the final submission. AI use is disclosed here as required.

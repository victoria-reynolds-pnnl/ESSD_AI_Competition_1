# Error Analysis Summary

On the test set, logistic regression is the less conservative model: it reaches recall 0.853 with precision 0.695, while XGBoost reaches recall 0.625 with much higher precision 0.938. This means logistic regression catches more true high-risk cases but also produces many more false positives, whereas XGBoost is more selective and misses more positives.

The false-negative burden is materially larger for XGBoost (2566 missed high-risk cases) than for logistic regression (1007 missed high-risk cases). Relative to their correctly identified positives, the missed cases in both models tend to show weaker heat-wave intensity and shorter duration signals on average, which suggests the models struggle most on borderline high-risk events rather than on the most extreme events.

Region-level error analysis is limited in this dataset because the test records only expose a single `nerc_id` category, so there is no meaningful cross-region contrast to evaluate. The grouped error plot still documents the imbalance in false positives versus false negatives by model.

Average feature profile of missed high-risk cases:
- Logistic regression false negatives: mean intensity 298.11, mean annual duration 59.31, mean event duration 32.27
- Logistic regression true positives: mean intensity 314.61, mean annual duration 55.65, mean event duration 8.40
- XGBoost false negatives: mean intensity 308.15, mean annual duration 60.89, mean event duration 22.73
- XGBoost true positives: mean intensity 314.61, mean annual duration 53.37, mean event duration 5.43

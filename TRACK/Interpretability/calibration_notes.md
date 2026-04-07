# Calibration Note

Brier score compares probability calibration quality, with lower values indicating better calibrated probabilities.
On the test set, xgboost has the lower Brier score and therefore better overall calibration under this metric.
Calibration curves are included to show where each model is over-confident or under-confident across probability ranges.

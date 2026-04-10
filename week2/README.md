# Heat Wave & Cold Snap — ML Pipeline

## Overview
End-to-end data cleaning, feature engineering, and ML training pipeline
for predicting **regional probabilities of heat waves and cold snaps**
by NERC region and date.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your data file
cp your_data.csv data/raw/data.csv

# 3. Run the pipeline
python pipeline.py --input data/raw/data.csv \
                   --output_dir outputs/ \
                   --query_date 2024-07-15
```

---

## Output Files

| File | Description |
|---|---|
| `data/processed/original.csv` | Unmodified copy of input data |
| `data/processed/cleaned.csv` | After missing values, duplicates, coercion, outlier handling |
| `data/processed/ml_ready.csv` | After feature engineering; ready for model input |
| `outputs/split_summary.txt` | Train / val / test date ranges and class distributions |
| `outputs/feature_importance.png` | Top-20 feature importances from trained model |
| `outputs/calibration_curve.png` | Probability calibration curve vs. perfect calibration |
| `outputs/pipeline_model.joblib` | Saved model + label encoder + feature metadata |

---

## Data Preparation Approach

Raw data passes through a sequential, reproducible pipeline:

1. **Missing values** — Anchor dates (start_date, centroid_date) are
   required; rows missing them are dropped. Numeric columns receive
   grouped median imputation (NERC region × hazard_type) with a global
   median fallback. Categorical gaps are filled with `'UNKNOWN'` to
   preserve rows while allowing the model to learn this as a level.

2. **Duplicates** — Full row duplicates are removed first. Logical key
   duplicates (same event anchor recorded multiple times) keep the
   most spatially representative observation (highest spatial_coverage_pct).

3. **Type coercion** — Dates are parsed with `errors='coerce'`; numeric
   columns cast to float64. `duration_days` is recomputed from
   `end_date − start_date + 1` as the authoritative source of truth.
   Categorical strings are whitespace-stripped and lowercased.

4. **Outlier treatment** — Values outside physical bounds (e.g., Kelvin
   temperature range) are hard-clipped first, then IQR Winsorisation
   (×1.5) is applied. Removal is avoided — extreme events are the signal.

5. **Feature engineering** — Three leakage-safe features are added:
   cyclical day-of-year encoding (sin/cos), 30-day trailing regional
   event rate, and days-since-last-same-type-event.

6. **Train/val/test split** — Chronological 70/15/15 split on
   centroid_date. No shuffling; future data never enters training.
   Rolling features use only observations strictly before the event date.

---

## Model Selection

| Model | Role | Why |
|---|---|---|
| **Gradient Boosting (GBM)** | Primary model | Best accuracy on tabular data with mixed feature types; handles non-linear region × season interactions; native feature importance |
| **Random Forest** | Alternative | Robust to outliers; fast; good baseline comparison |
| **Logistic Regression** | Interpretable baseline | Useful for coefficient inspection; probability output with reliable calibration |
| **Isotonic Calibration** | Post-processing (all models) | Ensures predicted probabilities are reliable for travel queries ("P=0.3" truly means ~30% chance) |

### Why Calibrated Probabilities Matter
The objective is not just classification but **reliable probability
estimation** for queries like "lowest heat-wave probability region on
date X". Without calibration, a model might output 0.9 for events that
only occur 60% of the time. Isotonic calibration (fit on held-out
validation data) corrects this systematic bias.

### Recommended Evaluation Metrics

| Metric | Purpose |
|---|---|
| **Brier Score** | Measures probability accuracy (0 = perfect, 0.25 = random) |
| **ROC-AUC** | Discrimination ability across all probability thresholds |
| **Average Precision** | Useful when class imbalance exists |
| **Calibration Curve** | Visual check that probabilities are reliable |

---

## Feature Engineering Summary

| Feature | Description | Why It Helps |
|---|---|---|
| `doy_sin` / `doy_cos` | Cyclical encoding of day-of-year | Heat waves peak in summer, cold snaps in winter; cyclic encoding keeps Dec 31 ≈ Jan 1 |
| `region_hazard_rate_30d` | 30-day trailing event rate per NERC region | Captures regional climatological base rate and event clustering |
| `days_since_last_event` | Days since prior same-type event in same region | Extreme events cluster; recent event predicts next |

---

## Data Dictionary

See [`data_dictionary.md`](data_dictionary.md) for full column
descriptions, valid ranges, engineered feature definitions, and
cleaning decision log.
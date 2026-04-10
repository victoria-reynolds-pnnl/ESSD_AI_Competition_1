# Heat Wave & Cold Snap - Week 2 Pipeline

## Overview
Week 2 covers two executable notebooks:
1. `week2_data_cleaning.ipynb`
2. `week2_data_preparation_ml.ipynb`

They transform the combined event library into cleaned data, then into panel-based ML features and artifacts.

> Note: large generated outputs are not fully committed to AI Competition GitHub repository. Please re-run notebooks to reproduce files.

---

## What Is In `scripts/`

| File | Actual role |
|---|---|
| `combine_data.py` | Consolidates 6 raw ZIP libraries into one CSV. Current script writes to `week1/outputs/combined_extreme_thermal_event_library.csv`. |
| `week2_data_cleaning.ipynb` | Cleans + validates event data and converts temperature from Kelvin to Celsius. |
| `week2_data_preparation_ml.ipynb` | Engineers definition-aware features, builds daily panel target, splits/encodes/scales, saves artifacts. |

---

## Input Data Assumptions

- Raw ZIP source (per script comments): PNNL event libraries.
- Week 2 notebooks read from `week2/data/combined_extreme_thermal_event_library.csv` and write to `week2/data/processed/`.

If you use `combine_data.py`, copy or move its output CSV into `week2/data/` before running notebooks.

---

## How To Run

Run from `week2/scripts/` in this order:

```bash
# 1) Data cleaning
jupyter notebook week2_data_cleaning.ipynb

# 2) ML preparation
jupyter notebook week2_data_preparation_ml.ipynb
```

Optional non-interactive execution:

```bash
jupyter nbconvert --to notebook --execute --inplace week2_data_cleaning.ipynb
jupyter nbconvert --to notebook --execute --inplace week2_data_preparation_ml.ipynb
```

---

## Cleaning Notebook (`week2_data_cleaning.ipynb`)

### Implemented cleaning logic
1. Schema/type validation and coercion.
2. Missing-value handling (grouped median for numeric, categorical fill where needed).
3. Duplicate handling using logical key:
   - `(hazard_type, definition_id, start_date, NERC_ID)`
4. Duration consistency checks.
5. **Temperature conversion (Step 5.7):**
   - `extreme_temperature_K` -> `extreme_temperature_C`
   - Formula: `C = K - 273.15`
   - Column renamed and downstream numeric schema updated to Celsius.
6. Outlier handling via physical bounds + IQR capping stats.

### Saved outputs
| Path | Description |
|---|---|
| `data/processed/data_raw.csv` | Raw copy used in notebook |
| `data/processed/data_cleaned.csv` | Cleaned dataset (temperature in **Celsius**) |
| `data/processed/cleaning_log.csv` | Audit log of cleaning steps |
| `data/processed/outlier_cap_stats.csv` | Outlier cap thresholds and affected counts |
| `figures/data_cleaning/` | EDA/cleaning diagnostics |

---

## ML Prep Notebook (`week2_data_preparation_ml.ipynb`)

### Important feature engineering (including definition handling)

The feature engineering design evolved through four iterative cycles of testing and debugging. 
- Iteration 1: The original design treated the problem as a binary classification of event type (heat wave vs. cold snap) using all 12 event definitions mixed together, with extreme_temperature_C as a feature and definition_id dropped entirely. The first issue discovered was that ignoring definition_id caused massive pseudo-duplication (the same physical event appeared up to 12 times) and inflated the engineered trailing-rate and days-since-last-event features. 
- Iteration 2: Filtering to a single definition (Def 2) as a quick fix collapsed the dataset to ~1/12th its size, causing temperature to achieve perfect feature importance of 1.0 while all other features dropped to zero, because temperature is the definition threshold. Reverting to all definitions and engineering interpretable features from the definition metadata (def_percentile, def_temp_metric, def_min_duration) resolved the data volume problem, but the query function still produced identical risk levels across all regions because 9 of 11 input features were static all-time medians that never changed with the query date. 
- Iteration 3: Making those features seasonal (±30-day window around the query day-of-year) introduced date sensitivity but still failed to differentiate regions, because the underlying model had learned that temperature alone separates heat waves from cold snaps — a trivially correct but operationally useless insight. 
- Iteration 4: The final resolution was to reframe the ML task entirely: instead of classifying what type of event occurred (which temperature answers perfectly), the model now predicts whether an event occurs on a given (region, date) combination using a daily panel structure, with temperature excluded from features since it is unknown at query time, forcing the model to learn genuine regional and seasonal occurrence patterns.

1. **Definition metadata mapping from `definition_id`**
   - `Def1`..`Def12` are mapped to:
     - `def_temp_metric` (daily_mean / daily_max / daily_min)
     - `def_percentile` (e.g. 90, 95, 97.5, 99)
     - `def_min_duration` (2 or 3)
   - `definition_id` is then treated as metadata source, not a direct model feature.
   - Unmapped definitions are filled with defaults (`unknown`, `95.0`, `2.0`).

2. **Seasonality features**
   - `doy_sin`, `doy_cos`

3. **Historical recurrence features**
   - `region_hazard_rate_30d`
   - `days_since_last_event`
   - Computed using grouped history that includes `definition_id` in grouping to reduce cross-definition inflation.

4. **Daily panel conversion (critical change)**
   - Converts event rows into daily rows with binary target `event_occurred`.
   - Includes non-event days (negative sampling) so model learns occurrence probability, not just event class.

5. **Panel modeling feature set actually used for encoding/scaling**
   - Numeric: `def_percentile`, `def_min_duration`, `doy_sin`, `doy_cos`
   - Categorical: `NERC_ID`, `aggregation_method`, `def_temp_metric`, `hazard_query`
   - Explicitly excluded at inference-time feature stage: `extreme_temperature_C`, `duration_days`, `spatial_coverage_pct`, and history-derived fields.

### Split/encoding/scaling behavior
- Chronological split: train/val/test (70/15/15)
- One-hot fit on train, aligned for val/test
- StandardScaler fit on train numeric columns only

### Saved outputs

#### Data
| Path | Description |
|---|---|
| `data/processed/ml_ready.csv` | Feature-engineered dataset before final matrix export |
| `data/processed/train.csv` / `val.csv` / `test.csv` | Human-readable chronological splits |
| `data/processed/X_train.csv` / `X_val.csv` / `X_test.csv` | Encoded + scaled feature matrices |
| `data/processed/y_train.csv` / `y_val.csv` / `y_test.csv` | Binary target (`event_occurred`) |

#### Artifacts
| Path | Description |
|---|---|
| `data/processed/ml_artifacts/scaler.joblib` | Fitted scaler |
| `data/processed/ml_artifacts/label_encoder.joblib` | Label encoder |
| `data/processed/ml_artifacts/feature_names.json` | Ordered feature names |
| `data/processed/ml_artifacts/onehot_columns.json` | One-hot metadata for alignment |
| `data/processed/ml_artifacts/split_summary.txt` | Split report |
| `data/processed/ml_artifacts/feature_summary.csv` | Feature summary table |
| `figures/ml_preparation/` | ML prep visuals |

---

## References / Domain Notes

- Hazard types: `heat_wave`, `cold_snap`
- Aggregation methods: `SM`, `MWA`, `MWP`
- Definition IDs: `Def1`..`Def12` (mapped to physically meaningful definition features)
- Temperature unit in cleaned and ML-prep stages: **degree Celsius (C)**
- Kelvin appears only in raw input and pre-conversion cleaning steps.

# Week 2 Plan: Data Preparation and Documentation

## Context

We are team "Zero Shot Hopefuls" in the PNNL ESSD AI Competition, currently on Week 2. Our problem: **forecast compound hazards (co-occurring heatwave+drought, or wildfire during those conditions) 1-2 weeks ahead across CONUS**. The dataset is the ComExDBM_CONUS benchmark (~5.2 GB of NetCDF files, daily 0.5deg resolution, 2001-2020). Week 2 requires cleaned data, engineered features, data dictionary, cleaning scripts, ML algorithm selection, and documentation.

---

## Deliverable Checklist

| #   | Deliverable                                              | File                                                                                       |
| --- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| 1   | Original un-cleaned data                                 | `Data/original_sample.csv` + README pointer to NetCDF source                               |
| 2   | Cleaned dataset (CSV + JSON) with 3+ engineered features | `Data/cleaned_compound_hazard_weekly.csv` + `Data/cleaned_compound_hazard_weekly.json`     |
| 3   | Data dictionary (JSON)                                   | `Data/data_dictionary.json`                                                                |
| 4   | Cleaning scripts (AI-cited)                              | `Scripts/01_load_and_merge.py`, `Scripts/02_clean_and_engineer.py`, `Scripts/03_export.py` |
| 5   | ML algorithm selection + justification                   | In `README.md`                                                                             |
| 6   | requirements.txt                                         | `requirements.txt`                                                                         |
| 7   | Data preparation paragraph                               | In `README.md`                                                                             |

---

## Step 1: Branch & Folder Setup

- Create branch `zero-shot-hopefuls_week_2` from `week_2`
- Create folder structure:

```
zero-shot-hopefuls/
├── Data/
├── Scripts/
├── requirements.txt
└── README.md
```

## Step 2: Write `requirements.txt`

```
xarray>=2024.1.0
netCDF4>=1.6.5
pandas>=2.1.0
numpy>=1.26.0
scikit-learn>=1.4.0
lightgbm>=4.3.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

- Minimal and installable. No heavy geospatial stack needed (data is already on a regular grid).

## Step 3: Write `Scripts/01_load_and_merge.py` — Load & Merge NetCDF Data

**What it does:**

1. For each year (2001-2020), open 4 NetCDF files with `xarray.open_dataset()`:
   - `01_HeatWave/ComExDBM_{YYYY}_HeatWave_V02.nc` — keep: tmax, tmin, tmean, HI02, HI05
   - `02_FireData/ComExDBM_{YYYY}_Fires_V02.nc` — keep: FHS_c9, maxFRP_max_c9, BA_km2
   - `03_DroughtData/ComExDBM_{YYYY}_Drought_V02.nc` — keep: pdsi, pdsi_category, spi90d, spei90d
   - `04_Meteorological_Variables/ComExDBM_{YYYY}_MetVars_V02.nc` — keep: air_sfc, air_apcp, soilm, lhtfl, shtfl, rhum_2m, uwnd, vwnd, BCOC, LAI, NDVI, FWI
2. Merge on shared (time, lat, lon) dimensions with `xr.merge()`
3. Convert to DataFrame, filter to fire season (April-October)
4. Save intermediate parquet per year (memory management)
5. Also export 1000-row raw daily sample as `original_sample.csv`

**Variable selection rationale:** Drop redundant HI indices (keep HI02/HI05), drop lower-confidence fire data (FHS_c8c9), drop short-timescale drought indices (14d/30d — too noisy for weekly aggregation), drop upper-atmosphere variables (shum_250/850, uwnd_250/850, vwnd_250/850 — surface variables suffice for compound hazard forecasting).

**Key reference files:**

- `dataset/ComExDBM_CONUS/04_OutputData/README.md` — variable definitions
- `dataset/ComExDBM_CONUS/02_Code/08_Save_data_to_netcdf.py` — how NetCDFs were built (time dimension naming, variable naming conventions)

## Step 4: Write `Scripts/02_clean_and_engineer.py` — Clean & Engineer Features

### Cleaning Steps

1. **NaN detection & logging**: Count NaNs per variable per year, log summary
2. **Temperature NaN removal**: Drop rows where tmax/tmin/tmean are all NaN (following dataset creator's approach in `07_Compound_extremes_analysis.py`)
3. **Fire zero-fill**: Replace NaN in fire variables with 0 (no fire = 0, not missing)
4. **Met var forward-fill**: Forward-fill along time within each grid cell for LAI, NDVI, soilm (physical continuity)
5. **Outlier flagging**: Flag physically impossible values (negative precip, temp outside [-60, 60]C, negative FRP)
6. **Duplicate check**: Verify no duplicate (date, lat, lon) entries
7. **Type enforcement**: HI indicators as int (0/1), drought categories as int

### Engineered Features (7 new features, well above the 3 minimum)

| Feature                         | Definition                                                        | Rationale                                                    |
| ------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------ |
| `compound_hw_drought`           | HI02 == 1 AND pdsi_category <= 3                                  | Core compound event label — heatwave + mild-or-worse drought |
| `fire_during_compound`          | compound_hw_drought == 1 AND FHS_c9 > 0                           | Triple-compound: fire during heatwave+drought                |
| `wind_speed`                    | sqrt(uwnd^2 + vwnd^2) at surface                                  | Known wildfire driver, not directly in dataset               |
| `seasonal_sin` / `seasonal_cos` | sin/cos(2*pi*DOY/365)                                             | Cyclical seasonality encoding, avoids DOY discontinuity      |
| `temp_anomaly_7d`               | 7-day rolling mean of tmean minus grid-cell long-term weekly mean | Captures anomalous warming leading to heatwaves              |
| `drought_trend_14d`             | pdsi - pdsi_14_days_ago                                           | Worsening drought signal for forecasting                     |
| `fwi_7d_max`                    | 7-day rolling max of FWI                                          | Persistent fire-favorable conditions                         |

### Weekly Aggregation

- **Binary vars** (HI02, compound_hw_drought, fire_during_compound): weekly max (1 if any day)
- **Continuous vars** (temps, met vars, wind_speed): weekly mean
- **Fire vars** (FHS_c9, BA_km2): weekly sum
- **Rolling features**: value at end of week

### Target Variables (for Week 3 forecasting)

- `target_compound_1wk`: compound_hw_drought value NEXT week (1-week lead)
- `target_compound_2wk`: compound_hw_drought value 2 weeks ahead

## Step 5: Write `Scripts/03_export.py` — Export to Deliverable Formats

1. Concatenate all yearly weekly data
2. Export full cleaned dataset to CSV
3. Export representative sample (or full if size allows) to JSON (records format)
4. Generate and export `data_dictionary.json`
5. Export `original_sample.csv` (raw daily data before cleaning)

### Data Size Management

- Full weekly fire-season dataset: ~3,200 cells x 30 weeks x 20 years = ~1.9M rows x ~33 cols
- Estimated CSV size: ~200-300 MB (too large for comfortable git)
- **Strategy**: Commit the CSV gzipped (`cleaned_compound_hazard_weekly.csv.gz`), provide JSON as a 10K-row sample, and document that scripts regenerate the full dataset
- **Fallback**: If gzip is still too large, subset to 2011-2020 (10 years, ~100 MB)

## Step 6: Write `Data/data_dictionary.json`

JSON array with one entry per field:

```json
{
  "field_name": "compound_hw_drought",
  "description": "Binary: co-occurrence of heatwave (HI02=1) and drought (PDSI category <= 3)",
  "data_type": "int",
  "example_value": 1,
  "source": "Engineered from HI02 + pdsi_category",
  "context": "Core target for compound hazard forecasting"
}
```

Cover all ~33 columns: identifiers (year, week, lat, lon), raw variables, engineered features, targets.

## Step 7: Write `README.md`

### ML Algorithm Selection

**Primary — LightGBM**: Gradient-boosted trees excel on tabular data with mixed feature types and class imbalance (compound events are rare), provide built-in feature importance for interpretability, and handle spatial-temporal structure efficiently.

**Baseline — Logistic Regression (L2)**: Establishes a performance floor and provides a fully interpretable linear reference point to measure gains from the more complex model.

### Data Dictionary Table

Full table of all fields with description, data type, example.

### Data Preparation Paragraph

Covers: loading from NetCDF via xarray, merging 4 data sources on (date, lat, lon), cleaning steps (NaN handling, zero-fill, forward-fill, outlier detection), feature engineering (compound labels, derived meteorological features, temporal rolling statistics, seasonal encoding), weekly aggregation aligned with 1-2 week forecast horizon, fire season restriction (Apr-Oct), target variable creation via forward shift. Role of AI: Claude used for script generation (cited in comments), data exploration, and feature engineering design.

---

## ML Algorithm Justification (looking ahead to Weeks 3-5)

- Week 3 needs 2+ models → LightGBM + Logistic Regression
- Week 4 compares ML vs LLM → LightGBM's SHAP values make comparison rich
- Binary classification with class imbalance → LightGBM handles natively (`is_unbalance`)
- Rubric rewards "creative application" → LightGBM over vanilla Random Forest

---

## AI Citation Convention

```python
# AI-GENERATED: [tool] ([provider]). Prompt: "[summary]". Reviewed by [name].
```

---

## Verification

1. `uv pip install -r zero-shot-hopefuls/requirements.txt`
2. `python zero-shot-hopefuls/Scripts/01_load_and_merge.py` — produces intermediate parquet + original_sample.csv
3. `python zero-shot-hopefuls/Scripts/02_clean_and_engineer.py` — produces cleaned weekly parquet
4. `python zero-shot-hopefuls/Scripts/03_export.py` — produces final CSV, JSON, data_dictionary.json
5. Verify: CSV has 3+ engineered features, JSON is valid, data_dictionary.json covers all columns
6. Verify: README has ML selection, data dictionary table, data prep paragraph
7. Commit to `zero-shot-hopefuls_week_2`, PR into `week_2`

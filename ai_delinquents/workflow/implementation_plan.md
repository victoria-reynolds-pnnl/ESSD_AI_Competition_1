# Implementation Plan — Water Supply Volume Forecasting

## Context

The team reviewed the technical workflow plan and is ready for implementation. This plan maps each workflow step to specific scripts, data sources (with URLs and site IDs), and Python packages. The deliverable is a working pipeline from raw data download through model evaluation, structured to meet the Week 2 competition requirements (due 4/2).

---

## Files to Create

```
ESSD_AI_Competition_1/
├── data/
│   ├── raw/                        # created by download script
│   └── clean/                      # created by cleaning script
├── scripts/
│   ├── 01_download_data.py         # all data acquisition
│   ├── 02_clean_data.py            # QA/QC and temporal alignment
│   ├── 03_feature_engineering.py   # build feature matrix + target
│   ├── 04_train_model.py           # XGBoost + MLR + climatology baseline
│   └── 05_evaluate.py              # metrics, plots, comparison
├── requirements.txt
├── data_dictionary.md
└── README.md                       # update with model justification + data prep description
```

---

## Step 1: `requirements.txt`

```
pandas>=2.0
numpy
xgboost>=2.0
scikit-learn
matplotlib
plotly
dataretrieval          # USGS NWIS Python client
requests               # for direct CSV/text downloads
properscoring          # CRPS calculation
optuna                 # hyperparameter tuning
```

---

## Step 2: `scripts/01_download_data.py`

Downloads all raw data to `data/raw/`. Each dataset saved as a separate CSV.

### 2a. Natural streamflow at The Dalles (target variable)

- **Source**: BPA Historical Streamflow Data
- **URL**: `https://www.bpa.gov/energy-and-services/power/historical-streamflow-data`
  - Download the modified flow / natural flow dataset (Excel/CSV). BPA publishes monthly natural flows at The Dalles and other Columbia Basin points.
- **Fallback**: UW digital library dataset referenced in the idea file: `https://digital.lib.washington.edu/server/api/core/bitstreams/82048a9b-9bdd-4069-97d1-9c071b6ffaf1/content`
- **Fallback 2**: Nature scientific data paper: `https://www.nature.com/articles/s41597-026-06865-5` (likely has a Zenodo/figshare data deposit)
- **Output**: `data/raw/natural_flow_the_dalles.csv`
- **Implementation**: Use `requests` to download. If the BPA page requires navigating to a download link, we may need to manually grab the direct URL first and hardcode it. The download script will document the exact source URL used.

### 2b. USGS observed streamflow (for QC comparison)

- **Source**: USGS NWIS via `dataretrieval` Python package
- **Site ID**: `14105700` (Columbia River at The Dalles, OR)
- **Parameter**: `00060` (discharge, cfs)
- **Call**: `dataretrieval.nwis.get_record(sites="14105700", service="dv", start="1950-01-01", end="2025-12-31", parameterCd="00060")`
- **Output**: `data/raw/usgs_the_dalles_daily_q.csv`

### 2c. SNOTEL SWE data

- **Source**: NRCS AWDB (Air-Water Database) web service
- **Method**: Direct REST API calls to `https://wcc.sc.egov.usda.gov/reportGenerator/` — the NRCS Report Generator produces CSV output
- **Key stations** (Columbia Basin above The Dalles — will start with ~10–15 high-elevation stations with long records):
  - Need to identify stations programmatically. Use the NRCS inventory endpoint or hardcode a curated list of stations in the upper Columbia, Snake, and Willamette basins with records back to at least 1985.
- **Variables**: SWE (daily or 1st-of-month), snow depth
- **Output**: `data/raw/snotel_swe.csv` (long format: date, station_id, swe_inches)
- **Implementation**: Build the Report Generator URL with parameters for station list, date range, and element (WTEQ for SWE). Parse the CSV response with `pandas`.

### 2d. Climate indices

- **Source**: NOAA PSL Climate Indices
- **Direct download URLs** (plain text, space-delimited, year x 12 months):
  - PDO: `https://psl.noaa.gov/pdo/data/pdo.timeseries.ersstv5.csv` or `https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat`
  - Nino 3.4: `https://psl.noaa.gov/enso/mei/data/meiv2.data` or `https://psl.noaa.gov/data/correlation/nina34.anom.data`
  - PNA: `https://psl.noaa.gov/data/correlation/pna.data`
  - AMO: `https://psl.noaa.gov/data/correlation/amon.us.data`
- **Output**: `data/raw/climate_indices.csv` (columns: year, month, pdo, nino34, pna, amo)
- **Implementation**: `requests.get()` each URL, parse the fixed-width text format with `pandas.read_fwf()` or custom parsing (these files have a header line then year + 12 monthly values per row). Merge all indices into one dataframe on year-month.

### 2e. Basin-average temperature and precipitation (stretch goal)

- **Source**: GridMET via OpenDAP or pre-aggregated downloads
- **Deferred**: If time permits, add after core pipeline works. The climate indices capture most of the large-scale signal.

---

## Step 3: `scripts/02_clean_data.py`

Reads from `data/raw/`, writes to `data/clean/`.

### Operations (using `pandas`):

1. **Natural flow**: Parse BPA data, ensure monthly time step, convert units to kcfs-days or acre-feet if needed. Compute **April-September volume** as the dependent variable (sum of monthly flows x days-in-month x conversion factor). Output: `data/clean/target_apr_sep_volume.csv` (columns: water_year, volume_kcfs_days)

2. **USGS observed flow**: Resample daily to monthly mean. Used only for QC (plot natural vs. observed to verify regulation signal is removed). Output: `data/clean/usgs_monthly_q.csv`

3. **SNOTEL SWE**:
   - Flag stations with >15% missing in the target period
   - For retained stations, fill short gaps (<5 days) with linear interpolation
   - Extract April 1 SWE per station per year
   - Compute basin-average April 1 SWE (simple mean across stations)
   - Output: `data/clean/snotel_apr1_swe.csv`

4. **Climate indices**:
   - Align to common date index
   - Check for missing months, flag and note
   - Output: `data/clean/climate_indices_monthly.csv`

5. **Cleaning log**: Print summary stats (record lengths, missing %, stations dropped) to `data/clean/cleaning_log.txt`

---

## Step 4: `scripts/03_feature_engineering.py`

Reads cleaned data, produces a single feature matrix for modeling.

### Features computed:

| Feature column | Computation | Source file |
|---|---|---|
| `apr1_swe_anomaly_pct` | (Apr 1 SWE - median) / median x 100 | `snotel_apr1_swe.csv` |
| `oct_mar_precip_anomaly` | Oct-Mar cumulative precip anomaly (from SNOTEL cumulative precip as proxy, or GridMET if available) | `snotel_apr1_swe.csv` or GridMET |
| `djf_nino34` | Dec-Feb mean Nino 3.4 anomaly | `climate_indices_monthly.csv` |
| `jan_mar_mean_q` | Jan-Mar mean naturalized Q at The Dalles | `target_apr_sep_volume.csv` (monthly source) |
| `djf_pdo` | Dec-Feb mean PDO index | `climate_indices_monthly.csv` |

### Output:
- `data/clean/feature_matrix.csv` — one row per water year, columns: `water_year`, all features above, plus `target_volume`
- This is the primary input to modeling scripts

---

## Step 5: `scripts/04_train_model.py`

### Packages: `xgboost`, `scikit-learn`, `optuna`

### Models:

1. **Climatology baseline**: predict the long-term median volume for every year. No training needed — just compute median from training fold.

2. **Multiple Linear Regression (MLR)**:
   - `sklearn.linear_model.LinearRegression`
   - Fit on the feature matrix
   - LOYO cross-validation: for each year, fit on all others, predict held-out year

3. **XGBoost**:
   - `xgboost.XGBRegressor` for point predictions
   - Hyperparameter tuning via `optuna` (within the LOYO loop, use inner CV or a simple train/val split on the training fold to avoid leakage)
   - Key constraints: `max_depth` 3-6, `n_estimators` 50-300, `learning_rate` 0.01-0.3
   - For **probabilistic forecasts**: train three models with `objective="reg:quantile"` and `quantile_alpha` set to 0.1, 0.5, 0.9

### Cross-validation:
- LOYO: outer loop over each water year
- Store predictions in a dataframe: `water_year, obs, pred_clim, pred_mlr, pred_xgb, pred_xgb_q10, pred_xgb_q50, pred_xgb_q90`

### Outputs:
- `results/cv_predictions.csv` — all cross-validated predictions
- `results/xgb_feature_importance.csv` — feature importance scores from full-dataset model
- `results/model_xgb_final.json` — saved final XGBoost model (trained on all data)

---

## Step 6: `scripts/05_evaluate.py`

### Packages: `scikit-learn`, `properscoring`, `matplotlib`, `plotly`

### Deterministic metrics (computed from `cv_predictions.csv`):
- NSE, KGE (custom functions — ~5 lines each), MAPE, skill score vs. climatology
- Produce a summary table: `results/metrics_summary.csv`

### Probabilistic metrics:
- CRPS via `properscoring.crps_ensemble` or `properscoring.crps_gaussian`
- Prediction interval coverage: % of observations falling within [q10, q90] (target: ~80%)
- Pinball loss for each quantile

### Plots:
- Observed vs. predicted time series (all three models)
- 1:1 scatter plot with prediction intervals
- Feature importance bar chart
- Reliability diagram (binned coverage)
- All saved to `results/` as PNG

### Comparison against operational forecasts:
- If NRCS/NWRFC/BPA hindcast data can be obtained, compute the same metrics and add to summary table
- Otherwise, note published skill metrics from the literature for qualitative comparison

---

## Step 7: `data_dictionary.md`

Document all fields in `feature_matrix.csv` and `cv_predictions.csv`:

| Field | Description | Type | Units | Example |
|---|---|---|---|---|
| `water_year` | Water year (Oct Y-1 to Sep Y) | int | year | 2005 |
| `target_volume` | Apr-Sep naturalized flow volume | float | kcfs-days | 95000 |
| `apr1_swe_anomaly_pct` | Basin-avg Apr 1 SWE, % of median | float | % | 112.5 |
| ... | ... | ... | ... | ... |

---

## Step 8: Update `README.md`

Add the required sections:
- **ML Model Selection**: 1-2 sentences on XGBoost choice (per deliverable requirements)
- **Data Dictionary Table**: reference or embed from `data_dictionary.md`
- **Data Preparation Paragraph**: cleaning approach, normalization, feature engineering, role of AI at each step

---

## Implementation Order

1. `requirements.txt` — so everyone can set up their environment
2. `scripts/01_download_data.py` — get the data flowing first
3. `scripts/02_clean_data.py` — QA/QC pass
4. `scripts/03_feature_engineering.py` — build the feature matrix
5. `data_dictionary.md` — document what we have (good to do early)
6. `scripts/04_train_model.py` — train and cross-validate
7. `scripts/05_evaluate.py` — metrics and plots
8. Update `README.md` — final deliverable polish

---

## Verification

- After step 2: confirm `data/raw/` files are non-empty and have expected date ranges
- After step 3: check `data/clean/` files have no unexpected NaN, plot time series for visual QC
- After step 4: verify `feature_matrix.csv` has one row per water year, no missing target values
- After step 6: check that LOYO predictions exist for every year, NSE > 0 (better than mean), prediction intervals cover ~80%
- End-to-end: run all scripts in sequence (`python scripts/01_download_data.py && python scripts/02_...` etc.)

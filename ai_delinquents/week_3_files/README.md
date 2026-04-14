# ESSD AI Competetion - Week 3

**Project Title**: Water supply forecasts for the Columbia River Basin

**Team Name**: AI Delinquents

**Team Members**:

| Name | Role | Favorite Sci-fi Robot or AI | Expertise | Responsibilities |
| --- | --- | --- | --- | --- |
| @cameronbracken |  Domain, compute, data |  Chappy, Johnny 5, Wall-E | Hydrology, Water resources, Hydropower| data managment, AI techniques|
| @stefan-rose | Geospatial analysis, data processing | TARS, Baymax | GIS, Remote Sensing, Data Viz | data analysis and viz |
| @mdsturtevant-pnnl | Software Engineer | C3PO, Liberty Prime | Web Apps and UIs | Flashy Deliverables? |
| @amanda-lawter |Resarch Analyst, data processing  | Roz | Geochemistry, Subsurface, Contaminant Monitoring and Remediation | data analysis |
| @JanaSimo |Subsurface Science | Roz | Subsurface, fluid flow, reservoir geoscience | data analysis, application |

## Week 3

- [Idea](idea_water_supply_forecast.md)
- [Deliverable requirements](ai_delinquents_week_2_deliverable.md)
- [Implementation plan](implementation_plan.md)
- [Data dictionary](data_dictionary.md)

---

## ML Model Selection

We selected **XGBoost (Extreme Gradient Boosting)** for this water supply forecasting problem. XGBoost is well-suited because (1) it handles the moderate sample size (34 complete water years) without overfitting when properly regularized, (2) it captures nonlinear interactions between snowpack, climate indices, and antecedent flow conditions that linear models miss, (3) it natively supports quantile regression for producing probabilistic forecasts (prediction intervals), and (4) it provides built-in feature importance scores for interpretability. We also train a multiple linear regression (MLR) model and a climatology baseline for comparison, following standard practice in seasonal hydrologic forecasting.

## Data Dictionary

See the full [data dictionary](data_dictionary.md) for detailed field descriptions.

**Feature Matrix** (`data/clean/feature_matrix.csv`) — 34 water years (1985-2018), 6 engineered features:

| Field | Description | Units | Source |
|---|---|---|---|
| `target_volume` | Apr-Sep naturalized flow volume at The Dalles | kcfs-days | BPA |
| `apr1_swe_anomaly_pct` | April 1 SWE, % departure from median (r=0.79) | % | NRCS SNOTEL |
| `djf_nino34` | Dec-Feb Nino 3.4 SST anomaly (r=-0.46) | deg C | NOAA PSL |
| `djf_pdo` | Dec-Feb PDO index (r=-0.45) | unitless | NOAA PSL |
| `djf_pna` | Dec-Feb PNA pattern index (r=-0.40) | unitless | NOAA PSL |
| `jan_mar_mean_q_cfs` | Jan-Mar mean naturalized flow (r=0.56) | cfs | BPA |
| `oct_mar_volume_kcfs_days` | Oct-Mar cumulative flow volume (r=0.55) | kcfs-days | BPA |

## Data Preparation

### Data Sources

Raw data was acquired from four sources using automated download scripts:

1. **BPA Historical Streamflow Reports** — Monthly naturalized ("modified") streamflow at The Dalles, OR (control point TDA6M), WY 1929-2019. This removes reservoir regulation effects to represent natural basin response.
2. **USGS NWIS** (site 14105700) — Daily observed discharge at The Dalles, used for QC comparison against natural flow. Retrieved via the `dataretrieval` Python package.
3. **NRCS SNOTEL** — Monthly snow water equivalent from 16 stations spanning the Columbia Basin (Montana, Idaho, Oregon, Washington), 1985-2025. Retrieved via the NRCS Report Generator API.
4. **NOAA Physical Sciences Laboratory** — Monthly climate teleconnection indices (PDO, Nino 3.4, PNA, AMO) dating back to 1948.

### Cleaning

- BPA flow data was parsed from Excel files in the BPA zip archive and converted to monthly volumes (kcfs-days) using month-specific day counts
- SNOTEL stations were checked for completeness; all 16 stations had >85% coverage and were retained. April 1 values were extracted and averaged across stations to produce a basin-wide SWE index
- Climate indices were parsed from NOAA's fixed-width text format; missing values (coded as -99.9) were flagged as NaN

### Feature Engineering

Six features were engineered from the cleaned data, all representing conditions known *before* the April 1 forecast date:

1. **April 1 SWE anomaly** — Basin-average SWE as percent departure from the 1985-2018 median. This is the single strongest predictor (r=0.79), consistent with the well-known dominance of snowpack in Columbia Basin water supply.
2. **DJF Nino 3.4** — Winter-season ENSO signal. La Nina winters (negative values) tend to produce above-normal snowpack and runoff in the Pacific Northwest.
3. **DJF PDO** — Pacific Decadal Oscillation captures decadal-scale SST patterns that modulate ENSO's influence on PNW precipitation.
4. **DJF PNA** — Pacific-North American pattern reflects the large-scale atmospheric circulation; negative PNA favors enhanced precipitation in the PNW.
5. **Jan-Mar mean flow** — Antecedent naturalized flow captures early-season runoff and base flow conditions.
6. **Oct-Mar volume** — Cumulative antecedent volume integrates the full cold-season hydrologic response.

### Role of AI

All data download, cleaning, and feature engineering scripts (`scripts/01_download_data.py`, `scripts/02_clean_data.py`, `scripts/03_feature_engineering.py`) were generated with assistance from Claude (Anthropic). The human role focused on selecting data sources, curating SNOTEL station locations, choosing feature definitions based on domain knowledge of Columbia Basin hydroclimatology, and validating outputs. AI-generated code sections are marked in script headers.

## Week 3: Model Training & Evaluation

### Train/Validation/Test Split Methodology

We use a **temporal hold-out split** to prevent data leakage, consistent with how operational seasonal forecasts are issued. The dataset is divided chronologically: water years 1985–2012 (28 years) form the training set, and water years 2013–2018 (6 years, ~18% of data) are held out as the test set. The test set was not used in any way during model development. For hyperparameter tuning, we apply **Leave-One-Year-Out (LOYO) cross-validation** within the training set — in each fold, one year is withheld and the model is trained on all remaining years. This preserves the time structure of the data, avoids look-ahead bias, and reflects the real-world constraint that future conditions are always unknown. LOYO is preferred over k-fold for this dataset because the small sample size (28 training years) makes standard k-fold folds too small to be meaningful, and LOYO directly mirrors the sequential nature of seasonal forecasting.

### Model Performance Summary

*Run `pixi run python scripts/evaluate.py` to populate this section with computed metrics. Results are saved to `outputs/metrics_summary.csv` and `outputs/model_performance_summary.txt`.*

We compare three models: a **climatology baseline** (training-set median, constant prediction), **Multiple Linear Regression (MLR)** with StandardScaler-normalized features, and **XGBoost** with optuna-tuned hyperparameters and quantile regression (q10/q50/q90) for probabilistic prediction intervals. Primary deterministic metrics are **NSE** (Nash-Sutcliffe Efficiency; 1=perfect, 0=mean baseline) and **KGE** (Kling-Gupta Efficiency; penalizes bias, variance, and correlation errors simultaneously). These are the standard metrics in operational hydrology and allow direct comparison against published forecast skill from NRCS and NWRFC. Probabilistic performance is assessed using **CRPS** (Continuous Ranked Probability Score; lower is better) and prediction interval coverage (target: ~80% of observations fall within q10–q90).

### Human-in-the-Loop Domain Review

- **Does the model behavior make sense given domain expectations?** No, not at all. We see the XGboost, a sophisticated ML model is not doing as well as the less sophisticated Multi-linear regression (MLR) model based on the validation metrics provided by the model. This could be due to the AI setting up the model incorrectly. We expect the problem to be ML model to be as good or better than the MLR, because we expect the data to be non-linear. 

- **Any concerning spurious correlations?** No, correlations between volume and flow for the same months are highly correlated. SWE is highly correlated with spring flow as expected as most flow at that time of year is related to the snow pack. No concerns when looking at the feature importance maps - this matches our expert expectations. 

- **Any high-risk failure modes?** There were high points in 1997 and 2011. Neither model (MLR or XGBoost) predicted the 2011 peak but MLR closely predicted the 1997 peak. There was also a low in 2001 that MLR predicted well while XGBoost did not. 

### Failure and Limitations Review

To summarize, MLR unexpectedly outperformed the XGBoost model. This is unexpected because we do not expect linear results, and therefore there may be an issue with our model set up. Our next step will be to evaluate the model to see what went wrong with the XGBoost. 

---

## Running the Pipeline

```bash
# Install dependencies
pixi install
# Or: pip install -r requirements.txt

# Week 2: Data pipeline
pixi run python scripts/01_download_data.py        # Download raw data
pixi run python scripts/02_clean_data.py            # Clean and QA/QC
pixi run python scripts/03_feature_engineering.py   # Build feature matrix

# Week 3: EDA, training, evaluation, interpretability
pixi run python scripts/04_eda_plots.py             # EDA visualizations → plots/
pixi run python scripts/train.py                    # Train models → outputs/ + models/
pixi run python scripts/evaluate.py                 # Metrics → outputs/
pixi run python scripts/interpretability.py         # Model figures → visualizations/
```

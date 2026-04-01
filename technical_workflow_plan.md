# Technical Workflow Plan — Water Supply Volume Forecasting for the Columbia River Basin

## 1. Data Acquisition

### Target Variable
- **April–September naturalized streamflow volume at The Dalles** (acre-feet or kcfs-days)
- Source natural flow data from BPA historical streamflow records or the referenced UW/nature.com datasets
- If expanding beyond The Dalles, prioritize locations where NRCS/NWRFC issue operational forecasts (for direct comparison)

### Predictor Data

| Category | Source | Variables | Temporal Resolution |
|---|---|---|---|
| Snow | NRCS SNOTEL | SWE, snow depth, cumulative precipitation | Daily → monthly aggregates |
| Streamflow (antecedent) | USGS NWIS | Discharge at The Dalles and upstream gages | Daily → monthly aggregates |
| Climate indices | NOAA PSL | PDO, ENSO (Niño 3.4), PNA, AMO | Monthly |
| Temperature & precipitation | PRISM or GridMET | Basin-average T and P | Daily → monthly aggregates |

### Acquisition Approach
- Use `dataretrieval` (Python) for USGS NWIS pulls
- Use `climata` or direct CSV download for SNOTEL
- Script all downloads so they are reproducible (`scripts/01_download_data.py`)

---

## 2. Data Cleaning and QA/QC

1. **Gap detection**: flag missing values per station per month; drop stations with >15% missing in the training period
2. **Outlier screening**: flag values beyond ±4σ from the monthly climatology; inspect and annotate rather than auto-remove
3. **Temporal alignment**: resample all predictors to a common monthly time step; align water-year conventions (Oct–Sep)
4. **Natural flow QC**: cross-check BPA natural flows against USGS observed flows to confirm regulation effects are removed (regulated flows will be systematically lower during spring and higher during winter)
5. **Flow volume calculation**: For the dependent variable compute seasonal streamflow volume (April–September)

Deliverable: `data/raw/` (original downloads), `data/clean/` (processed), and a cleaning log

---

## 3. Feature Engineering (minimum 3 new features)

| # | Feature | Rationale |
|---|---|---|
| 1 | **April 1 SWE anomaly** (basin-average, % of median) | Single strongest predictor of spring/summer runoff in snowmelt-dominated basins |
| 2 | **Oct–Mar cumulative precipitation anomaly** | Captures total water input before the forecast date |
| 3 | **Winter ENSO state** (DJF Niño 3.4 average) | Teleconnection to Pacific NW precipitation; La Niña years tend to produce above-normal snowpack |
| 4 | **Antecedent streamflow index** (Jan–Mar mean Q at The Dalles) | Integrates baseflow and early-melt signals |
| 5 | **PDO phase** (winter-mean PDO index) | Modulates ENSO influence on PNW hydrology; in-phase PDO+ENSO amplifies signal |

All anomalies computed relative to a 1981–2010 (or available period) climatology.

---

## 4. Model Selection

### Recommended: Gradient Boosted Trees (XGBoost or LightGBM)

**Justification**:
- Handles mixed feature types (continuous indices + categorical climate states) without extensive preprocessing
- Captures non-linear predictor–response relationships (e.g., SWE–runoff curve is non-linear at extremes)
- Built-in feature importance for interpretability, which matters for comparison with physics-based operational forecasts
- Performs well on tabular datasets with moderate sample sizes (40–80 years of annual forecasts)
- Strong track record in hydrology competitions (e.g., USGS/Bureau of Reclamation forecast rodeos)

### Baseline Comparison Models
- **Multiple linear regression** (MLR): the traditional operational method; provides a benchmark
- **Climatology**: long-term median volume; the no-skill reference for skill scores

### Hyperparameter Tuning
- Use `optuna` or `sklearn.model_selection.RandomizedSearchCV`
- Key parameters: `max_depth`, `n_estimators`, `learning_rate`, `subsample`, `colsample_bytree`
- Constraint: keep tree depth shallow (3–6) to reduce overfitting given small sample size

---

## 5. Training and Validation Strategy

### Cross-Validation: Leave-One-Year-Out (LOYO) or Expanding Window

- **LOYO** (standard approach): for each year, train on all other years, predict the held-out year. Appropriate here because water supply forecasting is an annual prediction, and temporal autocorrelation between annual volumes is weak.
- **Stationary bootstrapping** (alternative more computationally intensive): .
- **Expanding window** (alternative): train on years 1–t, predict year t+1. More conservative; mimics real-time operations.


Both approaches avoid data leakage from future years.

---

## 6. Risk Quantification and Evaluation

### Deterministic Metrics

| Metric | Formula / Description | Purpose |
|---|---|---|
| **NSE** (Nash-Sutcliffe Efficiency) | 1 − Σ(obs−pred)² / Σ(obs−mean)² | Overall accuracy; >0.7 is good for annual volumes |
| **KGE** (Kling-Gupta Efficiency) | Decomposes into correlation, bias, variability | Better than NSE for diagnosing error sources |
| **MAPE** | Mean absolute percentage error | Interpretable error in % units |
| **Skill Score** | 1 − MSE_model / MSE_climatology | Quantifies improvement over no-skill baseline |

### Probabilistic / Uncertainty Metrics

To quantify forecast risk (not just point accuracy):

| Metric | Method | Purpose |
|---|---|---|
| **Prediction intervals** | Quantile regression (XGBoost with `quantile` objective) for 10th, 50th, 90th percentiles | Communicates forecast uncertainty range |
| **CRPS** (Continuous Ranked Probability Score) | Score the full predictive distribution against observations | Gold-standard probabilistic verification |
| **Reliability diagram** | Bin predicted probabilities vs. observed frequencies | Checks if 80% prediction intervals actually contain 80% of observations |
| **Pinball loss** | Asymmetric loss for each quantile | Verifies quantile calibration |

### Evaluation Against Trusted Forecasts

| Reference Forecast | Source | Comparison Method |
|---|---|---|
| NRCS statistical forecasts | NRCS predefined reports | Side-by-side skill scores (same verification period) |
| NWRFC ESP-based forecasts | NWRFC water supply page | Compare CRPS and reliability |
| BPA forecasts | BPA historical data | Bias and MAPE comparison |

Compute **relative skill**: skill_score = 1 − CRPS_model / CRPS_reference. A positive value means the ML model outperforms the reference.

---

## 7. Proposed Directory Structure

```
ESSD_AI_Competition_1/
├── data/
│   ├── raw/              # Original downloaded data (do not modify)
│   └── clean/            # Cleaned data with engineered features
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_clean_data.py
│   ├── 03_feature_engineering.py
│   ├── 04_train_model.py
│   └── 05_evaluate.py
├── notebooks/            # Exploratory analysis (optional)
├── results/              # Figures, tables, model outputs
├── requirements.txt
├── data_dictionary.md
└── README.md
```

---

## 8. Workflow Summary

```
Download raw data (scripted, reproducible)
        │
        ▼
QA/QC and cleaning (gap-fill, outlier flags, temporal alignment)
        │
        ▼
Feature engineering (SWE anomaly, precip anomaly, ENSO, antecedent Q, PDO)
        │
        ▼
Train XGBoost + MLR baseline (LOYO cross-validation)
        │
        ▼
Evaluate: deterministic (NSE, KGE, skill score) + probabilistic (CRPS, prediction intervals)
        │
        ▼
Compare against NRCS / NWRFC / BPA operational forecasts
        │
        ▼
Document results and deliverables
```

---

## 9. Key Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Small sample size (40–80 years) | Overfitting, unreliable validation | Shallow trees, regularization, LOYO CV |
| Natural flow data availability | Can't train without target variable | Multiple sources identified (BPA, UW); fall back to USGS observed if needed and note limitation |
| Non-stationarity (climate change) | Historical relationships may not hold | Include climate indices as features; test on recent decades specifically |
| SNOTEL station record length | Limits training period | Use only stations with records back to at least 1985 |

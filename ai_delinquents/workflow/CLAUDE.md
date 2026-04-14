# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **AI Delinquents** team repository for the ESSD AI Competition. The project goal is to build a machine learning model for **water supply forecasting in the Columbia River Basin** using streamflow, climate indices, and snow data.

Key focus: The Dalles as the primary forecast location, using **naturalized** (not regulated) streamflow data.

## Team

| GitHub | Expertise | Role |
|---|---|---|
| @cameronbracken | Hydrology, water resources, hydropower | Domain lead, compute, data management |
| @stefan-rose | GIS, remote sensing, data viz | Geospatial analysis, data processing |
| @mdsturtevant-pnnl | Web apps, software engineering | Deliverables and UI |
| @amanda-lawter | Geochemistry, subsurface science | Data analysis |
| @JanaSimo | Subsurface science, fluid flow | Data analysis, application |

Branch strategy: week-based branches off `main` (e.g., `ai_delinquents_week_2`, `ai_delinquents_week_3`).

## Python Environment

**Always use `pixi run python ...`** — the base conda environment has a pandas/numpy version conflict.

```bash
pixi install                        # set up environment
pixi run python scripts/foo.py      # run any script
pixi add <package>                  # add a new dependency
```

## What Was Done Each Week

### Week 1 (concept and planning)

- Defined the project concept: seasonal water supply volume forecasting at The Dalles, OR
- Selected The Dalles as primary forecast point (most important Columbia Basin control)
- Decided to use **naturalized** (BPA modified) flow, not regulated observed flow
- Identified data sources: BPA natural flow, USGS NWIS, NRCS SNOTEL, NOAA PSL climate indices
- Identified operational forecasts for comparison: NRCS, NWRFC, BPA
- Wrote `idea_water_supply_forecast.md`

### Week 2 (data pipeline and feature engineering, delivered 4/2)

**Deliverables completed:**
- `scripts/01_download_data.py` — downloads all raw data (BPA flow, USGS NWIS via `dataretrieval`, NRCS SNOTEL via Report Generator API, NOAA PSL climate indices)
- `scripts/02_clean_data.py` — QA/QC, temporal alignment, produces cleaned CSVs and `data/clean/cleaning_log.txt`
- `scripts/03_feature_engineering.py` — builds `feature_matrix.csv` (one row per water year)
- `data_dictionary.md` — full field documentation
- `requirements.txt` and `pyproject.toml` (pixi) for environment reproducibility
- `README.md` updated with ML model selection, data dictionary, and data preparation sections

**Model selection:** XGBoost (primary) + MLR + climatology baseline. XGBoost chosen for nonlinear interaction handling, probabilistic forecasts (stretch goal), and feature importance.

**Feature matrix** (`data/clean/feature_matrix.csv`): 34 water years (WY 1985–2018), 6 features + target:

| Feature | Description | r with target |
|---|---|---|
| `apr1_swe_inches` | Basin-avg April 1 SWE (16 SNOTEL stations) | — |
| `apr1_swe_anomaly_pct` | April 1 SWE % departure from median | 0.79 |
| `djf_nino34` | Dec-Feb Nino 3.4 SST anomaly | -0.46 |
| `djf_pdo` | Dec-Feb PDO index | -0.45 |
| `djf_pna` | Dec-Feb PNA pattern index | -0.40 |
| `jan_mar_mean_q_cfs` | Jan-Mar mean naturalized flow at The Dalles | 0.56 |
| `oct_mar_volume_kcfs_days` | Oct-Mar cumulative naturalized flow volume | 0.55 |
| `target_volume` | Apr-Sep naturalized flow volume (kcfs-days) | — |

### Week 3 (EDA and visualization)

- `scripts/04_eda_plots.py` — reproducible EDA script; reads `data/clean/`, saves 8 PNGs to `plots/`
  - `01_target_distribution.png` — histogram + KDE of target variable
  - `02_target_timeseries.png` — annual target volume with OLS trend line
  - `03_feature_correlations.png` — Pearson correlation heatmap of feature matrix
  - `04_swe_vs_target_enso.png` — SWE vs target scatter, colored by DJF ENSO phase
  - `05_monthly_hydrograph.png` — box plots of monthly naturalized flow (seasonal cycle)
  - `06_climate_indices_timeseries.png` — PDO, Nino3.4, PNA time series
  - `07_snotel_station_heatmap.png` — Apr 1 SWE per station × water year heatmap
  - `08_predictor_scatter_matrix.png` — pairplot of top 5 predictors
- `seaborn` added as a project dependency (`pixi add seaborn`)

## Next Steps (planned)

- `scripts/04_train_model.py` — XGBoost + MLR + climatology baseline, LOYO cross-validation, quantile forecasts
- `scripts/05_evaluate.py` — NSE, KGE, CRPS metrics; comparison against operational forecasts

## Data Sources

- **Natural flow**: BPA Historical Streamflow Reports (TDA6M control point, WY 1929-2019)
- **Observed flow (QC)**: USGS NWIS site 14105700 (Columbia River at The Dalles, OR)
- **Snow**: NRCS SNOTEL — 16 stations across Montana, Idaho, Oregon, Washington
- **Climate indices**: NOAA PSL (PDO, Nino3.4, PNA, AMO)
- **Validation benchmarks**: NRCS, NWRFC, BPA seasonal forecasts

## Key Files

- `idea_water_supply_forecast.md` — original project concept
- `implementation_plan.md` — detailed pipeline design (scripts 01–05)
- `data_dictionary.md` — full field documentation for all cleaned datasets
- `README.md` — week 2 deliverable (ML selection, data prep, data dictionary)
- `ai_delinquents_week_2_deliverable.md` — week 2 requirements checklist
- `week_2_resources.md` — team learning resources

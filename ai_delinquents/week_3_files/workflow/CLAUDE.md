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

### Week 3 (EDA, model training, and evaluation — delivered ~4/9)

**File location**: `ai_delinquents/week_3_files/`

EDA:
- `scripts/04_eda_plots.py` — reads `data/clean/`, saves 8 PNGs to `plots/`

Training and evaluation pipeline:
- `scripts/train.py` — temporal split (train WY 1985–2012 / test WY 2013–2018), optuna LOYO hyperparameter tuning, trains climatology + MLR + XGBoost (point + quantile), saves models to `models/` and predictions to `outputs/`
- `scripts/evaluate.py` — NSE, KGE, RMSE, CRPS, PI coverage; saves `outputs/metrics_summary.csv` and `outputs/model_performance_summary.txt`
- `scripts/interpretability.py` — 3 figures saved to `visualizations/`

**Test set results (WY 2013–2018):**

| Model | NSE | KGE | RMSE (kcfs-days) |
|---|---|---|---|
| Climatology | −0.03 | −0.41 | 8,488 |
| MLR | **0.77** | **0.74** | **4,034** |
| XGBoost | 0.46 | 0.50 | 6,170 |

MLR outperforms XGBoost on the test set — expected given small sample size (34 years). XGBoost PI coverage was ~67% (target ~80%).

### Week 4 (LLM benchmarking — delivered ~4/16)

**File location**: `ai_delinquents/week_4_files/`

- `Notebooks/llm_benchmark.ipynb` — benchmarks three locally hosted LLMs on the same regression task as Week 3; uses `DRY_RUN` flag for local testing (set to `False` on JupyterHub)
- `Prompts/prompt_template.txt` and `few_shot_samples.json` — versioned prompt artifacts
- `Results/` — per-model clean/raw CSVs and metrics summary

**LLM test set results (WY 2013–2018, same split as Week 3):**

| Model | Prompt | NSE | RMSE (kcfs-days) |
|---|---|---|---|
| gemma-3-12b-it | v2 few-shot | 0.33 | 6,817 |
| Phi-3.5-mini-instruct | v1 zero-shot | 0.12 | 7,824 |
| Phi-mini-MoE-instruct | v2 few-shot | −0.57 | 10,472 |

Key findings: 100% parse success; dominant LLM failure mode is regression to the mean (compressed prediction variance); gemma (12B) substantially outperforms Phi models; few-shot prompting benefits larger models but hurt Phi-3.5-mini; LLMs cannot match purpose-built regression models on low-dimensional tabular tasks.

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

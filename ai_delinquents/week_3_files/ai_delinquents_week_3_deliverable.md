# ESSD AI Competition — Week 3 Deliverable

**Project Title**: Water supply forecasts for the Columbia River Basin

**Team Name**: AI Delinquents

**Due**: Week 3

---

## Deliverables Checklist

### 1. Train/Validation/Test Split Methodology
- [x] Written in `README.md` (Week 3 section)
- **Summary**: Temporal hold-out — train WY 1985-2012, test WY 2013-2018. LOYO CV within training set for hyperparameter tuning. Prevents data leakage.

### 2. Reproducible Training Workflow
- [x] `scripts/train.py` — loads feature matrix, splits data, tunes XGBoost with optuna, trains MLR + XGBoost + climatology baseline, saves models and predictions
- [x] `requirements.txt` updated (added seaborn, joblib)
- Script sub-tasks:
  - [x] Load cleaned dataset from week 2 (`data/clean/feature_matrix.csv`)
  - [x] Perform train/validation/test split (temporal hold-out)
  - [x] Train models (climatology, MLR, XGBoost point + quantile)
  - [x] Tune XGBoost hyperparameters using LOYO validation (optuna, 50 trials)
  - [x] Evaluate final performance on test set

### 3. Data and Results Files
- [x] `outputs/train_split.csv` — WY 1985-2012 features + target
- [x] `outputs/test_split.csv` — WY 2013-2018 features + target
- [x] `outputs/loyo_cv_predictions.csv` — LOYO cross-validated predictions (all models)
- [x] `outputs/test_predictions.csv` — test set predictions + prediction intervals

### 4. Model Performance Summary
- [x] `scripts/evaluate.py` computes: NSE, KGE, RMSE, MAE, MAPE, skill score, CRPS, PI coverage
- [x] `outputs/metrics_summary.csv` — all metrics tabulated
- [x] `outputs/model_performance_summary.txt` — formatted text summary
- [ ] Written summary in `README.md` (Week 3 section) — **fill in after running evaluate.py**

### 5. Visualization
- [x] `scripts/interpretability.py` generates:
  - [x] `visualizations/obs_vs_pred_timeseries.png` — time series, all models, with PI band
  - [x] `visualizations/scatter_obs_vs_pred.png` — 1:1 scatter, all models
  - [x] `visualizations/feature_importance.png` — XGBoost gain-based importance

### 6. Human-in-the-Loop Review
- [ ] Domain review writeup in `README.md` (Week 3 section) — **assigned to Cameron or Jana**
  - [ ] Does model behavior make sense given domain expectations?
  - [ ] Any concerning spurious correlations?
  - [ ] Any high-risk failure modes?

### 7. Failure and Limitations Review
- [ ] Written paragraph in `README.md` (Week 3 section) — **complete after reviewing model outputs**

---

## How to Run

```bash
pixi install
pixi run python scripts/train.py
pixi run python scripts/evaluate.py
pixi run python scripts/interpretability.py
```

Expected outputs after running:
```
outputs/
  train_split.csv
  test_split.csv
  loyo_cv_predictions.csv
  test_predictions.csv
  metrics_summary.csv
  model_performance_summary.txt
models/
  xgb_final.json
  xgb_q10_final.json
  xgb_q50_final.json
  xgb_q90_final.json
  mlr_final.pkl
  mlr_scaler.pkl
visualizations/
  obs_vs_pred_timeseries.png
  scatter_obs_vs_pred.png
  feature_importance.png
```

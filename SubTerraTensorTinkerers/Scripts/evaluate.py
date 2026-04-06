import json
import os
import pickle
from pathlib import Path
import importlib

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import RegressorMixin

try:
    XGBRegressor = importlib.import_module("xgboost").XGBRegressor
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "xgboost is required for model training."
    ) from exc


class CompatibleXGBRegressor(XGBRegressor, RegressorMixin):
    # Explicit estimator type for strict sklearn compatibility checks.
    _estimator_type = "regressor"


EXCLUDED_FEATURE_TERMS = ("Phase", "Flag", "TS", "TU")  # Features focus on hot injection phase measurements (TC -> TL, TN)


################# Configuration and Paths #################
DATA_DIR = Path(r"../Data")
SPLIT_DATA_DIR = DATA_DIR / "3_split"

MODEL_DIR = Path(r"../Models")
MODEL_OUT_DIR = MODEL_DIR / "xgboost"

TEST_FILE = SPLIT_DATA_DIR / "FTES_1hour_test.csv"
PARAMS_FILE = MODEL_OUT_DIR / "xgb_best_params.json"
CONFIG_FILE = MODEL_OUT_DIR / "xgb_training_config.json"

METRICS_OUT = MODEL_OUT_DIR / "xgb_test_metrics.csv"
PRED_OUT = MODEL_OUT_DIR / "xgb_test_predictions.csv"


try:
    XGBRegressor = importlib.import_module("xgboost").XGBRegressor
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "xgboost is required for evaluation."
    ) from exc


def make_supervised_xy(split_df_in, targets, horizon_h):
    x_num = split_df_in.select_dtypes(include="number").copy()
    kept_cols = [
        c for c in x_num.columns
        if not any(term in c for term in EXCLUDED_FEATURE_TERMS)
    ]
    x_num = x_num[kept_cols]
    y = split_df_in[targets].shift(-horizon_h)

    valid_mask = ~y.isnull().any(axis=1)
    x_num = x_num.loc[valid_mask]
    y = y.loc[valid_mask]
    return x_num, y


def safe_pct_improvement(baseline_value, model_value):
    if baseline_value == 0:
        return np.nan
    return 100.0 * (baseline_value - model_value) / abs(baseline_value)


def load_trained_model(model_out_dir, safe_name):
    """Load model with JSON-first strategy to avoid pickle class-resolution issues."""
    json_path = model_out_dir / f"xgb_{safe_name}.json"
    pkl_path = model_out_dir / f"xgb_{safe_name}.pkl"

    if json_path.exists():
        model = CompatibleXGBRegressor()
        model.load_model(json_path)
        return model

    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    return None


script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if not TEST_FILE.exists():
    raise FileNotFoundError(f"Missing test split file: {TEST_FILE}")
if not PARAMS_FILE.exists():
    raise FileNotFoundError(f"Missing parameter file: {PARAMS_FILE}")
if not CONFIG_FILE.exists():
    raise FileNotFoundError(f"Missing training config file: {CONFIG_FILE}")

test_df = pd.read_csv(TEST_FILE, parse_dates=["Time"], index_col="Time")

with open(PARAMS_FILE, "r", encoding="utf-8") as f:
    params_by_target = json.load(f)
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    train_cfg = json.load(f)

target_cols = train_cfg.get("target_cols", list(params_by_target.keys()))
forecast_horizon_h = int(train_cfg.get("forecast_horizon_h", 1))  
feature_cols = train_cfg.get("feature_cols")
feature_cols_by_target = train_cfg.get("feature_cols_by_target", {})
target_cols = [t for t in target_cols if t in test_df.columns]

if not target_cols:
    raise ValueError("No valid target columns found in test data for evaluation.")

X_test, y_test = make_supervised_xy(test_df, target_cols, forecast_horizon_h)

if feature_cols:
    missing_feature_cols = [c for c in feature_cols if c not in X_test.columns]
    if missing_feature_cols:
        raise ValueError(f"Missing expected feature columns in test set: {missing_feature_cols[:10]}")
    X_test = X_test[feature_cols]

print("Supervised test matrices:")
print("  X_test:", X_test.shape, "y_test:", y_test.shape)

test_metrics = []
pred_df = pd.DataFrame(index=y_test.index)

for target in target_cols:
    safe_name = target.replace(" ", "_").replace("/", "_")
    model = load_trained_model(MODEL_OUT_DIR, safe_name)

    if model is None:
        print(f"Skipping {target}: missing model files xgb_{safe_name}.json/.pkl")
        continue

    target_feature_cols = feature_cols_by_target.get(target, feature_cols)
    if not target_feature_cols:
        raise ValueError(f"No feature columns configured for target: {target}")
    missing_target_feature_cols = [c for c in target_feature_cols if c not in X_test.columns]
    if missing_target_feature_cols:
        raise ValueError(f"Missing expected feature columns for {target}: {missing_target_feature_cols[:10]}")

    X_test_target = X_test[target_feature_cols]
    test_pred = model.predict(X_test_target)
    baseline_pred = test_df[target].loc[y_test.index].values

    pred_df[f"{target}__actual"] = y_test[target]
    pred_df[f"{target}__pred"] = test_pred
    pred_df[f"{target}__baseline_pred"] = baseline_pred

    model_rmse = float(np.sqrt(mean_squared_error(y_test[target], test_pred)))
    model_mae = float(mean_absolute_error(y_test[target], test_pred))
    model_r2 = float(r2_score(y_test[target], test_pred))

    baseline_rmse = float(np.sqrt(mean_squared_error(y_test[target], baseline_pred)))
    baseline_mae = float(mean_absolute_error(y_test[target], baseline_pred))
    baseline_r2 = float(r2_score(y_test[target], baseline_pred))

    test_metrics.append(
        {
            "target": target,
            "rmse": model_rmse,
            "mae": model_mae,
            "r2": model_r2,
            "baseline_rmse": baseline_rmse,
            "baseline_mae": baseline_mae,
            "baseline_r2": baseline_r2,
            "rmse_improvement_pct": safe_pct_improvement(baseline_rmse, model_rmse),
            "mae_improvement_pct": safe_pct_improvement(baseline_mae, model_mae),
        }
    )

if not test_metrics:
    raise RuntimeError("No models were evaluated. Ensure .pkl or .json model files exist.")

metrics_df = pd.DataFrame(test_metrics).sort_values("rmse")
metrics_df.to_csv(METRICS_OUT, index=False)
pred_df.to_csv(PRED_OUT, index=True)

print("\nXGBoost test evaluation complete.")
print(f"Metrics saved -> {METRICS_OUT}")
print(f"Predictions saved -> {PRED_OUT}")
print("\nTest metrics summary:")
print(metrics_df[["target", "rmse", "baseline_rmse", "rmse_improvement_pct", "r2", "baseline_r2"]].to_string(index=False))


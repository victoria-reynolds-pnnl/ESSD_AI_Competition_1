import json
import os
import pickle
from pathlib import Path
import importlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import RegressorMixin


################# Configuration #################

MODEL = "xgb"
PREDICT_DELTA = False  # Train on y(t+1)-y(t) instead of y(t+1) to anchor predictions to persistence

if MODEL == "xgb":
    try:
        XGBRegressor = importlib.import_module("xgboost").XGBRegressor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xgboost is required for model training."
        ) from exc

    class XGBRegressorClass(XGBRegressor, RegressorMixin):
        # Explicit estimator type for strict sklearn compatibility checks.
        _estimator_type = "regressor"

################# Paths #################

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

DATA_DIR = Path(r"../Data")
SPLIT_DATA_DIR = DATA_DIR / "03_split"

TEST_FILE = SPLIT_DATA_DIR / "FTES_1hour_test.csv"

if PREDICT_DELTA:
    SAVE_DIR = Path(r"../Models") / f"{MODEL}_delta"
else:
    SAVE_DIR = Path(r"../Models") / f"{MODEL}_original"

TRAIN_META = SAVE_DIR / "train_output"
PARAMS_FILE = TRAIN_META / f"{MODEL}_best_params.json"
CONFIG_FILE = TRAIN_META / f"{MODEL}_training_config.json"

EVAL_META = SAVE_DIR / "eval_output"
EVAL_META.mkdir(exist_ok=True)
METRICS_OUT = EVAL_META / f"{MODEL}_test_metrics.csv"
PRED_OUT = EVAL_META / f"{MODEL}_test_predictions.csv"

###########################################################

def safe_pct_improvement(baseline_err, model_err):
    if baseline_err == 0:
        return np.nan
    return 100.0 * (baseline_err - model_err) / abs(baseline_err)


def load_trained_model(model_out_dir, safe_name):
    """Load model with JSON-first strategy to avoid pickle class-resolution issues."""
    json_path = model_out_dir / f"{MODEL}_{safe_name}.json"
    pkl_path = model_out_dir / f"{MODEL}_{safe_name}.pkl"

    if json_path.exists():
        model = XGBRegressorClass()
        model.load_model(json_path)
        return model

    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    return None


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
predict_delta = train_cfg.get("predict_delta", False)
assert predict_delta == PREDICT_DELTA, "Predict delta setting mismatch between training config and evaluation script."
feature_cols = train_cfg.get("feature_cols")
target_cols = [t for t in target_cols if t in test_df.columns]

if not target_cols:
    raise ValueError("No valid target columns found in test data for evaluation.")

# ── Build X/y directly from saved test CSV ────────────────────────────────────
# X at time t, actual y from time t+horizon (read straight from the CSV).
# No target reconstruction needed — actuals are the raw saved values.
n_test = len(test_df)
x_rows = test_df.iloc[: n_test - forecast_horizon_h]        # features at time t
y_rows = test_df.iloc[forecast_horizon_h:]                   # actuals at time t+h

# Align indices: X keeps time-t index, y values are the future absolute readings.
x_num = x_rows.select_dtypes(include="number").copy()
X_test = x_num[feature_cols]

# Actual future values (absolute) pulled directly from the saved CSV.
y_actual = y_rows[target_cols].values                        # shape (n-h, n_targets)
# Persistence baseline: current value at time t for each target.
baseline_vals = x_rows[target_cols].values                   # shape (n-h, n_targets)

print("Supervised test matrices (from saved CSV):")
print("  X_test:", X_test.shape, f"y_actual: ({y_actual.shape[0]}, {y_actual.shape[1]})")

test_metrics = []
pred_df = pd.DataFrame(index=X_test.index)

for i, target in enumerate(target_cols):
    safe_name = target.replace(" ", "_").replace("/", "_")
    model = load_trained_model(SAVE_DIR, safe_name)

    if model is None:
        print(f"Skipping {target}: missing model files {MODEL}_{safe_name}.json/.pkl")
        continue

    X_test_target = X_test[feature_cols]
    raw_pred = model.predict(X_test_target)

    actual_vals = y_actual[:, i]
    baseline_pred = baseline_vals[:, i]

    if predict_delta:
        # Model predicted delta,  reconstruct absolute: y_hat(t+1) = y(t) + delta
        test_pred = baseline_pred + raw_pred
    else:
        test_pred = raw_pred

    pred_df[f"{target}__actual"] = actual_vals
    pred_df[f"{target}__pred"] = test_pred
    pred_df[f"{target}__baseline_pred"] = baseline_pred

    model_rmse = float(np.sqrt(mean_squared_error(actual_vals, test_pred)))
    model_mae = float(mean_absolute_error(actual_vals, test_pred))

    baseline_rmse = float(np.sqrt(mean_squared_error(actual_vals, baseline_pred)))
    baseline_mae = float(mean_absolute_error(actual_vals, baseline_pred))

    test_metrics.append(
        {
            "target": target,
            "baseline_mae": baseline_mae,
            "baseline_rmse": baseline_rmse,
            "mae": model_mae,
            "rmse": model_rmse,
            "mae_improvement_pct": safe_pct_improvement(baseline_mae, model_mae),
            "rmse_improvement_pct": safe_pct_improvement(baseline_rmse, model_rmse),
        }
    )

if not test_metrics:
    raise RuntimeError("No models were evaluated. Ensure .pkl or .json model files exist.")

metrics_df = pd.DataFrame(test_metrics).sort_values("mae_improvement_pct", ascending=False)
metrics_df.to_csv(METRICS_OUT, index=False)
pred_df = pred_df.shift(1)  # Align predictions with the correct time step (predicting t+1)
pred_df.to_csv(PRED_OUT, index=True)

print(f"\n{MODEL} test evaluation complete.")
print(f"Metrics saved -> {METRICS_OUT}")
print(f"Predictions saved -> {PRED_OUT}")
print("\nTest metrics summary:")
print(metrics_df.to_string(index=False))


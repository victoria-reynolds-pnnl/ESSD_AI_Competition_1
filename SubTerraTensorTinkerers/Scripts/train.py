# # WEEK 3: FTES Training Pipeline

# **Dataset:** DEMO-FTES Test 1 — 1-hour averaged telemetry  
# **Period:** Dec 10, 2024 – Mar 24, 2025  
#             Filtered to hot injection phase (2024-12-13 20:00 to 2025-02-22 17:26:06)
# **Wells:** TL, TN, TC, TU, TS (collar / interval / bottom sections)  
# **Measurements:** Flow rate (L/min), Pressure (psi), Temperature (°C via TEC thermocouples), Electrical Conductivity (EC), Packer depths (ft)

### Training Steps
# 1. Train/test split
# 2. Train and tune the model
# 3. Fit on best parameter set for final model

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import RegressorMixin
import json
import pickle
import os
import importlib
from typing import List, Tuple

warnings.filterwarnings("ignore")
print("Libraries loaded.")


################# Configuration #################

MODEL = "xgb"
PREDICT_DELTA = True  # Train on y(t+1)-y(t) instead of y(t+1) to anchor predictions to persistence
if PREDICT_DELTA:
    loss_func = "reg:pseudohubererror"  # Huber loss, robust to outliers when predicting deltas
else:
    loss_func = "reg:squarederror"  # Standard regression loss for absolute predictions

FORECAST_HORIZON_H = 1  # 1-hour
EXCLUDED_FEATURE_TERMS = ("Phase", "Flag", "TS", "TU", "12h", "24h")

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
CLEAN_DATA_DIR = DATA_DIR / "02_cleaned"
CLEAN_FILE = CLEAN_DATA_DIR / "FTES_1hour_cleaned.csv"
SPLIT_DATA_DIR = DATA_DIR / "03_split"
SPLIT_DATA_DIR.mkdir(exist_ok=True)

if PREDICT_DELTA:
    SAVE_DIR = Path(r"../Models") / f"{MODEL}_delta"
else:
    SAVE_DIR = Path(r"../Models") / f"{MODEL}_original"

TRAIN_META = SAVE_DIR / "train_output"
TRAIN_META.mkdir(parents=True, exist_ok=True)

###########################################################

# ── load ──────────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(CLEAN_FILE, parse_dates=["Time"], index_col="Time")
df = df_raw.copy()

# ── classify columns by sensor type ──────────────────────────────────────────
flow_cols  = [c for c in df.columns if "Flow" in c]
temp_cols  = [c for c in df.columns if "TEC" in c]           # thermocouple cols
ec_cols    = [c for c in df.columns if " EC" in c or c.endswith("EC")]
pres_cols  = [c for c in df.columns if "Pressure" in c]
depth_cols = [c for c in df.columns if "Depth" in c]

# ── Temporal split by % (chronological, no shuffle, taken from cleaning script) ────
# Split data into train/val/test by chronological order to prevent data leakage across time. 

df = df.copy()
df.index = pd.to_datetime(df.index)          # ensure datetime index
df = df.sort_index()                         # chronological

train_frac = 0.70
val_frac   = 0.15
test_frac  = 0.15
assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9

n = len(df)
i_train = int(np.floor(n * train_frac))
i_val   = int(np.floor(n * (train_frac + val_frac)))

train_df = df.iloc[:i_train].copy()
val_df   = df.iloc[i_train:i_val].copy()
test_df  = df.iloc[i_val:].copy()

print("Rows:", {"train": len(train_df), "val": len(val_df), "test": len(test_df)})
print("Time ranges:")
print("  train:", train_df.index.min(), "->", train_df.index.max())
print("  val  :", val_df.index.min(),   "->", val_df.index.max())
print("  test :", test_df.index.min(),  "->", test_df.index.max())

# Ensure same columns across splits (no leakage; consistent model inputs)
val_df  = val_df.reindex(columns=train_df.columns, fill_value=0)
test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

# Fill NaNs from lag/rolling/diff using train medians (no future leakage)
num_cols_all = train_df.select_dtypes(include="number").columns
train_medians = train_df[num_cols_all].median()
train_df[num_cols_all] = train_df[num_cols_all].fillna(train_medians)
val_df[num_cols_all] = val_df[num_cols_all].fillna(train_medians)
test_df[num_cols_all] = test_df[num_cols_all].fillna(train_medians)

# Keep chronological order and persist which split each row belongs to.
train_out = train_df.copy()
val_out = val_df.copy()
test_out = test_df.copy()
# OpenAI GPT‑5.3-Codex

# ── Fit scaler on train only, transform val/test ──────────────────────────────
# NOTE: excluded for XGBoost (tree-based models do not require feature scaling)
if MODEL != "xgb":

# ── Choose columns to scale (include engineered numeric columns if present) ───
    base_scale_cols = (
        flow_cols + pres_cols + temp_cols + ec_cols
        + depth_cols
        + [
            "Total_Production_Flow",
            "Pressure_Differential",
            "Producer_Impedance_Proxy",
            "TC_Impedance_Proxy",
        ]
    )

    # Include any mixing proxy engineered columns like "*__MixFrac", "*__NormDist_to_InjEC", etc.
    mix_proxy_cols = [c for c in train_df.columns if "__MixFrac" in c or "__NormDist_to_InjEC" in c or "__Delta_to_InjEC" in c]

    # Include generated time-series features, excluding binary jump flags
    ts_scale_cols = [
        c for c in train_df.columns
        if (
            "__lag_" in c
            or "__roll_" in c
            or "__roc_" in c
            or "__mean_24h" in c
        ) and not c.endswith("__jump_flag_24h")
    ]

    scale_cols = [c for c in (base_scale_cols + mix_proxy_cols + ts_scale_cols)
                if c in train_df.columns and not c.startswith("Phase_")]
    # OpenAI GPT‑5.3-Codex

    scaler = MinMaxScaler()

    train_scaled = train_df.copy()
    val_scaled   = val_df.copy()
    test_scaled  = test_df.copy()

    train_scaled[scale_cols] = scaler.fit_transform(train_df[scale_cols])
    val_scaled[scale_cols]   = scaler.transform(val_df[scale_cols])
    test_scaled[scale_cols]  = scaler.transform(test_df[scale_cols])

    print(f"\nMinMaxScaler fit on TRAIN, applied to {len(scale_cols)} columns.")
    print("Scaled column sample (first 5):", scale_cols[:5])

    # Optional quick check (train should be ~[0,1] for these cols)
    train_scaled[scale_cols].describe().round(4).loc[["min","max"]].head(10)

    train_out = train_scaled.copy()
    val_out = val_scaled.copy()
    test_out = test_scaled.copy()

# ── Save split-indexed datasets ─────────────────────────────────────────────

train_out["Split"] = "train"
val_out["Split"] = "val"
test_out["Split"] = "test"

split_df = pd.concat([train_out, val_out, test_out], axis=0)
split_df = split_df.sort_index()

# Use a MultiIndex (Split, Time) so split membership is embedded in the index.
split_df = split_df.reset_index().set_index(["Split", "Time"])

split_data_path = SPLIT_DATA_DIR / "FTES_1hour_split_indexed.csv"
split_df.to_csv(split_data_path)

# Optional convenience files for direct model consumption.
train_out.to_csv(SPLIT_DATA_DIR / "FTES_1hour_train.csv", index=True)
test_out.to_csv(SPLIT_DATA_DIR / "FTES_1hour_test.csv", index=True)
val_out.to_csv(SPLIT_DATA_DIR / "FTES_1hour_val.csv", index=True)

print(f"Split-indexed dataset saved -> {split_data_path}")
print(
    "Saved per-split files -> "
    f"{SPLIT_DATA_DIR / 'FTES_1hour_train.csv'}, {SPLIT_DATA_DIR / 'FTES_1hour_test.csv'}, {SPLIT_DATA_DIR / 'FTES_1hour_val.csv'}"
)
print(f"Final split-indexed shape: {split_df.shape}")

# ── Forecasting ──────────────────────────────────────
forecast_horizon_h = FORECAST_HORIZON_H

# Target set aligned with problem statement:
# 1) injection pressure
# 2) producer flows 
# 3) producer temperatures (interval-upper TEC sensors)
requested_targets = [  # (including only hot injection phase producer wells)
    "Injection Pressure", 
    "TC Interval Pressure", 
        "TC Bottom Pressure", 
        "TC Packer Pressure",
    "TL Interval Flow", "TN Interval Flow", 
        "TL Bottom Flow", "TN Bottom Flow", 
        "TL Collar Flow", "TN Collar Flow", 
    "TL-TEC-INT-U", "TN-TEC-INT-U", 
        "TL-TEC-INT-L", "TN-TEC-INT-L",
    "TL-TEC-BOT-U", "TN-TEC-BOT-U", 
        "TL-TEC-BOT-L", "TN-TEC-BOT-L"
]
target_cols = [c for c in requested_targets if c in df.columns]

if not target_cols:
    raise ValueError("No valid target columns found for training.")

missing_targets = [c for c in requested_targets if c not in target_cols]
if missing_targets:
    print(f"Skipped missing targets: {missing_targets}")

def make_supervised_xy(split_df_in, targets, horizon_h, predict_delta=False):
    x_num = split_df_in.select_dtypes(include="number").copy()
    kept_cols = [
        c for c in x_num.columns
        if not any(term in c for term in EXCLUDED_FEATURE_TERMS)
    ]
    x_num = x_num[kept_cols]

    if predict_delta:
        # Target is the CHANGE from current to next step: y(t+h) - y(t)
        # Model learns small corrections on top of persistence baseline.
        y = split_df_in[targets].shift(-horizon_h) - split_df_in[targets]
    else:
        y = split_df_in[targets].shift(-horizon_h)

    valid_mask = ~y.isnull().any(axis=1)
    x_num = x_num.loc[valid_mask]
    y = y.loc[valid_mask]
    return x_num, y

X_train, y_train = make_supervised_xy(train_out, target_cols, forecast_horizon_h, predict_delta=PREDICT_DELTA)
X_val, y_val = make_supervised_xy(val_out, target_cols, forecast_horizon_h, predict_delta=PREDICT_DELTA)

print(f"\nSupervised matrices (after shifting {forecast_horizon_h}h ahead):")
print("  X_train:", X_train.shape, "y_train:", y_train.shape)
print("  X_val  :", X_val.shape, "y_val  :", y_val.shape)

param_grid = [
    # Conservative: shallow trees, heavy regularization.
    {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.03,
        "colsample_bytree": 0.5,
        "subsample": 0.7,
        "min_child_weight": 10,
        "reg_lambda": 5.0,
        "reg_alpha": 1.0,
    },
    # Moderate: balanced capacity.
    {
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.02,
        "colsample_bytree": 0.6,
        "subsample": 0.7,
        "min_child_weight": 15,
        "reg_lambda": 5.0,
        "reg_alpha": 2.0,
    },
    # Deeper with little regularization.
    {
        "n_estimators": 800,
        "max_depth": 4,
        "learning_rate": 0.02,
        "colsample_bytree": 0.9,
        "subsample": 1.0,
        "min_child_weight": 10,
        "reg_lambda": 5.0,
        "reg_alpha": 0.0,
    },
    # High capacity with strong guards.
    {
        "n_estimators": 1000,
        "max_depth": 4,
        "learning_rate": 0.01,
        "colsample_bytree": 0.6,
        "subsample": 0.8,
        "min_child_weight": 20,
        "reg_lambda": 10.0,
        "reg_alpha": 5.0,
    },
]


def build_walk_forward_folds(
    n_samples: int,
    n_folds: int = 3,
    min_train_frac: float = 0.5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build expanding-window train/validation folds for time-series CV."""
    if n_samples < 20:
        raise ValueError(f"Insufficient samples for walk-forward validation: {n_samples}")

    min_train_end = int(np.floor(n_samples * min_train_frac))
    min_train_end = max(min_train_end, 10)
    remaining = n_samples - min_train_end
    fold_size = max(remaining // n_folds, 5)

    folds = []
    train_end = min_train_end
    for _ in range(n_folds):
        val_end = min(train_end + fold_size, n_samples)
        if val_end - train_end < 3:
            break

        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        folds.append((train_idx, val_idx))
        train_end = val_end

    if len(folds) < 2:
        raise ValueError("Could not construct enough walk-forward folds.")
    return folds

walk_forward_folds = build_walk_forward_folds(len(X_train), n_folds=3, min_train_frac=0.5)
all_best_params = {}  # accumulate per-target best params
for target in target_cols:
    target_feature_cols = X_train.columns.tolist()
    X_train_target = X_train[target_feature_cols]
    X_val_target = X_val[target_feature_cols]
    X_trainval_target = pd.concat([X_train_target, X_val_target], axis=0)
    y_trainval_target = pd.concat([y_train[target], y_val[target]], axis=0)

    best_mae = np.inf
    best_params = None
    best_n_estimators = None
    best_cv_metrics = {}

    for params in param_grid:
        fold_maes = []
        fold_rmses = []
        fold_baseline_maes = []
        fold_best_iters = []

        for fold_train_idx, fold_val_idx in walk_forward_folds:
            X_fold_train = X_train_target.iloc[fold_train_idx]
            y_fold_train = y_train[target].iloc[fold_train_idx]
            X_fold_val = X_train_target.iloc[fold_val_idx]
            y_fold_val = y_train[target].iloc[fold_val_idx]

            model = XGBRegressorClass(
                objective=loss_func,
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=20,
                **params,
            )
            model.fit(
                X_fold_train,
                y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False,
            )
            val_pred = model.predict(X_fold_val)

            # Persistence baseline: in delta mode predicting "no change" (0)
            # is the correct baseline; in absolute mode use current value.
            if PREDICT_DELTA:
                # Persistence baseline on DELTA: predict 0 change between time steps
                baseline_pred = np.zeros(len(y_fold_val))  
            else:
                # Persistence baseline on TARGET: predict current value at next time step
                baseline_pred = X_fold_val[target].values

            fold_maes.append(float(mean_absolute_error(y_fold_val, val_pred)))
            fold_rmses.append(float(np.sqrt(mean_squared_error(y_fold_val, val_pred))))
            fold_baseline_maes.append(float(mean_absolute_error(y_fold_val, baseline_pred)))

            best_iter = getattr(model, "best_iteration", None)
            if best_iter is not None:
                fold_best_iters.append(int(best_iter) + 1)

        cv_mae = float(np.mean(fold_maes))
        cv_rmse = float(np.mean(fold_rmses))
        cv_baseline_mae = float(np.mean(fold_baseline_maes))
        cv_best_n_estimators = int(np.median(fold_best_iters)) if fold_best_iters else int(params["n_estimators"])

        # Select by lowest MAE (aligned with Huber objective).
        if cv_mae < best_mae:
            best_mae = cv_mae
            best_params = params
            best_n_estimators = cv_best_n_estimators
            best_cv_metrics = {
                "walk_forward_mae": cv_mae,
                "walk_forward_rmse": cv_rmse,
                "walk_forward_baseline_mae": cv_baseline_mae,
                "walk_forward_mae_improvement_pct": float(
                    100.0 * (cv_baseline_mae - cv_mae) / cv_baseline_mae
                ) if cv_baseline_mae > 0 else 0.0,
            }

    if best_params is None:
        raise RuntimeError(f"No valid hyperparameter set selected for target: {target}")

    param_dump = {
        "final_n_estimators": best_n_estimators,
        **best_params,
        **best_cv_metrics,
        "walk_forward_folds": len(walk_forward_folds),
        "feature_count": len(target_feature_cols)
    }
    all_best_params[target] = param_dump
    final_params = {**best_params, "n_estimators": best_n_estimators}
    final_model = XGBRegressorClass(
        objective=loss_func,
        random_state=42,
        n_jobs=-1,
        **final_params,
    )
    final_model.fit(
        X_trainval_target,
        y_trainval_target,
        verbose=False,
    )

    safe_name = target.replace(" ", "_").replace("/", "_")
    with open(SAVE_DIR / f"{MODEL}_{safe_name}.pkl", "wb") as f:
        pickle.dump(final_model, f)
    final_model.save_model(SAVE_DIR / f"{MODEL}_{safe_name}.json")

params_path = TRAIN_META / f"{MODEL}_best_params.json"
config_path = TRAIN_META / f"{MODEL}_training_config.json"

with open(params_path, "w", encoding="utf-8") as f:
    json.dump(all_best_params, f, indent=2, ensure_ascii=False)
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "forecast_horizon_h": forecast_horizon_h,
            "predict_delta": PREDICT_DELTA,
            "target_cols": target_cols,
            "feature_cols": target_feature_cols
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print("\nTraining complete.")
print(f"Pickle models saved -> {SAVE_DIR}")
print(f"Best params saved -> {params_path}")
print(f"Training config saved -> {config_path}")

# OpenAI GPT‑5.3-Codex and Claude Opus 4.6
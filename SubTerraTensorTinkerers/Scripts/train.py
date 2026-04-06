# # WEEK 3: FTES Training Pipeline

# **Dataset:** DEMO-FTES Test 1 — 1-hour averaged telemetry  
# **Period:** Dec 10, 2024 – Mar 24, 2025  
#             Filtered to hot injection phase (2024-12-13 20:00 to 2025-02-22 17:26:06)
# **Wells:** TL, TN, TC, TU, TS (collar / interval / bottom sections)  
# **Measurements:** Flow rate (L/min), Pressure (psi), Temperature (°C via TEC thermocouples), Electrical Conductivity (EC), Packer depths (ft)

### Training Steps
# 1. Train/test split
# 2. Train and tune the XGBoost regressor model
# 3. Fit on best parameter set for final model

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import RegressorMixin
import json
import pickle
import os
import importlib
from typing import List, Tuple

warnings.filterwarnings("ignore")
print("Libraries loaded.")

EXCLUDED_FEATURE_TERMS = ("Phase", "Flag", "TS", "TU", "12h", "24h")
MODEL = "xgb"
FORECAST_HORIZON_H = 1

try:
    XGBRegressor = importlib.import_module("xgboost").XGBRegressor
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "xgboost is required for model training."
    ) from exc


class CompatibleXGBRegressor(XGBRegressor, RegressorMixin):
    # Explicit estimator type for strict sklearn compatibility checks.
    _estimator_type = "regressor"

################# Configuration and Paths #################
DATA_DIR = Path(r"../Data")
CLEAN_DATA_DIR = DATA_DIR / "2_cleaned"
CLEAN_FILE = CLEAN_DATA_DIR / "FTES_1hour_cleaned.csv"
SPLIT_DATA_DIR = DATA_DIR / "3_split"
SPLIT_DATA_DIR.mkdir(exist_ok=True)

MODEL_DIR = Path(r"../Models")
MODEL_DIR.mkdir(exist_ok=True)

###########################################################

# ── load ──────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
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
# NOTE: excluded for XGBoost (tree-based models do not require feature scaling; skews RMSE units)
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
            or "__accel_" in c
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

# ── XGBoost forecasting (1-hour ahead) ──────────────────────────────────────
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
    raise ValueError("No valid target columns found for XGBoost training.")

missing_targets = [c for c in requested_targets if c not in target_cols]
if missing_targets:
    print(f"Skipped missing targets: {missing_targets}")

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

X_train, y_train = make_supervised_xy(train_out, target_cols, forecast_horizon_h)
X_val, y_val = make_supervised_xy(val_out, target_cols, forecast_horizon_h)

print(f"\nSupervised matrices ({forecast_horizon_h}h ahead):")
print("  X_train:", X_train.shape, "y_train:", y_train.shape)
print("  X_val  :", X_val.shape, "y_val  :", y_val.shape)

# Capacity-aware manual search for 1-hour forecasting.
# Includes conservative, regularized, and moderate-capacity options.
param_grid = [
    # Conservative baseline: robust when targets are locally flat.
    {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "colsample_bytree": 0.6,
        "min_child_weight": 5,
        "reg_lambda": 3.0
    },
    # More regularized version of the same family to reduce overfit.
    {
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.03,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 3.0
    },
    # Medium baseline candidate with balanced regularization.
    {
        "n_estimators": 800,
        "max_depth": 4,
        "learning_rate": 0.02,
        "colsample_bytree": 0.9,
        "min_child_weight": 10,
        "reg_lambda": 5.0
    },
    # Moderate-capacity guardrail for dynamic targets.
    {
        "n_estimators": 1000,
        "max_depth": 4,
        "learning_rate": 0.01,
        "colsample_bytree": 1.0,
        "min_child_weight": 10,
        "reg_lambda": 5.0
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

model_out_dir = MODEL_DIR / "xgboost"
model_out_dir.mkdir(exist_ok=True)

walk_forward_folds = build_walk_forward_folds(len(X_train), n_folds=3, min_train_frac=0.5)
for target in target_cols:
    target_feature_cols = X_train.columns.tolist()
    X_train_target = X_train[target_feature_cols]
    X_val_target = X_val[target_feature_cols]
    X_trainval_target = pd.concat([X_train_target, X_val_target], axis=0)
    y_trainval_target = pd.concat([y_train[target], y_val[target]], axis=0)

    best_rmse_gain = -np.inf
    best_r2 = -np.inf
    best_params = None
    best_n_estimators = None

    for params in param_grid:
        fold_rmse_gains = []
        fold_r2s = []
        fold_best_iters = []

        for fold_train_idx, fold_val_idx in walk_forward_folds:
            X_fold_train = X_train_target.iloc[fold_train_idx]
            y_fold_train = y_train[target].iloc[fold_train_idx]
            X_fold_val = X_train_target.iloc[fold_val_idx]
            y_fold_val = y_train[target].iloc[fold_val_idx]

            model = CompatibleXGBRegressor(
                objective="reg:squarederror",
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
            baseline_pred = X_fold_val[target]  # Baseline prediction is the target itself at the current timestamp (persistence model)

            model_rmse = float(np.sqrt(mean_squared_error(y_fold_val, val_pred)))
            baseline_rmse = float(np.sqrt(mean_squared_error(y_fold_val, baseline_pred)))
            fold_rmse_gains.append(baseline_rmse - model_rmse)

            val_r2 = float(r2_score(y_fold_val, val_pred))
            fold_r2s.append(val_r2)

            best_iter = getattr(model, "best_iteration", None)
            if best_iter is not None:
                fold_best_iters.append(int(best_iter) + 1)

        cv_rmse_gain = float(np.mean(fold_rmse_gains))
        cv_r2 = float(np.mean(fold_r2s))
        cv_best_n_estimators = int(np.median(fold_best_iters)) if fold_best_iters else int(params["n_estimators"])

        if (
            cv_rmse_gain > best_rmse_gain
            or (np.isclose(cv_rmse_gain, best_rmse_gain) and cv_r2 > best_r2)
        ):
            best_rmse_gain = cv_rmse_gain
            best_r2 = cv_r2
            best_params = params
            best_n_estimators = cv_best_n_estimators

    if best_params is None:
        raise RuntimeError(f"No valid hyperparameter set selected for target: {target}")

    param_dump = {
        "final_n_estimators": best_n_estimators,
        **best_params,
        "walk_forward_rmse_gain": best_rmse_gain,
        "walk_forward_r2": best_r2,
        "walk_forward_folds": len(walk_forward_folds),
        "selected_feature_count": len(target_feature_cols),
        "feature_cols": target_feature_cols,
    }
    final_params = {**best_params, "n_estimators": best_n_estimators}
    final_model = CompatibleXGBRegressor(
        objective="reg:squarederror",
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
    with open(model_out_dir / f"xgb_{safe_name}.pkl", "wb") as f:
        pickle.dump(final_model, f)
    final_model.save_model(model_out_dir / f"xgb_{safe_name}.json")

params_path = model_out_dir / "xgb_best_params.json"
config_path = model_out_dir / "xgb_training_config.json"

with open(params_path, "w", encoding="utf-8") as f:
    json.dump(param_dump, f, indent=2, ensure_ascii=False)
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "forecast_horizon_h": forecast_horizon_h,
            "target_cols": target_cols,
            "feature_cols": target_feature_cols
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print("\nXGBoost training complete.")
print(f"Pickle models saved -> {model_out_dir}")
print(f"Best params saved -> {params_path}")
print(f"Training config saved -> {config_path}")

# OpenAI GPT‑5.3-Codex
"""
train.py — Reproducible training workflow for water supply volume forecasting.

Loads the cleaned feature matrix from Week 2, applies a temporal train/test
split, tunes XGBoost hyperparameters via LOYO cross-validation, trains final
models, and evaluates on the held-out test set.

Split strategy (temporal, prevents data leakage):
  Train: WY 1985-2012  (28 years)
  Test:  WY 2013-2018  (6 years, ~18% — held out completely)
  Validation: LOYO CV within training set for hyperparameter tuning

Models:
  1. Climatology baseline  — training-set median
  2. Multiple Linear Regression (MLR) — sklearn with StandardScaler
  3. XGBoost point forecast — tuned via optuna (n_trials=50)
  4. XGBoost quantile (q10/q50/q90) — probabilistic prediction intervals

Outputs:
  outputs/train_split.csv
  outputs/test_split.csv
  outputs/loyo_cv_predictions.csv
  outputs/test_predictions.csv
  models/xgb_final.json
  models/mlr_scaler.pkl
  models/mlr_final.pkl

Usage:
  pixi run python scripts/train.py

AI-generated: This script was generated with assistance from Claude (Anthropic).
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FEATURE_MATRIX = Path("data/clean/feature_matrix.csv")
OUTPUTS_DIR = Path("outputs")
MODELS_DIR = Path("models")

TRAIN_CUTOFF = 2012   # WY <= this value are training; > are test
TEST_START = 2013

FEATURE_COLS = [
    "apr1_swe_inches",
    "apr1_swe_anomaly_pct",
    "djf_pdo",
    "djf_nino34",
    "djf_pna",
    "jan_mar_mean_q_cfs",
    "oct_mar_volume_kcfs_days",
]
TARGET_COL = "target_volume"

OPTUNA_TRIALS = 50
RANDOM_SEED = 42

OUTPUTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Load and split
# ---------------------------------------------------------------------------

def load_and_split():
    df = pd.read_csv(FEATURE_MATRIX)
    df = df.sort_values("water_year").reset_index(drop=True)

    train = df[df["water_year"] <= TRAIN_CUTOFF].copy()
    test = df[df["water_year"] >= TEST_START].copy()

    print(f"Loaded {len(df)} water years ({df['water_year'].min()}-{df['water_year'].max()})")
    print(f"  Train: {len(train)} years ({train['water_year'].min()}-{train['water_year'].max()})")
    print(f"  Test:  {len(test)} years ({test['water_year'].min()}-{test['water_year'].max()})")

    train.to_csv(OUTPUTS_DIR / "train_split.csv", index=False)
    test.to_csv(OUTPUTS_DIR / "test_split.csv", index=False)
    print(f"  Saved train_split.csv and test_split.csv to {OUTPUTS_DIR}/")

    return train, test


# ---------------------------------------------------------------------------
# LOYO cross-validation helpers
# ---------------------------------------------------------------------------

def loyo_indices(n):
    """Yield (train_idx, val_idx) for leave-one-year-out CV."""
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        yield train_idx, [i]


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


# ---------------------------------------------------------------------------
# XGBoost hyperparameter tuning via optuna (LOYO inner loop)
# ---------------------------------------------------------------------------

def tune_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Use optuna to find best XGBoost hyperparameters via LOYO CV."""
    print(f"\nTuning XGBoost with {OPTUNA_TRIALS} optuna trials (LOYO CV)...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "random_state": RANDOM_SEED,
            "verbosity": 0,
        }
        preds = []
        for tr_idx, val_idx in loyo_indices(len(X_train)):
            model = XGBRegressor(**params)
            model.fit(X_train[tr_idx], y_train[tr_idx])
            preds.append(model.predict(X_train[val_idx])[0])
        return rmse(y_train, preds)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    best = study.best_params
    print(f"  Best LOYO RMSE: {study.best_value:,.0f} kcfs-days")
    print(f"  Best params: {best}")
    return best


# ---------------------------------------------------------------------------
# LOYO CV predictions (all models)
# ---------------------------------------------------------------------------

def run_loyo_cv(train: pd.DataFrame, best_xgb_params: dict) -> pd.DataFrame:
    """Run LOYO CV for all three models; return predictions dataframe."""
    print("\nRunning LOYO CV predictions for all models...")

    X = train[FEATURE_COLS].values
    y = train[TARGET_COL].values
    years = train["water_year"].values

    records = []
    for tr_idx, val_idx in loyo_indices(len(X)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val = X[val_idx]
        wy = years[val_idx[0]]
        obs = y[val_idx[0]]

        # 1. Climatology baseline
        pred_clim = float(np.median(y_tr))

        # 2. MLR
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)
        mlr = LinearRegression()
        mlr.fit(X_tr_sc, y_tr)
        pred_mlr = float(mlr.predict(X_val_sc)[0])

        # 3. XGBoost point
        xgb = XGBRegressor(**best_xgb_params, random_state=RANDOM_SEED, verbosity=0)
        xgb.fit(X_tr, y_tr)
        pred_xgb = float(xgb.predict(X_val)[0])

        # 4. XGBoost quantile intervals
        preds_q = {}
        for q in [0.1, 0.5, 0.9]:
            xgb_q = XGBRegressor(
                **best_xgb_params,
                objective="reg:quantileerror",
                quantile_alpha=q,
                random_state=RANDOM_SEED,
                verbosity=0,
            )
            xgb_q.fit(X_tr, y_tr)
            preds_q[f"xgb_q{int(q*100):02d}"] = float(xgb_q.predict(X_val)[0])

        records.append({
            "water_year": wy,
            "observed": obs,
            "pred_climatology": pred_clim,
            "pred_mlr": pred_mlr,
            "pred_xgb": pred_xgb,
            **preds_q,
            "split": "loyo_cv",
        })

    cv_df = pd.DataFrame(records).sort_values("water_year").reset_index(drop=True)
    out_path = OUTPUTS_DIR / "loyo_cv_predictions.csv"
    cv_df.to_csv(out_path, index=False)
    print(f"  Saved {len(cv_df)} LOYO CV predictions to {out_path}")
    return cv_df


# ---------------------------------------------------------------------------
# Train final models on full training set
# ---------------------------------------------------------------------------

def train_final_models(train: pd.DataFrame, best_xgb_params: dict):
    """Train final models on all training data; return fitted objects."""
    print("\nTraining final models on full training set...")

    X_tr = train[FEATURE_COLS].values
    y_tr = train[TARGET_COL].values

    # Climatology: just the median
    clim_value = float(np.median(y_tr))
    print(f"  Climatology median: {clim_value:,.0f} kcfs-days")

    # MLR
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    mlr = LinearRegression()
    mlr.fit(X_tr_sc, y_tr)
    joblib.dump(scaler, MODELS_DIR / "mlr_scaler.pkl")
    joblib.dump(mlr, MODELS_DIR / "mlr_final.pkl")
    print(f"  MLR R²={mlr.score(X_tr_sc, y_tr):.3f} (training, not CV)")

    # XGBoost point
    xgb_final = XGBRegressor(**best_xgb_params, random_state=RANDOM_SEED, verbosity=0)
    xgb_final.fit(X_tr, y_tr)
    xgb_final.save_model(MODELS_DIR / "xgb_final.json")
    print(f"  XGBoost model saved to {MODELS_DIR}/xgb_final.json")

    # XGBoost quantile models
    xgb_quantile_models = {}
    for q in [0.1, 0.5, 0.9]:
        xgb_q = XGBRegressor(
            **best_xgb_params,
            objective="reg:quantileerror",
            quantile_alpha=q,
            random_state=RANDOM_SEED,
            verbosity=0,
        )
        xgb_q.fit(X_tr, y_tr)
        xgb_q.save_model(MODELS_DIR / f"xgb_q{int(q*100):02d}_final.json")
        xgb_quantile_models[q] = xgb_q

    return clim_value, scaler, mlr, xgb_final, xgb_quantile_models


# ---------------------------------------------------------------------------
# Predict on test set
# ---------------------------------------------------------------------------

def predict_test(test: pd.DataFrame, clim_value: float, scaler, mlr,
                 xgb_final, xgb_quantile_models) -> pd.DataFrame:
    """Generate predictions on the held-out test set."""
    print("\nPredicting on test set...")

    X_test = test[FEATURE_COLS].values
    y_test = test[TARGET_COL].values

    X_test_sc = scaler.transform(X_test)

    records = []
    for i, (_, row) in enumerate(test.iterrows()):
        x = X_test[[i]]
        x_sc = X_test_sc[[i]]
        preds_q = {
            f"xgb_q{int(q*100):02d}": float(xgb_quantile_models[q].predict(x)[0])
            for q in [0.1, 0.5, 0.9]
        }
        records.append({
            "water_year": int(row["water_year"]),
            "observed": float(row[TARGET_COL]),
            "pred_climatology": clim_value,
            "pred_mlr": float(mlr.predict(x_sc)[0]),
            "pred_xgb": float(xgb_final.predict(x)[0]),
            **preds_q,
            "split": "test",
        })

    test_df = pd.DataFrame(records)
    out_path = OUTPUTS_DIR / "test_predictions.csv"
    test_df.to_csv(out_path, index=False)
    print(f"  Saved {len(test_df)} test predictions to {out_path}")
    return test_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Water Supply Forecast — Training Pipeline")
    print("=" * 60)

    train, test = load_and_split()

    X_train_arr = train[FEATURE_COLS].values
    y_train_arr = train[TARGET_COL].values

    best_params = tune_xgboost(X_train_arr, y_train_arr)

    cv_preds = run_loyo_cv(train, best_params)

    clim_val, scaler, mlr, xgb_final, xgb_q_models = train_final_models(train, best_params)

    test_preds = predict_test(test, clim_val, scaler, mlr, xgb_final, xgb_q_models)

    print("\n" + "=" * 60)
    print("Training complete. Outputs written to outputs/ and models/")
    print("Next: pixi run python scripts/evaluate.py")
    print("=" * 60)

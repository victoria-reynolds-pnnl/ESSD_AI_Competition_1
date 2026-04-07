# arima_vs_arimax_daily.py
# Compare ARIMA vs ARIMAX per monitoring_location_id with daily data.
#
# Outputs:
# - Data/data_cleaned_split.csv (daily reindexed + imputed flow_rate + split column)
# - Models/ARIMA_{id}_p1_d1_q1.pkl and Models/ARIMAX_{id}_p1_d1_q1.pkl (best per val RMSE)
# - Data/arima_arimax_results.csv (metrics on val and test for both models)
# - Visualizations/arima_arimax_{id}.png (plot per location)
#
# Requirements:
#   pip install pandas numpy statsmodels scikit-learn tqdm matplotlib

import os
import warnings
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

warnings.filterwarnings("ignore")


# -----------------------
# Configuration
# -----------------------
IN_PATH = "../Data/data_cleaned.csv"
SPLIT_OUT_PATH = "../Data/data_cleaned_split.csv"
RESULTS_OUT_PATH = "../Data/arima_arimax_results.csv"

MODELS_DIR = "../Models"
VIZ_DIR = "../Visualizations"

ID_COL = "monitoring_location_id"
TIME_COL = "time"
Y_COL = "flow_rate"
EXOG_COLS = ["seasonal_sin", "seasonal_cos", "trend", "rolling_7", "rolling_14", "rolling_21"]

SPLIT_COL = "split"  # train/val/test

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1

# Grid search: p in [0,1,2], q in [0,1,2], d=1
P_LIST = [0, 1, 2]
Q_LIST = [0, 1, 2]
D_FIXED = 1

EPS = 1e-9


# -----------------------
# Helpers
# -----------------------
def ensure_dirs():
    os.makedirs(os.path.dirname(SPLIT_OUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_OUT_PATH), exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)


def mean_relative_error(y_true, y_pred, eps=EPS):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps)))


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mre = mean_relative_error(y_true, y_pred)
    return rmse, mae, r2, mre


def daily_reindex_and_impute(g: pd.DataFrame) -> pd.DataFrame:
    """
    For one location:
    - sort by time
    - reindex to daily frequency between min and max time
    - impute missing flow_rate via time interpolation then ffill/bfill
    - impute missing exogenous features similarly (so ARIMAX won't choke)
    """
    g = g.copy()
    g[TIME_COL] = pd.to_datetime(g[TIME_COL], errors="coerce")
    g = g.dropna(subset=[TIME_COL]).sort_values(TIME_COL)

    full_idx = pd.date_range(g[TIME_COL].min(), g[TIME_COL].max(), freq="D")
    g = g.set_index(TIME_COL).reindex(full_idx)
    g.index.name = TIME_COL

    # restore id and carry static columns if present
    g[ID_COL] = g[ID_COL].dropna().iloc[0] if g[ID_COL].notna().any() else np.nan

    # Impute numeric columns used for modeling
    for c in [Y_COL] + EXOG_COLS:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")
            g[c] = g[c].interpolate(method="time").ffill().bfill()

    return g.reset_index()


def sequential_split(g: pd.DataFrame, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC) -> pd.DataFrame:
    g = g.copy().sort_values(TIME_COL)
    n = len(g)
    n_train = int(np.floor(train_frac * n))
    n_val = int(np.floor(val_frac * n))
    split = np.array(["train"] * n, dtype=object)
    split[n_train:n_train + n_val] = "val"
    split[n_train + n_val:] = "test"
    g[SPLIT_COL] = split
    return g


def fit_sarimax(y, order, exog=None):
    model = SARIMAX(
        y,
        order=order,
        exog=exog,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    # Increase maxiter a bit for robustness; still "minimal"
    res = model.fit(disp=False, method="lbfgs", maxiter=300)
    return res


def best_model_by_val_rmse_arima(train_y, val_y, orders):
    best = None  # (rmse, order)
    for order in orders:
        try:
            res = fit_sarimax(train_y, order=order, exog=None)
            pred_val = np.asarray(res.forecast(steps=len(val_y)))
            rmse, _, _, _ = compute_metrics(val_y, pred_val)
            if (best is None) or (rmse < best[0]):
                best = (rmse, order)
        except Exception:
            continue
    return best  # or None


def best_model_by_val_rmse_arimax(train_y, train_X, val_y, val_X, orders):
    best = None  # (rmse, order)
    for order in orders:
        try:
            res = fit_sarimax(train_y, order=order, exog=train_X)
            pred_val = np.asarray(res.forecast(steps=len(val_y), exog=val_X))
            rmse, _, _, _ = compute_metrics(val_y, pred_val)
            if (best is None) or (rmse < best[0]):
                best = (rmse, order)
        except Exception:
            continue
    return best  # or None


def plot_location(
    loc_id,
    g,
    train_end_time,
    val_end_time,
    pred_val_arima,
    pred_test_arima,
    pred_val_arimax,
    pred_test_arimax,
    out_path,
):
    g = g.sort_values(TIME_COL)
    train = g[g[SPLIT_COL] == "train"]
    val = g[g[SPLIT_COL] == "val"]
    test = g[g[SPLIT_COL] == "test"]

    plt.figure(figsize=(13, 5))

    # Actuals
    plt.plot(train[TIME_COL], train[Y_COL], color="tab:blue", linewidth=2, label="Actual (train)")
    plt.plot(
        pd.concat([val, test])[TIME_COL],
        pd.concat([val, test])[Y_COL],
        color="tab:blue",
        linestyle="--",
        linewidth=2,
        label="Actual (val+test)",
    )

    # Predictions (val+test)
    val_times = val[TIME_COL].to_numpy()
    test_times = test[TIME_COL].to_numpy()

    if pred_val_arima is not None:
        plt.plot(val_times, pred_val_arima, color="tab:orange", linewidth=2, label="ARIMA pred (val)")
    if pred_test_arima is not None:
        plt.plot(test_times, pred_test_arima, color="tab:orange", linewidth=2, label="ARIMA pred (test)")

    if pred_val_arimax is not None:
        plt.plot(val_times, pred_val_arimax, color="tab:green", linewidth=2, label="ARIMAX pred (val)")
    if pred_test_arimax is not None:
        plt.plot(test_times, pred_test_arimax, color="tab:green", linewidth=2, label="ARIMAX pred (test)")

    # Split cutoffs
    plt.axvline(train_end_time, color="k", linestyle=":", linewidth=1, label="Train/Val cutoff")
    plt.axvline(val_end_time, color="k", linestyle="--", linewidth=1, label="Val/Test cutoff")

    plt.title(f"ARIMA vs ARIMAX — Location {loc_id}")
    plt.xlabel("Time")
    plt.ylabel(Y_COL)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -----------------------
# Main pipeline
# -----------------------
def main():
    ensure_dirs()

    # 1) Load CSV
    df = pd.read_csv(IN_PATH)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[ID_COL, TIME_COL]).copy()

    # 2) Daily reindex + impute per location; sequential 80/10/10 split
    split_frames = []
    for loc_id, g in df.groupby(ID_COL, sort=False):
        g_daily = daily_reindex_and_impute(g)
        g_split = sequential_split(g_daily)
        split_frames.append(g_split)

    df_split = pd.concat(split_frames, ignore_index=True)
    df_split.to_csv(SPLIT_OUT_PATH, index=False)  # 3) save split + imputed flow_rate

    orders = [(p, D_FIXED, q) for p, q in product(P_LIST, Q_LIST)]

    results_rows = []

    # 4) Train models per location with progress bar
    for loc_id, g in tqdm(df_split.groupby(ID_COL, sort=False), desc="Fitting per location"):
        g = g.sort_values(TIME_COL)

        train = g[g[SPLIT_COL] == "train"]
        val = g[g[SPLIT_COL] == "val"]
        test = g[g[SPLIT_COL] == "test"]

        # Need enough data to fit and score
        if len(train) < 30 or len(val) < 5 or len(test) < 5:
            results_rows.append({
                ID_COL: loc_id,
                "status": "skipped_not_enough_data",
            })
            continue

        train_y = train[Y_COL].to_numpy()
        val_y = val[Y_COL].to_numpy()
        test_y = test[Y_COL].to_numpy()

        train_X = train[EXOG_COLS].to_numpy()
        val_X = val[EXOG_COLS].to_numpy()
        test_X = test[EXOG_COLS].to_numpy()

        # 5) Select best by val RMSE
        best_arima = best_model_by_val_rmse_arima(train_y, val_y, orders)
        best_arimax = best_model_by_val_rmse_arimax(train_y, train_X, val_y, val_X, orders)

        # If a whole family fails, we still want rows in the results
        # Refit best on train+val for saving and test forecasting
        trainval = pd.concat([train, val]).sort_values(TIME_COL)
        trainval_y = trainval[Y_COL].to_numpy()
        trainval_X = trainval[EXOG_COLS].to_numpy()

        # Prepare cutoffs for plotting
        train_end_time = train[TIME_COL].iloc[-1]
        val_end_time = val[TIME_COL].iloc[-1]

        # ---- ARIMA
        arima_order = None
        arima_model_path = None
        pred_val_arima = None
        pred_test_arima = None
        arima_val_metrics = (np.nan, np.nan, np.nan, np.nan)
        arima_test_metrics = (np.nan, np.nan, np.nan, np.nan)
        arima_status = "failed"

        if best_arima is not None:
            _, arima_order = best_arima
            p, d, q = arima_order
            try:
                # Fit on train (for val preds)
                res_arima_train = fit_sarimax(train_y, order=arima_order, exog=None)
                pred_val_arima = np.asarray(res_arima_train.forecast(steps=len(val_y)))
                arima_val_metrics = compute_metrics(val_y, pred_val_arima)

                # Fit on train+val (for test preds + saving)
                res_arima = fit_sarimax(trainval_y, order=arima_order, exog=None)
                pred_test_arima = np.asarray(res_arima.forecast(steps=len(test_y)))
                arima_test_metrics = compute_metrics(test_y, pred_test_arima)

                arima_model_path = os.path.join(MODELS_DIR, f"ARIMA_{loc_id}_p{p}_d{d}_q{q}.pkl")
                res_arima.save(arima_model_path)
                arima_status = "ok"
            except Exception as e:
                arima_status = f"failed: {type(e).__name__}"

        # ---- ARIMAX
        arimax_order = None
        arimax_model_path = None
        pred_val_arimax = None
        pred_test_arimax = None
        arimax_val_metrics = (np.nan, np.nan, np.nan, np.nan)
        arimax_test_metrics = (np.nan, np.nan, np.nan, np.nan)
        arimax_status = "failed"

        if best_arimax is not None:
            _, arimax_order = best_arimax
            p, d, q = arimax_order
            try:
                # Fit on train (for val preds)
                res_arimax_train = fit_sarimax(train_y, order=arimax_order, exog=train_X)
                pred_val_arimax = np.asarray(res_arimax_train.forecast(steps=len(val_y), exog=val_X))
                arimax_val_metrics = compute_metrics(val_y, pred_val_arimax)

                # Fit on train+val (for test preds + saving)
                res_arimax = fit_sarimax(trainval_y, order=arimax_order, exog=trainval_X)
                pred_test_arimax = np.asarray(res_arimax.forecast(steps=len(test_y), exog=test_X))
                arimax_test_metrics = compute_metrics(test_y, pred_test_arimax)

                arimax_model_path = os.path.join(MODELS_DIR, f"ARIMAX_{loc_id}_p{p}_d{d}_q{q}.pkl")
                res_arimax.save(arimax_model_path)
                arimax_status = "ok"
            except Exception as e:
                arimax_status = f"failed: {type(e).__name__}"

        # 6.3) Plot and save
        viz_path = os.path.join(VIZ_DIR, f"arima_arimax_{loc_id}.png")
        try:
            plot_location(
                loc_id=loc_id,
                g=g,
                train_end_time=train_end_time,
                val_end_time=val_end_time,
                pred_val_arima=pred_val_arima,
                pred_test_arima=pred_test_arima,
                pred_val_arimax=pred_val_arimax,
                pred_test_arimax=pred_test_arimax,
                out_path=viz_path,
            )
            viz_status = "ok"
        except Exception as e:
            viz_status = f"failed: {type(e).__name__}"
            viz_path = None

        # 6.2) Record metrics to unified CSV
        arima_val_rmse, arima_val_mae, arima_val_r2, arima_val_mre = arima_val_metrics
        arima_test_rmse, arima_test_mae, arima_test_r2, arima_test_mre = arima_test_metrics

        arimax_val_rmse, arimax_val_mae, arimax_val_r2, arimax_val_mre = arimax_val_metrics
        arimax_test_rmse, arimax_test_mae, arimax_test_r2, arimax_test_mre = arimax_test_metrics

        row = {
            ID_COL: loc_id,

            # Orders + paths
            "arima_order": str(arima_order) if arima_order is not None else None,
            "arimax_order": str(arimax_order) if arimax_order is not None else None,
            "arima_model_path": arima_model_path,
            "arimax_model_path": arimax_model_path,

            # Metrics
            "arima_val_rmse": arima_val_rmse,
            "arima_val_mae": arima_val_mae,
            "arima_val_r2": arima_val_r2,
            "arima_val_mre": arima_val_mre,
            "arima_test_rmse": arima_test_rmse,
            "arima_test_mae": arima_test_mae,
            "arima_test_r2": arima_test_r2,
            "arima_test_mre": arima_test_mre,

            "arimax_val_rmse": arimax_val_rmse,
            "arimax_val_mae": arimax_val_mae,
            "arimax_val_r2": arimax_val_r2,
            "arimax_val_mre": arimax_val_mre,
            "arimax_test_rmse": arimax_test_rmse,
            "arimax_test_mae": arimax_test_mae,
            "arimax_test_r2": arimax_test_r2,
            "arimax_test_mre": arimax_test_mre,

            # Status
            "arima_status": arima_status,
            "arimax_status": arimax_status,
            "viz_path": viz_path,
            "viz_status": viz_status,

            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "time_min": g[TIME_COL].min(),
            "time_max": g[TIME_COL].max(),
        }
        results_rows.append(row)

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(RESULTS_OUT_PATH, index=False)

    print(f"Saved split data:   {SPLIT_OUT_PATH}")
    print(f"Saved results:      {RESULTS_OUT_PATH}")
    print(f"Saved models to:    {MODELS_DIR}/")
    print(f"Saved plots to:     {VIZ_DIR}/")


if __name__ == "__main__":
    main()
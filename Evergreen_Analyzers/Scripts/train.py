import os
import warnings
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


# -----------------------
# Configuration
# -----------------------
IN_PATH = "../Data/data_cleaned.csv"
SPLIT_OUT_PATH = "../Data/data_cleaned_split.csv"
ORDERS_OUT_PATH = "../Data/model_orders.csv"
MODELS_DIR = "../Models"

ID_COL = "monitoring_location_id"
TIME_COL = "time"
Y_COL = "flow_rate"
EXOG_COLS = ["seasonal_sin", "seasonal_cos", "trend", "rolling_7", "rolling_14", "rolling_21"]

SPLIT_COL = "split"

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1

P_LIST = [0, 1, 2]
Q_LIST = [0, 1, 2]
D_FIXED = 1


# -----------------------
# Helpers
# -----------------------
def ensure_dirs():
    os.makedirs(os.path.dirname(SPLIT_OUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(ORDERS_OUT_PATH), exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


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

    g[ID_COL] = g[ID_COL].dropna().iloc[0] if g[ID_COL].notna().any() else np.nan

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
    res = model.fit(disp=False, method="lbfgs", maxiter=300)
    return res


def best_order_by_val_rmse(train_y, val_y, orders, train_X=None, val_X=None):
    best = None
    for order in orders:
        try:
            res = fit_sarimax(train_y, order=order, exog=train_X)
            pred_val = np.asarray(res.forecast(steps=len(val_y), exog=val_X))
            rmse = float(np.sqrt(np.mean((val_y - pred_val) ** 2)))
            if best is None or rmse < best[0]:
                best = (rmse, order)
        except Exception:
            continue
    return best


# -----------------------
# Main
# -----------------------
def main():
    ensure_dirs()

    df = pd.read_csv(IN_PATH)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[ID_COL, TIME_COL]).copy()

    split_frames = []
    for _, g in df.groupby(ID_COL, sort=False):
        g_daily = daily_reindex_and_impute(g)
        g_split = sequential_split(g_daily)
        split_frames.append(g_split)

    df_split = pd.concat(split_frames, ignore_index=True)
    df_split.to_csv(SPLIT_OUT_PATH, index=False)

    orders = [(p, D_FIXED, q) for p, q in product(P_LIST, Q_LIST)]
    order_rows = []

    for loc_id, g in tqdm(df_split.groupby(ID_COL, sort=False), desc="Training"):
        g = g.sort_values(TIME_COL)
        train = g[g[SPLIT_COL] == "train"]
        val = g[g[SPLIT_COL] == "val"]

        if len(train) < 30 or len(val) < 5:
            order_rows.append({ID_COL: loc_id, "arima_order": None, "arimax_order": None, "status": "skipped"})
            continue

        train_y = train[Y_COL].to_numpy()
        val_y = val[Y_COL].to_numpy()
        train_X = train[EXOG_COLS].to_numpy()
        val_X = val[EXOG_COLS].to_numpy()

        trainval = pd.concat([train, val]).sort_values(TIME_COL)
        trainval_y = trainval[Y_COL].to_numpy()
        trainval_X = trainval[EXOG_COLS].to_numpy()

        # ARIMA: grid search on val, then refit on train+val and save
        arima_order = None
        best_arima = best_order_by_val_rmse(train_y, val_y, orders)
        if best_arima is not None:
            _, arima_order = best_arima
            p, d, q = arima_order
            try:
                res = fit_sarimax(trainval_y, order=arima_order)
                res.save(os.path.join(MODELS_DIR, f"ARIMA_{loc_id}_p{p}_d{d}_q{q}.pkl"))
            except Exception:
                arima_order = None

        # ARIMAX: grid search on val, then refit on train+val and save
        arimax_order = None
        best_arimax = best_order_by_val_rmse(train_y, val_y, orders, train_X, val_X)
        if best_arimax is not None:
            _, arimax_order = best_arimax
            p, d, q = arimax_order
            try:
                res = fit_sarimax(trainval_y, order=arimax_order, exog=trainval_X)
                res.save(os.path.join(MODELS_DIR, f"ARIMAX_{loc_id}_p{p}_d{d}_q{q}.pkl"))
            except Exception:
                arimax_order = None

        order_rows.append({
            ID_COL: loc_id,
            "arima_order": str(arima_order) if arima_order else None,
            "arimax_order": str(arimax_order) if arimax_order else None,
            "status": "ok",
        })

    pd.DataFrame(order_rows).to_csv(ORDERS_OUT_PATH, index=False)

    print(f"Saved split data:    {SPLIT_OUT_PATH}")
    print(f"Saved model orders:  {ORDERS_OUT_PATH}")
    print(f"Saved models to:     {MODELS_DIR}/")


if __name__ == "__main__":
    main()

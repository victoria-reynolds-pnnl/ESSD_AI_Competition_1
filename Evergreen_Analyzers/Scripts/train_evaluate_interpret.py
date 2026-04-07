# AI incubator planning chat: https://ai-incubator-chat.pnnl.gov/s/8d8622f7-7289-475b-b64f-a6bb1342704e
# AI incubator code generation chat: 
# arima_vs_arimax_per_location.py
# Requirements: pandas, numpy, statsmodels, scikit-learn

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX


DATA_PATH = "Data/data_cleaned.csv"
SPLIT_OUT_PATH = "Data/data_cleaned_with_split.csv"
RESULTS_OUT_PATH = "Data/arima_arimax_results.csv"

ID_COL = "monitoring_location_id"
TIME_COL = "time"
TARGET_COL = "flow_rate"

EXOG_COLS = ["seasonal_sin", "seasonal_cos", "trend", "rolling_7", "rolling_14", "rolling_21"]

# Minimal grid; expand if you want
PDQ_GRID = {
    "p": [2],#[0, 1, 2],
    "d": [1,2],#[0, 1],
    "q": [2],#[0, 1, 2],
}

SPLIT_COL = "split"  # train / val / test

EPS = 1e-9


@dataclass
class ModelResult:
    kind: str  # "ARIMA" or "ARIMAX"
    pdq: Tuple[int, int, int]
    val_mae: float
    val_rmse: float
    val_r2: float
    test_mae: float
    test_rmse: float
    test_r2: float
    n_train: int
    n_val: int
    n_test: int
    status: str  # "ok" or error message


def make_daily_indexed_frame(g: pd.DataFrame) -> pd.DataFrame:
    """
    For one location:
    - sort by time
    - reindex to daily frequency from min to max
    - impute missing values (time gaps and NaNs) with time-based interpolation + ffill/bfill
    """
    g = g.copy()
    g[TIME_COL] = pd.to_datetime(g[TIME_COL])
    g = g.sort_values(TIME_COL)

    # Build daily index for full range
    full_idx = pd.date_range(g[TIME_COL].min(), g[TIME_COL].max(), freq="D")
    g = g.set_index(TIME_COL).reindex(full_idx)
    g.index.name = TIME_COL

    # Restore id
    g[ID_COL] = g[ID_COL].iloc[0] if g[ID_COL].notna().any() else np.nan

    # Impute numeric columns (including target and exogs)
    num_cols = [TARGET_COL] + EXOG_COLS
    for c in num_cols:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")
            g[c] = g[c].interpolate(method="time").ffill().bfill()

    return g.reset_index()


def sequential_split_by_time(g: pd.DataFrame, train_frac=0.8, val_frac=0.1) -> pd.DataFrame:
    g = g.copy().sort_values(TIME_COL)
    n = len(g)
    n_train = int(np.floor(train_frac * n))
    n_val = int(np.floor(val_frac * n))
    n_test = n - n_train - n_val

    split = np.array(["train"] * n, dtype=object)
    split[n_train:n_train + n_val] = "val"
    split[n_train + n_val:] = "test"
    g[SPLIT_COL] = split
    return g


def metrics(y_true, y_pred) -> Tuple[float, float, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def fit_and_forecast_arima(
    y_train: np.ndarray,
    steps: int,
    order: Tuple[int, int, int],
) -> Optional[np.ndarray]:
    model = SARIMAX(
        y_train,
        order=order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    fc = res.forecast(steps=steps)
    return np.asarray(fc)


def fit_and_forecast_arimax(
    y_train: np.ndarray,
    X_train: np.ndarray,
    X_future: np.ndarray,  # exog for forecast horizon
    order: Tuple[int, int, int],
) -> Optional[np.ndarray]:
    model = SARIMAX(
        y_train,
        exog=X_train,
        order=order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    fc = res.forecast(steps=X_future.shape[0], exog=X_future)
    return np.asarray(fc)


def grid_orders() -> List[Tuple[int, int, int]]:
    return list(product(PDQ_GRID["p"], PDQ_GRID["d"], PDQ_GRID["q"]))


def evaluate_location(g: pd.DataFrame) -> List[ModelResult]:
    """
    Choose best ARIMA and best ARIMAX by validation RMSE, then evaluate each on test.
    """
    g = g.sort_values(TIME_COL)

    train = g[g[SPLIT_COL] == "train"]
    val = g[g[SPLIT_COL] == "val"]
    test = g[g[SPLIT_COL] == "test"]

    # Require enough points to be meaningful
    if len(train) < 20 or len(val) < 5 or len(test) < 5:
        return [ModelResult(
            kind="ARIMA",
            pdq=(-1, -1, -1),
            val_mae=np.nan, val_rmse=np.nan, val_r2=np.nan,
            test_mae=np.nan, test_rmse=np.nan, test_r2=np.nan,
            n_train=len(train), n_val=len(val), n_test=len(test),
            status="skipped: not enough data",
        ), ModelResult(
            kind="ARIMAX",
            pdq=(-1, -1, -1),
            val_mae=np.nan, val_rmse=np.nan, val_r2=np.nan,
            test_mae=np.nan, test_rmse=np.nan, test_r2=np.nan,
            n_train=len(train), n_val=len(val), n_test=len(test),
            status="skipped: not enough data",
        )]

    y_train = train[TARGET_COL].to_numpy()
    y_val = val[TARGET_COL].to_numpy()
    y_test = test[TARGET_COL].to_numpy()

    X_train = train[EXOG_COLS].to_numpy()
    X_val = val[EXOG_COLS].to_numpy()
    X_test = test[EXOG_COLS].to_numpy()

    orders = grid_orders()

    # ---- ARIMA model selection on val
    best_arima = None
    best_arima_rmse = np.inf
    best_arima_status = "ok"

    for order in orders:
        try:
            pred_val = fit_and_forecast_arima(y_train, steps=len(val), order=order)
            mae, rmse, r2 = metrics(y_val, pred_val)
            if rmse < best_arima_rmse:
                best_arima_rmse = rmse
                best_arima = (order, mae, rmse, r2)
        except Exception as e:
            best_arima_status = f"fit_failed_some_orders: {type(e).__name__}"
            continue

    # ---- ARIMAX model selection on val
    best_arimax = None
    best_arimax_rmse = np.inf
    best_arimax_status = "ok"

    for order in tqdm(orders):
        try:
            pred_val = fit_and_forecast_arimax(
                y_train=y_train,
                X_train=X_train,
                X_future=X_val,
                order=order,
            )
            mae, rmse, r2 = metrics(y_val, pred_val)
            if rmse < best_arimax_rmse:
                best_arimax_rmse = rmse
                best_arimax = (order, mae, rmse, r2)
        except Exception as e:
            best_arimax_status = f"fit_failed_some_orders: {type(e).__name__}"
            continue

    results: List[ModelResult] = []

    # ---- Evaluate best ARIMA on test (refit on train+val, forecast test)
    if best_arima is None:
        results.append(ModelResult(
            kind="ARIMA",
            pdq=(-1, -1, -1),
            val_mae=np.nan, val_rmse=np.nan, val_r2=np.nan,
            test_mae=np.nan, test_rmse=np.nan, test_r2=np.nan,
            n_train=len(train), n_val=len(val), n_test=len(test),
            status=f"failed_all_orders ({best_arima_status})",
        ))
    else:
        order, v_mae, v_rmse, v_r2 = best_arima
        y_trainval = pd.concat([train, val])[TARGET_COL].to_numpy()
        try:
            pred_test = fit_and_forecast_arima(y_trainval, steps=len(test), order=order)
            t_mae, t_rmse, t_r2 = metrics(y_test, pred_test)
            results.append(ModelResult(
                kind="ARIMA",
                pdq=order,
                val_mae=v_mae, val_rmse=v_rmse, val_r2=v_r2,
                test_mae=t_mae, test_rmse=t_rmse, test_r2=t_r2,
                n_train=len(train), n_val=len(val), n_test=len(test),
                status="ok",
            ))
        except Exception as e:
            results.append(ModelResult(
                kind="ARIMA",
                pdq=order,
                val_mae=v_mae, val_rmse=v_rmse, val_r2=v_r2,
                test_mae=np.nan, test_rmse=np.nan, test_r2=np.nan,
                n_train=len(train), n_val=len(val), n_test=len(test),
                status=f"test_fit_failed: {type(e).__name__}",
            ))

    # ---- Evaluate best ARIMAX on test (refit on train+val with exog, forecast test with X_test)
    if best_arimax is None:
        results.append(ModelResult(
            kind="ARIMAX",
            pdq=(-1, -1, -1),
            val_mae=np.nan, val_rmse=np.nan, val_r2=np.nan,
            test_mae=np.nan, test_rmse=np.nan, test_r2=np.nan,
            n_train=len(train), n_val=len(val), n_test=len(test),
            status=f"failed_all_orders ({best_arimax_status})",
        ))
    else:
        order, v_mae, v_rmse, v_r2 = best_arimax
        trainval = pd.concat([train, val])
        y_trainval = trainval[TARGET_COL].to_numpy()
        X_trainval = trainval[EXOG_COLS].to_numpy()
        try:
            pred_test = fit_and_forecast_arimax(
                y_train=y_trainval,
                X_train=X_trainval,
                X_future=X_test,
                order=order,
            )
            t_mae, t_rmse, t_r2 = metrics(y_test, pred_test)
            results.append(ModelResult(
                kind="ARIMAX",
                pdq=order,
                val_mae=v_mae, val_rmse=v_rmse, val_r2=v_r2,
                test_mae=t_mae, test_rmse=t_rmse, test_r2=t_r2,
                n_train=len(train), n_val=len(val), n_test=len(test),
                status="ok",
            ))
        except Exception as e:
            results.append(ModelResult(
                kind="ARIMAX",
                pdq=order,
                val_mae=v_mae, val_rmse=v_rmse, val_r2=v_r2,
                test_mae=np.nan, test_rmse=np.nan, test_r2=np.nan,
                n_train=len(train), n_val=len(val), n_test=len(test),
                status=f"test_fit_failed: {type(e).__name__}",
            ))

    return results


def main():
    # 1) Load CSV
    df = pd.read_csv(DATA_PATH)

    # Basic parsing
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[ID_COL, TIME_COL]).copy()

    # 2) Per-location daily reindex + impute, then sequential split 80/10/10 and save split column
    out_frames = []
    for loc_id, g in df.groupby(ID_COL, sort=False):
        g2 = make_daily_indexed_frame(g)
        g2 = sequential_split_by_time(g2, 0.8, 0.1)
        out_frames.append(g2)

    df2 = pd.concat(out_frames, ignore_index=True)

    # Save split info back to CSV
    df2.to_csv(SPLIT_OUT_PATH, index=False)

    # 3-6) Train + grid search + select by val + evaluate on test + save results
    all_results = []
    for loc_id, g in df2.groupby(ID_COL, sort=False):
        loc_results = evaluate_location(g)
        for r in loc_results:
            all_results.append({
                ID_COL: loc_id,
                "model": r.kind,
                "p": r.pdq[0],
                "d": r.pdq[1],
                "q": r.pdq[2],
                "val_mae": r.val_mae,
                "val_rmse": r.val_rmse,
                "val_r2": r.val_r2,
                "test_mae": r.test_mae,
                "test_rmse": r.test_rmse,
                "test_r2": r.test_r2,
                "n_train": r.n_train,
                "n_val": r.n_val,
                "n_test": r.n_test,
                "status": r.status,
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_OUT_PATH, index=False)

    print(f"Saved split-annotated data to: {SPLIT_OUT_PATH}")
    print(f"Saved results to: {RESULTS_OUT_PATH}")


if __name__ == "__main__":
    main()
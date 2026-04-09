import os
import ast
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

warnings.filterwarnings("ignore")


# -----------------------
# Configuration
# -----------------------
SPLIT_PATH = "../Data/data_cleaned_split.csv"
ORDERS_PATH = "../Data/model_orders.csv"
RESULTS_OUT_PATH = "../Data/arima_arimax_results.csv"
MODELS_DIR = "../Models"

ID_COL = "monitoring_location_id"
TIME_COL = "time"
Y_COL = "flow_rate"
EXOG_COLS = ["seasonal_sin", "seasonal_cos", "trend", "rolling_7", "rolling_14", "rolling_21"]

SPLIT_COL = "split"

EPS = 1e-9


# -----------------------
# Helpers
# -----------------------
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


# -----------------------
# Main
# -----------------------
def main():
    df_split = pd.read_csv(SPLIT_PATH)
    df_split[TIME_COL] = pd.to_datetime(df_split[TIME_COL], errors="coerce")

    df_orders = pd.read_csv(ORDERS_PATH)

    results_rows = []

    for _, row in tqdm(df_orders.iterrows(), total=len(df_orders), desc="Evaluating"):
        loc_id = row[ID_COL]
        g = df_split[df_split[ID_COL] == loc_id].sort_values(TIME_COL)

        train = g[g[SPLIT_COL] == "train"]
        val = g[g[SPLIT_COL] == "val"]
        test = g[g[SPLIT_COL] == "test"]

        if row.get("status") == "skipped" or len(test) < 5:
            results_rows.append({ID_COL: loc_id, "status": "skipped_not_enough_data"})
            continue

        train_y = train[Y_COL].to_numpy()
        val_y = val[Y_COL].to_numpy()
        test_y = test[Y_COL].to_numpy()
        train_X = train[EXOG_COLS].to_numpy()
        val_X = val[EXOG_COLS].to_numpy()
        test_X = test[EXOG_COLS].to_numpy()

        result = {
            ID_COL: loc_id,
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "time_min": g[TIME_COL].min(),
            "time_max": g[TIME_COL].max(),
        }

        # ---- ARIMA evaluation
        arima_order_str = row.get("arima_order")
        result["arima_order"] = arima_order_str
        if pd.notna(arima_order_str):
            arima_order = ast.literal_eval(arima_order_str)
            p, d, q = arima_order
            try:
                # Val metrics: fit on train only
                res_train = fit_sarimax(train_y, order=arima_order)
                pred_val = np.asarray(res_train.forecast(steps=len(val_y)))
                result["arima_val_rmse"], result["arima_val_mae"], result["arima_val_r2"], result["arima_val_mre"] = compute_metrics(val_y, pred_val)

                # Test metrics: load saved model (fit on train+val)
                model_path = os.path.join(MODELS_DIR, f"ARIMA_{loc_id}_p{p}_d{d}_q{q}.pkl")
                res_saved = SARIMAXResults.load(model_path)
                pred_test = np.asarray(res_saved.forecast(steps=len(test_y)))
                result["arima_test_rmse"], result["arima_test_mae"], result["arima_test_r2"], result["arima_test_mre"] = compute_metrics(test_y, pred_test)

                result["arima_model_path"] = model_path
                result["arima_status"] = "ok"
            except Exception as e:
                result["arima_status"] = f"failed: {type(e).__name__}"
        else:
            result["arima_status"] = "failed"

        # ---- ARIMAX evaluation
        arimax_order_str = row.get("arimax_order")
        result["arimax_order"] = arimax_order_str
        if pd.notna(arimax_order_str):
            arimax_order = ast.literal_eval(arimax_order_str)
            p, d, q = arimax_order
            try:
                # Val metrics: fit on train only
                res_train = fit_sarimax(train_y, order=arimax_order, exog=train_X)
                pred_val = np.asarray(res_train.forecast(steps=len(val_y), exog=val_X))
                result["arimax_val_rmse"], result["arimax_val_mae"], result["arimax_val_r2"], result["arimax_val_mre"] = compute_metrics(val_y, pred_val)

                # Test metrics: load saved model (fit on train+val)
                model_path = os.path.join(MODELS_DIR, f"ARIMAX_{loc_id}_p{p}_d{d}_q{q}.pkl")
                res_saved = SARIMAXResults.load(model_path)
                pred_test = np.asarray(res_saved.forecast(steps=len(test_y), exog=test_X))
                result["arimax_test_rmse"], result["arimax_test_mae"], result["arimax_test_r2"], result["arimax_test_mre"] = compute_metrics(test_y, pred_test)

                result["arimax_model_path"] = model_path
                result["arimax_status"] = "ok"
            except Exception as e:
                result["arimax_status"] = f"failed: {type(e).__name__}"
        else:
            result["arimax_status"] = "failed"

        result["status"] = "ok"
        results_rows.append(result)

    results_df = pd.DataFrame(results_rows)
    os.makedirs(os.path.dirname(RESULTS_OUT_PATH), exist_ok=True)
    results_df.to_csv(RESULTS_OUT_PATH, index=False)

    print(f"Saved results: {RESULTS_OUT_PATH}")


if __name__ == "__main__":
    main()

# evaluate.py
# Load saved models, compute val/test metrics per location, and print aggregate summary.
#
# Outputs:
#   - Data/arima_arimax_results.csv  (per-station RMSE, MAE, R2, MRE for val and test)
#   - Console: aggregate performance summary across all stations
#
# AI tools used:
#   - PNNL AI Incubator chat: https://ai-incubator-chat.pnnl.gov/s/9feb50bb-b740-40b6-a6d4-d3b68c7767f2
#   - GitHub Copilot

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
    print_aggregate_summary(results_df)


def print_aggregate_summary(df):
    """Print aggregate model performance statistics across all stations."""
    # Filter to stations that were successfully evaluated
    ok = df[df.get("arima_status", pd.Series()) == "ok"].copy()
    if ok.empty:
        # Fallback: try rows that have numeric metrics
        ok = df.dropna(subset=["arima_val_rmse", "arimax_val_rmse"])
    if ok.empty:
        print("No successfully evaluated stations to summarize.")
        return

    n = len(ok)
    print(f"\n{'='*60}")
    print(f"AGGREGATE MODEL PERFORMANCE SUMMARY ({n} stations)")
    print(f"{'='*60}")

    for model in ["arima", "arimax"]:
        print(f"\n--- {model.upper()} ---")
        for split in ["val", "test"]:
            rmse_col = f"{model}_{split}_rmse"
            mae_col = f"{model}_{split}_mae"
            r2_col = f"{model}_{split}_r2"

            rmse_vals = pd.to_numeric(ok[rmse_col], errors="coerce").dropna()
            mae_vals = pd.to_numeric(ok[mae_col], errors="coerce").dropna()
            r2_vals = pd.to_numeric(ok[r2_col], errors="coerce").dropna()

            print(f"  {split.upper()} (n={len(r2_vals)}):")
            print(f"    RMSE  median={rmse_vals.median():.2f}  mean={rmse_vals.mean():.2f}")
            print(f"    MAE   median={mae_vals.median():.2f}  mean={mae_vals.mean():.2f}")
            print(f"    R2    median={r2_vals.median():.4f}  mean={r2_vals.mean():.4f}")
            print(f"    R2>0  {(r2_vals > 0).sum()}/{len(r2_vals)} stations")

    # Head-to-head comparison
    both_ok = ok.dropna(subset=["arima_test_r2", "arimax_test_r2"])
    if not both_ok.empty:
        arima_r2 = pd.to_numeric(both_ok["arima_test_r2"])
        arimax_r2 = pd.to_numeric(both_ok["arimax_test_r2"])
        arimax_wins = (arimax_r2 > arima_r2).sum()
        print(f"\n--- HEAD-TO-HEAD (test R2) ---")
        print(f"  ARIMAX beats ARIMA: {arimax_wins}/{len(both_ok)} stations")

        best_idx = arimax_r2.idxmax()
        best_id = both_ok.loc[best_idx, ID_COL]
        best_r2 = arimax_r2.loc[best_idx]
        print(f"  Best ARIMAX test R2: {best_id} = {best_r2:.4f}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

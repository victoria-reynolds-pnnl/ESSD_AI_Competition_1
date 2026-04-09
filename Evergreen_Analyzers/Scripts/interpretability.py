import os
import ast
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

warnings.filterwarnings("ignore")


# -----------------------
# Configuration
# -----------------------
SPLIT_PATH = "../Data/data_cleaned_split.csv"
ORDERS_PATH = "../Data/model_orders.csv"
MODELS_DIR = "../Models"
VIZ_DIR = "../Visualizations"

ID_COL = "monitoring_location_id"
TIME_COL = "time"
Y_COL = "flow_rate"
EXOG_COLS = ["seasonal_sin", "seasonal_cos", "trend", "rolling_7", "rolling_14", "rolling_21"]

SPLIT_COL = "split"


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

    plt.plot(train[TIME_COL], train[Y_COL], color="tab:blue", linewidth=2, label="Actual (train)")
    plt.plot(
        pd.concat([val, test])[TIME_COL],
        pd.concat([val, test])[Y_COL],
        color="tab:blue",
        linestyle="--",
        linewidth=2,
        label="Actual (val+test)",
    )

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
# Main
# -----------------------
def main():
    os.makedirs(VIZ_DIR, exist_ok=True)

    df_split = pd.read_csv(SPLIT_PATH)
    df_split[TIME_COL] = pd.to_datetime(df_split[TIME_COL], errors="coerce")

    df_orders = pd.read_csv(ORDERS_PATH)

    for _, row in tqdm(df_orders.iterrows(), total=len(df_orders), desc="Plotting"):
        loc_id = row[ID_COL]
        if row.get("status") == "skipped":
            continue

        g = df_split[df_split[ID_COL] == loc_id].sort_values(TIME_COL)
        train = g[g[SPLIT_COL] == "train"]
        val = g[g[SPLIT_COL] == "val"]
        test = g[g[SPLIT_COL] == "test"]

        if len(val) < 2 or len(test) < 2:
            continue

        train_y = train[Y_COL].to_numpy()
        val_y = val[Y_COL].to_numpy()
        test_y = test[Y_COL].to_numpy()
        train_X = train[EXOG_COLS].to_numpy()
        val_X = val[EXOG_COLS].to_numpy()
        test_X = test[EXOG_COLS].to_numpy()

        pred_val_arima = pred_test_arima = None
        pred_val_arimax = pred_test_arimax = None

        # ARIMA predictions for plotting
        arima_order_str = row.get("arima_order")
        if pd.notna(arima_order_str):
            arima_order = ast.literal_eval(arima_order_str)
            p, d, q = arima_order
            try:
                res_train = fit_sarimax(train_y, order=arima_order)
                pred_val_arima = np.asarray(res_train.forecast(steps=len(val_y)))

                model_path = os.path.join(MODELS_DIR, f"ARIMA_{loc_id}_p{p}_d{d}_q{q}.pkl")
                res_saved = SARIMAXResults.load(model_path)
                pred_test_arima = np.asarray(res_saved.forecast(steps=len(test_y)))
            except Exception:
                pass

        # ARIMAX predictions for plotting
        arimax_order_str = row.get("arimax_order")
        if pd.notna(arimax_order_str):
            arimax_order = ast.literal_eval(arimax_order_str)
            p, d, q = arimax_order
            try:
                res_train = fit_sarimax(train_y, order=arimax_order, exog=train_X)
                pred_val_arimax = np.asarray(res_train.forecast(steps=len(val_y), exog=val_X))

                model_path = os.path.join(MODELS_DIR, f"ARIMAX_{loc_id}_p{p}_d{d}_q{q}.pkl")
                res_saved = SARIMAXResults.load(model_path)
                pred_test_arimax = np.asarray(res_saved.forecast(steps=len(test_y), exog=test_X))
            except Exception:
                pass

        train_end_time = train[TIME_COL].iloc[-1]
        val_end_time = val[TIME_COL].iloc[-1]

        plot_location(
            loc_id=loc_id,
            g=g,
            train_end_time=train_end_time,
            val_end_time=val_end_time,
            pred_val_arima=pred_val_arima,
            pred_test_arima=pred_test_arima,
            pred_val_arimax=pred_val_arimax,
            pred_test_arimax=pred_test_arimax,
            out_path=os.path.join(VIZ_DIR, f"arima_arimax_{loc_id}.png"),
        )

    print(f"Saved plots to: {VIZ_DIR}/")


if __name__ == "__main__":
    main()

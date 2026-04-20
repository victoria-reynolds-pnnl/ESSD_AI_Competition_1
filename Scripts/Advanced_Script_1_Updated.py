"""
Advanced Yakima River Forecasting Model
---------------------------------------

This script creates a prediction system for:
1. Single-step predictions of Mabton and Kiona gage heights.
2. Multi-step forecasting using autoregressive gage height features.
3. Seasonal predictors (month, day-of-year).
4. Lagged flow and lagged gage height features.
5. Uncertainty intervals from RF ensemble distribution.

Inputs:
- merged_GH_UD.csv (must contain timestamped flow + gage height values)

Requirements:
  • pandas
  • numpy
  • scikit-learn
  • matplotlib
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

# =========================================================
# Load Data
# =========================================================

DATA_PATH = Path(__file__).resolve().parent / 'merged_GH_UD.csv'
df = pd.read_csv(DATA_PATH, parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)

# =========================================================
# Create Time-Based Features
# =========================================================

df["dayofyear"] = df["time"].dt.dayofyear

df["doy_sin"] = np.sin(2*np.pi*df["dayofyear"]/365)
df["doy_cos"] = np.cos(2*np.pi*df["dayofyear"]/365)

# =========================================================
# Create Lagged Features
# =========================================================

LAGS = 6  # number of autoregressive timesteps to include

    # Flow interactions
df["flow_diff"] = df["Mabton_UD"] - df["Union_Gap_UD"]
df["flow_ratio"] = (np.log1p(df["Mabton_UD"]) - np.log1p(df["Union_Gap_UD"]))

for lag in range(1, LAGS + 1):
    # Lagged gage heights
    df[f"Mabton_GH_lag{lag}"] = df["Mabton_GH"].shift(lag)
    df[f"Kiona_GH_lag{lag}"] = df["Kiona_GH"].shift(lag)

    # Lagged flows
    df[f"Union_Gap_UD_lag{lag}"] = df["Union_Gap_UD"].shift(lag)
    df[f"Mabton_UD_lag{lag}"] = df["Mabton_UD"].shift(lag)



df = df.dropna().reset_index(drop=True)

# =========================================================
# Feature Matrix
# =========================================================

feature_cols = (
    [
        "Union_Gap_UD",
        "Mabton_UD",
        "flow_diff",
        "flow_ratio",
        "doy_sin",
        "doy_cos",
    ] +
    [f"Mabton_GH_lag{lag}" for lag in range(1, LAGS + 1)] +
    [f"Kiona_GH_lag{lag}" for lag in range(1, LAGS + 1)] +
    [f"Union_Gap_UD_lag{lag}" for lag in range(1, LAGS + 1)] +
    [f"Mabton_UD_lag{lag}" for lag in range(1, LAGS + 1)]
)

X = df[feature_cols]
y_mab = df["Mabton_GH"]
y_kio = df["Kiona_GH"]

# =========================================================
# Train Models
# =========================================================

#forced diversity in RF hyperparameters to get better uncertainty estimates from ensemble distribution
mod_mab = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=1, max_features="sqrt", bootstrap=True, random_state=42, n_jobs=-1)
mod_kio = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=1, max_features="sqrt", bootstrap=True, random_state=42, n_jobs=-1)

mod_mab.fit(X, y_mab)
mod_kio.fit(X, y_kio)

# =========================================================
# Utility: Ensemble Prediction Interval
# =========================================================

def prediction_interval(model, Xrow, alpha=0.05):
    X_vals = Xrow.to_numpy()
    preds = np.array([est.predict(X_vals)[0] for est in model.estimators_])
    mean = preds.mean()
    lo = np.percentile(preds, 100*(alpha/2))
    hi = np.percentile(preds, 100*(1-alpha/2))
    return mean, lo, hi, preds

# =========================================================
# Prepare Input Vector for Forecasting
# =========================================================

def make_feature_row(history, flows, timestamp):
    """
    history: dictionary storing recent lagged values
    flows: (UnionGap, MabtonFlow)
    timestamp: datetime for seasonal features
    """
    
    ug, mb = flows

    doy = timestamp.timetuple().tm_yday

    row = {
        "Union_Gap_UD": ug,
        "Mabton_UD": mb,
        "flow_diff": mb - ug,
        "flow_ratio": mb / (ug + 1e-6),
        "doy_sin": np.sin(2*np.pi*doy/365),
        "doy_cos": np.cos(2*np.pi*doy/365),

        }
    
    # Add lagged gage heights
    for lag in range(1, LAGS + 1):
        row[f"Mabton_GH_lag{lag}"] = history["mab"][-lag]
        row[f"Kiona_GH_lag{lag}"] = history["kio"][-lag]
        row[f"Union_Gap_UD_lag{lag}"] = history["ug"][-lag]
        row[f"Mabton_UD_lag{lag}"] = history["mb"][-lag]

    return pd.DataFrame([row])[feature_cols]  # ensure correct column order

# =========================================================
# Multi-step Forecasting with AR Dynamics
# =========================================================

def forecast_gage_heights(flows, start_time, init_history, steps=6, alpha=0.05):
    """
    flows: (UnionGapFlow, MabtonFlow)
    start_time: datetime of first forecast step
    init_history: dict containing last LAGS values for GH + flows
    steps: number of steps to forecast forward
    """

    ug, mb = flows
    results = []

    history = {
        "mab": list(init_history["mab"]),
        "kio": list(init_history["kio"]),
        "ug": list(init_history["ug"]),
        "mb": list(init_history["mb"]),
    }

    time = start_time
    
    for s in range(steps):
        Xrow = make_feature_row(history, (ug, mb), time)

        mab_mean, mab_lo, mab_hi, _ = prediction_interval(mod_mab, Xrow, alpha)
        kio_mean, kio_lo, kio_hi, _ = prediction_interval(mod_kio, Xrow, alpha)

        results.append({
            "step": s+1,
            "time": time,
            "Mabton_pred": mab_mean,
            "Mabton_lo": mab_lo,
            "Mabton_hi": mab_hi,
            "Kiona_pred": kio_mean,
            "Kiona_lo": kio_lo,
            "Kiona_hi": kio_hi,
        })

        # Update autoregressive history
        history["mab"].append(mab_mean)
        history["kio"].append(kio_mean)
        history["ug"].append(ug)
        history["mb"].append(mb)

        # Keep only last LAGS entries
        for key in history:
            history[key] = history[key][-LAGS:]

        # Advance time (assume 1 timestep = 15 minutes)
        time = time + pd.Timedelta(minutes=15)

    return pd.DataFrame(results)

# =========================================================
# Flood Detection
# =========================================================

def Kflood_within_period(forecast_df, threshold=14):
    return (
        forecast_df["Kiona_pred"].max() >= threshold
    )

def Mflood_within_period(forecast_df, threshold=16):
    return (
        forecast_df["Mabton_pred"].max() >= threshold
    )

# =========================================================
# USER INPUT (REAL-TIME DATA)
# =========================================================

if __name__ == "__main__":

    # Actual timestamp of measurement
    start_time = pd.Timestamp("2026-01-15 11:30:00+00:00") # Change to current time when running

    # Real-time discharge inputs
    ug_flow = 5040   # Union Gap discharge (cfs) - Change to real-time value when running
    mb_flow = 7530   # Mabton discharge (cfs) - Change to real-time value when running

    # ==========================================
    # Build initial history from MOST RECENT OBSERVED DATA
    # ==========================================

    init = {
        "mab": list(df["Mabton_GH"].iloc[-LAGS:]),
        "kio": list(df["Kiona_GH"].iloc[-LAGS:]),
        "ug": list(df["Union_Gap_UD"].iloc[-LAGS:]),
        "mb": list(df["Mabton_UD"].iloc[-LAGS:])
    }

    # IMPORTANT: Overwrites most recent flows with real-time values
    init["ug"][-1] = ug_flow
    init["mb"][-1] = mb_flow

    # ==========================================
    # Forecast
    # ==========================================

    fc = forecast_gage_heights(
        flows=(ug_flow, mb_flow),
        start_time=start_time,
        init_history=init,
        steps=6
    )

    print(fc.head())

    # DEBUG FIRST STEP (CRITICAL)
    print("\nFirst-step Mabton prediction:", fc.iloc[0]["Mabton_pred"])
    print("Last observed Mabton (Debug check):", init["mab"][-1])

    print("\nFirst-step Kiona prediction:", fc.iloc[0]["Kiona_pred"])
    print("Last observed Kiona (Debug check):", init["kio"][-1])

    print("\nMabton Flood in next 48 hours?:", Mflood_within_period(fc))
    print("Kiona Flood in next 48 hours?:", Kflood_within_period(fc))
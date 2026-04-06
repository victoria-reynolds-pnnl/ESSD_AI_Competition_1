# # WEEK 3: FTES Training Pipeline

# **Dataset:** DEMO-FTES Test 1 — 1-hour averaged telemetry  
# **Period:** Dec 10, 2024 – Mar 24, 2025  
#             Filtered to hot injection phase (2024-12-13 20:00 to 2025-02-22 17:26:06)
# **Wells:** TL, TN, TC, TU, TS (collar / interval / bottom sections)  
# **Measurements:** Flow rate (L/min), Pressure (psi), Temperature (°C via TEC thermocouples), Electrical Conductivity (EC), Packer depths (ft)

### Training Steps
# 1. Train/test split
# 2. Feature scaling / normalization
# 3. 

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import json
import os

warnings.filterwarnings("ignore")
print("Libraries loaded.")

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
# OpenAI GPT‑5.3-Codex

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
        or "__delta_vs_24hmean" in c
        or "__z_24h" in c
    ) and not c.endswith("__jump_flag_24h")
]

scale_cols = [c for c in (base_scale_cols + mix_proxy_cols + ts_scale_cols)
              if c in train_df.columns and not c.startswith("Phase_")]
# OpenAI GPT‑5.3-Codex

# ── Fit scaler on train only, transform val/test ──────────────────────────────
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

# ── Save split-indexed datasets ─────────────────────────────────────────────
# Keep chronological order and persist which split each row belongs to.
train_out = train_scaled.copy()
val_out = val_scaled.copy()
test_out = test_scaled.copy()

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


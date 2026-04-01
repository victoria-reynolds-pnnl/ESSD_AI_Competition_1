# # FTES Data Cleaning and Preprocessing Pipeline

# **Dataset:** DEMO-FTES Test 1 — 1-hour averaged telemetry  
# **Period:** Dec 10, 2024 – Mar 24, 2025  
# **Wells:** TL, TN, TC, TU, TS (collar / interval / bottom sections)  
# **Measurements:** Flow rate (L/min), Pressure (psi), Temperature (°C via TEC thermocouples), Electrical Conductivity (EC), Packer depths (ft)

# ### Cleaning steps
# 1. Load and inspect data  
# 2. Handle missing / invalid values  
# 3. Detect and correct anomalies  
# 4. Feature engineering and phase labeling  
# 5. Normalize and scale features  
# 6. Train/test split and final validation

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import json

warnings.filterwarnings("ignore")
print("Libraries loaded.")

################# Configuration and Paths #################
DATA_DIR = Path(r"../data")
CSV_FILE = DATA_DIR / "FTES-Full_Test_1hour_avg.csv"
OUTPUT_DIR = DATA_DIR
OUTPUT_DIR.mkdir(exist_ok=True)

filter_to_hot_injection = True  # Set to True to filter dataset to hot injection phase only (2024-12-13 20:00 to 2025-02-22 17:26:06)
###########################################################

# ── load ──────────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(CSV_FILE, parse_dates=["Time"], index_col="Time")
df = df_raw.copy()

# ── timestamp checks ──────────────────────────────────────────────────────────
expected_freq = "1h"
full_idx = pd.date_range(df.index.min(), df.index.max(), freq=expected_freq)
missing_ts = full_idx.difference(df.index)
duplicate_ts = df.index[df.index.duplicated()]

if len(duplicate_ts):
    print("  → Dropping duplicates (keep first)")
    df = df[~df.index.duplicated(keep="first")]

# Reindex to a complete hourly grid so gaps become NaN rows
df = df.reindex(full_idx)
df.index.name = "Time"

# ── identify known placeholder values ─────────────────────────────────────────
# Placeholder sentinel: -500 seen in pressure / depth columns
PRESSURE_PLACEHOLDER = -500.0
depth_cols   = [c for c in df.columns if "Depth" in c]
pressure_cols = [c for c in df.columns if "Pressure" in c]

impute_log = []   # running log of all imputation actions

# Replace -500 with NaN (it is not a physically meaningful value)
for col in pressure_cols + depth_cols:
    n = (df[col] == PRESSURE_PLACEHOLDER).sum()
    if n:
        df[col] = df[col].replace(PRESSURE_PLACEHOLDER, np.nan)
        impute_log.append({"column": col, "action": f"replaced {n} placeholder (-500) → NaN"})
        print(f"  [{col}]  replaced {n} placeholder values")

print(f"\nTotal NaN per column (before imputation):\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ── drop columns where > 5 % of values are missing ───────────────────────────
n_rows = len(df)
miss_frac = df.isnull().sum() / n_rows
drop_cols = miss_frac[miss_frac > 0.05].index.tolist()
if drop_cols:
    print("Dropping columns with > 5% missing values:")
    for c in drop_cols:
        print(f"  {c}  ({miss_frac[c]*100:.1f}% missing)")
    df.drop(columns=drop_cols, inplace=True)
    impute_log.append({"column": str(drop_cols), "action": "dropped (>5% missing)"})
else:
    print("No columns exceed 5% missing threshold.")

# ── imputation: forward-fill for short gaps (<=3 h), then median ──────────────
for col in df.select_dtypes(include="number").columns:
    null_before = df[col].isnull().sum()
    if null_before == 0:
        continue

    # Forward-fill propagates at most 3 steps (hours)
    df[col] = df[col].ffill(limit=3)

    # Any remaining NaN → column median
    remaining = df[col].isnull().sum()
    if remaining:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        impute_log.append({
            "column": col,
            "action": f"ffill(3h) for {null_before - remaining} NaNs; median ({median_val:.4f}) for {remaining} remaining"
        })
    else:
        impute_log.append({
            "column": col,
            "action": f"ffill(3h) filled all {null_before} NaNs"
        })

total_nulls_after = df.isnull().sum().sum()
print(f"Remaining NaN values after imputation: {total_nulls_after}")
print(f"Total imputation actions logged       : {len(impute_log)}")

# Filter to hot injection phase only 2024-12-13 17:17:06 |	2025-02-23 17:26:06
if filter_to_hot_injection := True:
    start_time = pd.to_datetime("2024-12-13 20:00")
    end_time = pd.to_datetime("2025-02-22 17:26:06") # adjusted end date to remove flow spike at 2025-02-23
    df = df.loc[start_time:end_time]
    print(f"\nFiltered to hot injection phase: {df.shape[0]} rows from {df.index.min()} to {df.index.max()}")

# ── classify columns by sensor type ──────────────────────────────────────────
flow_cols  = [c for c in df.columns if "Flow" in c]
temp_cols  = [c for c in df.columns if "TEC" in c]           # thermocouple cols
ec_cols    = [c for c in df.columns if " EC" in c or c.endswith("EC")]
pres_cols  = [c for c in df.columns if "Pressure" in c]
depth_cols = [c for c in df.columns if "Depth" in c]

anomaly_log = []

# 1. Flows: clip negative values to 0
for col in flow_cols:
    neg = (df[col] < 0).sum()
    if neg:
        df[col] = df[col].clip(lower=0)
        anomaly_log.append({"column": col, "action": f"clipped {neg} negative flows → 0"})

# 2. Pressures: clip values < -5 psi → 0
for col in pres_cols:
    bad = (df[col] < -5).sum()
    if bad:
        df[col] = df[col].clip(lower=0)
        anomaly_log.append({"column": col, "action": f"clipped {bad} values < -5 psi → 0"})

# 3. Temperatures: enforce 0–100 °C
for col in temp_cols:
    bad_low  = (df[col] < 0).sum()
    bad_high = (df[col] > 100).sum()
    if bad_low or bad_high:
        df[col] = df[col].clip(lower=0, upper=100)
        anomaly_log.append({"column": col, "action": f"clipped temps: {bad_low} < 0°C, {bad_high} > 100°C"})

# 4. EC: flag negatives without removing (may be directional)
ec_flag_cols = []
for col in ec_cols:
    neg = (df[col] < 0).sum()
    if neg:
        flag_col = col + "_neg_flag"
        df[flag_col] = (df[col] < 0).astype(int)
        ec_flag_cols.append(flag_col)
        anomaly_log.append({"column": col, "action": f"flagged {neg} negative EC values (HITL review needed)"})
print(f"EC flag columns created: {ec_flag_cols}")

# 5. Depths: replace 0 with column median (~150 ft range observed)
for col in depth_cols:
    zero_mask = df[col] == 0
    n_zeros = zero_mask.sum()
    if n_zeros:
        median_val = df.loc[df[col] > 0, col].median()
        df.loc[zero_mask, col] = median_val
        anomaly_log.append({"column": col, "action": f"replaced {n_zeros} zero depths with median ({median_val:.1f} ft)"})

print(f"\nAnomaly corrections applied: {len(anomaly_log)}")
for entry in anomaly_log[:20]:
    print(f"  [{entry['column']}] {entry['action']}")

# ── spike detection: z-score > 3 capped; high-variance smoothing ─────────────
NUM_COLS = df.select_dtypes(include="number").columns.tolist()
# Exclude flag columns from spike processing
NUM_COLS = [c for c in NUM_COLS if not c.endswith("_neg_flag")]

Z_THRESH = 3.0
overall_std = df[NUM_COLS].std()

spike_count = 0
smooth_count = 0

for col in NUM_COLS:
    col_std = overall_std[col]
    if col_std == 0:
        continue

    # Z-score capping
    col_mean = df[col].mean()
    upper = col_mean + Z_THRESH * col_std
    lower = col_mean - Z_THRESH * col_std
    spikes = ((df[col] > upper) | (df[col] < lower)).sum()
    if spikes:
        df[col] = df[col].clip(lower=lower, upper=upper)
        spike_count += spikes

    # Rolling-variance smoothing: if hourly variance > 10x overall variance
    rolling_var = df[col].rolling(3, min_periods=1).var()
    high_var_mask = rolling_var > 10 * col_std**2
    n_smooth = high_var_mask.sum()
    if n_smooth:
        rolling_median = df[col].rolling(3, center=True, min_periods=1).median()
        df.loc[high_var_mask, col] = rolling_median[high_var_mask]
        smooth_count += n_smooth

print(f"Z-score capped values (|z| > {Z_THRESH}): {spike_count}")
print(f"Rolling-median smoothed values            : {smooth_count}")

# ── Operational phase labels ─────────────────────────────────────────────────
# Phase date ranges from FTES data description and QC chat analysis
HOT_START       = pd.Timestamp("2024-12-13 20:00")
HOT_END         = pd.Timestamp("2025-02-23 11:00")
AMBIENT_START   = pd.Timestamp("2025-03-12 13:00")
AMBIENT_END     = pd.Timestamp("2025-03-24 22:00")

def assign_phase(ts):
    if HOT_START <= ts <= HOT_END:
        return "Hot Injection"
    elif AMBIENT_START <= ts <= AMBIENT_END:
        return "Ambient Injection"
    else:
        return "Maintenance/Testing"

df["Phase"] = df.index.map(assign_phase)
print("Phase distribution:")
print(df["Phase"].value_counts())

# ── Engineered features ───────────────────────────────────────────────────────

# Feature 1: Total Production Flow (sum of available interval flows for producer wells)
# Producer wells = TL, TN, TU, TS  (TC is the main injection well)
producer_interval_flows = [c for c in flow_cols if "Interval Flow" in c
                            and not c.startswith("TC")]
df["Total_Production_Flow"] = df[producer_interval_flows].sum(axis=1)
print("Producer interval flow columns summed:", producer_interval_flows)

# Feature 2: Pressure Differential (injection pressure minus mean producer interval pressures)
inj_pres_col = "Injection Pressure"
producer_pres = [c for c in pres_cols if "Interval Pressure" in c
                 and not c.startswith("TC")]
if inj_pres_col in df.columns and producer_pres:
    df["Pressure_Differential"] = df[inj_pres_col] - df[producer_pres].mean(axis=1)
    print(f"Pressure differential: {inj_pres_col} minus mean of {producer_pres}")

# Feature 3: Depth Consistency Flag
# Binary flag = 1 if any packer depth deviates > 10 ft from its own median
def depth_consistency_flag(row):
    for col in depth_cols:
        if col in row.index:
            med = df[col].median()
            if abs(row[col] - med) > 10:
                return 1
    return 0

df["Depth_Consistency_Flag"] = df.apply(depth_consistency_flag, axis=1)
print(f"\nRows with inconsistent packer depths: {df['Depth_Consistency_Flag'].sum()}")

# Feature 4: TC (injection) impedance-like proxy = TC Pressure / (Net Flow + ε)
# Goal: gauge "how much pressure per unit flow" at TC (rising pressure at same/lower flow => proxy increases)

tc_pres_col = "TC Interval Pressure"   # change to "TC Bottom Pressure" or "Injection Pressure" if preferred
net_flow_col = "Net Flow"
eps = 1e-3  # L/min, prevents divide-by-zero blowups (tune)

if tc_pres_col in df.columns and net_flow_col in df.columns:
    denom = df[net_flow_col].clip(lower=0) + eps
    df["TC_Impedance_Proxy"] = df[tc_pres_col] / denom

    print(f"\nTC impedance-like proxy: {tc_pres_col} / (max({net_flow_col}, 0) + {eps})")
else:
    print(f"Missing columns. Need: {tc_pres_col} and {net_flow_col}")

print(f"\nEngineered feature columns:\n  Total_Production_Flow, Pressure_Differential, Depth_Consistency_Flag, Injection_Impedance_Proxy")

# Mixing Proxy Features: distance of producer EC to injection EC, normalized by baseline EC difference
# Columns
inj_ec_col = "Injection EC"

producer_ec_cols = [
    c for c in df.columns
    if c.endswith("EC")
    and c != inj_ec_col
    and "__" not in c
    and (c.startswith("TL") or c.startswith("TN") or c.startswith("TU") or c.startswith("TS"))
]

baseline_start = "2024-12-10"
baseline_stop  = "2024-12-20"

def get_baseline(series):
    if baseline_start is not None and baseline_stop is not None and "Time" in df.columns:
        m = (df["Time"] >= pd.to_datetime(baseline_start)) & (df["Time"] <= pd.to_datetime(baseline_stop))
        return series.loc[m].median()
    # if Time is index:
    m = (df.index >= pd.to_datetime(baseline_start)) & (df.index <= pd.to_datetime(baseline_stop))
    return series.loc[m].median()

eps = 1e-6
created = []

if inj_ec_col in df.columns and producer_ec_cols:
    for col in producer_ec_cols:
        base = get_baseline(df[col])
        out = f"{col}__MixFrac"

        if out not in df.columns:  # safe to re-run
            f = (df[col] - base) / (df[inj_ec_col] - base + eps)
            df[out] = f.clip(0, 1)

        created.append(out)
        print(f"{col}: baseline EC = {base:.2f} µS/cm -> created {out}")
else:
    print(f"Need '{inj_ec_col}' and producer EC columns. Found: {producer_ec_cols}")

print("\nMixFrac columns:")
print(created)

# OpenAI GPT‑5.2

# ── Temporal split by % (chronological, no shuffle) ────
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

# ── One-hot encode Phase (fit on train; align val/test) ───────────────────────
train_df = pd.get_dummies(train_df, columns=["Phase"], prefix="Phase", drop_first=False)
val_df   = pd.get_dummies(val_df,   columns=["Phase"], prefix="Phase", drop_first=False)
test_df  = pd.get_dummies(test_df,  columns=["Phase"], prefix="Phase", drop_first=False)

# Ensure same columns across splits (no leakage; consistent model inputs)
val_df  = val_df.reindex(columns=train_df.columns, fill_value=0)
test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

phase_cols = [c for c in train_df.columns if c.startswith("Phase_")]
print("One-hot Phase columns (train):", phase_cols)

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

scale_cols = [c for c in (base_scale_cols + mix_proxy_cols)
              if c in train_df.columns and not c.startswith("Phase_")]

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

# ── Save cleaned dataset ──────────────────────────────────────────────────────
out_path = OUTPUT_DIR / "FTES_1hour_cleaned.csv"
df.to_csv(out_path)
print(f"Cleaned dataset saved → {out_path}")
print(f"Final shape: {df.shape}")

# ── Data dictionary ───────────────────────────────────────────────────────────
# Build a reference table of all columns with type, units, and transformations

unit_map = {}
for c in df.columns:
    if c.startswith("Phase_") or c == "Phase":
        unit_map[c] = "categorical"
    elif c.endswith("_neg_flag") or c.endswith("_flag") or "Flag" in c:
        unit_map[c] = "binary"
    elif "__MixFrac" in c:
        unit_map[c] = "fraction (0–1)"
    elif "__NormDist_to_InjEC" in c:
        unit_map[c] = "dimensionless"
    elif "__Delta_to_InjEC" in c:
        unit_map[c] = "µS/cm"
    elif "Impedance_Proxy" in c:
        unit_map[c] = "psi / (L/min)"
    elif "Flow" in c:
        unit_map[c] = "L/min"
    elif "Pressure" in c:
        unit_map[c] = "psi"
    elif "TEC" in c:
        unit_map[c] = "°C"
    elif " EC" in c or c.endswith("EC"):
        unit_map[c] = "µS/cm"
    elif "Depth" in c:
        unit_map[c] = "ft"
    else:
        unit_map[c] = "—"

transform_map = {}
for c in df.columns:
    transforms = []

    # raw sensor transforms (your existing rules)
    if c in flow_cols:   transforms.append("clipped negative→0")
    if c in pres_cols:   transforms.append("clipped <-5psi→0")
    if c in temp_cols:   transforms.append("clipped 0–100°C")
    if c in ec_cols:     transforms.append("negatives flagged (HITL)")
    if c in depth_cols:  transforms.append("zeros→median; z-score cap")

    # engineered feature provenance
    if c == "Total_Production_Flow":      transforms.append("engineered: sum(producer interval flows)")
    if c == "Pressure_Differential":      transforms.append("engineered: Injection Pressure − mean(producer interval pressures)")
    if c.endswith("Impedance_Proxy"):     transforms.append("engineered: Pressure/(Net Flow+ε)")
    if "__Delta_to_InjEC" in c:           transforms.append("engineered: EC_prod − EC_inj")
    if "__NormDist_to_InjEC" in c:        transforms.append("engineered: |EC_prod−EC_inj|/|EC_inj−EC_baseline|")
    if "__MixFrac" in c:                  transforms.append("engineered: (EC_prod−EC_base)/(EC_inj−EC_base); clipped 0–1")

    # encoding / scaling
    if c == "Phase":                      transforms.append("one-hot encoded (see Phase_*)")
    if c.startswith("Phase_"):            transforms.append("one-hot column")
    if "scale_cols" in globals() and c in scale_cols:
        transforms.append("MinMaxScaled (fit on train)")

    transform_map[c] = "; ".join(transforms) if transforms else "None"

data_dict = pd.DataFrame({
    "Column"         : df.columns,
    "dtype"          : [str(df[c].dtype) for c in df.columns],
    "Units"          : [unit_map.get(c, "—") for c in df.columns],
    "Transformations": [transform_map.get(c, "None") for c in df.columns],
    "Non-null count" : [df[c].notna().sum() for c in df.columns],
})

dict_path = OUTPUT_DIR / "FTES_1hour_cleaned_data_dictionary.csv"
data_dict.to_csv(dict_path, index=False)
print(f"Data dictionary saved → {dict_path}")

data_dict
# OpenAI GPT‑5.2

dataset_name = "FTES_1hour_cleaned"

def build_json_data_dictionary(df, output_path, dataset_name="cleaned_dataset", max_examples=5):
    # ---- Helpers ----
    def infer_units(col):
        if col.startswith("Phase_") or col == "Phase": return "categorical"
        if col.endswith("_neg_flag") or col.endswith("_flag") or "Flag" in col: return "binary"
        if "__MixFrac" in col: return "fraction (0–1)"
        if "Impedance_Proxy" in col: return "psi/(L/min)"
        if "Flow" in col: return "L/min"
        if "Pressure" in col: return "psi"
        if "TEC" in col: return "°C"
        if " EC" in col or col.endswith("EC"): return "µS/cm"
        if "Depth" in col: return "ft"
        return "—"

    def infer_description(col):
        if col.endswith("__MixFrac"):
            base = col.replace("__MixFrac", "")
            return (f"Estimated fraction of injection-like water at '{base}' using two-endmember EC mixing: "
                    f"(EC_prod - EC_baseline)/(Injection EC - EC_baseline), clipped to [0,1].")
        if col == "Total_Production_Flow":
            return "Engineered: sum of available producer interval flows (excludes TC)."
        if col == "Pressure_Differential":
            return "Engineered: Injection Pressure minus mean of producer interval pressures (excludes TC)."
        if col.endswith("Impedance_Proxy"):
            return "Engineered impedance-like proxy: Pressure/(Net Flow + ε) (higher implies higher resistance for given flow)."
        if col.startswith("Phase_"):
            return f"One-hot encoded indicator for Phase == '{col.replace('Phase_', '')}'."
        if col == "Phase":
            return "Operational phase label (categorical)."
        if "Interval EC" in col: return "Electrical conductivity measured at interval sensor."
        if "Bottom EC" in col: return "Electrical conductivity measured at bottom sensor."
        if "Collar EC" in col: return "Electrical conductivity measured at collar sensor."
        if "Injection EC" in col: return "Electrical conductivity of injection stream."
        if "Interval Flow" in col: return "Flow rate measured at interval."
        if "Bottom Flow" in col: return "Flow rate measured at bottom."
        if "Collar Flow" in col: return "Flow rate measured at collar."
        if "Net Flow" in col: return "Net flow rate (system-level)."
        if "Injection Pressure" in col: return "Injection line pressure."
        if "Interval Pressure" in col: return "Pressure measured at interval."
        if "Bottom Pressure" in col: return "Pressure measured at bottom."
        if "Packer Pressure" in col: return "Pressure measured at packer."
        if "TEC" in col: return "Temperature sensor (TEC)."
        if "Depth" in col: return "Depth measurement/setting."
        if col.endswith("_neg_flag"): return "Flag indicating negative values were detected for this signal."
        return "Column from cleaned dataset."

    def infer_transformations(col):
        transforms = []
        if "Flow" in col and "__" not in col and not col.startswith("Phase_"):
            transforms.append("may be clipped negative→0 (if applied during cleaning)")
        if "Pressure" in col and "__" not in col and not col.startswith("Phase_"):
            transforms.append("may be clipped below threshold→0 (if applied during cleaning)")
        if "TEC" in col:
            transforms.append("may be clipped to plausible range (if applied during cleaning)")
        if (" EC" in col or col.endswith("EC")) and "__" not in col:
            transforms.append("may have negatives flagged (if applied during cleaning)")
        if col.endswith("__MixFrac"):
            transforms.append("engineered from Injection EC + baseline window; clipped [0,1]")
        if col.startswith("Phase_"):
            transforms.append("one-hot encoded from Phase")
        return transforms

    def example_values(series, k=max_examples):
        s = series.dropna()
        if s.empty:
            return []
        vals = s.unique()[:k]
        out = []
        for v in vals:
            if isinstance(v, (np.datetime64, pd.Timestamp)):
                out.append(pd.to_datetime(v).isoformat())
            elif isinstance(v, (np.floating, float)):
                out.append(float(v))
            elif isinstance(v, (np.integer, int)):
                out.append(int(v))
            elif isinstance(v, (np.bool_, bool)):
                out.append(bool(v))
            else:
                out.append(str(v))
        return out

    def dtype_format(series):
        if pd.api.types.is_datetime64_any_dtype(series): return (str(series.dtype), "ISO-8601 datetime")
        if pd.api.types.is_bool_dtype(series):          return (str(series.dtype), "boolean")
        if pd.api.types.is_integer_dtype(series):       return (str(series.dtype), "integer")
        if pd.api.types.is_float_dtype(series):         return (str(series.dtype), "float")
        return (str(series.dtype), "string/categorical")

    def numeric_stats(series):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return {"min": None, "mean": None, "max": None}
        return {"min": float(s.min()), "mean": float(s.mean()), "max": float(s.max())}

    # ---- Dataset-level metadata ----
    dd = {
        "dataset_name": dataset_name,
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "index": {
            "name": df.index.name if df.index.name else "index",
            "dtype": str(df.index.dtype),
            "is_datetime_index": bool(isinstance(df.index, pd.DatetimeIndex)),
            "min": df.index.min().isoformat() if isinstance(df.index, pd.DatetimeIndex) and len(df) else None,
            "max": df.index.max().isoformat() if isinstance(df.index, pd.DatetimeIndex) and len(df) else None,
        },
        "fields": []
    }

    # ---- Field-level entries ----
    for col in df.columns:
        ser = df[col]
        dtype_str, fmt = dtype_format(ser)

        entry = {
            "name": col,
            "description": infer_description(col),
            "dtype": dtype_str,
            "format": fmt,
            "units": infer_units(col),
            "transformations": infer_transformations(col),
            "non_null_count": int(ser.notna().sum()),
            "null_count": int(ser.isna().sum()),
            "stats": numeric_stats(ser) if pd.api.types.is_numeric_dtype(ser) else None,
            "source_context": (
                "cleaned dataset column; sensor channels typically prefixed by well (TL/TN/TU/TS/TC). "
                "Engineered fields noted in 'transformations'."
            ),
        }
        dd["fields"].append(entry)

    with open(str(output_path), "w", encoding="utf-8") as f:
        json.dump(dd, f, indent=2, ensure_ascii=False)

    return dd


# ---- Usage ----
json_path = OUTPUT_DIR / "FTES_1hour_cleaned_data_dictionary.json"
dd_json = build_json_data_dictionary(df, json_path, dataset_name="df_cleaned", max_examples=5)
print(f"JSON data dictionary saved → {json_path}")
# OpenAI GPT‑5.2

# ── RAW data dictionary (CSV + JSON) ──────────────────────────────────────────
# Matches raw columns exactly; includes units + basic stats (min/max/mean) for numeric cols.

unit_map = {}
for c in df_raw.columns:
    if c.startswith("Phase_") or c == "Phase":
        unit_map[c] = "categorical"
    elif c.endswith("_neg_flag") or c.endswith("_flag") or "Flag" in c:
        unit_map[c] = "binary"
    elif "__MixFrac" in c:
        unit_map[c] = "fraction (0–1)"
    elif "__NormDist_to_InjEC" in c:
        unit_map[c] = "dimensionless"
    elif "__Delta_to_InjEC" in c:
        unit_map[c] = "µS/cm"
    elif "Impedance_Proxy" in c:
        unit_map[c] = "psi / (L/min)"
    elif "Flow" in c:
        unit_map[c] = "L/min"
    elif "Pressure" in c:
        unit_map[c] = "psi"
    elif "TEC" in c:
        unit_map[c] = "°C"
    elif " EC" in c or c.endswith("EC"):
        unit_map[c] = "µS/cm"
    elif "Depth" in c:
        unit_map[c] = "ft"
    else:
        unit_map[c] = "—"

# For raw: no cleaning/engineering transformations applied (keep explicit)
transform_map = {c: "None (raw)" for c in df_raw.columns}

def num_stat(series, fn):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(fn(s)) if len(s) else np.nan

raw_data_dict = pd.DataFrame({
    "Column"         : df_raw.columns,
    "dtype"          : [str(df_raw[c].dtype) for c in df_raw.columns],
    "Units"          : [unit_map.get(c, "—") for c in df_raw.columns],
    "Transformations": [transform_map.get(c, "None (raw)") for c in df_raw.columns],
    "Non-null count" : [int(df_raw[c].notna().sum()) for c in df_raw.columns],
    "min"            : [num_stat(df_raw[c], np.min)  if pd.api.types.is_numeric_dtype(df_raw[c]) else np.nan for c in df_raw.columns],
    "max"            : [num_stat(df_raw[c], np.max)  if pd.api.types.is_numeric_dtype(df_raw[c]) else np.nan for c in df_raw.columns],
    "mean"           : [num_stat(df_raw[c], np.mean) if pd.api.types.is_numeric_dtype(df_raw[c]) else np.nan for c in df_raw.columns],
})

raw_csv_path  = OUTPUT_DIR / "FTES-Full_Test_1hour_avg_data_dictionary.csv"   # (keep your path)
raw_json_path = OUTPUT_DIR / "FTES-Full_Test_1hour_avg_data_dictionary.json"  # (keep your path)

raw_data_dict.to_csv(raw_csv_path, index=False)
print(f"Raw data dictionary CSV saved → {raw_csv_path}")

# JSON version (same content as CSV)
dd_json = {
    "dataset_name": "raw_data",
    "generated_utc": pd.Timestamp.utcnow().isoformat(),
    "row_count": int(df_raw.shape[0]),
    "column_count": int(df_raw.shape[1]),
    "index": {
        "name": df_raw.index.name if df_raw.index.name else "index",
        "dtype": str(df_raw.index.dtype),
        "is_datetime_index": bool(isinstance(df_raw.index, pd.DatetimeIndex)),
        "min": df_raw.index.min().isoformat() if isinstance(df_raw.index, pd.DatetimeIndex) and len(df_raw) else None,
        "max": df_raw.index.max().isoformat() if isinstance(df_raw.index, pd.DatetimeIndex) and len(df_raw) else None,
    },
    "fields": raw_data_dict.to_dict(orient="records"),
}

with open(raw_json_path, "w", encoding="utf-8") as f:
    json.dump(dd_json, f, indent=2, ensure_ascii=False)

print(f"Raw data dictionary JSON saved → {raw_json_path}")

raw_data_dict
# OpenAI GPT‑5.2
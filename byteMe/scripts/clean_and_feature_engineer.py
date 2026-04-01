#!/usr/bin/env python3
"""
clean_and_feature_engineer.py
------------------------------
Data cleaning and feature engineering for the FTES dataset.

Inputs  (rawData/):
  - FTES-Full_Test_1hour_avg.csv          (~2 609 rows, primary ML dataset)
  - FTES-Full_Test_1sec_system_processed.csv  (~6.5 GB, processed in chunks)

Outputs (processedData/):
  - FTES_cleaned_1hour.csv
  - FTES_cleaned_1sec_1min_resample.csv   (1-second data resampled to 1-minute)

AI-generated sections:
  The entire script was generated with AI assistance (GitHub Copilot / Claude Sonnet 4.6),
  including Feature engineering logic, experimental design alignment, and pipeline
  structure.

-------------------------------------------------------------------------------
DESIGN EXPERIMENT ALIGNMENT — KEY CHANGES
-------------------------------------------------------------------------------
The following changes were made to align this script with the formal experimental
design document produced by our LLM Engineer (goal2_experiment_design.html). 
Each change is keyed to the design's Run Order steps where relevant.

[1] T0 per-well ambient baseline  (Run Order Step 1)
    ORIGINAL : No ambient baseline was computed. All features used raw temperature
               values, meaning the model would have to learn each well's absolute
               underground offset independently, hurting generalizability.
    UPDATED  : compute_T0() loads the pre-injection window (Dec 10–13) BEFORE
               Phase-1 filtering removes those rows. It computes T0[well] =
               median((TEC-INT-U + TEC-INT-L) / 2) per well. This anchors the
               new delta_T_above_T0_{well} features, which are the primary
               prediction targets in the design (temperature rise above ambient,
               not absolute temperature).

[2] Phase-1 filter boundary  (Run Order Step 2)
    ORIGINAL : Filter started at midnight 2024-12-13, which included several
               hours before the pump was first turned on.
    UPDATED  : Filter now starts at INJECTION_START = 2024-12-13 20:00, matching
               the actual injection start timestamp used throughout the design.

[3] Ramp-up window detection and exclusion  (Run Order Step 2)
    ORIGINAL : No ramp-up exclusion. The first ~6 hours of transient pump spin-up
               were included in training data, which distorts modeled steady-state
               thermal patterns.
    UPDATED  : detect_ramp_up_end() identifies the end of the ramp-up window
               dynamically: the first timestamp at which Net Flow has been ≥ 80%
               of its 6-hr rolling maximum for ≥ 60 consecutive minutes.
               Rows in this window are labeled 'ramp_up' in the split column and
               must be excluded from model training.

[4] Train/test split labels  (Run Order Steps 2, 5)
    ORIGINAL : No split information was written to the output CSV. The ML engineer
               had no reliable way to enforce the strictly chronological partition
               required by the design (no shuffling of any kind).
    UPDATED  : add_split_labels() writes a 'split' column to every output row:
                 'ramp_up' — transient spin-up, excluded from training/evaluation
                 'train'   — ~50 days post-ramp-up (through ~Feb 1 20:00)
                 'test'    — last ~22 days of Phase 1, touched once after all
                             model decisions are frozen
               This enforces the expanding-window CV structure described in the
               design's Data Splits tab.

[5] Monitor well sensor drift detrending  (Run Order Step 3)
    ORIGINAL : No drift correction. TU and TS are monitor wells far from the
               injection well TC; a linear temperature rise in those wells over
               the 72-day Phase 1 period is instrument drift, not real thermal
               response. Including drifted values corrupts Family C features.
    UPDATED  : check_monitor_drift() fits a linear trend to TEC-INT-U for TU
               and TS over the full hot-phase window. If |slope| > 0.12 °C/hr
               (= 0.002 °C/min, the design's threshold), a linear detrend is
               applied to all TEC columns of that well before features are built.

[6] Interval mean temperature columns
    ORIGINAL : Raw upper (TEC-INT-U) and lower (TEC-INT-L) sensor values were
               used separately. Noise from individual sensors propagates directly
               into rolling stats, delta calculations, and the cumulative heat
               feature.
    UPDATED  : compute_interval_means() averages each well's upper/lower pair
               into XX_INT_mean and XX_BOT_mean before feature construction.
               These averaged columns are used as the primary temperature signal
               throughout feature families A, B, and C.

[7] cumulative_heat_input replaces cumulative_injected_volume as primary C-family
    feature  (Design: Family C — Engineered physics features)
    ORIGINAL : cumulative_injected_volume accumulated Net Flow × Δt only.
               This ignores that hotter injected water carries more thermal energy
               per unit volume, which is the physical mechanism driving production
               well breakthrough.
    UPDATED  : cumulative_heat_input = Σ(Net_Flow × TC_INT_mean) × Δt, encoding
               both volumetric and thermal energy delivery. The design rates this
               feature as Critical importance — the most powerful single engineered
               feature. cumulative_injected_volume is retained for backward
               compatibility but is superseded by this column for modeling.

[8] New engineered features added  (Design: Feature Families A, B, C)
    The following features were added to match the design specification:
      TC_INT_delta          (A) — rate of change of injection temp (°C/hr);
                                  captures acceleration/deceleration of hot-water
                                  delivery, a High-importance Family A feature.
      dT_TL_dt / dT_TN_dt  (B) — rate of temperature rise at each production
                                  well; Family B autoregressive feature tracking
                                  how fast the thermal front is currently moving.
      elapsed_injection_min (D1) — minutes since injection start; supersedes
                                  days_since_injection as the design specifies
                                  minutes as the unit for elapsed time features.
      delta_T_above_T0_TL,
      delta_T_above_T0_TN  (D2) — temperature rise above T0 per production well;
                                  the primary target variable. High importance per
                                  the design; removes well-to-well absolute offsets.
      T_gradient_INT_{well} (D4) — TEC-INT-U minus TEC-INT-L; vertical thermal
                                  gradient within the 26-inch packer interval.
                                  Sharpens near the thermal front arrival.
-------------------------------------------------------------------------------
"""

import os
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RAW_DIR   = os.path.join(BASE_DIR, '..', 'rawData')
OUT_DIR   = os.path.join(BASE_DIR, '..', 'processedData')
os.makedirs(OUT_DIR, exist_ok=True)

HOURLY_FILE = os.path.join(RAW_DIR, 'FTES-Full_Test_1hour_avg.csv')
SEC_FILE    = os.path.join(RAW_DIR, 'FTES-Full_Test_1sec_system_processed.csv')

# Timeline boundaries
PRE_INJECTION_START = pd.Timestamp('2024-12-10')           # data collection begins
INJECTION_START     = pd.Timestamp('2024-12-13 20:00:00')  # hot water injection begins
PHASE1_END          = pd.Timestamp('2025-02-23 23:59:59')  # Phase 1 ends
TRAIN_END           = pd.Timestamp('2025-02-01 20:00:00')  # ~50 days post-injection → train/test split

# Ramp-up detection parameters (Run Order Step 2)
RAMP_UP_FLOW_MIN   = 0.5   # L/min — Net Flow above this signals pump is on
RAMP_UP_STEADY_PCT = 0.80  # fraction of 6-hr rolling max Net Flow = "steady-state"
RAMP_UP_CONSEC_HRS = 1     # consecutive hours at steady-state to declare ramp-up done

# Sensor drift detrending threshold for monitor wells (Run Order Step 3)
DRIFT_THRESHOLD_C_PER_HR = 0.12  # 0.002 °C/min × 60 min/hr

# Physical plausibility bounds (domain knowledge)
TEMP_MIN,  TEMP_MAX  = -5.0, 150.0   # °C
PRESS_MAX            = 500.0          # bar/psi
EC_MIN,    EC_MAX    =  0.0, 5_000.0  # µS/cm

# Well roles
WELLS            = ['TL', 'TN', 'TC', 'TU', 'TS']
INJECTION_WELL   = 'TC'
PRODUCTION_WELLS = ['TL', 'TN']
MONITOR_WELLS    = ['TU', 'TS']

# 1-second processing: set to False to skip (the file is ~6.5 GB)
PROCESS_1SEC  = True
SEC_CHUNKSIZE = 200_000  # rows per chunk


# ---------------------------------------------------------------------------
# Helper identifiers
# ---------------------------------------------------------------------------
def _temp_cols(df):
    return [c for c in df.columns if 'TEC' in c]

def _pressure_cols(df):
    return [c for c in df.columns if 'Pressure' in c or c.startswith('PT ')]

def _ec_cols(df):
    return [c for c in df.columns if ' EC' in c or c == 'Injection EC']

def _flow_cols(df):
    return [c for c in df.columns if 'Flow' in c]


# ---------------------------------------------------------------------------
# Section 1 – Load
# ---------------------------------------------------------------------------
def load_hourly(path):
    df = pd.read_csv(path, parse_dates=['Time'])
    df = df.rename(columns={'Time': 'timestamp'}).set_index('timestamp').sort_index()
    return df


def _parse_1sec_chunk(chunk):
    """Normalize a raw chunk from the 1-second CSV."""
    # Drop the leading integer index column emitted by the logger
    drop_cols = [c for c in chunk.columns if c == '' or c.startswith('Unnamed')]
    chunk = chunk.drop(columns=drop_cols, errors='ignore')
    chunk = chunk.rename(columns={'Time': 'timestamp'}).set_index('timestamp').sort_index()

    # Convert True/False boolean string columns to int (0/1)
    bool_cols = [c for c in chunk.columns if '(True/False)' in c]
    for col in bool_cols:
        chunk[col] = chunk[col].map({'True': 1, 'False': 0, True: 1, False: 0})

    # Map Triplex On/Off to binary
    if 'Triplex On/Off' in chunk.columns:
        chunk['Triplex On/Off'] = chunk['Triplex On/Off'].map(
            {'on': 1, 'off': 0, 1: 1, 0: 0}
        ).fillna(0).astype(int)

    return chunk


# ---------------------------------------------------------------------------
# Section 2 – Pre-injection T0 baseline (Run Order Step 1)
# ---------------------------------------------------------------------------
def compute_T0(df_full):
    """
    Compute per-well ambient baseline temperature T0 from the pre-injection
    window (PRE_INJECTION_START → INJECTION_START, Dec 10–13).

    T0[well] = median of (TEC-INT-U + TEC-INT-L) / 2 over that window.

    This MUST be called on the full raw dataframe before Phase-1 filtering
    removes the pre-injection rows.  T0 anchors all delta_T_above_T0
    calculations and is the intended primary target variable per the
    experimental design.

    Returns a dict: {well: T0_float}  e.g. {'TL': 24.7, 'TN': 24.6, ...}
    """
    pre = df_full.loc[PRE_INJECTION_START:INJECTION_START]
    T0 = {}
    for well in WELLS:
        upper = f'{well}-TEC-INT-U'
        lower = f'{well}-TEC-INT-L'
        if upper in pre.columns and lower in pre.columns:
            T0[well] = float(((pre[upper] + pre[lower]) / 2.0).median())
        else:
            T0[well] = np.nan
    print("\nT0 per well (pre-injection ambient baseline):")
    for w, v in T0.items():
        tag = f'{v:.3f} °C' if not np.isnan(v) else 'N/A (columns missing)'
        print(f"  {w}: {tag}")
    return T0


# ---------------------------------------------------------------------------
# Section 3 – Cleaning
# ---------------------------------------------------------------------------
def clean(df, label=''):
    print(f"\n[{label}] Raw shape: {df.shape}")

    # 2a. Remove duplicate timestamps
    n_dup = df.index.duplicated().sum()
    if n_dup:
        print(f"  Removing {n_dup} duplicate timestamps")
        df = df[~df.index.duplicated(keep='first')]

    # 2b. Filter Phase 1 only (hot-water injection period)
    df = df.loc[INJECTION_START:PHASE1_END]
    print(f"  After Phase-1 filter ({INJECTION_START} – {PHASE1_END.date()}): {df.shape}")

    # 2c. Drop constant columns (zero information; e.g. fixed packer depths)
    nuniq = df.select_dtypes(include=[np.number]).nunique(dropna=False)
    const_cols = nuniq[nuniq <= 1].index.tolist()
    if const_cols:
        print(f"  Dropping constant columns: {const_cols}")
        df = df.drop(columns=const_cols, errors='ignore')

    # 2d. Clamp physically implausible sensor values to NaN
    for col in _temp_cols(df):
        mask = (df[col] < TEMP_MIN) | (df[col] > TEMP_MAX)
        n = mask.sum()
        if n:
            print(f"  {col}: clamping {n} temperature outliers to NaN")
            df.loc[mask, col] = np.nan

    for col in _pressure_cols(df):
        mask = df[col] > PRESS_MAX
        n = mask.sum()
        if n:
            print(f"  {col}: clamping {n} pressure spikes (>{PRESS_MAX}) to NaN")
            df.loc[mask, col] = np.nan

    for col in _ec_cols(df):
        mask = (df[col] < EC_MIN) | (df[col] > EC_MAX)
        n = mask.sum()
        if n:
            print(f"  {col}: clamping {n} EC outliers to NaN")
            df.loc[mask, col] = np.nan

    # 2e. Drop columns that are entirely NaN — done after outlier clamping so
    #     columns whose values were all out-of-range are also caught here.
    all_nan = df.columns[df.isna().all()].tolist()
    if all_nan:
        print(f"  Dropping all-NaN columns (post-clamp): {all_nan}")
        df = df.drop(columns=all_nan)

    # 2f. Interpolate short gaps (≤ 2 consecutive missing values) using time
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = (
        df[numeric_cols]
        .interpolate(method='time', limit=2, limit_direction='forward')
    )

    # 2g. Drop columns still >50% NaN after interpolation (not recoverable)
    nan_frac = df.isna().mean()
    unrecoverable = nan_frac[nan_frac > 0.50].index.tolist()
    if unrecoverable:
        print(f"  Dropping columns with >50% NaN after interpolation: {unrecoverable}")
        df = df.drop(columns=unrecoverable)
        nan_frac = df.isna().mean()

    # 2h. Report residual NaN fraction (>5%) for any remaining columns
    high_nan = nan_frac[nan_frac > 0.05]
    if not high_nan.empty:
        print(f"  Columns with >5% NaN remaining after interpolation:\n"
              f"{high_nan.round(3).to_string()}")

    print(f"[{label}] Clean shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Section 4 – Split labeling (Run Order Step 2)
# ---------------------------------------------------------------------------
def detect_ramp_up_end(df, freq='1h'):
    """
    Identify ramp-up end: first timestamp at which Net Flow has been ≥ 80% of
    its 6-hr rolling maximum for at least 60 consecutive minutes (1 hr).
    Falls back to INJECTION_START + 6 h if detection fails.
    """
    if 'Net Flow' not in df.columns:
        return INJECTION_START + pd.Timedelta(hours=6)

    rows_per_hour = 1 if freq == '1h' else 60
    window_rows = 6 * rows_per_hour   # 6-hr rolling max window
    consec_rows = 1 * rows_per_hour   # 60 min of steady flow

    flow = df['Net Flow'].fillna(0)
    rolling_max = flow.rolling(window=window_rows, min_periods=1).max()
    is_steady = (flow >= RAMP_UP_STEADY_PCT * rolling_max) & (flow > RAMP_UP_FLOW_MIN)

    # Consecutive steady count
    streak = is_steady.groupby((~is_steady).cumsum()).cumsum()
    reached = streak[streak >= consec_rows]

    if reached.empty:
        ts = INJECTION_START + pd.Timedelta(hours=6)
        print(f"  Ramp-up detection: no steady window found; defaulting to {ts}")
    else:
        ts = reached.index[0]
        print(f"  Ramp-up end detected: {ts}")
    return ts


def add_split_labels(df, ramp_up_end):
    """
    Add a string 'split' column marking each row's role in the ML pipeline:
      'ramp_up' – transient pump spin-up, excluded from model training
      'train'   – ~50 days post-ramp-up, used to fit models
      'test'    – last ~22 days of Phase 1, touched once after all decisions frozen
    """
    df['split'] = 'train'
    df.loc[df.index <= ramp_up_end, 'split'] = 'ramp_up'
    df.loc[df.index > TRAIN_END, 'split'] = 'test'
    print(f"  Split counts: {df['split'].value_counts().to_dict()}")
    return df


# ---------------------------------------------------------------------------
# Section 5 – Sensor drift detrending for monitor wells (Run Order Step 3)
# ---------------------------------------------------------------------------
def check_monitor_drift(df):
    """
    For each monitor well (TU, TS), fit a linear trend to TEC-INT-U over the
    full cleaned period.  Monitor wells are far from the injection well TC, so
    a sustained linear rise indicates instrument drift rather than real thermal
    response.  If |slope| > DRIFT_THRESHOLD_C_PER_HR (0.12 °C/hr = 0.002 °C/min),
    subtract the linear trend from all TEC columns of that well.
    """
    for well in MONITOR_WELLS:
        col = f'{well}-TEC-INT-U'
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) < 10:
            continue
        x_fit = (series.index - df.index[0]).total_seconds().values
        slope, intercept, _, _, _ = scipy_stats.linregress(x_fit, series.values)
        slope_per_hr = slope * 3600
        print(f"  Drift check {well}: {slope_per_hr:+.4f} °C/hr "
              f"(threshold ±{DRIFT_THRESHOLD_C_PER_HR:.4f})")
        if abs(slope_per_hr) > DRIFT_THRESHOLD_C_PER_HR:
            print(f"  *** Linear detrend applied to {well} TEC columns ***")
            x_all = pd.Series(
                (df.index - df.index[0]).total_seconds(), index=df.index
            )
            for tc in [c for c in df.columns if c.startswith(f'{well}-TEC')]:
                df[tc] = df[tc] - slope * x_all
        else:
            print(f"  {well}: drift within threshold, no detrend applied.")
    return df


# ---------------------------------------------------------------------------
# Section 6 – Intermediate computed columns
# ---------------------------------------------------------------------------
def compute_interval_means(df):
    """
    Compute per-well interval and bottom mean temperatures:
      XX_INT_mean = (XX-TEC-INT-U + XX-TEC-INT-L) / 2
      XX_BOT_mean = (XX-TEC-BOT-U + XX-TEC-BOT-L) / 2

    Averaging the upper/lower sensor pair reduces noise and is the primary
    temperature signal used in feature families A, C, and D.
    """
    for well in WELLS:
        u_int, l_int = f'{well}-TEC-INT-U', f'{well}-TEC-INT-L'
        u_bot, l_bot = f'{well}-TEC-BOT-U', f'{well}-TEC-BOT-L'
        if u_int in df.columns and l_int in df.columns:
            df[f'{well}_INT_mean'] = (df[u_int] + df[l_int]) / 2.0
        if u_bot in df.columns and l_bot in df.columns:
            df[f'{well}_BOT_mean'] = (df[u_bot] + df[l_bot]) / 2.0
    return df


# ---------------------------------------------------------------------------
# Section 7 – Feature Engineering
# (AI-assisted: GitHub Copilot / Claude Sonnet 4.6)
# ---------------------------------------------------------------------------
def engineer_features(df, T0, freq='1h'):
    """
    Constructs engineered features aligned with the experimental design's five
    feature families. All features use only past or current information to
    prevent data leakage (no future values used).

    Family A — Injection temperature history (primary causal signal)
      A1: TC_INT_delta        — rate of change of TC_INT_mean (°C/hr)
      A2: net_flow_rolling_6h — 6-hr trailing mean of Net Flow (L/min)

    Family B — Autoregressive target history
      B1: dT_TL_dt / dT_TN_dt — rate of temperature rise at each producer (°C/hr)

    Family C — Engineered physics features (encode cumulative thermal state)
      C1: elapsed_injection_min    — minutes since INJECTION_START
      C2: delta_T_above_T0_{well}  — T_well_INT_mean - T0[well] (primary target)
      C3: cumulative_heat_input    — Σ[Net_Flow × TC_INT_mean] × Δt (°C·L·s)
      C4: T_gradient_INT_{well}    — TEC-INT-U minus TEC-INT-L (vertical gradient)

    Legacy features (retained for backward compatibility)
      L1: days_since_injection      — derived from elapsed_injection_min
      L2: hour_sin / hour_cos       — cyclic hour-of-day encoding
      L3: delta_T_inj_prod          — instantaneous injection vs. production ΔT
      L4: cumulative_injected_volume — flow-only cumsum (superseded by C3)
    """
    dt_seconds = 3600 if freq == '1h' else 60

    # ---- A1: Rate of change of injection interval temperature ----
    if 'TC_INT_mean' in df.columns:
        df['TC_INT_delta'] = df['TC_INT_mean'].diff() / (dt_seconds / 3600.0)

    # ---- A2: 6-hour rolling mean of Net Flow ----
    if 'Net Flow' in df.columns:
        window = 6 if freq == '1h' else 360
        df['net_flow_rolling_6h'] = (
            df['Net Flow'].rolling(window=window, min_periods=1).mean()
        )

    # ---- B1: Rate of temperature rise at each production well ----
    for well in PRODUCTION_WELLS:
        col = f'{well}_INT_mean'
        if col in df.columns:
            df[f'dT_{well}_dt'] = df[col].diff() / (dt_seconds / 3600.0)

    # ---- C1: Elapsed time since injection start (minutes) ----
    df['elapsed_injection_min'] = (
        pd.Series((df.index - INJECTION_START).total_seconds(), index=df.index)
        / 60.0
    ).clip(lower=0)

    # ---- C2: Temperature rise above pre-injection ambient per production well ----
    for well in PRODUCTION_WELLS:
        col = f'{well}_INT_mean'
        t0_val = T0.get(well, np.nan)
        if col in df.columns and not np.isnan(t0_val):
            df[f'delta_T_above_T0_{well}'] = df[col] - t0_val

    # ---- C3: Cumulative heat input (flow × injection temperature × Δt) ----
    # More physically correct than flow-only: accounts for both volume and
    # temperature of injected fluid. Rated Critical importance in the design.
    if 'Net Flow' in df.columns and 'TC_INT_mean' in df.columns:
        positive_flow = df['Net Flow'].fillna(0).clip(lower=0)
        inj_temp = df['TC_INT_mean'].ffill()
        df['cumulative_heat_input'] = (positive_flow * inj_temp).cumsum() * dt_seconds

    # ---- C4: Vertical thermal gradient within the packer interval ----
    for well in WELLS:
        u, l = f'{well}-TEC-INT-U', f'{well}-TEC-INT-L'
        if u in df.columns and l in df.columns:
            df[f'T_gradient_INT_{well}'] = df[u] - df[l]

    # ---- Legacy L1: days since injection (derived from D1) ----
    df['days_since_injection'] = df['elapsed_injection_min'] / 1440.0

    # ---- Legacy L2: Cyclic hour-of-day encoding ----
    frac_hour = df.index.hour + df.index.minute / 60.0 + df.index.second / 3600.0
    df['hour_sin'] = np.sin(2 * np.pi * frac_hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * frac_hour / 24.0)

    # ---- Legacy L3: Injection-to-production temperature contrast ----
    inj_col = f'{INJECTION_WELL}-TEC-INT-U'
    prod_upper = [f'{w}-TEC-INT-U' for w in PRODUCTION_WELLS if f'{w}-TEC-INT-U' in df.columns]
    if inj_col in df.columns and prod_upper:
        df['delta_T_inj_prod'] = df[inj_col] - df[prod_upper].mean(axis=1)

    # ---- Legacy L4: Cumulative injected volume (flow-only, superseded by D3) ----
    if 'Net Flow' in df.columns:
        df['cumulative_injected_volume'] = (
            df['Net Flow'].fillna(0).clip(lower=0).cumsum() * dt_seconds
        )

    return df


# ---------------------------------------------------------------------------
# Section 4 – Main pipeline
# ---------------------------------------------------------------------------
def process_hourly():
    print("=" * 60)
    print("Processing hourly-average data")
    print("=" * 60)

    # Step 1: Load full dataset (pre-injection through Phase 1 end)
    df_full = load_hourly(HOURLY_FILE)

    # Step 2: Compute T0 BEFORE Phase-1 filtering removes pre-injection rows
    T0 = compute_T0(df_full)

    # Step 3: Clean (dedup, Phase-1 filter, outlier clamp, interpolate, drop bad cols)
    df = clean(df_full, label='Hourly')

    # Step 4: Compute interval mean temperatures (needed for drift check + features)
    df = compute_interval_means(df)

    # Step 5: Check and detrend monitor well sensor drift
    df = check_monitor_drift(df)

    # Step 6: Identify ramp-up window and assign train/test split labels
    ramp_up_end = detect_ramp_up_end(df, freq='1h')
    df = add_split_labels(df, ramp_up_end)

    # Step 7: Engineer features
    df = engineer_features(df, T0=T0, freq='1h')

    out_path = os.path.join(OUT_DIR, 'FTES_cleaned_1hour.csv')
    df.to_csv(out_path)
    print(f"\nSaved → {out_path}")
    print(f"Final shape: {df.shape}")
    print(f"Columns ({len(df.columns)}):\n  " + "\n  ".join(df.columns.tolist()))
    return df


def process_1sec():
    """
    Reads the 6.5 GB 1-second CSV in chunks, filters Phase 1, concatenates,
    cleans, resamples to 1-minute averages, engineers features, and saves.
    """
    print("\n" + "=" * 60)
    print("Processing 1-second data (chunked, then resampled to 1-minute)")
    print("=" * 60)

    chunks = []
    total_raw = 0
    for i, chunk in enumerate(
        pd.read_csv(
            SEC_FILE,
            parse_dates=['Time'],
            chunksize=SEC_CHUNKSIZE,
            low_memory=False,
        )
    ):
        chunk = _parse_1sec_chunk(chunk)

        # Phase 1 filter per chunk for early-exit efficiency
        chunk = chunk.loc[
            (chunk.index >= INJECTION_START) & (chunk.index <= PHASE1_END)
        ]
        if not chunk.empty:
            chunks.append(chunk)
            total_raw += len(chunk)

        if (i + 1) % 10 == 0:
            print(f"  Processed chunk {i + 1} — Phase-1 rows so far: {total_raw:,}")

    if not chunks:
        print("  No Phase-1 records found in 1-second file.")
        return None

    print(f"\n  Concatenating {len(chunks)} chunks ({total_raw:,} rows)…")
    df_s = pd.concat(chunks).sort_index()

    # Clean at 1-second resolution before resampling
    df_s = clean(df_s, label='1-second')

    # Resample to 1-minute means (reduces ~6 M rows → ~100 K rows)
    print("  Resampling to 1-minute averages…")
    numeric = df_s.select_dtypes(include=[np.number])
    df_1min = numeric.resample('1min').mean()

    # T0 from hourly file (same pre-injection window)
    df_hourly_full = load_hourly(HOURLY_FILE)
    T0 = compute_T0(df_hourly_full)

    df_1min = compute_interval_means(df_1min)
    df_1min = check_monitor_drift(df_1min)
    ramp_up_end = detect_ramp_up_end(df_1min, freq='1min')
    df_1min = add_split_labels(df_1min, ramp_up_end)
    df_1min = engineer_features(df_1min, T0=T0, freq='1min')

    out_path = os.path.join(OUT_DIR, 'FTES_cleaned_1sec_1min_resample.csv')
    df_1min.to_csv(out_path)
    print(f"\nSaved → {out_path}")
    print(f"Final shape: {df_1min.shape}")
    return df_1min


def main():
    process_hourly()

    if PROCESS_1SEC:
        process_1sec()
    else:
        print(
            "\n[INFO] 1-second processing skipped (PROCESS_1SEC = False).\n"
            "       Set PROCESS_1SEC = True at the top of this script to enable it."
        )


if __name__ == '__main__':
    main()

"""
FTES Dataset Cleaning for AI/ML
================================
Produces 4 cleaned variants:

1. FTES_cleaned_basic.csv
   - Sorted by time, ML-friendly column names, zero-variance column dropped,
     values rounded to 6 d.p., row index added.

2. FTES_cleaned_outliers_clipped.csv
   - Everything in (1) plus percentile-based Winsorization (1st / 99th pct)
     to remove probable sensor-fault sentinel values (e.g. the ~-1100 floor
     seen across pressure channels) without destroying legitimate operational
     peaks in bimodal columns.

3. FTES_cleaned_with_features.csv
   - Everything in (1) plus derived time features: hour_of_day, day_of_week,
     month, and cyclical sine/cosine encodings for hour and day_of_week, so
     temporal periodicity is captured without ordinal bias.

4. FTES_cleaned_normalized.csv
   - Everything in (2) (outliers clipped) then min-max scaled to [0, 1] on
     all numeric columns. Time is kept as-is; a separate
     FTES_normalization_params.csv contains the min/max used, allowing
     inverse-transform of predictions.
"""

import csv
import math
import statistics
import os

INPUT_FILE = "FTES-Full_Test_1hour_avg.csv"
OUTPUT_DIR = "."  # same folder

# ── helpers ──────────────────────────────────────────────────────────────────

def make_ml_name(col: str) -> str:
    """Convert column name to snake_case ML-friendly identifier."""
    return (col
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_per_")
            .replace("(", "")
            .replace(")", "")
            .lower())


def read_csv(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames[:]
    return fieldnames, rows


def write_csv(path: str, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {path}  ({len(rows)} rows × {len(fieldnames)} cols)")


def round_row(row: dict, numeric_cols, dp=6) -> dict:
    out = dict(row)
    for col in numeric_cols:
        try:
            out[col] = round(float(out[col]), dp)
        except (ValueError, TypeError):
            pass
    return out


# ── percentile clipping ───────────────────────────────────────────────────────

def compute_percentile_bounds(values: list, lo_pct=1.0, hi_pct=99.0):
    """Return (lower, upper) bounds at the given percentiles."""
    sorted_v = sorted(float(v) for v in values)
    n = len(sorted_v)

    def percentile(p):
        idx = (p / 100.0) * (n - 1)
        lo_i = int(idx)
        hi_i = min(lo_i + 1, n - 1)
        frac = idx - lo_i
        return sorted_v[lo_i] * (1 - frac) + sorted_v[hi_i] * frac

    return percentile(lo_pct), percentile(hi_pct)


def clip_value(val, lo, hi):
    v = float(val)
    return max(lo, min(hi, v))


# ── min-max normalisation ────────────────────────────────────────────────────

def minmax_scale(val, lo, hi):
    if hi == lo:
        return 0.0
    return (float(val) - lo) / (hi - lo)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD & BASELINE PREP
# ─────────────────────────────────────────────────────────────────────────────

print("Loading data …")
orig_fieldnames, orig_rows = read_csv(INPUT_FILE)

# Sort by time (rows are already ordered but make it explicit)
orig_rows.sort(key=lambda r: r["Time"])

numeric_cols_orig = [c for c in orig_fieldnames if c != "Time"]

# Identify zero-variance columns via min == max (avoids floating-point stdev noise)
zero_var_cols = []
for col in numeric_cols_orig:
    vals = [float(r[col]) for r in orig_rows]
    if (max(vals) - min(vals)) < 1e-9:
        zero_var_cols.append(col)

print(f"  Zero-variance columns to drop: {zero_var_cols}")

# Build ML-friendly name mapping
orig_to_ml = {c: make_ml_name(c) for c in orig_fieldnames}
# Ensure uniqueness (shouldn't be an issue but guard anyway)
seen = {}
for orig, ml in list(orig_to_ml.items()):
    if ml in seen:
        orig_to_ml[orig] = ml + "_2"
    else:
        seen[ml] = orig

# Columns kept after dropping zero-variance
kept_orig = [c for c in orig_fieldnames if c not in zero_var_cols]
kept_ml   = [orig_to_ml[c] for c in kept_orig]
numeric_ml = [orig_to_ml[c] for c in kept_orig if c != "Time"]

print(f"  Columns kept: {len(kept_orig)} / {len(orig_fieldnames)}")

# ─────────────────────────────────────────────────────────────────────────────
# FILE 1 — Basic Cleaning
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Basic cleaning …")

basic_rows = []
for i, row in enumerate(orig_rows):
    new = {"row_index": i}
    for c in kept_orig:
        new[orig_to_ml[c]] = row[c]
    new = round_row(new, numeric_ml, dp=6)
    basic_rows.append(new)

basic_fields = ["row_index"] + kept_ml
write_csv(os.path.join(OUTPUT_DIR, "FTES_cleaned_basic.csv"), basic_fields, basic_rows)

# ─────────────────────────────────────────────────────────────────────────────
# FILE 2 — Outlier Clipping (IQR 1.5×)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Outlier clipping (1st / 99th percentile) …")

# Compute bounds per numeric column from the basic (renamed) rows
pct_bounds = {}
clipped_counts = {}
for col in numeric_ml:
    vals = [float(r[col]) for r in basic_rows]
    lo, hi = compute_percentile_bounds(vals, lo_pct=1.0, hi_pct=99.0)
    pct_bounds[col] = (lo, hi)
    clipped_counts[col] = sum(1 for v in vals if v < lo or v > hi)

total_clipped = sum(clipped_counts.values())
cols_with_clips = [(c, n) for c, n in clipped_counts.items() if n > 0]
print(f"  Total cell values clipped: {total_clipped}")
print(f"  Columns with clipped values (1st–99th pct bounds):")
for col, n in sorted(cols_with_clips, key=lambda x: -x[1]):
    lo, hi = pct_bounds[col]
    print(f"    {col}: {n} values  (bounds [{lo:.3f}, {hi:.3f}])")

clipped_rows = []
for row in basic_rows:
    new = dict(row)
    for col in numeric_ml:
        lo, hi = pct_bounds[col]
        new[col] = round(clip_value(row[col], lo, hi), 6)
    clipped_rows.append(new)

write_csv(os.path.join(OUTPUT_DIR, "FTES_cleaned_outliers_clipped.csv"),
          basic_fields, clipped_rows)

# ─────────────────────────────────────────────────────────────────────────────
# FILE 3 — With Temporal Features
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Adding temporal features …")

def parse_time(s):
    # "2024-12-10 21:00:00"
    date_part, time_part = s.split(" ")
    y, mo, d = map(int, date_part.split("-"))
    h, mi, sec = map(int, time_part.split(":"))
    # weekday: Monday=0 … Sunday=6
    import datetime
    dt = datetime.datetime(y, mo, d, h, mi, sec)
    return dt

feat_rows = []
for row in basic_rows:
    new = dict(row)
    dt = parse_time(row["time"])
    new["hour_of_day"]     = dt.hour
    new["day_of_week"]     = dt.weekday()          # 0=Mon, 6=Sun
    new["month"]           = dt.month
    new["is_weekend"]      = int(dt.weekday() >= 5)
    # Cyclical encoding avoids ordinal cliff (23→0 and 6→0 wrap)
    new["hour_sin"]        = round(math.sin(2 * math.pi * dt.hour / 24), 6)
    new["hour_cos"]        = round(math.cos(2 * math.pi * dt.hour / 24), 6)
    new["day_of_week_sin"] = round(math.sin(2 * math.pi * dt.weekday() / 7), 6)
    new["day_of_week_cos"] = round(math.cos(2 * math.pi * dt.weekday() / 7), 6)
    feat_rows.append(new)

feat_extra = ["hour_of_day", "day_of_week", "month", "is_weekend",
              "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos"]
feat_fields = basic_fields + feat_extra
write_csv(os.path.join(OUTPUT_DIR, "FTES_cleaned_with_features.csv"),
          feat_fields, feat_rows)

# ─────────────────────────────────────────────────────────────────────────────
# FILE 4 — Normalized (built on top of clipped rows)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Min-max normalisation …")

minmax = {}
for col in numeric_ml:
    vals = [float(r[col]) for r in clipped_rows]
    minmax[col] = (min(vals), max(vals))

norm_rows = []
for row in clipped_rows:
    new = dict(row)
    for col in numeric_ml:
        lo, hi = minmax[col]
        new[col] = round(minmax_scale(row[col], lo, hi), 6)
    norm_rows.append(new)

write_csv(os.path.join(OUTPUT_DIR, "FTES_cleaned_normalized.csv"),
          basic_fields, norm_rows)

# Save normalisation params so predictions can be inverse-transformed
param_fields = ["column", "min", "max"]
param_rows = [{"column": col, "min": lo, "max": hi}
              for col, (lo, hi) in minmax.items()]
write_csv(os.path.join(OUTPUT_DIR, "FTES_normalization_params.csv"),
          param_fields, param_rows)

print("\nDone.")

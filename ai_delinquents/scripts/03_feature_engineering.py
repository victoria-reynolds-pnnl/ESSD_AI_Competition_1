"""
03_feature_engineering.py — Build feature matrix for water supply forecasting.

Reads cleaned data from data/clean/, produces a single feature matrix
with one row per water year for modeling.

Engineered features:
  1. apr1_swe_anomaly_pct  — April 1 SWE as % departure from median
  2. djf_nino34            — Dec-Feb mean Nino 3.4 index
  3. djf_pdo               — Dec-Feb mean PDO index
  4. djf_pna               — Dec-Feb mean PNA index
  5. jan_mar_mean_q_cfs    — Jan-Mar mean naturalized flow at The Dalles
  6. oct_mar_volume_kcfs_days — Oct-Mar cumulative volume (antecedent flow)

AI-generated: This script was generated with assistance from Claude (Anthropic).
"""

from pathlib import Path

import pandas as pd

# %%
# Settings

CLEAN_DIR = Path("data/clean")


# %%
# Load cleaned data

def load_data():
    target = pd.read_csv(CLEAN_DIR / "target_apr_sep_volume.csv")
    monthly_flow = pd.read_csv(CLEAN_DIR / "natural_flow_monthly.csv")
    snotel = pd.read_csv(CLEAN_DIR / "snotel_apr1_swe.csv")
    climate = pd.read_csv(CLEAN_DIR / "climate_indices_monthly.csv")
    return target, monthly_flow, snotel, climate


# %%
# Feature: April 1 SWE anomaly (% departure from median)

def compute_swe_anomaly(snotel):
    """Apr 1 SWE as percent of median: (SWE - median) / median * 100."""
    median_swe = snotel["apr1_swe_inches"].median()
    snotel = snotel.copy()
    snotel["apr1_swe_anomaly_pct"] = (
        (snotel["apr1_swe_inches"] - median_swe) / median_swe * 100
    )
    print(f"  SWE median: {median_swe:.1f} inches")
    return snotel[["water_year", "apr1_swe_inches", "apr1_swe_anomaly_pct"]]


# %%
# Feature: DJF climate indices (Dec-Feb mean for each water year)

def compute_djf_indices(climate):
    """Compute Dec-Feb mean for each climate index.

    DJF for water year Y uses Dec of Y-1 and Jan-Feb of Y.
    """
    climate = climate.copy()

    # Assign water year: Oct-Dec -> next water year, Jan-Sep -> same year
    climate["water_year"] = climate["year"]
    climate.loc[climate["month"] >= 10, "water_year"] = climate["year"] + 1

    # Filter to DJF months only
    djf = climate[climate["month"].isin([12, 1, 2])]

    indices = ["pdo", "nino34", "pna"]
    result = djf.groupby("water_year")[indices].mean()
    result.columns = [f"djf_{idx}" for idx in indices]
    result = result.reset_index()
    print(f"  DJF indices computed for {len(result)} water years")
    return result


# %%
# Feature: Jan-Mar mean naturalized flow

def compute_jan_mar_flow(monthly_flow):
    """Compute Jan-Mar mean flow from the BPA monthly natural flow data."""
    jan_mar = monthly_flow[monthly_flow["cal_month"].isin([1, 2, 3])]
    result = (
        jan_mar.groupby("water_year")["mean_flow_cfs"]
        .mean()
        .reset_index()
        .rename(columns={"mean_flow_cfs": "jan_mar_mean_q_cfs"})
    )
    print(f"  Jan-Mar flow computed for {len(result)} water years")
    return result


# %%
# Feature: Oct-Mar antecedent volume

def compute_oct_mar_volume(monthly_flow):
    """Compute Oct-Mar cumulative volume (kcfs-days) as antecedent conditions."""
    oct_mar = monthly_flow[monthly_flow["cal_month"].isin([10, 11, 12, 1, 2, 3])]
    result = (
        oct_mar.groupby("water_year")["volume_kcfs_days"]
        .sum()
        .reset_index()
        .rename(columns={"volume_kcfs_days": "oct_mar_volume_kcfs_days"})
    )
    print(f"  Oct-Mar volume computed for {len(result)} water years")
    return result


# %%
# Merge into feature matrix

def build_feature_matrix():
    """Merge all features and target into a single matrix."""
    print("Building feature matrix...")
    target, monthly_flow, snotel, climate = load_data()

    swe_features = compute_swe_anomaly(snotel)
    djf_features = compute_djf_indices(climate)
    jan_mar_flow = compute_jan_mar_flow(monthly_flow)
    oct_mar_vol = compute_oct_mar_volume(monthly_flow)

    # Start with target variable
    fm = target.copy()

    # Merge features one by one
    fm = fm.merge(swe_features, on="water_year", how="left")
    fm = fm.merge(djf_features, on="water_year", how="left")
    fm = fm.merge(jan_mar_flow, on="water_year", how="left")
    fm = fm.merge(oct_mar_vol, on="water_year", how="left")

    # Rename target for clarity
    fm = fm.rename(columns={"apr_sep_volume_kcfs_days": "target_volume"})

    # Report completeness
    feature_cols = [
        "apr1_swe_anomaly_pct", "djf_nino34", "djf_pdo", "djf_pna",
        "jan_mar_mean_q_cfs", "oct_mar_volume_kcfs_days",
    ]
    print(f"\n  Feature matrix: {len(fm)} rows x {len(fm.columns)} columns")
    print(f"  Water years: {fm['water_year'].min()}-{fm['water_year'].max()}")
    print("\n  Completeness:")
    for col in feature_cols:
        valid = fm[col].notna().sum()
        print(f"    {col}: {valid}/{len(fm)}")

    # Drop rows where any feature or target is missing
    complete = fm.dropna(subset=["target_volume"] + feature_cols)
    print(f"\n  Complete cases: {len(complete)} water years "
          f"({complete['water_year'].min()}-{complete['water_year'].max()})")

    out_path = CLEAN_DIR / "feature_matrix.csv"
    complete.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    # Quick correlation check
    print("\n  Correlations with target_volume:")
    for col in feature_cols:
        r = complete["target_volume"].corr(complete[col])
        print(f"    {col}: r={r:.3f}")

    return complete


# %%
# Main

if __name__ == "__main__":
    fm = build_feature_matrix()
    print("\nFeature engineering complete.")

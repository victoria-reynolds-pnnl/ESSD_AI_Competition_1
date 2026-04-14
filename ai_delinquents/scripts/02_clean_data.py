"""
02_clean_data.py — Clean and QA/QC raw data for water supply forecasting.

Reads from data/raw/, writes to data/clean/.

Operations:
  1. BPA natural flow: parse monthly, compute Apr-Sep volume per water year
  2. USGS observed flow: resample daily to monthly (for QC comparison only)
  3. SNOTEL SWE: filter stations, extract April 1 values, compute basin average
  4. Climate indices: align to common date index

AI-generated: This script was generated with assistance from Claude (Anthropic).
"""

import calendar
from pathlib import Path

import pandas as pd

# %%
# Settings

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Water years with complete BPA natural flow data
MIN_WATER_YEAR = 1929
MAX_WATER_YEAR = 2019


# %%
# 1. BPA Natural Flow — compute target variable

def clean_natural_flow():
    """Parse BPA modified (natural) flow and compute Apr-Sep volume.

    BPA data is monthly mean flow in cfs by water year (Oct-Sep).
    Target = April-September total volume in kcfs-days.
    Also saves the full monthly series for use as a feature.
    """
    print("Cleaning BPA natural flow data...")
    df = pd.read_csv(RAW_DIR / "bpa_the_dalles_natural_monthly.csv")

    months = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar",
              "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
    apr_sep = ["Apr", "May", "Jun", "Jul", "Aug", "Sep"]

    # Convert monthly mean cfs to monthly volume in kcfs-days
    # Volume = mean_cfs * days_in_month / 1000
    # Note: BPA water year rows have Oct of year Y-1 through Sep of year Y
    records = []
    monthly_records = []
    for _, row in df.iterrows():
        wy = int(row["water_year"])

        # Map each month to its calendar month/year for days-in-month
        month_info = {
            "Oct": (10, wy - 1), "Nov": (11, wy - 1), "Dec": (12, wy - 1),
            "Jan": (1, wy), "Feb": (2, wy), "Mar": (3, wy),
            "Apr": (4, wy), "May": (5, wy), "Jun": (6, wy),
            "Jul": (7, wy), "Aug": (8, wy), "Sep": (9, wy),
        }

        apr_sep_volume = 0.0
        all_valid = True
        for m in months:
            cal_month, cal_year = month_info[m]
            days = calendar.monthrange(cal_year, cal_month)[1]
            flow_cfs = row[m]
            if pd.isna(flow_cfs):
                if m in apr_sep:
                    all_valid = False
                continue
            vol_kcfs_days = float(flow_cfs) * days / 1000.0
            monthly_records.append({
                "water_year": wy,
                "month": m,
                "cal_year": cal_year,
                "cal_month": cal_month,
                "mean_flow_cfs": float(flow_cfs),
                "volume_kcfs_days": vol_kcfs_days,
            })
            if m in apr_sep:
                apr_sep_volume += vol_kcfs_days

        if all_valid:
            records.append({
                "water_year": wy,
                "apr_sep_volume_kcfs_days": round(apr_sep_volume, 1),
            })

    # Save target variable
    target_df = pd.DataFrame(records)
    target_path = CLEAN_DIR / "target_apr_sep_volume.csv"
    target_df.to_csv(target_path, index=False)
    print(f"  Target variable: {len(target_df)} water years, "
          f"range {target_df['water_year'].min()}-{target_df['water_year'].max()}")
    print(f"  Apr-Sep volume stats (kcfs-days): "
          f"mean={target_df['apr_sep_volume_kcfs_days'].mean():.0f}, "
          f"min={target_df['apr_sep_volume_kcfs_days'].min():.0f}, "
          f"max={target_df['apr_sep_volume_kcfs_days'].max():.0f}")

    # Save full monthly series (needed for Jan-Mar flow feature)
    monthly_df = pd.DataFrame(monthly_records)
    monthly_path = CLEAN_DIR / "natural_flow_monthly.csv"
    monthly_df.to_csv(monthly_path, index=False)
    print(f"  Monthly series: {len(monthly_df)} rows saved to {monthly_path}")

    return target_df


# %%
# 2. USGS Observed Flow (QC only)

def clean_usgs_flow():
    """Resample USGS daily flow to monthly mean for QC comparison."""
    print("Cleaning USGS observed flow...")
    df = pd.read_csv(RAW_DIR / "usgs_the_dalles_daily_q.csv", parse_dates=["date"])
    df = df.set_index("date")
    df["discharge_cfs"] = pd.to_numeric(df["discharge_cfs"], errors="coerce")

    monthly = df["discharge_cfs"].resample("MS").mean()
    monthly = monthly.dropna()

    out = monthly.reset_index()
    out.columns = ["date", "mean_discharge_cfs"]
    out_path = CLEAN_DIR / "usgs_monthly_q.csv"
    out.to_csv(out_path, index=False)
    print(f"  {len(out)} months, {out['date'].min()} to {out['date'].max()}")
    return out


# %%
# 3. SNOTEL SWE

def clean_snotel_swe():
    """Extract April 1 SWE per station per year, compute basin average.

    Filters stations with >15% missing April values.
    """
    print("Cleaning SNOTEL SWE data...")
    df = pd.read_csv(RAW_DIR / "snotel_swe.csv")

    # Parse date — format is "Mon YYYY" (e.g., "Apr 1985")
    df["date"] = pd.to_datetime(df["date"], format="%b %Y")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Extract April values (start-of-month = April 1 SWE)
    apr = df[df["month"] == 4].copy()
    apr["swe_inches"] = pd.to_numeric(apr["swe_inches"], errors="coerce")

    # Check station completeness
    n_years = apr["year"].nunique()
    station_coverage = apr.groupby("station_id")["swe_inches"].apply(
        lambda x: x.notna().sum() / n_years
    )
    good_stations = station_coverage[station_coverage >= 0.85].index.tolist()
    dropped = set(apr["station_id"].unique()) - set(good_stations)
    if dropped:
        print(f"  Dropped {len(dropped)} stations with >15% missing: {dropped}")
    print(f"  Retained {len(good_stations)} stations")

    # Filter to good stations
    apr = apr[apr["station_id"].isin(good_stations)]

    # Per-station April 1 SWE
    station_apr = apr.pivot_table(
        index="year", columns="station_id", values="swe_inches"
    )

    # Basin-average April 1 SWE (simple mean across stations)
    basin_avg = station_apr.mean(axis=1).rename("apr1_swe_inches")
    basin_avg = basin_avg.reset_index().rename(columns={"year": "water_year"})
    # April of year Y falls in water year Y
    basin_avg = basin_avg.dropna()

    out_path = CLEAN_DIR / "snotel_apr1_swe.csv"
    basin_avg.to_csv(out_path, index=False)
    print(f"  {len(basin_avg)} years of basin-avg April 1 SWE")
    print(f"  SWE stats (inches): mean={basin_avg['apr1_swe_inches'].mean():.1f}, "
          f"min={basin_avg['apr1_swe_inches'].min():.1f}, "
          f"max={basin_avg['apr1_swe_inches'].max():.1f}")

    # Also save per-station data for reference
    station_path = CLEAN_DIR / "snotel_apr1_by_station.csv"
    station_apr.to_csv(station_path)
    print(f"  Per-station data saved to {station_path}")

    return basin_avg


# %%
# 4. Climate Indices

def clean_climate_indices():
    """Align climate indices to common monthly date index."""
    print("Cleaning climate indices...")
    df = pd.read_csv(RAW_DIR / "climate_indices.csv")

    # Check completeness
    indices = ["pdo", "nino34", "pna", "amo"]
    for idx in indices:
        valid = df[idx].notna().sum()
        total = len(df)
        print(f"  {idx}: {valid}/{total} valid months "
              f"({df.loc[df[idx].notna(), 'year'].min()}-"
              f"{df.loc[df[idx].notna(), 'year'].max()})")

    out_path = CLEAN_DIR / "climate_indices_monthly.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows to {out_path}")
    return df


# %%
# 5. Cleaning Summary

def write_cleaning_log(target, usgs, snotel, climate):
    """Write a summary of the cleaning process."""
    log_path = CLEAN_DIR / "cleaning_log.txt"
    lines = [
        "=== Data Cleaning Summary ===",
        "",
        "BPA Natural Flow (The Dalles):",
        f"  Water years: {target['water_year'].min()}-{target['water_year'].max()} "
        f"({len(target)} years)",
        "  Target: Apr-Sep volume (kcfs-days)",
        f"  Mean: {target['apr_sep_volume_kcfs_days'].mean():.0f}, "
        f"Std: {target['apr_sep_volume_kcfs_days'].std():.0f}",
        "",
        "USGS Observed Flow (QC reference):",
        f"  Months: {len(usgs)}",
        f"  Date range: {usgs['date'].min()} to {usgs['date'].max()}",
        "",
        "SNOTEL April 1 SWE (basin average):",
        f"  Years: {len(snotel)}",
        f"  Mean: {snotel['apr1_swe_inches'].mean():.1f} inches",
        "",
        "Climate Indices:",
        f"  Months: {len(climate)}",
        "  Indices: PDO, Nino 3.4, PNA, AMO",
    ]
    log_path.write_text("\n".join(lines))
    print(f"\nCleaning log saved to {log_path}")


# %%
# Main

if __name__ == "__main__":
    target = clean_natural_flow()
    usgs = clean_usgs_flow()
    snotel = clean_snotel_swe()
    climate = clean_climate_indices()
    write_cleaning_log(target, usgs, snotel, climate)
    print("\nAll cleaning complete. Check data/clean/ for output files.")

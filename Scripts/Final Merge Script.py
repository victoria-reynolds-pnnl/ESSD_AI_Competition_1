# merge_cleaned_yakima_minimal.py
# Purpose: Merge cleaned USGS CSVs (time, value) by timestamp and optionally resample hourly.

import pandas as pd

# --- Update these paths to your cleaned files ---
CLEANED_FILES = {
    "Mabton": "cleaned_1 Yakima River at Mabton, WA - USGS-12508990 010125-123125.csv",
    "Kiona":  "cleaned_2 Yakima River at Kiona, WA - USGS-12510500 010125-123125.csv",
    "UnionGap": "cleaned_3 Yakima River Above Ahtanum Creek at Union Gap, WA - USGS-12500450 010125-123125.csv"
}

OUTPUT_MERGED = "merged_yakima.csv"
OUTPUT_RESAMPLED_HOURLY = "merged_resampled_yakima.csv"
RESAMPLE = True  # Set to False if you do not want hourly resampling

def load_series(path: str) -> pd.Series:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "value"]).drop_duplicates(subset=["time"])
    df = df.sort_values("time").set_index("time")
    return df["value"]

def main():
    # Load cleaned series and rename to site labels
    series = {site: load_series(path) for site, path in CLEANED_FILES.items()}

    # Outer-join on time to preserve all timestamps across sites
    merged = pd.concat(series, axis=1).sort_index()

    # (Optional) resample to hourly mean and drop rows with missing values
    if RESAMPLE:
        merged = merged.resample("1H").mean()
        merged = merged.dropna(how="any")  # keep hours where all sites have data
        merged.to_csv(OUTPUT_RESAMPLED_HOURLY, index=True)
        print(f"Saved hourly merged dataset: {OUTPUT_RESAMPLED_HOURLY} (rows={len(merged)})")
    else:
        merged.to_csv(OUTPUT_MERGED, index=True)
        print(f"Saved merged dataset (native cadence): {OUTPUT_MERGED} (rows={len(merged)})")

if __name__ == "__main__":
    main()
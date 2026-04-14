"""
01_download_data.py — Download all raw data for water supply forecasting model.

Data sources:
  - BPA natural/modified streamflow (monthly, all control points)
  - USGS observed streamflow at The Dalles (daily) via dataretrieval
  - SNOTEL SWE from key Columbia Basin stations (monthly) via NRCS Report Generator
  - NOAA PSL climate indices: PDO, Nino 3.4, PNA, AMO (monthly)

Outputs saved to data/raw/. Run from the project root directory.

AI-generated: This script was generated with assistance from Claude (Anthropic).
"""

import io
import zipfile
from pathlib import Path

import dataretrieval.nwis as nwis
import numpy as np
import pandas as pd
import requests

# %%
# Settings

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# USGS site for The Dalles, OR
USGS_SITE = "14105700"

# BPA monthly natural flow data (zip of xlsx files)
BPA_MONTHLY_URL = (
    "https://www.bpa.gov/-/media/Aep/power/historical-streamflow-reports/"
    "historic-streamflow-all-monthly-data.zip"
)

# SNOTEL stations: curated list of long-record stations spanning the Columbia
# Basin above The Dalles. Verified against NRCS station inventory.
# Format: "station_id:state:SNTL"
SNOTEL_STATIONS = {
    # Upper Columbia / Montana
    "530:MT:SNTL": "Hoodoo Basin, MT",
    "562:MT:SNTL": "Kraft Creek, MT",
    "662:MT:SNTL": "Nez Perce Camp, MT",
    "664:MT:SNTL": "Noisy Basin, MT",
    "787:MT:SNTL": "Stahl Peak, MT",
    # Snake River / Idaho
    "489:ID:SNTL": "Galena, ID",
    "550:ID:SNTL": "Jackson Peak, ID",
    "601:ID:SNTL": "Lost-Wood Divide, ID",
    "370:ID:SNTL": "Brundage Reservoir, ID",
    "830:ID:SNTL": "Trinity Mtn, ID",
    # Oregon Cascades (Willamette/Deschutes headwaters)
    "351:OR:SNTL": "Blazed Alder, OR",
    "651:OR:SNTL": "Mt Hood Test Site, OR",
    "395:OR:SNTL": "Chemult Alternate, OR",
    "302:OR:SNTL": "Aneroid Lake #2, OR",
    # Washington Cascades
    "788:WA:SNTL": "Stampede Pass, WA",
    "791:WA:SNTL": "Stevens Pass, WA",
}

# NOAA PSL climate indices (fixed-width text, year x 12 months)
CLIMATE_INDEX_URLS = {
    "pdo": "https://psl.noaa.gov/data/correlation/pdo.data",
    "nino34": "https://psl.noaa.gov/data/correlation/nina34.anom.data",
    "pna": "https://psl.noaa.gov/data/correlation/pna.data",
    "amo": "https://psl.noaa.gov/data/correlation/amon.us.data",
}


# %%
# 1. BPA Natural Flow Data

def _parse_bpa_xlsx(zf, xlsx_path):
    """Parse a BPA monthly streamflow xlsx file from inside a zip archive."""
    data = zf.read(xlsx_path)
    df = pd.read_excel(io.BytesIO(data), header=None, skiprows=1)
    df.columns = [
        "water_year", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar",
        "Apr", "May", "Jun", "Jul", "Aug", "Sep", "AVE",
    ]
    df = df[df["water_year"].apply(lambda x: str(x).isdigit())]
    df["water_year"] = df["water_year"].astype(int)
    return df


def download_bpa_natural_flow():
    """Download BPA monthly natural/modified streamflow data (zip of xlsx).

    Extracts TDA6M (The Dalles, modified/natural flow) and TDA6A (actual/regulated
    flow, for QC comparison). The 'M' suffix = modified (natural) flow.
    """
    print("Downloading BPA monthly streamflow data...")
    resp = requests.get(BPA_MONTHLY_URL, timeout=120)
    resp.raise_for_status()

    zip_path = RAW_DIR / "bpa_monthly_streamflow.zip"
    zip_path.write_bytes(resp.content)
    print(f"  Saved archive to {zip_path}")

    targets = {
        "monthly/monthly_M/TDA6M_monthly.xlsx": "bpa_the_dalles_natural_monthly.csv",
        "monthly/monthly_A/TDA6A_monthly.xlsx": "bpa_the_dalles_actual_monthly.csv",
    }
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for xlsx_path, csv_name in targets.items():
            df = _parse_bpa_xlsx(zf, xlsx_path)
            out_path = RAW_DIR / csv_name
            df.to_csv(out_path, index=False)
            print(f"  {xlsx_path} -> {out_path} ({len(df)} water years)")


# %%
# 2. USGS Observed Streamflow (via dataretrieval)

def download_usgs_streamflow():
    """Download daily streamflow at The Dalles from USGS NWIS.

    Uses the dataretrieval package (nwis.get_dv).
    """
    print(f"Downloading USGS daily streamflow for site {USGS_SITE}...")
    df, _ = nwis.get_dv(
        sites=USGS_SITE,
        start="1950-01-01",
        end="2025-12-31",
        parameterCd="00060",
    )

    # Rename discharge column for clarity
    q_col = [c for c in df.columns if "00060" in c and "cd" not in c.lower()][0]
    df = df.rename(columns={q_col: "discharge_cfs"})
    df = df[["discharge_cfs"]].copy()
    df.index.name = "date"

    out_path = RAW_DIR / "usgs_the_dalles_daily_q.csv"
    df.to_csv(out_path)
    print(f"  Saved {len(df)} rows to {out_path}")


# %%
# 3. SNOTEL SWE Data (via NRCS Report Generator)

def download_snotel_swe():
    """Download monthly start-of-period SWE for selected SNOTEL stations.

    Uses the NRCS Report Generator REST API which returns CSV.
    Docs: https://www.nrcs.usda.gov/sites/default/files/2023-03/Report%20Generator%20Help%20Guide.pdf
    """
    print("Downloading SNOTEL SWE data...")
    all_data = []

    for station_code, station_name in SNOTEL_STATIONS.items():
        station_id = station_code.split(":")[0]
        url = (
            "https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/"
            "customMultiTimeSeriesGroupByStationReport/monthly/start_of_period/"
            f"{station_code}%7Cid=%22%22%7Cname/1985-01-01,2025-04-01/WTEQ::value"
        )
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            lines = [
                line for line in resp.text.splitlines() if not line.startswith("#")
            ]
            if len(lines) < 2:
                print(f"  WARNING: No data for {station_name} ({station_id})")
                continue
            df = pd.read_csv(io.StringIO("\n".join(lines)))
            df.columns = ["date", "swe_inches"]
            df["station_id"] = station_id
            df["station_name"] = station_name
            all_data.append(df)
            print(f"  {station_name}: {len(df)} months")
        except Exception as e:
            print(f"  ERROR downloading {station_name}: {e}")

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        out_path = RAW_DIR / "snotel_swe.csv"
        result.to_csv(out_path, index=False)
        print(f"  Saved {len(result)} total rows to {out_path}")
    else:
        print("  WARNING: No SNOTEL data downloaded!")


# %%
# 4. Climate Indices

def parse_noaa_psl_index(text, index_name):
    """Parse NOAA PSL fixed-width climate index file.

    Format: first line has start/end years, then year + 12 monthly values.
    Missing values are large negative numbers (-99.9, -99.99, etc.).
    """
    lines = text.strip().splitlines()
    header = lines[0].split()
    start_year = int(header[0])
    end_year = int(header[1])

    records = []
    months = list(range(1, 13))
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 13:
            continue
        year = int(parts[0])
        if year < start_year or year > end_year:
            continue
        for month, val_str in zip(months, parts[1:13]):
            val = float(val_str)
            if val < -90:
                val = np.nan
            records.append({"year": year, "month": month, index_name: val})
    return pd.DataFrame(records)


def download_climate_indices():
    """Download and merge NOAA PSL climate indices."""
    print("Downloading climate indices...")
    merged = None

    for name, url in CLIMATE_INDEX_URLS.items():
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = parse_noaa_psl_index(resp.text, name)
        print(f"  {name}: {df[name].notna().sum()} valid months")

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["year", "month"], how="outer")

    merged = merged.sort_values(["year", "month"]).reset_index(drop=True)
    out_path = RAW_DIR / "climate_indices.csv"
    merged.to_csv(out_path, index=False)
    print(f"  Saved {len(merged)} rows to {out_path}")


# %%
# Main

if __name__ == "__main__":
    download_bpa_natural_flow()
    download_usgs_streamflow()
    download_snotel_swe()
    download_climate_indices()
    print("\nAll downloads complete. Check data/raw/ for output files.")

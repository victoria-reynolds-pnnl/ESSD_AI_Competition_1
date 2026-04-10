#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path

import pandas as pd


ZIP_FILES = [
    "cold_snap_library_NERC_average.zip",
    "cold_snap_library_NERC_average_area.zip",
    "cold_snap_library_NERC_average_pop.zip",
    "heat_wave_library_NERC_average.zip",
    "heat_wave_library_NERC_average_area.zip",
    "heat_wave_library_NERC_average_pop.zip",
]

def infer_hazard_type(zip_name: str) -> str:
    return "cold_snap" if zip_name.startswith("cold_snap") else "heat_wave"

def infer_aggregation_method(zip_name: str) -> str:
    # Per documentation: average=SM, average_area=MWA, average_pop=MWP [1]
    if zip_name.endswith("_average.zip"):
        return "SM"
    if zip_name.endswith("_average_area.zip"):
        return "MWA"
    if zip_name.endswith("_average_pop.zip"):
        return "MWP"
    return "UNKNOWN"

def infer_definition_id(member_name: str) -> str:
    # filenames end with _def1.csv ... _def12.csv
    m = re.search(r"_def(\d{1,2})\.csv$", member_name, flags=re.IGNORECASE)
    if not m:
        return "UnknownDef"
    return f"Def{int(m.group(1))}"

def standardize(df: pd.DataFrame, hazard_type: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Common event structure [1]
    required = {"start_date", "end_date", "centroid_date", "duration", "NERC_ID", "spatial_coverage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing}. Found: {list(df.columns)}")

    # Parse dates [1]
    for c in ["start_date", "end_date", "centroid_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    # Hazard-specific intensity column [1]
    if hazard_type == "heat_wave":
        if "highest_temperature" not in df.columns:
            raise ValueError("Heat wave file missing 'highest_temperature' [1].")
        df["extreme_temperature_K"] = pd.to_numeric(df["highest_temperature"], errors="coerce")
    else:
        if "lowest_temperature" not in df.columns:
            raise ValueError("Cold snap file missing 'lowest_temperature' [1].")
        df["extreme_temperature_K"] = pd.to_numeric(df["lowest_temperature"], errors="coerce")

    df["duration_days"] = pd.to_numeric(df["duration"], errors="coerce")
    df["spatial_coverage_pct"] = pd.to_numeric(df["spatial_coverage"], errors="coerce")

    # Final tidy schema
    return df[
        [
            "start_date",
            "end_date",
            "centroid_date",
            "extreme_temperature_K",
            "duration_days",
            "NERC_ID",
            "spatial_coverage_pct",
        ]
    ].copy()

def load_all(root: Path) -> pd.DataFrame:
    frames = []

    for zip_name in ZIP_FILES:
        zip_path = root / zip_name
        if not zip_path.exists():
            raise FileNotFoundError(f"Expected zip not found: {zip_path}")

        hazard_type = infer_hazard_type(zip_name)
        aggregation_method = infer_aggregation_method(zip_name)

        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".csv")]

            # Optional: keep only *_def*.csv
            members = [m for m in members if re.search(r"_def\d{1,2}\.csv$", m, flags=re.IGNORECASE)]

            for member in members:
                definition_id = infer_definition_id(member)
                with zf.open(member) as f:
                    df = pd.read_csv(f)

                df = standardize(df, hazard_type)

                df.insert(0, "hazard_type", hazard_type)
                df.insert(1, "aggregation_method", aggregation_method)
                df.insert(2, "definition_id", definition_id)

                frames.append(df)

    if not frames:
        raise RuntimeError("No CSV event libraries were loaded. Check ZIP contents/paths.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        ["hazard_type", "aggregation_method", "definition_id", "NERC_ID", "start_date"],
        kind="mergesort",
    )
    return combined

def main():

    DATA_ROOT = Path("C:/WorkSpace/Tools/AI-ML/ESSD_AI_Competition/heat_wave_cold_snap_nerc")  # folder containing the 6 zip files
    df = load_all(DATA_ROOT)
    os.makedirs("week1/outputs", exist_ok=True)

    out_csv = Path("week1/outputs/combined_extreme_thermal_event_library.csv")

    df.to_csv(out_csv, index=False)

    print(f"Wrote {len(df):,} rows")
    print(f"  {out_csv.resolve()}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""Extract yearly merged CONUS fire occurrence source tables."""

# AI-assistance note: OpenAI Codex helped draft this script. Team review is still required before challenge submission.

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _fire_pipeline_common import (
    AI_ASSISTANCE_NOTE,
    DATASET_VARIABLES,
    DEFAULT_INPUT_ROOT,
    add_common_arguments,
    add_identifiers,
    build_base_manifest,
    dataset_path,
    ensure_directory,
    load_netcdf_frame,
    normalize_frame_columns,
    optimize_numeric_dtypes,
    parse_years,
    partition_data_path,
    partition_manifest_path,
    validate_unique_keys,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Directory that contains the four ComExDBM CONUS output folders. Defaults to {DEFAULT_INPUT_ROOT}.",
    )
    add_common_arguments(parser)
    return parser.parse_args()


def build_merged_year_frame(input_root: Path, year: int) -> pd.DataFrame:
    fire_frame = load_netcdf_frame(dataset_path(input_root, "fire", year), DATASET_VARIABLES["fire"])
    heatwave_frame = load_netcdf_frame(dataset_path(input_root, "heatwave", year), DATASET_VARIABLES["heatwave"])
    drought_frame = load_netcdf_frame(dataset_path(input_root, "drought", year), DATASET_VARIABLES["drought"])
    meteorology_frame = load_netcdf_frame(
        dataset_path(input_root, "meteorology", year),
        DATASET_VARIABLES["meteorology"],
    )

    merged = fire_frame.merge(heatwave_frame, on=["date", "lat", "lon"], how="left", validate="one_to_one")
    merged = merged.merge(drought_frame, on=["date", "lat", "lon"], how="left", validate="one_to_one")
    merged = merged.merge(meteorology_frame, on=["date", "lat", "lon"], how="left", validate="one_to_one")
    merged = normalize_frame_columns(merged)
    validate_unique_keys(merged, f"merged_{year}")

    merged = add_identifiers(merged)
    merged = merged.sort_values(["date", "lat", "lon"], kind="stable").reset_index(drop=True)
    return optimize_numeric_dtypes(merged)


def write_year_outputs(frame: pd.DataFrame, artifact_root: Path, input_root: Path, year: int) -> None:
    data_path = partition_data_path(artifact_root, "raw", year)
    manifest_path = partition_manifest_path(artifact_root, "raw", year)
    ensure_directory(data_path.parent)
    frame.to_parquet(data_path, index=False)

    manifest = build_base_manifest("raw", year, frame)
    manifest["source_files"] = {
        dataset_name: str(dataset_path(input_root, dataset_name, year))
        for dataset_name in DATASET_VARIABLES
    }
    manifest["fire_positive_rows"] = int((frame["FHS_c9"].fillna(0) > 0).sum())
    manifest["ai_assistance_note"] = AI_ASSISTANCE_NOTE
    write_json(manifest_path, manifest)


def main() -> None:
    args = parse_args()
    years = parse_years(args.years)

    for year in years:
        data_path = partition_data_path(args.artifact_root, "raw", year)
        manifest_path = partition_manifest_path(args.artifact_root, "raw", year)
        if data_path.exists() and manifest_path.exists() and not args.overwrite:
            print(f"[extract] skipping {year}: {data_path} already exists")
            continue

        frame = build_merged_year_frame(args.input_root, year)
        write_year_outputs(frame, args.artifact_root, args.input_root, year)
        print(f"[extract] wrote {len(frame):,} raw rows for {year} to {data_path}")


if __name__ == "__main__":
    main()

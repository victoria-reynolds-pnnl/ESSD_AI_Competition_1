#!/usr/bin/env python
"""Export Week 2 CSV, JSON, and documentation artifacts from pipeline parquet outputs."""

# AI-assistance note: OpenAI Codex helped draft this script. Team review is still required before challenge submission.

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import pandas as pd

from _fire_pipeline_common import (
    AI_ASSISTANCE_NOTE,
    DEFAULT_EXPORT_ROOT,
    add_common_arguments,
    aggregate_manifests,
    build_field_metadata_records,
    discover_available_years,
    ensure_directory,
    load_manifests,
    load_partition_frame,
    month_filtered,
    parse_years,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EXPORT_ROOT,
        help=f"Directory for Week 2 export files. Defaults to {DEFAULT_EXPORT_ROOT}.",
    )
    return parser.parse_args()


def resolve_years_or_available(args: argparse.Namespace) -> list[int]:
    if args.years:
        return parse_years(args.years)

    raw_years = set(discover_available_years(args.artifact_root, "raw"))
    cleaned_years = set(discover_available_years(args.artifact_root, "cleaned"))
    available_years = sorted(raw_years & cleaned_years)
    if not available_years:
        raise FileNotFoundError(
            "No overlapping raw and cleaned parquet partitions were found. "
            "Run the extraction and feature engineering scripts first."
        )
    return available_years


def ensure_writable_outputs(output_dir: Path, overwrite: bool) -> None:
    output_paths = [
        output_dir / "original_dataset.csv.gz",
        output_dir / "original_dataset.json.gz",
        output_dir / "cleaned_dataset.csv.gz",
        output_dir / "cleaned_dataset.json.gz",
        output_dir / "data_dictionary.json",
        output_dir / "qc_summary.json",
        output_dir / "week2_summary.md",
    ]
    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths and not overwrite:
        raise FileExistsError(
            "Week 2 export files already exist. Re-run with --overwrite to replace them: "
            + ", ".join(str(path) for path in existing_paths)
        )
    ensure_directory(output_dir)


def append_frame_to_exports(
    frame: pd.DataFrame,
    csv_handle,
    json_handle,
    include_header: bool,
) -> None:
    frame.to_csv(csv_handle, index=False, header=include_header)
    frame.to_json(json_handle, orient="records", lines=True, date_format="iso")


def build_week2_summary(years: list[int], artifact_root: Path, output_dir: Path) -> str:
    year_text = ", ".join(str(year) for year in years)
    return f"""# Week 2 Summary

## ML Algorithms
- `LogisticRegression`: interpretable baseline for rare-event binary classification.
- `HistGradientBoostingClassifier`: efficient non-linear tabular model that can capture threshold effects and interactions while still producing calibrated scores for later evaluation.

## Data Preparation Steps
- Merge yearly heatwave, fire, drought, and meteorology NetCDF files on `date`, `lat`, and `lon`.
- Normalize source aliases such as `time -> date` and `apcp -> air_apcp`, then add deterministic `cell_id` and `row_id`.
- Engineer the locked v1 feature set with past-only lag and rolling windows before filtering the cleaned modeling table to May-October.
- Preserve raw leakage-sensitive fields only in the merged source export; exclude them from the cleaned modeling export.

## AI Role
- OpenAI Codex assisted with drafting the pipeline scripts and this summary. Human review is still required before submission.

## Artifact Locations
- Artifact root: `{artifact_root}`
- Week 2 export directory: `{output_dir}`
- Years included in this export: {year_text}
"""


def main() -> None:
    args = parse_args()
    years = resolve_years_or_available(args)
    ensure_writable_outputs(args.output_dir, args.overwrite)

    raw_csv_path = args.output_dir / "original_dataset.csv.gz"
    raw_json_path = args.output_dir / "original_dataset.json.gz"
    cleaned_csv_path = args.output_dir / "cleaned_dataset.csv.gz"
    cleaned_json_path = args.output_dir / "cleaned_dataset.json.gz"

    raw_columns: list[str] | None = None
    cleaned_columns: list[str] | None = None
    raw_row_count = 0
    cleaned_row_count = 0

    with gzip.open(raw_csv_path, "wt", encoding="utf-8", newline="") as raw_csv_handle, gzip.open(
        raw_json_path,
        "wt",
        encoding="utf-8",
    ) as raw_json_handle:
        for index, year in enumerate(years):
            raw_frame = load_partition_frame(args.artifact_root, "raw", year)
            raw_frame = month_filtered(raw_frame)
            if raw_columns is None:
                raw_columns = list(raw_frame.columns)
            elif list(raw_frame.columns) != raw_columns:
                raise ValueError(f"Raw columns changed in year {year}; expected {raw_columns}, found {list(raw_frame.columns)}")
            append_frame_to_exports(raw_frame, raw_csv_handle, raw_json_handle, include_header=index == 0)
            raw_row_count += len(raw_frame)

    with gzip.open(cleaned_csv_path, "wt", encoding="utf-8", newline="") as cleaned_csv_handle, gzip.open(
        cleaned_json_path,
        "wt",
        encoding="utf-8",
    ) as cleaned_json_handle:
        for index, year in enumerate(years):
            cleaned_frame = load_partition_frame(args.artifact_root, "cleaned", year)
            if cleaned_columns is None:
                cleaned_columns = list(cleaned_frame.columns)
            elif list(cleaned_frame.columns) != cleaned_columns:
                raise ValueError(
                    f"Cleaned columns changed in year {year}; expected {cleaned_columns}, found {list(cleaned_frame.columns)}"
                )
            append_frame_to_exports(
                cleaned_frame,
                cleaned_csv_handle,
                cleaned_json_handle,
                include_header=index == 0,
            )
            cleaned_row_count += len(cleaned_frame)

    raw_columns = raw_columns or []
    cleaned_columns = cleaned_columns or []
    data_dictionary = build_field_metadata_records(raw_columns, cleaned_columns)
    write_json(args.output_dir / "data_dictionary.json", data_dictionary)

    raw_manifests = load_manifests(args.artifact_root, "raw", years)
    cleaned_manifests = load_manifests(args.artifact_root, "cleaned", years)
    qc_summary = {
        "artifact_root": str(args.artifact_root),
        "output_dir": str(args.output_dir),
        "years": years,
        "raw": aggregate_manifests(raw_manifests),
        "cleaned": aggregate_manifests(cleaned_manifests),
        "export_row_counts": {
            "original_dataset": raw_row_count,
            "cleaned_dataset": cleaned_row_count,
        },
        "ai_assistance_note": AI_ASSISTANCE_NOTE,
    }
    write_json(args.output_dir / "qc_summary.json", qc_summary)

    summary_path = args.output_dir / "week2_summary.md"
    summary_path.write_text(build_week2_summary(years, args.artifact_root, args.output_dir), encoding="utf-8")

    print(f"[export] wrote Week 2 artifacts to {args.output_dir}")
    print(f"[export] original_dataset rows: {raw_row_count:,}")
    print(f"[export] cleaned_dataset rows: {cleaned_row_count:,}")


if __name__ == "__main__":
    main()

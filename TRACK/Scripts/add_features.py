#!/usr/bin/env python3
"""Batch-add engineered features to all cleaned event CSV files.

NOTE: All sections of this script were created with an LLM.

This script reads every CSV under Data/cleaned and writes matching outputs under:
  - Data/cleaned_with_features/csv
  - Data/cleaned_with_features/json

Each output preserves the same relative folder structure as Data/cleaned.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

INTER_EVENT_RECOVERY_FEATURE = "inter_event_recovery_interval_days"
CUMULATIVE_HEAT_STRESS_FEATURE = "cumulative_heat_stress_index"
YEARLY_MAX_INTENSITY_FEATURE = "yearly_max_heat_wave_intensity"
YEARLY_MAX_DURATION_FEATURE = "yearly_max_heat_wave_duration"
INTENSITY_TREND_FEATURE = "yearly_max_heat_wave_intensity_trend"
DURATION_TREND_FEATURE = "yearly_max_heat_wave_duration_trend"

REGION_CANDIDATES = ("nerc_id", "NERC_ID")
START_DATE_CANDIDATES = ("start_date", "START_DATE")
END_DATE_CANDIDATES = ("end_date", "END_DATE")
DURATION_CANDIDATES = ("duration_days", "duration", "DURATION_DAYS", "DURATION")
TEMPERATURE_CANDIDATES = (
    "highest_temperature_k",
    "lowest_temperature_k",
    "highest_temperature",
    "lowest_temperature",
    "HIGHEST_TEMPERATURE_K",
    "LOWEST_TEMPERATURE_K",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add all engineered features to every cleaned CSV and write mirrored "
            "CSV/JSON outputs under Data/cleaned_with_features."
        )
    )
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--input-root",
        default=str(project_root / "Data" / "cleaned"),
        help="Root folder containing cleaned CSV files.",
    )
    parser.add_argument(
        "--output-root",
        default=str(project_root / "Data" / "cleaned_with_features"),
        help="Root folder where csv/ and json/ outputs are written.",
    )
    return parser.parse_args()


def pick_column(columns: pd.Index, candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Missing required {label} column. Expected one of: {list(candidates)}")


def resolve_schema(df: pd.DataFrame) -> dict[str, str]:
    return {
        "region": pick_column(df.columns, REGION_CANDIDATES, "region"),
        "start_date": pick_column(df.columns, START_DATE_CANDIDATES, "start date"),
        "end_date": pick_column(df.columns, END_DATE_CANDIDATES, "end date"),
        "duration": pick_column(df.columns, DURATION_CANDIDATES, "duration"),
        "temperature": pick_column(df.columns, TEMPERATURE_CANDIDATES, "temperature"),
    }


def add_inter_event_recovery_interval(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    work = out.copy()
    work["_original_order"] = range(len(work))
    work["_start_dt"] = pd.to_datetime(work[schema["start_date"]], errors="coerce")
    work["_end_dt"] = pd.to_datetime(work[schema["end_date"]], errors="coerce")

    if work["_start_dt"].isna().all() or work["_end_dt"].isna().all():
        raise ValueError("Start/end date columns must contain parseable dates.")

    work = work.sort_values(
        by=[schema["region"], "_start_dt", "_end_dt", "_original_order"],
        kind="mergesort",
    )
    next_start = work.groupby(schema["region"], dropna=False)["_start_dt"].shift(-1)
    work[INTER_EVENT_RECOVERY_FEATURE] = (next_start - work["_end_dt"]).dt.days
    work = work.sort_values("_original_order", kind="mergesort")
    out[INTER_EVENT_RECOVERY_FEATURE] = work[INTER_EVENT_RECOVERY_FEATURE].to_numpy()
    return out


def add_cumulative_heat_stress_index(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    work = out.copy()
    work["_original_order"] = range(len(work))
    work["_start_dt"] = pd.to_datetime(work[schema["start_date"]], errors="coerce")

    if work["_start_dt"].isna().all():
        raise ValueError("Start date column must contain parseable dates.")

    work["_period_year"] = work["_start_dt"].dt.year
    work["_event_heat_stress"] = (
        pd.to_numeric(work[schema["temperature"]], errors="coerce")
        * pd.to_numeric(work[schema["duration"]], errors="coerce")
    )
    work = work.sort_values(
        by=[schema["region"], "_period_year", "_start_dt", "_original_order"],
        kind="mergesort",
    )
    work[CUMULATIVE_HEAT_STRESS_FEATURE] = work.groupby(
        [schema["region"], "_period_year"], dropna=False
    )["_event_heat_stress"].cumsum()
    work = work.sort_values("_original_order", kind="mergesort")
    out[CUMULATIVE_HEAT_STRESS_FEATURE] = work[CUMULATIVE_HEAT_STRESS_FEATURE].to_numpy()
    return out


def add_max_intensity_duration_trends(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    work = out.copy()
    work["_original_order"] = range(len(work))
    work["_start_dt"] = pd.to_datetime(work[schema["start_date"]], errors="coerce")

    if work["_start_dt"].isna().all():
        raise ValueError("Start date column must contain parseable dates.")

    work["_period_year"] = work["_start_dt"].dt.year
    work["_temperature_num"] = pd.to_numeric(work[schema["temperature"]], errors="coerce")
    work["_duration_num"] = pd.to_numeric(work[schema["duration"]], errors="coerce")

    yearly = (
        work.groupby([schema["region"], "_period_year"], dropna=False)
        .agg(
            **{
                YEARLY_MAX_INTENSITY_FEATURE: ("_temperature_num", "max"),
                YEARLY_MAX_DURATION_FEATURE: ("_duration_num", "max"),
            }
        )
        .sort_index()
        .reset_index()
    )

    yearly[INTENSITY_TREND_FEATURE] = yearly.groupby(schema["region"], dropna=False)[
        YEARLY_MAX_INTENSITY_FEATURE
    ].diff()
    yearly[DURATION_TREND_FEATURE] = yearly.groupby(schema["region"], dropna=False)[
        YEARLY_MAX_DURATION_FEATURE
    ].diff()

    enriched = work.merge(yearly, on=[schema["region"], "_period_year"], how="left")
    enriched = enriched.sort_values("_original_order", kind="mergesort")

    out[YEARLY_MAX_INTENSITY_FEATURE] = enriched[YEARLY_MAX_INTENSITY_FEATURE].to_numpy()
    out[YEARLY_MAX_DURATION_FEATURE] = enriched[YEARLY_MAX_DURATION_FEATURE].to_numpy()
    out[INTENSITY_TREND_FEATURE] = enriched[INTENSITY_TREND_FEATURE].to_numpy()
    out[DURATION_TREND_FEATURE] = enriched[DURATION_TREND_FEATURE].to_numpy()
    return out


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    schema = resolve_schema(df)
    engineered = add_inter_event_recovery_interval(df, schema)
    engineered = add_cumulative_heat_stress_index(engineered, schema)
    engineered = add_max_intensity_duration_trends(engineered, schema)
    return engineered


def process_file(input_csv: Path, input_root: Path, output_root: Path) -> None:
    relative_path = input_csv.relative_to(input_root)
    csv_output_path = output_root / "csv" / relative_path
    json_output_path = output_root / "json" / relative_path.with_suffix(".json")

    df = pd.read_csv(input_csv)
    engineered = add_all_features(df)

    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)

    engineered.to_csv(csv_output_path, index=False)
    engineered.to_json(json_output_path, orient="records", date_format="iso", indent=2)

    print(f"Processed: {relative_path}")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    csv_files = sorted(input_root.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {input_root}")

    failures: list[tuple[Path, str]] = []
    for input_csv in csv_files:
        try:
            process_file(input_csv, input_root, output_root)
        except Exception as exc:  # pylint: disable=broad-except
            failures.append((input_csv, str(exc)))
            print(f"Failed: {input_csv} -> {exc}")

    print(f"Completed {len(csv_files) - len(failures)} of {len(csv_files)} files.")
    if failures:
        failed_files = "\n".join(f"  - {path}: {message}" for path, message in failures)
        raise RuntimeError(f"Feature generation failed for {len(failures)} file(s):\n{failed_files}")


if __name__ == "__main__":
    main()

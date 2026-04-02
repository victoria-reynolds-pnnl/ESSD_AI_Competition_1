#!/usr/bin/env python3
"""Add the inter-event recovery interval feature to a heat-wave CSV file.

Example:
    python Scripts/add_features.py \
        --input Data/heat_wave_library_NERC_average_def1.csv \
        --output Data/heat_wave_library_NERC_average_def1_features.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FEATURE_NAME = "inter_event_recovery_interval_days"
CUMULATIVE_HEAT_STRESS_FEATURE = "cumulative_heat_stress_index"
YEARLY_MAX_INTENSITY_FEATURE = "yearly_max_heat_wave_intensity"
YEARLY_MAX_DURATION_FEATURE = "yearly_max_heat_wave_duration"
INTENSITY_TREND_FEATURE = "yearly_max_heat_wave_intensity_trend"
DURATION_TREND_FEATURE = "yearly_max_heat_wave_duration_trend"
REGION_COLUMN = "NERC_ID"
START_DATE_COLUMN = "start_date"
END_DATE_COLUMN = "end_date"
TEMPERATURE_COLUMN = "highest_temperature"
DURATION_COLUMN = "duration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add requested engineered features to a heat-wave CSV dataset."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", required=True, help="Path to output CSV file.")
    parser.add_argument(
        "--add-cumulative-heat-stress-index",
        action="store_true",
        help=(
            "Also add cumulative heat stress per NERC subregion by calendar year, "
            "based on highest_temperature * duration."
        ),
    )
    parser.add_argument(
        "--add-max-intensity-duration-trends",
        action="store_true",
        help=(
            "Also add yearly max heat-wave intensity/duration per NERC subregion "
            "and year-over-year trend deltas."
        ),
    )
    return parser.parse_args()


def validate_required_columns(
    df: pd.DataFrame,
    include_cumulative_heat_stress: bool,
    include_max_trends: bool,
) -> None:
    required_columns = [REGION_COLUMN, START_DATE_COLUMN, END_DATE_COLUMN]
    if include_cumulative_heat_stress or include_max_trends:
        required_columns.extend([TEMPERATURE_COLUMN, DURATION_COLUMN])

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required column(s): {missing_columns}")


def add_inter_event_recovery_interval(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    work = out.copy()
    work["_original_order"] = range(len(work))
    work["_start_dt"] = pd.to_datetime(work[START_DATE_COLUMN], errors="coerce")
    work["_end_dt"] = pd.to_datetime(work[END_DATE_COLUMN], errors="coerce")

    if work["_start_dt"].isna().all() or work["_end_dt"].isna().all():
        raise ValueError(
            f"Columns '{START_DATE_COLUMN}' and '{END_DATE_COLUMN}' must contain parseable dates."
        )

    work = work.sort_values(
        by=[REGION_COLUMN, "_start_dt", "_end_dt", "_original_order"],
        kind="mergesort",
    )

    next_start = work.groupby(REGION_COLUMN, dropna=False)["_start_dt"].shift(-1)
    work[FEATURE_NAME] = (next_start - work["_end_dt"]).dt.days

    work = work.sort_values("_original_order", kind="mergesort")
    out[FEATURE_NAME] = work[FEATURE_NAME].to_numpy()
    return out


def add_cumulative_heat_stress_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    work = out.copy()
    work["_original_order"] = range(len(work))
    work["_start_dt"] = pd.to_datetime(work[START_DATE_COLUMN], errors="coerce")

    if work["_start_dt"].isna().all():
        raise ValueError(f"Column '{START_DATE_COLUMN}' must contain parseable dates.")

    work["_period_year"] = work["_start_dt"].dt.year
    work["_event_heat_stress"] = (
        pd.to_numeric(work[TEMPERATURE_COLUMN], errors="coerce")
        * pd.to_numeric(work[DURATION_COLUMN], errors="coerce")
    )

    work = work.sort_values(
        by=[REGION_COLUMN, "_period_year", "_start_dt", "_original_order"],
        kind="mergesort",
    )

    work[CUMULATIVE_HEAT_STRESS_FEATURE] = work.groupby(
        [REGION_COLUMN, "_period_year"],
        dropna=False,
    )["_event_heat_stress"].cumsum()

    work = work.sort_values("_original_order", kind="mergesort")
    out[CUMULATIVE_HEAT_STRESS_FEATURE] = work[CUMULATIVE_HEAT_STRESS_FEATURE].to_numpy()
    return out


def add_max_intensity_duration_trends(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    work = out.copy()
    work["_start_dt"] = pd.to_datetime(work[START_DATE_COLUMN], errors="coerce")
    if work["_start_dt"].isna().all():
        raise ValueError(f"Column '{START_DATE_COLUMN}' must contain parseable dates.")

    work["_period_year"] = work["_start_dt"].dt.year

    yearly = (
        work.groupby([REGION_COLUMN, "_period_year"], dropna=False)
        .agg(
            **{
                YEARLY_MAX_INTENSITY_FEATURE: (TEMPERATURE_COLUMN, "max"),
                YEARLY_MAX_DURATION_FEATURE: (DURATION_COLUMN, "max"),
            }
        )
        .sort_index()
        .reset_index()
    )

    yearly[INTENSITY_TREND_FEATURE] = yearly.groupby(REGION_COLUMN, dropna=False)[
        YEARLY_MAX_INTENSITY_FEATURE
    ].diff()
    yearly[DURATION_TREND_FEATURE] = yearly.groupby(REGION_COLUMN, dropna=False)[
        YEARLY_MAX_DURATION_FEATURE
    ].diff()

    enriched = work.merge(
        yearly,
        on=[REGION_COLUMN, "_period_year"],
        how="left",
    )

    out[YEARLY_MAX_INTENSITY_FEATURE] = enriched[YEARLY_MAX_INTENSITY_FEATURE].to_numpy()
    out[YEARLY_MAX_DURATION_FEATURE] = enriched[YEARLY_MAX_DURATION_FEATURE].to_numpy()
    out[INTENSITY_TREND_FEATURE] = enriched[INTENSITY_TREND_FEATURE].to_numpy()
    out[DURATION_TREND_FEATURE] = enriched[DURATION_TREND_FEATURE].to_numpy()
    return out


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    validate_required_columns(
        df,
        include_cumulative_heat_stress=args.add_cumulative_heat_stress_index,
        include_max_trends=args.add_max_intensity_duration_trends,
    )
    engineered = add_inter_event_recovery_interval(df)
    if args.add_cumulative_heat_stress_index:
        engineered = add_cumulative_heat_stress_index(engineered)
    if args.add_max_intensity_duration_trends:
        engineered = add_max_intensity_duration_trends(engineered)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    engineered.to_csv(output_path, index=False)

    print(f"Wrote {len(engineered.columns)} columns to: {output_path}")


if __name__ == "__main__":
    main()

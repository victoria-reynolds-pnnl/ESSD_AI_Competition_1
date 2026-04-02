#!/usr/bin/env python
"""Engineer cleaned May-October modeling features from raw yearly fire tables."""

# AI-assistance note: OpenAI Codex helped draft this script. Team review is still required before challenge submission.

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from _fire_pipeline_common import (
    AI_ASSISTANCE_NOTE,
    CLEANED_EXPORT_COLUMNS,
    DEFAULT_ARTIFACT_ROOT,
    EXPORT_MONTHS,
    YEAR_TO_SPLIT_BUCKET,
    add_common_arguments,
    build_base_manifest,
    discover_available_years,
    ensure_directory,
    load_partition_frame,
    optimize_numeric_dtypes,
    parse_years,
    partition_data_path,
    partition_manifest_path,
    write_json,
)


@dataclass
class CellHistoryState:
    apcp_history: list[float] = field(default_factory=list)
    fwi_history: list[float] = field(default_factory=list)
    fhs_history: list[float] = field(default_factory=list)
    last_fire_date: pd.Timestamp | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    return parser.parse_args()


def historical_window_feature(
    current_values: np.ndarray,
    history_values: list[float],
    reducer: str,
) -> np.ndarray:
    full_series = pd.Series([*history_values, *current_values.tolist()], dtype="float64")
    shifted = full_series.shift(1)
    if reducer == "sum":
        feature = shifted.rolling(window=7, min_periods=1).sum()
    elif reducer == "mean":
        feature = shifted.rolling(window=7, min_periods=1).mean()
    else:
        raise ValueError(f"Unsupported reducer: {reducer}")
    return feature.iloc[len(history_values) :].to_numpy(dtype="float32")


def lag1_feature(current_values: np.ndarray, history_values: list[float]) -> np.ndarray:
    full_series = pd.Series([*history_values, *current_values.tolist()], dtype="float64")
    return full_series.shift(1).iloc[len(history_values) :].to_numpy(dtype="float32")


def days_since_fire_feature(
    dates: pd.Series,
    fhs_values: np.ndarray,
    last_fire_date: pd.Timestamp | None,
) -> tuple[np.ndarray, pd.Timestamp | None]:
    result = np.full(len(dates), np.nan, dtype="float32")
    current_last_fire = last_fire_date
    for index, (date_value, fhs_value) in enumerate(zip(pd.to_datetime(dates), fhs_values, strict=True)):
        if current_last_fire is not None:
            result[index] = float((date_value - current_last_fire).days)
        if pd.notna(fhs_value) and float(fhs_value) > 0:
            current_last_fire = pd.Timestamp(date_value)
    return result, current_last_fire


def update_history(history_values: list[float], current_values: np.ndarray) -> list[float]:
    combined = [*history_values, *current_values.astype("float64").tolist()]
    return combined[-7:]


def engineer_year_frame(
    raw_frame: pd.DataFrame,
    state_by_cell: dict[str, CellHistoryState],
) -> pd.DataFrame:
    engineered = raw_frame.sort_values(["cell_id", "date"], kind="stable").reset_index(drop=True).copy()
    row_count = len(engineered)
    apcp_7d_sum = np.full(row_count, np.nan, dtype="float32")
    fwi_7d_mean = np.full(row_count, np.nan, dtype="float32")
    fhs_c9_lag1 = np.full(row_count, np.nan, dtype="float32")
    fhs_c9_lag7_sum = np.full(row_count, np.nan, dtype="float32")
    days_since_fire = np.full(row_count, np.nan, dtype="float32")

    indices_by_cell = engineered.groupby("cell_id", sort=False).indices
    for cell_id, index_array in indices_by_cell.items():
        positions = np.asarray(index_array, dtype=int)
        cell_frame = engineered.iloc[positions]
        cell_state = state_by_cell.setdefault(cell_id, CellHistoryState())

        apcp_values = cell_frame["air_apcp"].to_numpy(dtype="float32", copy=False)
        fwi_values = cell_frame["FWI"].to_numpy(dtype="float32", copy=False)
        fhs_values = cell_frame["FHS_c9"].to_numpy(dtype="float32", copy=False)

        apcp_7d_sum[positions] = historical_window_feature(apcp_values, cell_state.apcp_history, reducer="sum")
        fwi_7d_mean[positions] = historical_window_feature(fwi_values, cell_state.fwi_history, reducer="mean")
        fhs_c9_lag1[positions] = lag1_feature(fhs_values, cell_state.fhs_history)
        fhs_c9_lag7_sum[positions] = historical_window_feature(fhs_values, cell_state.fhs_history, reducer="sum")
        days_result, updated_last_fire = days_since_fire_feature(
            cell_frame["date"],
            fhs_values,
            cell_state.last_fire_date,
        )
        days_since_fire[positions] = days_result

        cell_state.apcp_history = update_history(cell_state.apcp_history, apcp_values)
        cell_state.fwi_history = update_history(cell_state.fwi_history, fwi_values)
        cell_state.fhs_history = update_history(cell_state.fhs_history, fhs_values)
        cell_state.last_fire_date = updated_last_fire

    engineered["year"] = engineered["date"].dt.year.astype("int16")
    engineered["month"] = engineered["date"].dt.month.astype("int8")
    engineered["split_bucket"] = engineered["year"].map(YEAR_TO_SPLIT_BUCKET)
    engineered["fire_occurrence"] = (engineered["FHS_c9"].fillna(0) > 0).astype("int8")
    engineered["wind_speed"] = np.sqrt(engineered["uwnd"] ** 2 + engineered["vwnd"] ** 2).astype("float32")
    engineered["temp_range"] = (engineered["tmax"] - engineered["tmin"]).astype("float32")
    engineered["pdsi_severe_flag"] = np.where(engineered["pdsi"] <= -4, 1, 0).astype("int8")
    engineered["apcp_7d_sum"] = apcp_7d_sum
    engineered["fwi_7d_mean"] = fwi_7d_mean
    engineered["fhs_c9_lag1"] = fhs_c9_lag1
    engineered["fhs_c9_lag7_sum"] = fhs_c9_lag7_sum
    engineered["days_since_fire"] = days_since_fire

    day_of_year = engineered["date"].dt.dayofyear.astype("float32")
    engineered["doy_sin"] = np.sin((2.0 * np.pi * day_of_year) / 366.0).astype("float32")
    engineered["doy_cos"] = np.cos((2.0 * np.pi * day_of_year) / 366.0).astype("float32")

    cleaned = engineered.loc[engineered["month"].isin(EXPORT_MONTHS), CLEANED_EXPORT_COLUMNS].copy()
    cleaned = optimize_numeric_dtypes(cleaned)
    return cleaned


def build_cleaned_manifest(cleaned_frame: pd.DataFrame, year: int) -> dict[str, object]:
    manifest = build_base_manifest("cleaned", year, cleaned_frame)
    positive_rows = int(cleaned_frame["fire_occurrence"].sum())
    manifest["class_balance"] = {
        "positive_rows": positive_rows,
        "negative_rows": int(len(cleaned_frame) - positive_rows),
        "positive_rate": float(positive_rows / len(cleaned_frame)) if len(cleaned_frame) else 0.0,
    }
    manifest["rows_by_split_bucket"] = {
        split_bucket: int(count)
        for split_bucket, count in cleaned_frame["split_bucket"].value_counts(dropna=False).sort_index().items()
    }
    manifest["ai_assistance_note"] = AI_ASSISTANCE_NOTE
    return manifest


def resolve_years_or_available(args: argparse.Namespace) -> list[int]:
    if args.years:
        return parse_years(args.years)
    available_years = discover_available_years(args.artifact_root, "raw")
    if not available_years:
        raise FileNotFoundError(
            "No raw parquet partitions were found under the artifact root. "
            "Run scripts/extract_conus_fire_table.py first."
        )
    return available_years


def main() -> None:
    args = parse_args()
    years = resolve_years_or_available(args)
    state_by_cell: dict[str, CellHistoryState] = {}

    for year in years:
        raw_path = partition_data_path(args.artifact_root, "raw", year)
        cleaned_path = partition_data_path(args.artifact_root, "cleaned", year)
        manifest_path = partition_manifest_path(args.artifact_root, "cleaned", year)
        if cleaned_path.exists() and manifest_path.exists() and not args.overwrite:
            print(f"[engineer] skipping {year}: {cleaned_path} already exists")
            continue

        raw_frame = load_partition_frame(args.artifact_root, "raw", year)
        cleaned_frame = engineer_year_frame(raw_frame, state_by_cell)

        ensure_directory(cleaned_path.parent)
        cleaned_frame.to_parquet(cleaned_path, index=False)
        write_json(manifest_path, build_cleaned_manifest(cleaned_frame, year))
        print(f"[engineer] wrote {len(cleaned_frame):,} cleaned rows for {year} to {cleaned_path}")


if __name__ == "__main__":
    main()

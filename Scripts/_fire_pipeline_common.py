"""Shared helpers for the CONUS fire occurrence v1 Week 2 pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

AI_ASSISTANCE_NOTE = (
    "AI-assistance note: OpenAI Codex helped draft this script. "
    "Team review is still required before challenge submission."
)

DEFAULT_INPUT_ROOT = Path("/workspace/data/ComExDBM_CONUS/04_OutputData")
DEFAULT_ARTIFACT_ROOT = Path("/workspace/data/artifacts/conus_fire_occurrence_v1")
DEFAULT_EXPORT_ROOT = DEFAULT_ARTIFACT_ROOT / "exports" / "week2"

ALL_YEARS = tuple(range(2001, 2021))
EXPORT_MONTHS = (5, 6, 7, 8, 9, 10)
MERGE_KEYS = ["date", "lat", "lon"]
IDENTIFIER_COLUMNS = ["cell_id", "row_id"]

HEATWAVE_FIELDS = ["tmax", "tmin", "HI02", "HI04", "HI05", "HI06", "HI09", "HI10"]
FIRE_FIELDS = ["FHS_c9", "FHS_c8c9", "maxFRP_c9", "maxFRP_c8c9", "BA_km2"]
DROUGHT_FIELDS = ["pdsi", "spi14d", "spi30d", "spi90d", "spei14d", "spei30d", "spei90d"]
MET_FIELDS = ["air_sfc", "air_apcp", "soilm", "rhum_2m", "uwnd", "vwnd", "LAI", "NDVI", "FWI"]
ENGINEERED_FIELDS = [
    "wind_speed",
    "temp_range",
    "pdsi_severe_flag",
    "apcp_7d_sum",
    "fwi_7d_mean",
    "fhs_c9_lag1",
    "fhs_c9_lag7_sum",
    "days_since_fire",
    "doy_sin",
    "doy_cos",
]

RAW_EXPORT_COLUMNS = MERGE_KEYS + IDENTIFIER_COLUMNS + HEATWAVE_FIELDS + FIRE_FIELDS + DROUGHT_FIELDS + MET_FIELDS
CLEANED_EXPORT_COLUMNS = [
    "date",
    "lat",
    "lon",
    "cell_id",
    "row_id",
    "year",
    "month",
    "split_bucket",
    "fire_occurrence",
    *DROUGHT_FIELDS,
    *MET_FIELDS,
    "tmax",
    "tmin",
    *ENGINEERED_FIELDS,
]

DATASET_VARIABLES = {
    "heatwave": HEATWAVE_FIELDS,
    "fire": FIRE_FIELDS,
    "drought": DROUGHT_FIELDS,
    "meteorology": MET_FIELDS,
}

DATASET_FILE_TEMPLATES = {
    "heatwave": Path("01_HeatWave") / "ComExDBM_{year}_HeatWave_V02.nc",
    "fire": Path("02_FireData") / "ComExDBM_{year}_Fires_V02.nc",
    "drought": Path("03_DroughtData") / "ComExDBM_{year}_Drought_V02.nc",
    "meteorology": Path("04_Meteorological_Variables") / "ComExDBM_{year}_MetVars_V02.nc",
}

COLUMN_ALIASES = {
    "time": "date",
    "apcp": "air_apcp",
    "maxFRP_max_c9": "maxFRP_c9",
    "maxFRP_max_c8c9": "maxFRP_c8c9",
}

YEAR_TO_SPLIT_BUCKET = {
    **{year: "train" for year in range(2001, 2017)},
    2017: "validation",
    2018: "validation",
    2019: "test",
    2020: "test",
}

FIELD_METADATA: dict[str, dict[str, str]] = {
    "date": {
        "stage": "both",
        "dtype": "datetime64[ns]",
        "description": "Daily timestamp for the grid cell observation.",
        "example": "2001-05-01T00:00:00",
        "source_or_formula": "Normalized from the NetCDF `time` coordinate.",
        "role": "key",
    },
    "lat": {
        "stage": "both",
        "dtype": "float32",
        "description": "Latitude of the CONUS grid cell center.",
        "example": "34.7500",
        "source_or_formula": "NetCDF `lat` coordinate.",
        "role": "key",
    },
    "lon": {
        "stage": "both",
        "dtype": "float32",
        "description": "Longitude of the CONUS grid cell center.",
        "example": "-118.2500",
        "source_or_formula": "NetCDF `lon` coordinate.",
        "role": "key",
    },
    "cell_id": {
        "stage": "both",
        "dtype": "string",
        "description": "Deterministic spatial identifier derived from latitude and longitude.",
        "example": "lat=34.7500|lon=-118.2500",
        "source_or_formula": "Formatted as `lat={lat:.4f}|lon={lon:.4f}`.",
        "role": "identifier",
    },
    "row_id": {
        "stage": "both",
        "dtype": "string",
        "description": "Deterministic row identifier for a unique cell-day.",
        "example": "2001-05-01|lat=34.7500|lon=-118.2500",
        "source_or_formula": "Concatenation of ISO date and `cell_id`.",
        "role": "identifier",
    },
    "tmax": {
        "stage": "both",
        "dtype": "float32",
        "description": "Daily maximum near-surface air temperature.",
        "example": "33.2",
        "source_or_formula": "Heatwave product raw field.",
        "role": "predictor",
    },
    "tmin": {
        "stage": "both",
        "dtype": "float32",
        "description": "Daily minimum near-surface air temperature.",
        "example": "17.9",
        "source_or_formula": "Heatwave product raw field.",
        "role": "predictor",
    },
    "HI02": {
        "stage": "raw",
        "dtype": "float32",
        "description": "Heatwave indicator from the source benchmark, kept only in the raw merged table.",
        "example": "0",
        "source_or_formula": "Heatwave product raw field.",
        "role": "excluded_predictor",
    },
    "HI04": {
        "stage": "raw",
        "dtype": "float32",
        "description": "Heatwave indicator from the source benchmark, kept only in the raw merged table.",
        "example": "0",
        "source_or_formula": "Heatwave product raw field.",
        "role": "excluded_predictor",
    },
    "HI05": {
        "stage": "raw",
        "dtype": "float32",
        "description": "Heatwave indicator from the source benchmark, kept only in the raw merged table.",
        "example": "0",
        "source_or_formula": "Heatwave product raw field.",
        "role": "excluded_predictor",
    },
    "HI06": {
        "stage": "raw",
        "dtype": "float32",
        "description": "Heatwave indicator from the source benchmark, kept only in the raw merged table.",
        "example": "0",
        "source_or_formula": "Heatwave product raw field.",
        "role": "excluded_predictor",
    },
    "HI09": {
        "stage": "raw",
        "dtype": "float32",
        "description": "Heatwave indicator from the source benchmark, kept only in the raw merged table.",
        "example": "0",
        "source_or_formula": "Heatwave product raw field.",
        "role": "excluded_predictor",
    },
    "HI10": {
        "stage": "raw",
        "dtype": "float32",
        "description": "Heatwave indicator from the source benchmark, kept only in the raw merged table.",
        "example": "0",
        "source_or_formula": "Heatwave product raw field.",
        "role": "excluded_predictor",
    },
    "FHS_c9": {
        "stage": "raw",
        "dtype": "float32",
        "description": "Highest-confidence fire hotspot count used to derive the binary target.",
        "example": "2",
        "source_or_formula": "Fire product raw field.",
        "role": "target_source",
    },
    "FHS_c8c9": {
        "stage": "raw",
        "dtype": "float32",
        "description": "Combined confidence fire hotspot count retained only in the raw merged table.",
        "example": "3",
        "source_or_formula": "Fire product raw field.",
        "role": "excluded_predictor",
    },
    "maxFRP_c9": {
        "stage": "raw",
        "dtype": "float32",
        "description": "Same-day fire radiative power proxy retained only in the raw merged table.",
        "example": "8.7",
        "source_or_formula": "Fire product raw field, normalized from `maxFRP_max_c9` if needed.",
        "role": "excluded_predictor",
    },
    "maxFRP_c8c9": {
        "stage": "raw",
        "dtype": "float32",
        "description": "Same-day fire radiative power proxy retained only in the raw merged table.",
        "example": "9.4",
        "source_or_formula": "Fire product raw field, normalized from `maxFRP_max_c8c9` if needed.",
        "role": "excluded_predictor",
    },
    "BA_km2": {
        "stage": "raw",
        "dtype": "float32",
        "description": "Same-day burned area retained only in the raw merged table.",
        "example": "0.6",
        "source_or_formula": "Fire product raw field.",
        "role": "excluded_predictor",
    },
    "pdsi": {
        "stage": "both",
        "dtype": "float32",
        "description": "Palmer Drought Severity Index at the grid cell-day.",
        "example": "-3.1",
        "source_or_formula": "Drought product raw field.",
        "role": "predictor",
    },
    "spi14d": {
        "stage": "both",
        "dtype": "float32",
        "description": "14-day Standardized Precipitation Index.",
        "example": "-0.8",
        "source_or_formula": "Drought product raw field.",
        "role": "predictor",
    },
    "spi30d": {
        "stage": "both",
        "dtype": "float32",
        "description": "30-day Standardized Precipitation Index.",
        "example": "-1.2",
        "source_or_formula": "Drought product raw field.",
        "role": "predictor",
    },
    "spi90d": {
        "stage": "both",
        "dtype": "float32",
        "description": "90-day Standardized Precipitation Index.",
        "example": "-1.9",
        "source_or_formula": "Drought product raw field.",
        "role": "predictor",
    },
    "spei14d": {
        "stage": "both",
        "dtype": "float32",
        "description": "14-day Standardized Precipitation Evapotranspiration Index.",
        "example": "-0.7",
        "source_or_formula": "Drought product raw field.",
        "role": "predictor",
    },
    "spei30d": {
        "stage": "both",
        "dtype": "float32",
        "description": "30-day Standardized Precipitation Evapotranspiration Index.",
        "example": "-1.0",
        "source_or_formula": "Drought product raw field.",
        "role": "predictor",
    },
    "spei90d": {
        "stage": "both",
        "dtype": "float32",
        "description": "90-day Standardized Precipitation Evapotranspiration Index.",
        "example": "-1.5",
        "source_or_formula": "Drought product raw field.",
        "role": "predictor",
    },
    "air_sfc": {
        "stage": "both",
        "dtype": "float32",
        "description": "Daily near-surface air temperature from the meteorology product.",
        "example": "297.4",
        "source_or_formula": "Meteorology product raw field.",
        "role": "predictor",
    },
    "air_apcp": {
        "stage": "both",
        "dtype": "float32",
        "description": "Daily accumulated precipitation.",
        "example": "1.7",
        "source_or_formula": "Meteorology raw field `apcp`, normalized to canonical name `air_apcp`.",
        "role": "predictor",
    },
    "soilm": {
        "stage": "both",
        "dtype": "float32",
        "description": "Daily soil moisture proxy from the meteorology product.",
        "example": "0.27",
        "source_or_formula": "Meteorology product raw field.",
        "role": "predictor",
    },
    "rhum_2m": {
        "stage": "both",
        "dtype": "float32",
        "description": "Daily relative humidity at 2 meters.",
        "example": "42.5",
        "source_or_formula": "Meteorology product raw field.",
        "role": "predictor",
    },
    "uwnd": {
        "stage": "both",
        "dtype": "float32",
        "description": "Daily zonal wind component.",
        "example": "3.2",
        "source_or_formula": "Meteorology product raw field.",
        "role": "predictor",
    },
    "vwnd": {
        "stage": "both",
        "dtype": "float32",
        "description": "Daily meridional wind component.",
        "example": "-1.1",
        "source_or_formula": "Meteorology product raw field.",
        "role": "predictor",
    },
    "LAI": {
        "stage": "both",
        "dtype": "float32",
        "description": "Leaf Area Index for the grid cell-day.",
        "example": "2.8",
        "source_or_formula": "Meteorology product raw field.",
        "role": "predictor",
    },
    "NDVI": {
        "stage": "both",
        "dtype": "float32",
        "description": "Normalized Difference Vegetation Index for the grid cell-day.",
        "example": "0.41",
        "source_or_formula": "Meteorology product raw field.",
        "role": "predictor",
    },
    "FWI": {
        "stage": "both",
        "dtype": "float32",
        "description": "Fire Weather Index for the grid cell-day.",
        "example": "17.3",
        "source_or_formula": "Meteorology product raw field.",
        "role": "predictor",
    },
    "year": {
        "stage": "cleaned",
        "dtype": "int16",
        "description": "Calendar year derived from the normalized date.",
        "example": "2001",
        "source_or_formula": "Derived from `date`.",
        "role": "partition",
    },
    "month": {
        "stage": "cleaned",
        "dtype": "int8",
        "description": "Calendar month derived from the normalized date.",
        "example": "5",
        "source_or_formula": "Derived from `date`.",
        "role": "partition",
    },
    "split_bucket": {
        "stage": "cleaned",
        "dtype": "string",
        "description": "Fixed benchmark split assignment for the row year.",
        "example": "train",
        "source_or_formula": "2001-2016=train, 2017-2018=validation, 2019-2020=test.",
        "role": "split",
    },
    "fire_occurrence": {
        "stage": "cleaned",
        "dtype": "int8",
        "description": "Binary target equal to 1 when `FHS_c9 > 0`, else 0.",
        "example": "1",
        "source_or_formula": "Derived from source field `FHS_c9`.",
        "role": "target",
    },
    "wind_speed": {
        "stage": "cleaned",
        "dtype": "float32",
        "description": "Wind speed magnitude derived from zonal and meridional wind components.",
        "example": "3.38",
        "source_or_formula": "sqrt(uwnd^2 + vwnd^2).",
        "role": "engineered_feature",
    },
    "temp_range": {
        "stage": "cleaned",
        "dtype": "float32",
        "description": "Daily temperature range.",
        "example": "15.3",
        "source_or_formula": "tmax - tmin.",
        "role": "engineered_feature",
    },
    "pdsi_severe_flag": {
        "stage": "cleaned",
        "dtype": "int8",
        "description": "Flag for severe drought conditions.",
        "example": "1",
        "source_or_formula": "1 when pdsi <= -4, else 0.",
        "role": "engineered_feature",
    },
    "apcp_7d_sum": {
        "stage": "cleaned",
        "dtype": "float32",
        "description": "Trailing 7-day precipitation sum using only prior days for the same cell.",
        "example": "6.9",
        "source_or_formula": "Rolling 7-day sum of `air_apcp`, shifted by one day.",
        "role": "engineered_feature",
    },
    "fwi_7d_mean": {
        "stage": "cleaned",
        "dtype": "float32",
        "description": "Trailing 7-day Fire Weather Index mean using only prior days for the same cell.",
        "example": "14.2",
        "source_or_formula": "Rolling 7-day mean of `FWI`, shifted by one day.",
        "role": "engineered_feature",
    },
    "fhs_c9_lag1": {
        "stage": "cleaned",
        "dtype": "float32",
        "description": "Previous-day highest-confidence fire hotspot count for the same cell.",
        "example": "0.0",
        "source_or_formula": "Lag-1 of source field `FHS_c9` by `cell_id`.",
        "role": "engineered_feature",
    },
    "fhs_c9_lag7_sum": {
        "stage": "cleaned",
        "dtype": "float32",
        "description": "Trailing 7-day sum of fire hotspot counts using only prior days for the same cell.",
        "example": "2.0",
        "source_or_formula": "Rolling 7-day sum of `FHS_c9`, shifted by one day.",
        "role": "engineered_feature",
    },
    "days_since_fire": {
        "stage": "cleaned",
        "dtype": "float32",
        "description": "Calendar days since the last prior day with `FHS_c9 > 0` for the same cell.",
        "example": "12.0",
        "source_or_formula": "Difference between `date` and the most recent prior fire date by `cell_id`.",
        "role": "engineered_feature",
    },
    "doy_sin": {
        "stage": "cleaned",
        "dtype": "float32",
        "description": "Sine encoding of day-of-year seasonality.",
        "example": "0.8660",
        "source_or_formula": "sin(2 * pi * day_of_year / 366).",
        "role": "engineered_feature",
    },
    "doy_cos": {
        "stage": "cleaned",
        "dtype": "float32",
        "description": "Cosine encoding of day-of-year seasonality.",
        "example": "-0.5000",
        "source_or_formula": "cos(2 * pi * day_of_year / 366).",
        "role": "engineered_feature",
    },
}


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help=f"Root directory for generated pipeline artifacts. Defaults to {DEFAULT_ARTIFACT_ROOT}.",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        help="Years to process. Supports tokens like `2001`, `2001-2003`, or `2001,2005`.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing parquet, manifest, and export files.",
    )
    return parser


def parse_years(year_tokens: Sequence[str] | None) -> list[int]:
    if not year_tokens:
        return list(ALL_YEARS)

    years: set[int] = set()
    for token in year_tokens:
        for piece in token.split(","):
            value = piece.strip()
            if not value:
                continue
            if "-" in value:
                start_text, end_text = value.split("-", maxsplit=1)
                start = int(start_text)
                end = int(end_text)
                if start > end:
                    raise ValueError(f"Invalid year range: {value}")
                years.update(range(start, end + 1))
            else:
                years.add(int(value))

    invalid_years = sorted(year for year in years if year not in ALL_YEARS)
    if invalid_years:
        raise ValueError(f"Requested years are outside the supported 2001-2020 range: {invalid_years}")
    return sorted(years)


def discover_available_years(artifact_root: Path, stage: str) -> list[int]:
    years: list[int] = []
    stage_root = artifact_root / stage
    if not stage_root.exists():
        return years

    for path in stage_root.glob("year=*/data.parquet"):
        try:
            years.append(int(path.parent.name.split("=", maxsplit=1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(years))


def dataset_path(input_root: Path, dataset_name: str, year: int) -> Path:
    return input_root / DATASET_FILE_TEMPLATES[dataset_name].as_posix().format(year=year)


def partition_dir(artifact_root: Path, stage: str, year: int) -> Path:
    return artifact_root / stage / f"year={year}"


def partition_data_path(artifact_root: Path, stage: str, year: int) -> Path:
    return partition_dir(artifact_root, stage, year) / "data.parquet"


def partition_manifest_path(artifact_root: Path, stage: str, year: int) -> Path:
    return partition_dir(artifact_root, stage, year) / "manifest.json"


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_column_names(columns: Iterable[str]) -> list[str]:
    return [COLUMN_ALIASES.get(column, column) for column in columns]


def normalize_frame_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(columns=COLUMN_ALIASES)
    if "date" in renamed.columns:
        renamed["date"] = pd.to_datetime(renamed["date"]).dt.normalize()
    return renamed


def optimize_numeric_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    optimized = frame.copy()
    for column in optimized.columns:
        series = optimized[column]
        if pd.api.types.is_float_dtype(series):
            optimized[column] = pd.to_numeric(series, downcast="float")
        elif pd.api.types.is_integer_dtype(series):
            optimized[column] = pd.to_numeric(series, downcast="integer")
    return optimized


def load_netcdf_frame(file_path: Path, variables: Sequence[str]) -> pd.DataFrame:
    import xarray as xr

    if not file_path.exists():
        raise FileNotFoundError(f"Missing source file: {file_path}")

    with xr.open_dataset(file_path) as dataset:
        rename_map = {
            source_name: target_name
            for source_name, target_name in COLUMN_ALIASES.items()
            if source_name in dataset.data_vars or source_name in dataset.coords
        }
        normalized_dataset = dataset.rename(rename_map) if rename_map else dataset

        missing_variables = [column for column in variables if column not in normalized_dataset.data_vars]
        if missing_variables:
            raise KeyError(f"{file_path.name} is missing required variables: {missing_variables}")

        frame = normalized_dataset[list(variables)].to_dataframe().reset_index()

    frame = normalize_frame_columns(frame)
    frame = optimize_numeric_dtypes(frame)
    validate_unique_keys(frame, file_path.name)
    return frame


def validate_unique_keys(frame: pd.DataFrame, label: str) -> None:
    duplicate_count = int(frame.duplicated(subset=MERGE_KEYS).sum())
    if duplicate_count:
        raise ValueError(f"{label} contains {duplicate_count} duplicate {MERGE_KEYS} rows.")


def add_identifiers(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["cell_id"] = result.apply(
        lambda row: f"lat={row['lat']:.4f}|lon={row['lon']:.4f}",
        axis=1,
    )
    result["row_id"] = result["date"].dt.strftime("%Y-%m-%d") + "|" + result["cell_id"]
    return result


def build_schema_map(frame: pd.DataFrame) -> dict[str, str]:
    return {column: str(dtype) for column, dtype in frame.dtypes.items()}


def build_base_manifest(stage: str, year: int, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "stage": stage,
        "year": year,
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
        "schema": build_schema_map(frame),
        "date_min": frame["date"].min().strftime("%Y-%m-%d") if len(frame) else None,
        "date_max": frame["date"].max().strftime("%Y-%m-%d") if len(frame) else None,
        "missing_counts": {
            column: int(count)
            for column, count in frame.isna().sum().items()
            if int(count) > 0
        },
        "duplicate_key_count": int(frame.duplicated(subset=MERGE_KEYS).sum()),
    }


def write_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_partition_frame(artifact_root: Path, stage: str, year: int) -> pd.DataFrame:
    data_path = partition_data_path(artifact_root, stage, year)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {stage} parquet partition for year {year}: {data_path}")
    frame = pd.read_parquet(data_path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def month_filtered(frame: pd.DataFrame) -> pd.DataFrame:
    months = pd.to_datetime(frame["date"]).dt.month
    return frame.loc[months.isin(EXPORT_MONTHS)].reset_index(drop=True)


def build_field_metadata_records(raw_columns: Sequence[str], cleaned_columns: Sequence[str]) -> list[dict[str, str]]:
    ordered_columns: list[str] = []
    for column in list(raw_columns) + list(cleaned_columns):
        if column not in ordered_columns:
            ordered_columns.append(column)

    raw_set = set(raw_columns)
    cleaned_set = set(cleaned_columns)
    records: list[dict[str, str]] = []
    for column in ordered_columns:
        metadata = dict(FIELD_METADATA.get(column, {}))
        metadata.setdefault("dtype", "unknown")
        metadata.setdefault("description", f"Column `{column}` from the CONUS fire occurrence v1 pipeline.")
        metadata.setdefault("example", "")
        metadata.setdefault("source_or_formula", "See pipeline scripts for derivation details.")
        metadata.setdefault("role", "field")

        if column in raw_set and column in cleaned_set:
            stage = "both"
        elif column in raw_set:
            stage = "raw"
        else:
            stage = "cleaned"

        records.append(
            {
                "name": column,
                "stage": stage,
                "dtype": metadata["dtype"],
                "description": metadata["description"],
                "example": metadata["example"],
                "source_or_formula": metadata["source_or_formula"],
                "role": metadata["role"],
            }
        )
    return records


def aggregate_manifests(manifests: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not manifests:
        return {"total_rows": 0, "rows_by_year": {}, "years": []}

    rows_by_year = {str(manifest["year"]): int(manifest["row_count"]) for manifest in manifests}
    years = sorted(int(year) for year in rows_by_year)
    total_rows = int(sum(rows_by_year.values()))
    all_columns: list[str] = []
    for manifest in manifests:
        for column in manifest.get("columns", []):
            if column not in all_columns:
                all_columns.append(column)

    missing_totals: dict[str, int] = {}
    for manifest in manifests:
        for column, count in manifest.get("missing_counts", {}).items():
            missing_totals[column] = missing_totals.get(column, 0) + int(count)

    return {
        "years": years,
        "total_rows": total_rows,
        "rows_by_year": rows_by_year,
        "columns": all_columns,
        "missing_counts": missing_totals,
        "date_min": min(manifest["date_min"] for manifest in manifests if manifest.get("date_min")),
        "date_max": max(manifest["date_max"] for manifest in manifests if manifest.get("date_max")),
    }


def load_manifests(artifact_root: Path, stage: str, years: Sequence[int]) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for year in years:
        manifest_path = partition_manifest_path(artifact_root, stage, year)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing {stage} manifest for year {year}: {manifest_path}")
        manifests.append(read_json(manifest_path))
    return manifests


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")

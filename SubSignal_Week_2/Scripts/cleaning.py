#!/usr/bin/env python3
"""Week 2 cleaning scaffold for SubSignal AI competition data."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


DELIVERABLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = DELIVERABLE_ROOT


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    input_path: Path
    time_format: str
    expected_gap_seconds: int
    has_triplex_state: bool


DATASETS = {
    "ftes_1sec": DatasetConfig(
        name="ftes_1sec",
        input_path=DELIVERABLE_ROOT / "data" / "original" / "FTES-Full_Test_1sec_system_processed.csv",
        time_format="%m/%d/%y %H:%M:%S",
        expected_gap_seconds=1,
        has_triplex_state=True,
    ),
    "ftes_1hour": DatasetConfig(
        name="ftes_1hour",
        input_path=DELIVERABLE_ROOT / "data" / "original" / "FTES-Full_Test_1hour_avg.csv",
        time_format="%Y-%m-%d %H:%M:%S",
        expected_gap_seconds=3600,
        has_triplex_state=False,
    ),
}


def normalize_header(header: str, fallback: str) -> str:
    value = header.strip()
    if not value:
        return fallback
    value = value.replace("(True/False)", "")
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or fallback


def make_header_mapping(headers: list[str]) -> list[tuple[str, str]]:
    mapping: list[tuple[str, str]] = []
    seen: Counter[str] = Counter()
    for idx, source in enumerate(headers):
        fallback = "raw_row_index" if idx == 0 and not source.strip() else f"column_{idx + 1}"
        clean = normalize_header(source, fallback)
        seen[clean] += 1
        if seen[clean] > 1:
            clean = f"{clean}_{seen[clean]}"
        mapping.append((source, clean))
    return mapping


def detect_boolean_fields(mapping: list[tuple[str, str]]) -> set[str]:
    return {
        clean
        for source, clean in mapping
        if "(True/False)" in source
    }


def detect_numeric_fields(mapping: list[tuple[str, str]], dataset: DatasetConfig) -> set[str]:
    excluded = {"time", "time_raw", "source_dataset"}
    if dataset.has_triplex_state:
        excluded.add("triplex_on_off")
    boolean_fields = detect_boolean_fields(mapping)
    numeric_fields = set()
    for source, clean in mapping:
        if clean in excluded or clean in boolean_fields:
            continue
        if clean == "raw_row_index":
            continue
        numeric_fields.add(clean)
    return numeric_fields


def normalize_triplex_state(value: str) -> str:
    return value.strip().lower()


def normalize_boolean(value: str) -> str:
    lowered = value.strip().lower()
    if lowered == "true":
        return "true"
    if lowered == "false":
        return "false"
    return lowered


def boolean_to_ml(value: str) -> str:
    normalized = normalize_boolean(value)
    if normalized == "true":
        return "1"
    if normalized == "false":
        return "0"
    return ""


def triplex_to_ml(value: str) -> str:
    normalized = normalize_triplex_state(value)
    if normalized == "on":
        return "1"
    if normalized == "off":
        return "0"
    return ""


def safe_parse_time(value: str, time_format: str) -> datetime | None:
    try:
        return datetime.strptime(value.strip(), time_format)
    except ValueError:
        return None


def to_output_time(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.strftime("%Y-%m-%dT%H:%M:%S")


def build_output_header(mapping: list[tuple[str, str]], dataset: DatasetConfig) -> list[str]:
    clean_headers = [clean for _, clean in mapping if clean != "time"]
    output_header = ["source_dataset", "time_raw", "time"]
    output_header.extend(clean_headers)
    output_header.extend(
        [
            "flag_bad_timestamp_format",
            "flag_non_monotonic_time",
            "flag_duplicate_timestamp",
            "flag_time_gap_gt_expected",
            "flag_row_length_mismatch",
            "flag_non_numeric_in_numeric_field",
        ]
    )
    return output_header


def build_ml_output_header(mapping: list[tuple[str, str]], dataset: DatasetConfig) -> list[str]:
    ml_fields = ["source_dataset", "time"]
    for _, clean in mapping:
        if clean in {"time", "raw_row_index"}:
            continue
        ml_fields.append(clean)
    ml_fields.extend(
        [
            "flag_bad_timestamp_format",
            "flag_non_monotonic_time",
            "flag_duplicate_timestamp",
            "flag_time_gap_gt_expected",
            "flag_row_length_mismatch",
            "flag_non_numeric_in_numeric_field",
        ]
    )
    return ml_fields


def iter_rows(reader: Iterable[list[str]], row_limit: int | None) -> Iterable[list[str]]:
    for idx, row in enumerate(reader, start=1):
        if row_limit is not None and idx > row_limit:
            break
        yield row


def process_dataset(dataset: DatasetConfig, mode: str, row_limit: int | None, output_root: Path) -> dict[str, object]:
    cleaned_dir = output_root / "data" / ("cleaned" if mode == "full" else mode)
    qa_dir = output_root / "qa" if mode == "full" else output_root / "qa" / mode
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = cleaned_dir / f"{dataset.name}_cleaned.csv"
    ml_ready_path = cleaned_dir / f"{dataset.name}_ml_ready.csv"
    summary_path = qa_dir / f"{dataset.name}_summary.json"

    with dataset.input_path.open(newline="") as infile:
        reader = csv.reader(infile)
        source_headers = next(reader)
        mapping = make_header_mapping(source_headers)
        boolean_fields = detect_boolean_fields(mapping)
        numeric_fields = detect_numeric_fields(mapping, dataset)
        output_header = build_output_header(mapping, dataset)
        ml_output_header = build_ml_output_header(mapping, dataset)
        time_clean_name = dict(mapping)["Time"]
        prev_dt: datetime | None = None
        time_min: datetime | None = None
        time_max: datetime | None = None
        timestamp_occurrences: Counter[str] = Counter()
        duplicate_run_lengths: Counter[int] = Counter()
        current_duplicate_run = 1

        row_count = 0
        mismatch_count = 0
        bad_time_count = 0
        non_monotonic_count = 0
        duplicate_count = 0
        gap_count = 0
        non_numeric_row_count = 0
        offending_numeric_columns: Counter[str] = Counter()

        with cleaned_path.open("w", newline="") as outfile, ml_ready_path.open("w", newline="") as mlfile:
            writer = csv.DictWriter(outfile, fieldnames=output_header)
            ml_writer = csv.DictWriter(mlfile, fieldnames=ml_output_header)
            writer.writeheader()
            ml_writer.writeheader()

            for row in iter_rows(reader, row_limit):
                row_count += 1
                row_length_mismatch = len(row) != len(source_headers)
                if row_length_mismatch:
                    mismatch_count += 1
                    if len(row) < len(source_headers):
                        row = row + [""] * (len(source_headers) - len(row))
                    else:
                        row = row[: len(source_headers)]

                raw_values = {clean: row[idx].strip() for idx, (_, clean) in enumerate(mapping)}
                time_raw = raw_values.get(time_clean_name, "")
                parsed_time = safe_parse_time(time_raw, dataset.time_format)
                bad_timestamp = parsed_time is None
                if bad_timestamp:
                    bad_time_count += 1

                non_monotonic = False
                duplicate = False
                gap_gt_expected = False
                if parsed_time is not None:
                    parsed_time_key = to_output_time(parsed_time)
                    timestamp_occurrences[parsed_time_key] += 1
                    time_min = parsed_time if time_min is None or parsed_time < time_min else time_min
                    time_max = parsed_time if time_max is None or parsed_time > time_max else time_max
                    if prev_dt is not None:
                        delta = int((parsed_time - prev_dt).total_seconds())
                        if delta < 0:
                            if current_duplicate_run > 1:
                                duplicate_run_lengths[current_duplicate_run] += 1
                            current_duplicate_run = 1
                            non_monotonic = True
                            non_monotonic_count += 1
                        elif delta == 0:
                            current_duplicate_run += 1
                            duplicate = True
                            duplicate_count += 1
                        elif delta > dataset.expected_gap_seconds:
                            if current_duplicate_run > 1:
                                duplicate_run_lengths[current_duplicate_run] += 1
                            current_duplicate_run = 1
                            gap_gt_expected = True
                            gap_count += 1
                        else:
                            if current_duplicate_run > 1:
                                duplicate_run_lengths[current_duplicate_run] += 1
                            current_duplicate_run = 1
                    prev_dt = parsed_time
                else:
                    if current_duplicate_run > 1:
                        duplicate_run_lengths[current_duplicate_run] += 1
                    current_duplicate_run = 1

                non_numeric = False
                for clean_name in numeric_fields:
                    value = raw_values.get(clean_name, "")
                    if value == "":
                        continue
                    try:
                        float(value)
                    except ValueError:
                        non_numeric = True
                        offending_numeric_columns[clean_name] += 1
                if non_numeric:
                    non_numeric_row_count += 1

                output_row: dict[str, str] = {
                    "source_dataset": dataset.name,
                    "time_raw": time_raw,
                    "time": to_output_time(parsed_time),
                }
                for _, clean_name in mapping:
                    if clean_name == time_clean_name:
                        continue
                    value = raw_values.get(clean_name, "")
                    if clean_name == "triplex_on_off":
                        value = normalize_triplex_state(value)
                    elif clean_name in boolean_fields:
                        value = normalize_boolean(value)
                    output_row[clean_name] = value

                output_row.update(
                    {
                        "flag_bad_timestamp_format": str(bad_timestamp).lower(),
                        "flag_non_monotonic_time": str(non_monotonic).lower(),
                        "flag_duplicate_timestamp": str(duplicate).lower(),
                        "flag_time_gap_gt_expected": str(gap_gt_expected).lower(),
                        "flag_row_length_mismatch": str(row_length_mismatch).lower(),
                        "flag_non_numeric_in_numeric_field": str(non_numeric).lower(),
                    }
                )
                writer.writerow(output_row)

                ml_row: dict[str, str] = {
                    "source_dataset": dataset.name,
                    "time": to_output_time(parsed_time),
                }
                for _, clean_name in mapping:
                    if clean_name in {"time", "raw_row_index"}:
                        continue
                    value = raw_values.get(clean_name, "")
                    if clean_name in numeric_fields:
                        if value == "":
                            ml_row[clean_name] = ""
                        else:
                            try:
                                ml_row[clean_name] = str(float(value))
                            except ValueError:
                                ml_row[clean_name] = ""
                    elif clean_name == "triplex_on_off":
                        ml_row[clean_name] = triplex_to_ml(value)
                    elif clean_name in boolean_fields:
                        ml_row[clean_name] = boolean_to_ml(value)
                    else:
                        ml_row[clean_name] = value

                ml_row.update(
                    {
                        "flag_bad_timestamp_format": "1" if bad_timestamp else "0",
                        "flag_non_monotonic_time": "1" if non_monotonic else "0",
                        "flag_duplicate_timestamp": "1" if duplicate else "0",
                        "flag_time_gap_gt_expected": "1" if gap_gt_expected else "0",
                        "flag_row_length_mismatch": "1" if row_length_mismatch else "0",
                        "flag_non_numeric_in_numeric_field": "1" if non_numeric else "0",
                    }
                )
                ml_writer.writerow(ml_row)

    if current_duplicate_run > 1:
        duplicate_run_lengths[current_duplicate_run] += 1

    duplicate_examples = [
        {"time": timestamp, "count": count}
        for timestamp, count in sorted(timestamp_occurrences.items())
        if count > 1
    ]

    summary = {
        "dataset": dataset.name,
        "mode": mode,
        "row_limit": row_limit,
        "input_path": str(dataset.input_path),
        "output_path": str(cleaned_path),
        "ml_ready_output_path": str(ml_ready_path),
        "rows_processed": row_count,
        "row_length_mismatches": mismatch_count,
        "timestamp_parse_failures": bad_time_count,
        "non_monotonic_time_rows": non_monotonic_count,
        "duplicate_timestamp_rows": duplicate_count,
        "duplicate_timestamp_keys": len(duplicate_examples),
        "duplicate_timestamp_examples": duplicate_examples[:10],
        "consecutive_duplicate_run_lengths": {
            str(run_length): count
            for run_length, count in sorted(duplicate_run_lengths.items())
        },
        "time_gap_gt_expected_rows": gap_count,
        "rows_with_non_numeric_fields": non_numeric_row_count,
        "offending_numeric_columns": dict(offending_numeric_columns),
        "time_min": to_output_time(time_min),
        "time_max": to_output_time(time_max),
        "header_mapping": [
            {"source": source, "clean": clean}
            for source, clean in mapping
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 2 data cleaning scaffold")
    parser.add_argument("--mode", choices=["sample", "full"], default="sample")
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASETS.keys()],
        default="all",
        help="Dataset to process",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=1000,
        help="Rows per dataset in sample mode",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for generated artifacts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = DATASETS.values() if args.dataset == "all" else [DATASETS[args.dataset]]
    row_limit = args.sample_rows if args.mode == "sample" else None
    summaries = [process_dataset(dataset, args.mode, row_limit, args.output_root) for dataset in selected]
    print(json.dumps({"summaries": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

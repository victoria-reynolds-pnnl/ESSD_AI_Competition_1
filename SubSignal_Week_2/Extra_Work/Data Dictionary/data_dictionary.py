import pandas as pd
import numpy as np
from pathlib import Path
import re

# =========================================================
# CONFIGURATION
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"

# Update the filename here only if your file is named differently
INPUT_FILE = DATA_DIR / "FTES-Full_Test_1hour_avg.csv"
OUTPUT_FILE = BASE_DIR / "FTES_data_dictionary.xlsx"

# If you later use Excel instead of CSV, this sheet index will be used
INPUT_SHEET = 0

# =========================================================
# REFERENCE DEFINITIONS FROM FTES DESCRIPTION DOCUMENT
# =========================================================
REFERENCE_DEFINITIONS = {
    "Time": "Date/time of the observation.",
    "Net Flow": "Injection water flow rate from Triplex pump.",
    "Injection EC": "Electrical conductivity associated with injected water.",
    "Injection Pressure": "Injection pressure associated with the active injection interval.",
    "TL Interval Flow": "Production water flow rate TL interval.",
    "TL Bottom Flow": "Production water flow rate TL bottom.",
    "TL Collar Flow": "Production water flow rate TL collar.",
    "TL Interval EC": "Water EC TL interval.",
    "TL Bottom EC": "Water EC TL bottom.",
    "TL Interval Pressure": "Water pressure TL interval.",
    "TL Bottom Pressure": "Water pressure TL bottom.",
    "TL Packer Pressure": "Packer pressure TL.",
    "TL-TEC-INT-U": "Temperature TL interval upper.",
    "TL-TEC-INT-L": "Temperature TL interval lower.",
    "TL-TEC-BOT-U": "Temperature TL bottom upper.",
    "TL-TEC-BOT-L": "Temperature TL bottom lower.",
    "TN Interval Flow": "Production water flow rate TN interval.",
    "TN Bottom Flow": "Production water flow rate TN bottom.",
    "TN Collar Flow": "Production water flow rate TN collar.",
    "TN Interval EC": "Water EC TN interval.",
    "TN Bottom EC": "Water EC TN bottom.",
    "TN Interval Pressure": "Water pressure TN interval.",
    "TN Bottom Pressure": "Water pressure TN bottom.",
    "TN Packer Pressure": "Packer pressure TN.",
    "TN-TEC-INT-U": "Temperature TN interval upper.",
    "TN-TEC-INT-L": "Temperature TN interval lower.",
    "TN-TEC-BOT-U": "Temperature TN bottom upper.",
    "TN-TEC-BOT-L": "Temperature TN bottom lower.",
    "TC Interval Flow": "Production water flow rate TC interval.",
    "TC Bottom Flow": "Production water flow rate TC bottom.",
    "TC Collar Flow": "Production water flow rate TC collar.",
    "TC Interval EC": "Water EC TC interval.",
    "TC Bottom EC": "Water EC TC bottom.",
    "TC Interval Pressure": "Water pressure TC interval.",
    "TC Bottom Pressure": "Water pressure TC bottom.",
    "TC Packer Pressure": "Packer pressure TC.",
    "TC-TEC-INT-U": "Temperature TC interval upper.",
    "TC-TEC-INT-L": "Temperature TC interval lower.",
    "TC-TEC-BOT-U": "Temperature TC bottom upper.",
    "TC-TEC-BOT-L": "Temperature TC bottom lower.",
    "TU Interval Flow": "Production water flow rate TU interval.",
    "TU Bottom Flow": "Production water flow rate TU bottom.",
    "TU Collar Flow": "Production water flow rate TU collar.",
    "TU Interval EC": "Water EC TU interval.",
    "TU Bottom EC": "Water EC TU bottom.",
    "TU Interval Pressure": "Water pressure TU interval.",
    "TU Bottom Pressure": "Water pressure TU bottom.",
    "TU Packer Pressure": "Packer pressure TU.",
    "TU-TEC-INT-U": "Temperature TU interval upper.",
    "TU-TEC-INT-L": "Temperature TU interval lower.",
    "TU-TEC-BOT-U": "Temperature TU bottom upper.",
    "TU-TEC-BOT-L": "Temperature TU bottom lower.",
    "TS Interval Flow": "Production water flow rate TS interval.",
    "TS Bottom Flow": "Production water flow rate TS bottom.",
    "TS Collar Flow": "Production water flow rate TS collar.",
    "TS Interval EC": "Water EC TS interval.",
    "TS Bottom EC": "Water EC TS bottom.",
    "TS Interval Pressure": "Water pressure TS interval.",
    "TS Bottom Pressure": "Water pressure TS bottom.",
    "TS Packer Pressure": "Packer pressure TS.",
    "PT 403": "Pressure transducer reading from sensor PT 403.",
    "PT 503": "Pressure transducer reading from sensor PT 503.",
    "PT 504": "Pressure transducer reading from sensor PT 504.",
    "TL Packer Center Depth": "Center depth of TL packer.",
    "TN Packer Center Depth": "Center depth of TN packer.",
    "TC Packer Center Depth": "Center depth of TC packer.",
    "TU Packer Center Depth": "Center depth of TU packer.",
    "TS Packer Center Depth": "Center depth of TS packer.",
}

# =========================================================
# HELPERS
# =========================================================
def load_table(file_path):
    file_path = Path(file_path)

    print(f"Current working directory: {Path.cwd()}")
    print(f"Resolved input path: {file_path}")

    if not file_path.exists():
        print("\nInput file was not found.")
        print("Files in Data directory:")
        if DATA_DIR.exists():
            for f in DATA_DIR.iterdir():
                print(f" - {f.name}")
        else:
            print(" - Data directory does not exist")
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(file_path, sheet_name=INPUT_SHEET)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def try_parse_datetime(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if series.dtype == object:
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().mean() > 0.8:
            return parsed
    return series


def detect_type(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "string"


def infer_definition(col):
    if col in REFERENCE_DEFINITIONS:
        return REFERENCE_DEFINITIONS[col]

    c = col.lower()

    if "flow" in c:
        return f"Flow measurement for {col}."
    if "pressure" in c or re.match(r"pt\s*\d+", c):
        return f"Pressure measurement for {col}."
    if "ec" in c:
        return f"Electrical conductivity measurement for {col}."
    if "tec" in c:
        return f"Thermocouple temperature measurement for {col}."
    if "depth" in c:
        return f"Depth measurement for {col}."
    if c == "time":
        return "Date/time of the observation."
    return f"Field representing {col}."


def infer_unit(col):
    c = col.lower()

    if c == "time":
        return "datetime"
    if "flow" in c or col == "Net Flow":
        return "L/min"
    if "pressure" in c or re.match(r"pt\s*\d+", c):
        return "psi"
    if "tec" in c:
        return "C"
    if "depth" in c:
        return "Feet"
    if "ec" in c:
        return "EC units not explicitly stated"
    return ""


def summarize_numeric(series):
    s = pd.to_numeric(series, errors="coerce")
    nonnull = s.dropna()

    if nonnull.empty:
        return {
            "Minimum": np.nan,
            "Maximum": np.nan,
            "Mean": np.nan,
            "Median": np.nan,
            "Range": np.nan,
            "Std Dev": np.nan,
            "Count": 0,
            "Missing Count": int(series.isna().sum()),
            "Missing Percent": round(series.isna().mean() * 100, 2),
            "Unique Count": 0,
        }

    return {
        "Minimum": nonnull.min(),
        "Maximum": nonnull.max(),
        "Mean": nonnull.mean(),
        "Median": nonnull.median(),
        "Range": nonnull.max() - nonnull.min(),
        "Std Dev": nonnull.std(),
        "Count": int(nonnull.count()),
        "Missing Count": int(series.isna().sum()),
        "Missing Percent": round(series.isna().mean() * 100, 2),
        "Unique Count": int(nonnull.nunique()),
    }


def summarize_datetime(series):
    s = pd.to_datetime(series, errors="coerce")
    nonnull = s.dropna()

    if nonnull.empty:
        return {
            "Minimum": np.nan,
            "Maximum": np.nan,
            "Mean": np.nan,
            "Median": np.nan,
            "Range": np.nan,
            "Std Dev": np.nan,
            "Count": 0,
            "Missing Count": int(series.isna().sum()),
            "Missing Percent": round(series.isna().mean() * 100, 2),
            "Unique Count": 0,
        }

    vals = nonnull.astype("int64")
    mean_dt = pd.to_datetime(int(vals.mean()))
    median_dt = pd.to_datetime(int(np.median(vals)))

    return {
        "Minimum": nonnull.min(),
        "Maximum": nonnull.max(),
        "Mean": mean_dt,
        "Median": median_dt,
        "Range": nonnull.max() - nonnull.min(),
        "Std Dev": np.nan,
        "Count": int(nonnull.count()),
        "Missing Count": int(series.isna().sum()),
        "Missing Percent": round(series.isna().mean() * 100, 2),
        "Unique Count": int(nonnull.nunique()),
    }


def summarize_text(series):
    nonnull = series.dropna().astype(str)
    examples = ", ".join(nonnull.unique()[:5]) if len(nonnull) > 0 else ""

    return {
        "Minimum": np.nan,
        "Maximum": np.nan,
        "Mean": np.nan,
        "Median": np.nan,
        "Range": np.nan,
        "Std Dev": np.nan,
        "Count": int(nonnull.count()),
        "Missing Count": int(series.isna().sum()),
        "Missing Percent": round(series.isna().mean() * 100, 2),
        "Unique Count": int(nonnull.nunique()),
        "Example Values": examples,
    }


def build_data_dictionary(input_file, output_file):
    df = load_table(input_file)
    df = clean_columns(df)

    # Parse possible datetime columns
    for col in df.columns:
        df[col] = try_parse_datetime(df[col])

    rows = []

    for col in df.columns:
        series = df[col]
        dtype = detect_type(series)
        definition = infer_definition(col)
        unit = infer_unit(col)

        if dtype in ["integer", "float", "numeric"]:
            stats = summarize_numeric(series)
        elif dtype == "datetime":
            stats = summarize_datetime(series)
        else:
            stats = summarize_text(series)

        row = {
            "Term": col,
            "Definition": definition,
            "Type": dtype,
            "Minimum": stats.get("Minimum"),
            "Maximum": stats.get("Maximum"),
            "Mean": stats.get("Mean"),
            "Median": stats.get("Median"),
            "Range": stats.get("Range"),
            "Unit": unit,
            "Std Dev": stats.get("Std Dev"),
            "Count": stats.get("Count"),
            "Missing Count": stats.get("Missing Count"),
            "Missing Percent": stats.get("Missing Percent"),
            "Unique Count": stats.get("Unique Count"),
            "Example Values": stats.get("Example Values", ""),
        }

        rows.append(row)

    dd = pd.DataFrame(rows)

    ordered_cols = [
        "Term",
        "Definition",
        "Type",
        "Minimum",
        "Maximum",
        "Mean",
        "Median",
        "Range",
        "Unit",
        "Std Dev",
        "Count",
        "Missing Count",
        "Missing Percent",
        "Unique Count",
        "Example Values",
    ]
    dd = dd[[c for c in ordered_cols if c in dd.columns]]

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        dd.to_excel(writer, index=False, sheet_name="Data Dictionary")

        summary = pd.DataFrame({
            "Dataset File": [str(input_file)],
            "Rows": [len(df)],
            "Columns": [len(df.columns)]
        })
        summary.to_excel(writer, index=False, sheet_name="Dataset Summary")

    print("\nData dictionary created successfully:")
    print(output_file)


if __name__ == "__main__":
    build_data_dictionary(INPUT_FILE, OUTPUT_FILE)
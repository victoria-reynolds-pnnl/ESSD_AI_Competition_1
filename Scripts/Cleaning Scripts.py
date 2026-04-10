# clean_original_yakima_exact.py

import pandas as pd
import os

# List of all original CSV files provided
FILES = [
    '1 Yakima River at Mabton, WA - USGS-12508990 010125-123125.csv',
    '2 Yakima River at Kiona, WA - USGS-12510500 010125-123125.csv',
    '3 Yakima River Above Ahtanum Creek at Union Gap, WA - USGS-12500450 010125-123125.csv',
    '1 Gage height USGS-12508990 010125-123125.csv',
    '2 Gage height USGS-12510500 010125-123125.csv',
    '3 Gage height USGS-12500450 010125-123125.csv'
]

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exact cleaning steps used previously:
    - Keep only 'time' and 'value' columns
    - Parse 'time' to datetime (coercing errors to NaT)
    - Drop rows where time/value is missing
    - Drop duplicate timestamps
    - Sort by time and reset index
    """
    # Keep only required columns
    df = df[['time', 'value']].copy()

    # Parse datetime
    df['time'] = pd.to_datetime(df['time'], errors='coerce')

    # Drop invalid timestamps or missing values
    df = df.dropna(subset=['time', 'value'])

    # Remove duplicate timestamps
    df = df.drop_duplicates(subset=['time'])

    # Sort chronologically
    df = df.sort_values('time').reset_index(drop=True)

    return df

def main():
    cleaned_files = []
    for file in FILES:
        df = pd.read_csv(file)
        df_clean = clean_df(df)
        cleaned_name = f"cleaned_{os.path.basename(file)}"
        df_clean.to_csv(cleaned_name, index=False)
        cleaned_files.append(cleaned_name)
        print(f"Saved: {cleaned_name}  (rows: {len(df_clean)})")

    print("\nAll cleaned files:\n" + "\n".join(cleaned_files))

if __name__ == "__main__":
    main()
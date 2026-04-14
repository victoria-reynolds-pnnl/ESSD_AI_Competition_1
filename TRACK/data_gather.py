import os
import glob
import pandas as pd

# Run this script from the TRACK project root
INPUT_ROOT = os.path.join("Data", "cleaned_with_features", "csv")
OUTPUT_PATH = os.path.join("Data", "week2_clean.csv")

def main():
    csv_files = glob.glob(os.path.join(INPUT_ROOT, "**", "*.csv"), recursive=True)

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {INPUT_ROOT}")

    print(f"Found {len(csv_files)} CSV files.\n")

    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df["source_file"] = file
            dfs.append(df)
            print(f"Loaded: {file} | shape={df.shape}")
        except Exception as e:
            print(f"Skipping {file} due to error: {e}")

    if not dfs:
        raise ValueError("No CSV files were successfully loaded.")

    combined = pd.concat(dfs, ignore_index=True, sort=False)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"\nCombined dataset saved to: {OUTPUT_PATH}")
    print(f"Final shape: {combined.shape}")
    print("\nColumns:")
    print(combined.columns.tolist())

if __name__ == "__main__":
    main()
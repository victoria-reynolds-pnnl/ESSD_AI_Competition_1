from pathlib import Path
import pandas as pd

# Update these paths after running the notebook.
INPUT_ROW_LEVEL_CSV = Path("Results/week4_llm_simple/<run_timestamp>/row_level_results.csv")
OUTPUT_CLEAN_CSV = Path("SubSignal_Week4/Results/microsoft_phi_3_5_mini_instruct_results_clean_1.csv")


def build_clean_results(df: pd.DataFrame) -> pd.DataFrame:
    core_cols = [
        "window_idx",
        "timestamp",
        "parse_ok",
        "latency_sec",
        "prompt",
        "raw_output",
        "parsed_prediction",
    ]

    value_cols = [
        c
        for c in df.columns
        if c.startswith("true_") or c.startswith("ml_") or c.startswith("llm_")
    ]

    keep_cols = [c for c in core_cols if c in df.columns] + value_cols
    cleaned = df[keep_cols].copy()
    cleaned.insert(0, "model_name", "microsoft/Phi-3.5-mini-instruct")
    cleaned.insert(1, "iteration_number", 1)
    return cleaned


def main() -> None:
    if not INPUT_ROW_LEVEL_CSV.exists():
        raise FileNotFoundError(
            f"Could not find input CSV at: {INPUT_ROW_LEVEL_CSV}. "
            "Update INPUT_ROW_LEVEL_CSV to your run folder path."
        )

    df = pd.read_csv(INPUT_ROW_LEVEL_CSV)
    cleaned = build_clean_results(df)

    OUTPUT_CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_CLEAN_CSV, index=False)

    print(f"Wrote cleaned results: {OUTPUT_CLEAN_CSV}")
    print(f"Rows: {len(cleaned)}")
    print("Columns:")
    for c in cleaned.columns:
        print(f" - {c}")


if __name__ == "__main__":
    main()

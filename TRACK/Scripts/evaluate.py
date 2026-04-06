import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix
)

# =========================
# CONFIG
# =========================
DATA_PATH = "Data/week2_clean.csv"
SPLIT_PATH = "Data/splits_indexed.csv"

MODEL_1_PATH = "Models/model_1.pkl"
MODEL_2_PATH = "Models/model_2.pkl"

HDBSCAN_PREPROCESSOR_PATH = "Models/hdbscan_preprocessor.pkl"
TIER_MAPPER_PATH = "Models/tier_mapper.pkl"

OUTPUT_PATH = "ml_results.csv"

DATE_COL = "centroid_date"

FEATURES = [
    "lowest_temperature_k",
    "duration_days",
    "nerc_id",
    "spatial_coverage",
    "yearly_max_heat_wave_intensity",
    "yearly_max_heat_wave_duration",
    "yearly_max_heat_wave_intensity_trend",
    "yearly_max_heat_wave_duration_trend",
]


# =========================
# HELPERS
# =========================
def load_data():
    return pd.read_csv(DATA_PATH)


def sort_by_time(df):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


def load_splits():
    return pd.read_csv(SPLIT_PATH)


def attach_splits(df, splits_df):
    return df.merge(splits_df[["row_id", "split"]], on="row_id", how="inner")


def assign_tiers(X_raw, unsup_preprocessor, tier_mapper):
    X_transformed = unsup_preprocessor.transform(X_raw)
    tiers = tier_mapper.predict(X_transformed)
    return tiers


def collapse_to_binary_high_risk(df):
    df = df.copy()
    df["high_risk"] = (df["risk_tier_3"] == 2).astype(int)
    return df


def evaluate_model(model, X, y, split_name, model_name):
    preds = model.predict(X)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)
    else:
        probs = None
        auc = np.nan

    f1 = f1_score(y, preds)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    accuracy = accuracy_score(y, preds)

    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()

    return {
        "model": model_name,
        "split": split_name,
        "f1": f1,
        "auc_roc": auc,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "n_rows": len(X)
    }


# =========================
# MAIN
# =========================
def main():
    print("Starting evaluation workflow...")

    required_files = [
        DATA_PATH,
        SPLIT_PATH,
        MODEL_1_PATH,
        MODEL_2_PATH,
        HDBSCAN_PREPROCESSOR_PATH,
        TIER_MAPPER_PATH,
    ]

    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    print("Loading data...")
    df = load_data()

    print("Sorting chronologically...")
    df = sort_by_time(df)

    print("Loading split assignments...")
    splits_df = load_splits()
    df = attach_splits(df, splits_df)

    print("Loading HDBSCAN tiering artifacts...")
    unsup_preprocessor = joblib.load(HDBSCAN_PREPROCESSOR_PATH)
    tier_mapper = joblib.load(TIER_MAPPER_PATH)

    print("Assigning 3-tier labels using saved training-derived mapper...")
    X_all = df[FEATURES].copy()
    df["risk_tier_3"] = assign_tiers(X_all, unsup_preprocessor, tier_mapper)

    print("Collapsing to binary high_risk...")
    df = collapse_to_binary_high_risk(df)

    print("Preparing validation and test sets...")
    val_mask = df["split"] == "val"
    test_mask = df["split"] == "test"

    X_val = df.loc[val_mask, FEATURES].copy()
    y_val = df.loc[val_mask, "high_risk"].copy()

    X_test = df.loc[test_mask, FEATURES].copy()
    y_test = df.loc[test_mask, "high_risk"].copy()

    print(f"Validation rows: {len(X_val)}")
    print(f"Test rows: {len(X_test)}")

    print("Loading trained supervised models...")
    model_1 = joblib.load(MODEL_1_PATH)
    model_2 = joblib.load(MODEL_2_PATH)

    print("Evaluating models...")
    results = [
        evaluate_model(model_1, X_val, y_val, "val", "logistic_regression"),
        evaluate_model(model_1, X_test, y_test, "test", "logistic_regression"),
        evaluate_model(model_2, X_val, y_val, "val", "xgboost"),
        evaluate_model(model_2, X_test, y_test, "test", "xgboost"),
    ]

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved evaluation results to: {OUTPUT_PATH}")
    print(results_df)


if __name__ == "__main__":
    main()
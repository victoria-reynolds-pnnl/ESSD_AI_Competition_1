#!/usr/bin/env python3
"""
Week 3 Training Script - CONUS Fire Occurrence Classification

This script trains LogisticRegression and HistGradientBoostingClassifier models
on the Week 2 cleaned dataset following the locked v1 benchmark specification.

Locked decisions from 02.03 CONUS fire occurrence v1 pipeline plan:
- Target: fire_occurrence (binary, derived from FHS_c9 > 0)
- Split: train=2001-2016, validation=2017-2018, test=2019-2020
- Models: LogisticRegression (baseline), HistGradientBoostingClassifier (main)
- Negative sampling: 10:1 ratio in training only
- Threshold selection: best F1 on validation set

Memory-optimized for large datasets (20M+ rows).
"""

import gc
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_feature_columns() -> list:
    """
    Return the list of predictor columns following the v1 feature set.

    Excludes: date, lat, lon, cell_id, row_id, year, month, split_bucket, fire_occurrence
    """
    # All features from the data dictionary
    feature_cols = [
        "pdsi",
        "spi14d",
        "spi30d",
        "spi90d",
        "spei14d",
        "spei30d",
        "spei90d",
        "air_sfc",
        "air_apcp",
        "soilm",
        "rhum_2m",
        "uwnd",
        "vwnd",
        "LAI",
        "NDVI",
        "FWI",
        "tmax",
        "tmin",
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
    return feature_cols


def load_split_data(
    data_path: Path, split_name: str, feature_cols: list, sample_negatives: bool = False
):
    """
    Load a specific split from the dataset with memory optimization.

    Args:
        data_path: Path to cleaned dataset
        split_name: 'train', 'validation', or 'test'
        feature_cols: List of feature column names
        sample_negatives: If True, apply 10:1 negative sampling (for training only)

    Returns:
        X, y arrays
    """
    print(f"\nLoading {split_name} split...")

    # Define efficient dtypes
    dtype_dict = {
        "year": "int16",
        "month": "int8",
        "fire_occurrence": "int8",
        "pdsi_severe_flag": "int8",
    }

    # Columns to load
    cols_to_load = ["split_bucket", "fire_occurrence", "year", "month"] + feature_cols

    # Load only the needed split in chunks
    chunks = []
    chunksize = 500000

    for chunk in pd.read_csv(
        data_path, usecols=cols_to_load, dtype=dtype_dict, chunksize=chunksize
    ):
        # Filter to the desired split
        split_chunk = chunk[chunk["split_bucket"] == split_name].copy()

        if len(split_chunk) > 0:
            if sample_negatives and split_name == "train":
                # Apply negative sampling within this chunk
                pos = split_chunk[split_chunk["fire_occurrence"] == 1]
                neg = split_chunk[split_chunk["fire_occurrence"] == 0]

                # Sample negatives by year-month
                sampled_neg = []
                for (year, month), group in neg.groupby(["year", "month"]):
                    pos_count = len(
                        pos[(pos["year"] == year) & (pos["month"] == month)]
                    )
                    if pos_count > 0:
                        n_sample = min(len(group), pos_count * 10)
                        sampled = group.sample(n=n_sample, random_state=42)
                        sampled_neg.append(sampled)

                if sampled_neg:
                    split_chunk = pd.concat([pos] + sampled_neg, ignore_index=True)
                else:
                    split_chunk = pos

            chunks.append(split_chunk)

        del chunk, split_chunk
        gc.collect()

    if not chunks:
        raise ValueError(f"No data found for split: {split_name}")

    df = pd.concat(chunks, ignore_index=True)
    print(f"  Loaded {len(df):,} rows")

    # Check class distribution
    pos = (df["fire_occurrence"] == 1).sum()
    neg = (df["fire_occurrence"] == 0).sum()
    print(f"  Positives: {pos:,} ({100*pos/len(df):.2f}%), Negatives: {neg:,}")

    # Extract features and target
    X = df[feature_cols].fillna(0).astype("float32").values
    y = df["fire_occurrence"].values

    del df, chunks
    gc.collect()

    print(f"  X shape: {X.shape}, memory: {X.nbytes / 1024**2:.1f} MB")

    return X, y


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple:
    """
    Find the optimal classification threshold by maximizing F1 score.

    Returns:
        (best_threshold, best_f1, precision_at_threshold, recall_at_threshold)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # Calculate F1 for each threshold
    f1_scores = []
    for p, r in zip(precisions[:-1], recalls[:-1]):
        if p + r > 0:
            f1_scores.append(2 * p * r / (p + r))
        else:
            f1_scores.append(0.0)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]

    return best_threshold, best_f1, best_precision, best_recall


def train_and_evaluate_model(
    model,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: Path,
) -> dict:
    """
    Train a model, find optimal threshold on validation set, and save artifacts.

    Returns:
        Dictionary with model metadata and validation metrics
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    # Train model
    print("Fitting model...")
    model.fit(X_train, y_train)
    print("Training complete.")

    # Get validation predictions
    print("Generating validation predictions...")
    y_val_proba = model.predict_proba(X_val)[:, 1]

    # Find best threshold on validation set
    print("Finding optimal threshold on validation set...")
    best_threshold, best_f1, best_precision, best_recall = find_best_threshold(
        y_val, y_val_proba
    )

    # Calculate validation metrics
    y_val_pred = (y_val_proba >= best_threshold).astype(int)
    val_pr_auc = average_precision_score(y_val, y_val_proba)
    val_roc_auc = roc_auc_score(y_val, y_val_proba)

    print(f"\nValidation Results:")
    print(f"  Optimal threshold: {best_threshold:.4f}")
    print(f"  PR-AUC: {val_pr_auc:.4f}")
    print(f"  ROC-AUC: {val_roc_auc:.4f}")
    print(f"  F1 @ threshold: {best_f1:.4f}")
    print(f"  Precision @ threshold: {best_precision:.4f}")
    print(f"  Recall @ threshold: {best_recall:.4f}")

    # Save model
    model_path = output_dir / f"{model_name}.pkl"
    print(f"\nSaving model to {model_path}...")
    joblib.dump(model, model_path)

    # Prepare metadata
    metadata = {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "model_path": str(model_path),
        "optimal_threshold": float(best_threshold),
        "validation_metrics": {
            "pr_auc": float(val_pr_auc),
            "roc_auc": float(val_roc_auc),
            "f1": float(best_f1),
            "precision": float(best_precision),
            "recall": float(best_recall),
        },
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "n_features": X_train.shape[1],
    }

    return metadata


def save_split_indexed_data(data_path: Path, output_path: Path):
    """
    Save validation and test splits for later evaluation.
    Only saves val+test to conserve memory.
    """
    print("\nSaving split-indexed data (validation + test)...")

    chunks = []
    chunksize = 500000

    for chunk in pd.read_csv(data_path, chunksize=chunksize):
        # Keep only validation and test
        filtered = chunk[chunk["split_bucket"].isin(["validation", "test"])].copy()
        if len(filtered) > 0:
            chunks.append(filtered)
        del chunk, filtered
        gc.collect()

    df = pd.concat(chunks, ignore_index=True)
    df.to_csv(output_path, index=False, compression="gzip")
    print(f"  Saved {len(df):,} rows to {output_path}")

    del df, chunks
    gc.collect()


def main():
    """Main training workflow."""
    print("=" * 60)
    print("Week 3 Training Pipeline - CONUS Fire Occurrence")
    print("=" * 60)

    # Setup paths
    repo_root = Path(__file__).parent.parent.parent
    week2_data_path = repo_root / "week_2" / "Data" / "cleaned_dataset.csv.gz"
    week3_data_dir = repo_root / "week_3" / "Data"
    week3_models_dir = repo_root / "week_3" / "Models"

    week3_data_dir.mkdir(parents=True, exist_ok=True)
    week3_models_dir.mkdir(parents=True, exist_ok=True)

    # Get feature columns
    feature_cols = get_feature_columns()
    print(f"\nUsing {len(feature_cols)} features")

    # Load training data with negative sampling
    print("\n" + "=" * 60)
    print("Loading Training Data")
    print("=" * 60)
    X_train, y_train = load_split_data(
        week2_data_path, "train", feature_cols, sample_negatives=True
    )

    # Load validation data
    print("\n" + "=" * 60)
    print("Loading Validation Data")
    print("=" * 60)
    X_val, y_val = load_split_data(
        week2_data_path, "validation", feature_cols, sample_negatives=False
    )

    # Save split-indexed data for evaluation
    split_indexed_path = week3_data_dir / "split_indexed_data.csv.gz"
    save_split_indexed_data(week2_data_path, split_indexed_path)

    # Copy input reference
    week3_input_path = week3_data_dir / "week_2_data.csv.gz"
    print(f"\nCreating reference link to input data at {week3_input_path}...")
    import shutil

    shutil.copy2(week2_data_path, week3_input_path)

    # Train models
    models_metadata = []

    # 1. Logistic Regression (baseline)
    print("\n" + "=" * 60)
    print("Model 1: Logistic Regression")
    print("=" * 60)
    lr_model = LogisticRegression(
        max_iter=100,
        random_state=42,
        solver="saga",
        class_weight="balanced",
        n_jobs=-1,
        verbose=1,
    )
    lr_metadata = train_and_evaluate_model(
        lr_model,
        "logistic_regression",
        X_train,
        y_train,
        X_val,
        y_val,
        week3_models_dir,
    )
    models_metadata.append(lr_metadata)

    # Free memory
    del lr_model
    gc.collect()

    # 2. Histogram Gradient Boosting (main model)
    print("\n" + "=" * 60)
    print("Model 2: Histogram Gradient Boosting")
    print("=" * 60)
    hgb_model = HistGradientBoostingClassifier(
        max_iter=50,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=5,
        validation_fraction=0.1,
        verbose=1,
    )
    hgb_metadata = train_and_evaluate_model(
        hgb_model,
        "hist_gradient_boosting",
        X_train,
        y_train,
        X_val,
        y_val,
        week3_models_dir,
    )
    models_metadata.append(hgb_metadata)

    # Save training summary
    summary = {
        "dataset": {
            "source": str(week2_data_path),
            "n_features": len(feature_cols),
            "feature_columns": feature_cols,
        },
        "splits": {
            "train": {"years": "2001-2016", "rows": len(X_train)},
            "validation": {"years": "2017-2018", "rows": len(X_val)},
            "test": {"years": "2019-2020", "note": "held out for evaluation"},
        },
        "sampling": {
            "method": "negative_downsampling",
            "ratio": "10:1",
            "applied_to": "train_only",
        },
        "models": models_metadata,
    }

    summary_path = week3_models_dir / "training_summary.json"
    print(f"\nSaving training summary to {summary_path}...")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Training pipeline complete!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  - Models: {week3_models_dir}")
    print(f"  - Split-indexed data: {split_indexed_path}")
    print(f"  - Training summary: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

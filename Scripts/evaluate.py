#!/usr/bin/env python3
"""
Week 3 Evaluation Script - CONUS Fire Occurrence Classification

This script evaluates trained models on the held-out test set and generates
final performance metrics following the locked v1 benchmark specification.

The evaluation uses the optimal thresholds determined during training on the
validation set and reports final test-set performance.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_model_and_metadata(
    model_path: Path, summary_path: Path, model_name: str
) -> tuple:
    """Load trained model and its metadata."""
    print(f"Loading {model_name}...")
    model = joblib.load(model_path)

    with open(summary_path, "r") as f:
        summary = json.load(f)

    # Find model metadata
    model_metadata = None
    for m in summary["models"]:
        if m["model_name"] == model_name:
            model_metadata = m
            break

    if model_metadata is None:
        raise ValueError(f"Model {model_name} not found in training summary")

    return model, model_metadata


def evaluate_model(
    model,
    model_name: str,
    threshold: float,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate model on test set using the pre-determined threshold.

    Returns:
        Dictionary with test metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on Test Set")
    print(f"{'='*60}")

    # Generate predictions
    print("Generating test predictions...")
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= threshold).astype(int)

    # Calculate metrics
    test_pr_auc = average_precision_score(y_test, y_test_proba)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    test_f1 = f1_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\nTest Results (threshold={threshold:.4f}):")
    print(f"  PR-AUC: {test_pr_auc:.4f}")
    print(f"  ROC-AUC: {test_roc_auc:.4f}")
    print(f"  F1: {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:,}  FP: {fp:,}")
    print(f"  FN: {fn:,}  TP: {tp:,}")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=["No Fire", "Fire"]))

    # Prepare results
    results = {
        "model_name": model_name,
        "threshold": float(threshold),
        "test_metrics": {
            "pr_auc": float(test_pr_auc),
            "roc_auc": float(test_roc_auc),
            "f1": float(test_f1),
            "precision": float(test_precision),
            "recall": float(test_recall),
        },
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "test_samples": len(y_test),
        "positive_samples": int((y_test == 1).sum()),
        "negative_samples": int((y_test == 0).sum()),
    }

    return results, y_test_proba, y_test_pred


def main():
    """Main evaluation workflow."""
    print("=" * 60)
    print("Week 3 Evaluation Pipeline - CONUS Fire Occurrence")
    print("=" * 60)

    # Setup paths
    repo_root = Path(__file__).parent.parent.parent
    week3_data_dir = repo_root / "week_3" / "Data"
    week3_models_dir = repo_root / "week_3" / "Models"

    split_indexed_path = week3_data_dir / "split_indexed_data.csv.gz"
    training_summary_path = week3_models_dir / "training_summary.json"

    # Load training summary to get feature columns and thresholds
    print(f"\nLoading training summary from {training_summary_path}...")
    with open(training_summary_path, "r") as f:
        training_summary = json.load(f)

    feature_cols = training_summary["dataset"]["feature_columns"]
    print(f"Using {len(feature_cols)} features from training")

    # Load split-indexed data
    print(f"\nLoading split-indexed data from {split_indexed_path}...")
    df = pd.read_csv(split_indexed_path)
    print(f"Loaded {len(df):,} rows")

    # Extract test set
    test_df = df[df["split_bucket"] == "test"].copy()
    print(f"\nTest set: {len(test_df):,} rows (years 2019-2020)")

    pos = (test_df["fire_occurrence"] == 1).sum()
    neg = (test_df["fire_occurrence"] == 0).sum()
    pos_pct = 100 * pos / len(test_df)
    positive_rate = pos / len(test_df)
    print(f"  Positives: {pos:,} ({pos_pct:.2f}%)")
    print(f"  Negatives: {neg:,}")

    # Prepare test features
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["fire_occurrence"]

    # Evaluate both models
    all_results = []
    predictions_data = []

    model_configs = [
        ("logistic_regression", "logistic_regression.pkl"),
        ("hist_gradient_boosting", "hist_gradient_boosting.pkl"),
    ]

    for model_name, model_filename in model_configs:
        model_path = week3_models_dir / model_filename
        model, metadata = load_model_and_metadata(
            model_path, training_summary_path, model_name
        )

        threshold = metadata["optimal_threshold"]

        results, y_proba, y_pred = evaluate_model(
            model,
            model_name,
            threshold,
            X_test,
            y_test,
        )
        results["data_prep"] = {"nan_fill_value": 0}

        all_results.append(results)

        # Store predictions for later analysis
        predictions_data.append(
            {
                "model_name": model_name,
                "y_proba": y_proba,
                "y_pred": y_pred,
            }
        )

    # Save evaluation results
    evaluation_summary = {
        "test_set": {
            "years": "2019-2020",
            "total_samples": len(test_df),
            "positive_samples": int(pos),
            "negative_samples": int(neg),
            "positive_rate": float(positive_rate),
        },
        "models": all_results,
    }

    eval_summary_path = week3_models_dir / "evaluation_summary.json"
    print(f"\nSaving evaluation summary to {eval_summary_path}...")
    with open(eval_summary_path, "w") as f:
        json.dump(evaluation_summary, f, indent=2)

    # Create ML_results.csv with predictions from both models
    print(f"\nCreating ML_results.csv...")
    ml_results = test_df[
        ["date", "lat", "lon", "cell_id", "row_id", "fire_occurrence"]
    ].copy()

    for pred_data in predictions_data:
        model_name = pred_data["model_name"]
        ml_results[f"{model_name}_proba"] = pred_data["y_proba"]
        ml_results[f"{model_name}_pred"] = pred_data["y_pred"]

    ml_results_path = week3_data_dir / "ML_results.csv"
    ml_results.to_csv(ml_results_path, index=False)
    print(f"  Saved to {ml_results_path}")
    print(f"  Columns: {list(ml_results.columns)}")

    # Print comparison summary
    print("\n" + "=" * 60)
    print("Model Comparison Summary")
    print("=" * 60)
    print(f"\n{'Model':<30} {'PR-AUC':<10} {'ROC-AUC':<10} {'F1':<10}")
    print("-" * 60)
    for result in all_results:
        model_name = result["model_name"]
        pr_auc = result["test_metrics"]["pr_auc"]
        roc_auc = result["test_metrics"]["roc_auc"]
        f1 = result["test_metrics"]["f1"]
        print(f"{model_name:<30} {pr_auc:<10.4f} {roc_auc:<10.4f} {f1:<10.4f}")

    print("\n" + "=" * 60)
    print("Evaluation pipeline complete!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  - Evaluation summary: {eval_summary_path}")
    print(f"  - ML results: {ml_results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

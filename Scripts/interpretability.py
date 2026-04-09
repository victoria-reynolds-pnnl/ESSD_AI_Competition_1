#!/usr/bin/env python3
"""
Week 3 Interpretability Script - CONUS Fire Occurrence Classification (Simplified)

This script generates essential interpretability visualizations for the trained models.
Memory-optimized version that processes data efficiently.
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

warnings.filterwarnings('ignore')


def setup_plot_style():
    """Configure matplotlib style for consistent visualizations."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")


def plot_feature_importance(model, model_name: str, feature_names: list, output_path: Path):
    """Plot feature importance for models that support it."""
    print(f"\nGenerating feature importance plot for {model_name}...")
    
    if hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
        title = f'{model_name.replace("_", " ").title()} - Feature Importance'
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        title = f'{model_name.replace("_", " ").title()} - Feature Importance'
    else:
        print(f"  Model {model_name} does not support feature importance")
        return
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    top_n = min(20, len(importance_df))
    top_features = importance_df.tail(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), top_features['importance'].values)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved to {output_path}")


def plot_pr_and_roc_curves(y_true, y_proba, model_name: str, output_dir: Path):
    """Plot PR and ROC curves together."""
    print(f"\nGenerating PR and ROC curves for {model_name}...")
    
    # Calculate curves
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # PR Curve
    ax1.plot(recall, precision, linewidth=2, label=f'PR-AUC = {pr_auc:.4f}')
    baseline = y_true.sum() / len(y_true)
    ax1.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline = {baseline:.4f}')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title(f'{model_name.replace("_", " ").title()} - Precision-Recall Curve')
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)
    
    # ROC Curve
    ax2.plot(fpr, tpr, linewidth=2, label=f'ROC-AUC = {roc_auc:.4f}')
    ax2.plot([0, 1], [0, 1], 'r--', label='Random')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f'{model_name.replace("_", " ").title()} - ROC Curve')
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"{model_name}_pr_roc_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved to {output_path}")


def plot_confusion_matrix(y_true, y_pred, model_name: str, output_path: Path):
    """Plot confusion matrix heatmap."""
    print(f"\nGenerating confusion matrix for {model_name}...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Fire', 'Fire'],
                yticklabels=['No Fire', 'Fire'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{model_name.replace("_", " ").title()} - Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved to {output_path}")


def plot_model_comparison(results: list, output_path: Path):
    """Create a comparison plot of model performance metrics."""
    print(f"\nGenerating model comparison plot...")
    
    metrics = ['pr_auc', 'roc_auc', 'f1', 'precision', 'recall']
    model_names = [r['model_name'].replace('_', ' ').title() for r in results]
    
    data = {metric: [r['test_metrics'][metric] for r in results] for metric in metrics}
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, model_name in enumerate(model_names):
        values = [data[metric][i] for metric in metrics]
        ax.bar(x + i * width, values, width, label=model_name)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison on Test Set')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([m.replace('_', ' ').upper() for m in metrics])
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved to {output_path}")


def main():
    """Main interpretability workflow."""
    print("="*60)
    print("Week 3 Interpretability Pipeline - CONUS Fire Occurrence")
    print("="*60)
    
    setup_plot_style()
    
    # Setup paths
    repo_root = Path(__file__).parent.parent.parent
    week3_data_dir = repo_root / "week_3" / "Data"
    week3_models_dir = repo_root / "week_3" / "Models"
    week3_viz_dir = repo_root / "week_3" / "Visualizations"
    
    week3_viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Load summaries
    print(f"\nLoading model summaries...")
    
    with open(week3_models_dir / "training_summary.json", 'r') as f:
        training_summary = json.load(f)
    
    with open(week3_models_dir / "evaluation_summary.json", 'r') as f:
        evaluation_summary = json.load(f)
    
    feature_cols = training_summary['dataset']['feature_columns']
    
    # Load ML results (already has predictions)
    print(f"\nLoading ML results...")
    ml_results = pd.read_csv(week3_data_dir / "ML_results.csv")
    print(f"Loaded {len(ml_results):,} test samples")
    
    y_test = ml_results['fire_occurrence'].values
    
    # Process each model
    model_configs = [
        ('logistic_regression', 'logistic_regression.pkl'),
        ('hist_gradient_boosting', 'hist_gradient_boosting.pkl'),
    ]
    
    for model_name, model_filename in model_configs:
        print(f"\n{'='*60}")
        print(f"Generating visualizations for {model_name}")
        print(f"{'='*60}")
        
        # Load model
        model_path = week3_models_dir / model_filename
        model = joblib.load(model_path)
        
        # Get predictions from ML results
        y_proba = ml_results[f'{model_name}_proba'].values
        y_pred = ml_results[f'{model_name}_pred'].values
        
        # Generate visualizations
        plot_feature_importance(
            model, model_name, feature_cols,
            week3_viz_dir / f"{model_name}_feature_importance.png"
        )
        
        plot_pr_and_roc_curves(
            y_test, y_proba, model_name, week3_viz_dir
        )
        
        plot_confusion_matrix(
            y_test, y_pred, model_name,
            week3_viz_dir / f"{model_name}_confusion_matrix.png"
        )
    
    # Generate comparison plot
    plot_model_comparison(
        evaluation_summary['models'],
        week3_viz_dir / "model_comparison.png"
    )
    
    print("\n" + "="*60)
    print("Interpretability pipeline complete!")
    print("="*60)
    
    viz_files = sorted(week3_viz_dir.glob("*.png"))
    print(f"\nGenerated {len(viz_files)} visualization files:")
    for viz_file in viz_files:
        print(f"  - {viz_file.name}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

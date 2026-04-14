"""
interpretability.py — Model output visualizations for interpretability.

Generates three publication-ready figures from the model predictions and the
final trained XGBoost model.

Outputs (all saved to visualizations/):
  obs_vs_pred_timeseries.png  — observed vs predicted time series, all models
  scatter_obs_vs_pred.png     — 1:1 scatter plots with XGBoost uncertainty band
  feature_importance.png      — XGBoost gain-based feature importance

Usage:
  pixi run python scripts/interpretability.py

AI-generated: This script was generated with assistance from Claude (Anthropic).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBRegressor

OUTPUTS_DIR = Path("outputs")
VIZ_DIR = Path("visualizations")
MODELS_DIR = Path("models")
FEATURE_MATRIX = Path("data/clean/feature_matrix.csv")
FIGURE_DPI = 150
STYLE = "seaborn-v0_8-whitegrid"

FEATURE_COLS = [
    "apr1_swe_inches",
    "apr1_swe_anomaly_pct",
    "djf_pdo",
    "djf_nino34",
    "djf_pna",
    "jan_mar_mean_q_cfs",
    "oct_mar_volume_kcfs_days",
]
FEATURE_LABELS = {
    "apr1_swe_inches": "Apr 1 SWE (in)",
    "apr1_swe_anomaly_pct": "Apr 1 SWE Anomaly (%)",
    "djf_pdo": "DJF PDO",
    "djf_nino34": "DJF Nino3.4",
    "djf_pna": "DJF PNA",
    "jan_mar_mean_q_cfs": "Jan-Mar Mean Q (cfs)",
    "oct_mar_volume_kcfs_days": "Oct-Mar Volume (kcfs-days)",
}
TRAIN_CUTOFF = 2012

plt.style.use(STYLE)
VIZ_DIR.mkdir(exist_ok=True)

MODEL_COLORS = {
    "Climatology": "#aec7e8",
    "MLR": "#ff7f0e",
    "XGBoost": "#1f77b4",
    "Observed": "black",
}


def save(fig, name):
    path = VIZ_DIR / name
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def load_data():
    cv = pd.read_csv(OUTPUTS_DIR / "loyo_cv_predictions.csv")
    test = pd.read_csv(OUTPUTS_DIR / "test_predictions.csv")
    all_preds = pd.concat([cv, test], ignore_index=True).sort_values("water_year")
    return cv, test, all_preds


# ---------------------------------------------------------------------------
# Plot 1: Observed vs predicted time series
# ---------------------------------------------------------------------------

def plot_timeseries(all_preds: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5))

    # Shade train/test regions
    ax.axvspan(all_preds["water_year"].min() - 0.5, TRAIN_CUTOFF + 0.5,
               color="#f0f0f0", zorder=0, label="Train (LOYO CV)")
    ax.axvspan(TRAIN_CUTOFF + 0.5, all_preds["water_year"].max() + 0.5,
               color="#fff7cc", zorder=0, label="Test (held out)")
    ax.axvline(TRAIN_CUTOFF + 0.5, color="gray", ls="--", lw=1)

    # XGBoost prediction interval
    ax.fill_between(all_preds["water_year"], all_preds["xgb_q10"], all_preds["xgb_q90"],
                    alpha=0.25, color=MODEL_COLORS["XGBoost"], label="XGBoost q10-q90")

    # Model predictions
    ax.plot(all_preds["water_year"], all_preds["pred_climatology"],
            color=MODEL_COLORS["Climatology"], lw=1.2, ls=":", label="Climatology")
    ax.plot(all_preds["water_year"], all_preds["pred_mlr"],
            color=MODEL_COLORS["MLR"], lw=1.5, ls="--", label="MLR")
    ax.plot(all_preds["water_year"], all_preds["pred_xgb"],
            color=MODEL_COLORS["XGBoost"], lw=2, label="XGBoost")

    # Observed
    ax.plot(all_preds["water_year"], all_preds["observed"],
            "o-", color=MODEL_COLORS["Observed"], lw=1.5, ms=5, label="Observed", zorder=5)

    ax.set_xlabel("Water Year")
    ax.set_ylabel("Apr-Sep Volume (kcfs-days)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.set_title("Observed vs Predicted Apr-Sep Naturalized Flow at The Dalles\n"
                 "(gray = LOYO CV region, yellow = held-out test)")
    ax.legend(ncol=3, fontsize=9, loc="upper left")
    ax.set_xlim(all_preds["water_year"].min() - 0.5, all_preds["water_year"].max() + 0.5)

    save(fig, "obs_vs_pred_timeseries.png")


# ---------------------------------------------------------------------------
# Plot 2: 1:1 scatter — all models side by side
# ---------------------------------------------------------------------------

def plot_scatter(all_preds: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    plot_specs = [
        ("pred_climatology", "Climatology", MODEL_COLORS["Climatology"]),
        ("pred_mlr", "MLR", MODEL_COLORS["MLR"]),
        ("pred_xgb", "XGBoost", MODEL_COLORS["XGBoost"]),
    ]

    obs_all = all_preds["observed"].values
    axis_min = min(obs_all.min(), all_preds[["pred_climatology", "pred_mlr", "pred_xgb"]].min().min()) * 0.92
    axis_max = max(obs_all.max(), all_preds[["pred_climatology", "pred_mlr", "pred_xgb"]].max().max()) * 1.05

    for ax, (pred_col, label, color) in zip(axes, plot_specs):
        pred = all_preds[pred_col].values

        # Separate train/test
        train_mask = all_preds["water_year"] <= TRAIN_CUTOFF
        ax.scatter(obs_all[train_mask], pred[train_mask], color=color, edgecolors="white",
                   s=55, alpha=0.85, label="Train (LOYO CV)", zorder=3)
        ax.scatter(obs_all[~train_mask], pred[~train_mask], color="gold", edgecolors=color,
                   s=70, linewidths=1.5, label="Test", zorder=4)

        if pred_col == "pred_xgb":
            for i, row in all_preds.iterrows():
                ax.plot([row["observed"], row["observed"]], [row["xgb_q10"], row["xgb_q90"]],
                        color=color, alpha=0.3, lw=1.5)

        # 1:1 line
        ax.plot([axis_min, axis_max], [axis_min, axis_max], "k--", lw=1, label="1:1")

        # OLS fit
        slope, intercept, r, _, _ = stats.linregress(obs_all, pred)
        x_fit = np.array([axis_min, axis_max])
        ax.plot(x_fit, slope * x_fit + intercept, color=color, lw=1.5, ls="-",
                label=f"OLS  r={r:.2f}")

        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_aspect("equal")
        ax.set_xlabel("Observed (kcfs-days)")
        if ax == axes[0]:
            ax.set_ylabel("Predicted (kcfs-days)")
        ax.set_title(label)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax.legend(fontsize=8)

    fig.suptitle("Observed vs Predicted — All Models\n(yellow = test years, vertical bars = XGBoost q10-q90)", y=1.01)
    fig.tight_layout()
    save(fig, "scatter_obs_vs_pred.png")


# ---------------------------------------------------------------------------
# Plot 3: XGBoost feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance():
    xgb = XGBRegressor()
    xgb.load_model(MODELS_DIR / "xgb_final.json")

    importance = xgb.get_booster().get_score(importance_type="gain")
    # Map internal feature names (f0, f1, ...) to readable labels
    feat_map = {f"f{i}": col for i, col in enumerate(FEATURE_COLS)}
    importance_named = {FEATURE_LABELS.get(feat_map.get(k, k), feat_map.get(k, k)): v
                        for k, v in importance.items()}

    # Pad missing features with 0
    for col in FEATURE_COLS:
        label = FEATURE_LABELS[col]
        if label not in importance_named:
            importance_named[label] = 0.0

    imp_df = (pd.Series(importance_named)
              .sort_values(ascending=True)
              .rename("gain"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#2171b5" if v > 0 else "#bdbdbd" for v in imp_df.values]
    imp_df.plot.barh(ax=ax, color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("XGBoost Feature Importance (Gain)\nFinal model trained on WY 1985-2012")
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    save(fig, "feature_importance.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for path in [OUTPUTS_DIR / "loyo_cv_predictions.csv",
                 OUTPUTS_DIR / "test_predictions.csv",
                 MODELS_DIR / "xgb_final.json"]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run scripts/train.py first.")

    print("Generating interpretability visualizations...")
    cv, test, all_preds = load_data()

    plot_timeseries(all_preds)
    plot_scatter(all_preds)
    plot_feature_importance()

    print(f"\nAll visualizations saved to {VIZ_DIR.resolve()}")

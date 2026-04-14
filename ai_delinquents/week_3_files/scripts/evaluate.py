"""
evaluate.py — Compute and report model performance metrics.

Reads prediction CSVs from outputs/, computes deterministic and probabilistic
metrics for both LOYO CV and the held-out test set, and writes a summary.

Metrics:
  Deterministic: NSE, KGE, RMSE, MAE, MAPE, Skill score vs climatology
  Probabilistic: CRPS, prediction interval coverage (q10-q90), pinball loss

Outputs:
  outputs/metrics_summary.csv
  outputs/model_performance_summary.txt

Usage:
  pixi run python scripts/evaluate.py

AI-generated: This script was generated with assistance from Claude (Anthropic).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import properscoring as ps

OUTPUTS_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def nse(obs, pred):
    """Nash-Sutcliffe Efficiency. 1=perfect, 0=mean baseline, <0=worse than mean."""
    obs, pred = np.array(obs), np.array(pred)
    return float(1 - np.sum((obs - pred) ** 2) / np.sum((obs - np.mean(obs)) ** 2))


def kge(obs, pred):
    """Kling-Gupta Efficiency. 1=perfect."""
    obs, pred = np.array(obs), np.array(pred)
    r = np.corrcoef(obs, pred)[0, 1]
    alpha = np.std(pred) / np.std(obs)
    beta = np.mean(pred) / np.mean(obs)
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def rmse(obs, pred):
    obs, pred = np.array(obs), np.array(pred)
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def mae(obs, pred):
    obs, pred = np.array(obs), np.array(pred)
    return float(np.mean(np.abs(obs - pred)))


def mape(obs, pred):
    obs, pred = np.array(obs), np.array(pred)
    return float(np.mean(np.abs((obs - pred) / obs)) * 100)


def skill_score(obs, pred, pred_baseline):
    """Skill score relative to baseline: 1 - MSE(model)/MSE(baseline)."""
    obs, pred, pred_baseline = np.array(obs), np.array(pred), np.array(pred_baseline)
    mse_model = np.mean((obs - pred) ** 2)
    mse_base = np.mean((obs - pred_baseline) ** 2)
    return float(1 - mse_model / mse_base)


def pi_coverage(obs, q10, q90):
    """Fraction of observations within the q10-q90 prediction interval."""
    obs, q10, q90 = np.array(obs), np.array(q10), np.array(q90)
    return float(np.mean((obs >= q10) & (obs <= q90)))


def pinball_loss(obs, pred_q, q):
    """Mean pinball (quantile) loss for quantile q."""
    obs, pred_q = np.array(obs), np.array(pred_q)
    err = obs - pred_q
    return float(np.mean(np.where(err >= 0, q * err, (q - 1) * err)))


def crps_from_quantiles(obs, q10, q50, q90):
    """Approximate CRPS using a 3-quantile Gaussian approximation via properscoring."""
    # Fit Gaussian from q10 and q90 (symmetric approximation)
    obs = np.array(obs)
    mu = np.array(q50)
    sigma = (np.array(q90) - np.array(q10)) / (2 * 1.282)  # z-score for 80% PI
    sigma = np.maximum(sigma, 1.0)  # avoid zero sigma
    scores = [ps.crps_gaussian(o, m, s) for o, m, s in zip(obs, mu, sigma)]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Compute metrics for one predictions dataframe
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame, label: str) -> list[dict]:
    """Compute all metrics for LOYO CV or test split; return list of row dicts."""
    obs = df["observed"].values
    clim = df["pred_climatology"].values
    mlr = df["pred_mlr"].values
    xgb = df["pred_xgb"].values
    q10 = df["xgb_q10"].values
    q50 = df["xgb_q50"].values
    q90 = df["xgb_q90"].values

    rows = []
    for model_name, pred in [("climatology", clim), ("mlr", mlr), ("xgb", xgb)]:
        row = {
            "split": label,
            "model": model_name,
            "n_years": len(obs),
            "nse": nse(obs, pred),
            "kge": kge(obs, pred),
            "rmse_kcfs_days": rmse(obs, pred),
            "mae_kcfs_days": mae(obs, pred),
            "mape_pct": mape(obs, pred),
            "skill_vs_climatology": skill_score(obs, pred, clim) if model_name != "climatology" else np.nan,
        }
        if model_name == "xgb":
            row["pi_coverage_q10_q90"] = pi_coverage(obs, q10, q90)
            row["pinball_q10"] = pinball_loss(obs, q10, 0.1)
            row["pinball_q50"] = pinball_loss(obs, q50, 0.5)
            row["pinball_q90"] = pinball_loss(obs, q90, 0.9)
            row["crps"] = crps_from_quantiles(obs, q10, q50, q90)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Format performance summary text
# ---------------------------------------------------------------------------

def format_summary(metrics_df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 65)
    lines.append("MODEL PERFORMANCE SUMMARY — Water Supply Forecasting")
    lines.append("Columbia River at The Dalles, OR  |  Apr-Sep Volume (kcfs-days)")
    lines.append("=" * 65)

    for split_label in ["loyo_cv", "test"]:
        sub = metrics_df[metrics_df["split"] == split_label]
        if sub.empty:
            continue
        header = "LOYO Cross-Validation (WY 1985-2012)" if split_label == "loyo_cv" else "Held-Out Test Set (WY 2013-2018)"
        lines.append(f"\n{header}")
        lines.append("-" * 65)
        lines.append(f"{'Model':<16} {'NSE':>6} {'KGE':>6} {'RMSE':>8} {'MAE':>8} {'MAPE%':>7} {'Skill':>7}")
        for _, row in sub.iterrows():
            skill = f"{row['skill_vs_climatology']:.3f}" if not np.isnan(row.get("skill_vs_climatology", np.nan)) else "  —   "
            lines.append(
                f"{row['model']:<16} {row['nse']:>6.3f} {row['kge']:>6.3f} "
                f"{row['rmse_kcfs_days']:>8,.0f} {row['mae_kcfs_days']:>8,.0f} "
                f"{row['mape_pct']:>6.1f}% {skill:>7}"
            )

        xgb_row = sub[sub["model"] == "xgb"]
        if not xgb_row.empty:
            r = xgb_row.iloc[0]
            lines.append(f"\n  XGBoost probabilistic (q10-q90):")
            lines.append(f"    PI coverage: {r['pi_coverage_q10_q90']:.1%}  (target: ~80%)")
            lines.append(f"    CRPS: {r['crps']:,.1f} kcfs-days")
            lines.append(f"    Pinball loss — q10: {r['pinball_q10']:,.0f}  q50: {r['pinball_q50']:,.0f}  q90: {r['pinball_q90']:,.0f}")

    lines.append("\n" + "=" * 65)
    lines.append("Metric notes:")
    lines.append("  NSE: Nash-Sutcliffe Efficiency (1=perfect, 0=mean baseline)")
    lines.append("  KGE: Kling-Gupta Efficiency (1=perfect, combines r/bias/variance)")
    lines.append("  Skill: 1 - MSE(model)/MSE(climatology baseline)")
    lines.append("  CRPS: Continuous Ranked Probability Score (lower=better)")
    lines.append("=" * 65)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cv_path = OUTPUTS_DIR / "loyo_cv_predictions.csv"
    test_path = OUTPUTS_DIR / "test_predictions.csv"

    if not cv_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Prediction CSVs not found. Run scripts/train.py first."
        )

    cv_df = pd.read_csv(cv_path)
    test_df = pd.read_csv(test_path)

    all_rows = compute_metrics(cv_df, "loyo_cv") + compute_metrics(test_df, "test")
    metrics_df = pd.DataFrame(all_rows)

    out_csv = OUTPUTS_DIR / "metrics_summary.csv"
    metrics_df.to_csv(out_csv, index=False)
    print(f"Saved metrics to {out_csv}")

    summary_text = format_summary(metrics_df)
    print("\n" + summary_text)

    out_txt = OUTPUTS_DIR / "model_performance_summary.txt"
    out_txt.write_text(summary_text)
    print(f"\nSaved summary to {out_txt}")

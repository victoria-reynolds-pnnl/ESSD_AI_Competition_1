# =============================================================================
# XGBoost Pressure Prediction — Baseline vs Kalman Filter
# Mirrors KF workflow: same split, same features, same metrics
# Single model: Injection Pressure, multi-horizon forecasting
# Fixes:
#   1. Absolute pressure target (not delta) — fixes recall cliff beyond +1h
#   2. Threshold-approach features added to prepare()
#   3. Corrected persistence baseline in evaluate_standard()
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
np.random.seed(42)

# =============================================================================
# 0. CONFIGURATION  — mirrors kalman_pressure.py exactly
# =============================================================================
DATA_DIR   = Path("../Data/02_cleaned")
CSV_FILE   = DATA_DIR / "FTES_1hour_cleaned.csv"

PRESSURE_COL       = "Injection Pressure"
INJECTION_FLOW_COL = "Net Flow"
PRODUCER_FLOW_COL  = "TN Interval Flow"

# Same split boundaries as KF
# In CONFIGURATION — extend Train to include first crossing
TRAIN_END = "2025-01-20 00:00:00"   # was Jan 11 — now includes ~3 days above threshold
TEST_END  = "2025-02-01 00:00:00"   # shift accordingly

# Same anomaly window
ANOMALY_START = "2025-01-11 00:00:00"
ANOMALY_END   = "2025-01-11 18:00:00"

# Same operational thresholds
PRESSURE_THRESHOLD = 5000.0
PRESSURE_ALERT     = 4800.0

# Same forecast horizons
HORIZONS = [1, 3, 6, 12, 24]

# Warmup rows to drop from Train evaluation (mirrors KF warmup)
WARMUP_STEPS = 48

# Output directory
VIZ_DIR = Path("../Visualizations/XGB/Pressure")

# XGBoost hyperparameters — conservative defaults for a baseline
XGB_PARAMS = {
    "n_estimators"    : 500,
    "max_depth"       : 4,
    "learning_rate"   : 0.05,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha"       : 0.1,
    "reg_lambda"      : 1.0,
    "random_state"    : 42,
    "n_jobs"          : -1,
    "verbosity"       : 0,
}

# Lag window for autoregressive features
LAG_HOURS  = [1, 2, 3, 6, 12, 24]
ROLL_HOURS = [3, 6, 12, 24]

# =============================================================================
# 1. LOAD & PREPARE
# =============================================================================
def load_data(csv_file: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_file, parse_dates=[0], index_col=0)
    df = df.sort_index()
    print(f"Loaded : {csv_file.name}  →  {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Period : {df.index[0]}  →  {df.index[-1]}")
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Physical features (mirrors KF) + autoregressive lags
    + threshold-approach features (fix for recall cliff).
    Built on full dataset before splitting so rolling windows
    at boundaries are correct.
    """
    cols = [PRESSURE_COL, INJECTION_FLOW_COL, PRODUCER_FLOW_COL]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df[cols].copy()

    # ── same physical features as KF ─────────────────────────────────────────
    out["net_subsurface"] = out[INJECTION_FLOW_COL] - out[PRODUCER_FLOW_COL]
    out["cumulative_net"] = out["net_subsurface"].cumsum()
    out["Q_squared"]      = out[INJECTION_FLOW_COL] ** 2
    out["dQ_dt"]          = out[INJECTION_FLOW_COL].diff().fillna(0)
    out["dP_dt"]          = out[PRESSURE_COL].diff().fillna(0)
    out["dP_dt_6h"]       = (out[PRESSURE_COL].diff(6) / 6).fillna(0)
    out["headroom"]       = PRESSURE_THRESHOLD - out[PRESSURE_COL]

    # ── threshold-approach features (fix 1) ──────────────────────────────────
    # Direct distance signal — tells model how close to threshold
    out["dist_to_threshold"] = PRESSURE_THRESHOLD - out[PRESSURE_COL]

    # Fraction of threshold reached — normalised proximity
    out["pct_of_threshold"]  = out[PRESSURE_COL] / PRESSURE_THRESHOLD

    # Binary flag: currently in alert band
    out["in_alert_band"] = (
        (out[PRESSURE_COL] >= PRESSURE_ALERT) &
        (out[PRESSURE_COL] < PRESSURE_THRESHOLD)
    ).astype(int)

    # Pressure acceleration (second derivative)
    out["d2P_dt2"] = out["dP_dt"].diff().fillna(0)

    # Rolling linear trend slope — is pressure consistently rising?
    out["P_trend_12h"] = (
        out[PRESSURE_COL]
        .rolling(12)
        .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
        .fillna(0)
    )

    # Rolling trend over 24h — slower reservoir signal
    out["P_trend_24h"] = (
        out[PRESSURE_COL]
        .rolling(24)
        .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
        .fillna(0)
    )

    # ── autoregressive lags ───────────────────────────────────────────────────
    for lag in LAG_HOURS:
        out[f"P_lag_{lag}h"]   = out[PRESSURE_COL].shift(lag)
        out[f"Q_lag_{lag}h"]   = out[INJECTION_FLOW_COL].shift(lag)
        out[f"net_lag_{lag}h"] = out["net_subsurface"].shift(lag)

    # ── rolling statistics ────────────────────────────────────────────────────
    for w in ROLL_HOURS:
        out[f"P_roll_mean_{w}h"] = out[PRESSURE_COL].rolling(w).mean()
        out[f"P_roll_std_{w}h"]  = out[PRESSURE_COL].rolling(w).std()
        out[f"Q_roll_mean_{w}h"] = out[INJECTION_FLOW_COL].rolling(w).mean()

    # ── anomaly flag ──────────────────────────────────────────────────────────
    out["anomaly"] = 0
    out.loc[ANOMALY_START:ANOMALY_END, "anomaly"] = 1

    # ── drop rows with NaN in core columns ───────────────────────────────────
    before = len(out)
    out = out.dropna(subset=[PRESSURE_COL, INJECTION_FLOW_COL, PRODUCER_FLOW_COL])
    print(f"Rows after dropna : {len(out)}  (dropped {before - len(out)})")
    print(f"Anomaly rows      : {out['anomaly'].sum()}")
    print(f"Feature count     : {len([c for c in out.columns if c not in ['anomaly']])}")
    return out


def event_split(df: pd.DataFrame, train_end: str, test_end: str):
    """Identical split logic to KF."""
    train    = df.loc[:train_end].copy()
    test     = df.loc[train_end:test_end].copy()
    validate = df.loc[test_end:].copy()

    test     = test.iloc[1:]
    validate = validate.iloc[1:]

    print(f"\nEvent-based split (Split C — mirrors KF):")
    print(f"  Train    : {len(train):>4} rows  "
          f"[{train.index[0]}  →  {train.index[-1]}]")
    print(f"  Test     : {len(test):>4} rows  "
          f"[{test.index[0]}  →  {test.index[-1]}]")
    print(f"  Validate : {len(validate):>4} rows  "
          f"[{validate.index[0]}  →  {validate.index[-1]}]")
    print(f"  ⚠️  Validate locked until final evaluation.")
    return train, test, validate


# =============================================================================
# 2. FEATURE MATRIX BUILDER
# =============================================================================
def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """All numeric columns except target and anomaly flag."""
    exclude = {PRESSURE_COL, "anomaly"}
    return [c for c in df.select_dtypes(include="number").columns
            if c not in exclude]


def build_Xy(df:           pd.DataFrame,
             horizon:      int,
             feature_cols: list[str],
             warmup:       int = 0
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build supervised learning matrices for horizon h.
    X at time t → absolute pressure P at time t+h.
    Anomaly rows at t+h excluded.
    Warmup rows dropped from front.

    Returns
    -------
    X_valid      : np.ndarray  (n_valid, n_features)
    y_valid      : np.ndarray  (n_valid,)  absolute pressure at t+h
    valid_mask   : np.ndarray  (n - warmup - horizon,)  bool
    """
    n       = len(df)
    max_t   = n - horizon

    X_raw   = df[feature_cols].iloc[warmup:max_t].values     # (max_t-warmup, F)
    y_raw   = df[PRESSURE_COL].iloc[warmup + horizon:n].values  # absolute P(t+h)
    anom    = df["anomaly"].iloc[warmup + horizon:n].values

    feat_ok    = ~np.isnan(X_raw).any(axis=1)
    y_ok       = ~np.isnan(y_raw)
    anom_ok    = anom == 0
    valid_mask = feat_ok & y_ok & anom_ok

    return X_raw[valid_mask], y_raw[valid_mask], valid_mask


# =============================================================================
# 3. TRAIN — absolute pressure target (fix 2)
# =============================================================================
def train_models(train_df:     pd.DataFrame,
                 feature_cols: list[str],
                 horizons:     list[int],
                 warmup:       int = WARMUP_STEPS
                 ) -> dict[int, xgb.XGBRegressor]:
    """
    Train one XGBRegressor per forecast horizon.
    Target: absolute pressure P(t+h)  — not delta.
    This prevents the recall cliff caused by delta prediction
    when training and evaluation regimes differ.
    """
    models = {}
    print("\nTraining XGBoost models (absolute pressure target):")

    for h in horizons:
        X_tr, y_tr, _ = build_Xy(train_df, h, feature_cols, warmup=warmup)

        if len(X_tr) == 0:
            print(f"  +{h}h : no training rows — skipping")
            continue

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_tr, y_tr)],
                  verbose=False)

        models[h] = model
        print(f"  +{h}h : trained on {len(X_tr)} rows  "
              f"(target range: {y_tr.min():.1f} → {y_tr.max():.1f} psi)")

    return models


# =============================================================================
# 4. PREDICT
# =============================================================================
def predict_split(split_df:     pd.DataFrame,
                  models:       dict[int, xgb.XGBRegressor],
                  feature_cols: list[str],
                  horizons:     list[int],
                  split_name:   str
                  ) -> dict[int, np.ndarray]:
    """
    Generate predictions for a split.
    Returns {horizon: array of length len(split_df)} with NaN padding —
    same convention as KF run() so metrics and plots are identical.
    """
    n     = len(split_df)
    preds = {h: np.full(n, np.nan) for h in horizons}

    for h in horizons:
        if h not in models:
            continue

        X_valid, _, valid_mask = build_Xy(split_df, h, feature_cols, warmup=0)

        if len(X_valid) == 0:
            continue

        # absolute pressure predictions
        abs_pred = models[h].predict(X_valid)

        # place at t+h positions in output array
        # valid_mask has length (n - h), maps t → bool
        # t+h index in the output array
        t_indices  = np.where(valid_mask)[0]       # positions in [0, n-h)
        th_indices = t_indices + h                 # t+h positions in [0, n)

        for idx, val in zip(th_indices, abs_pred):
            if idx < n:
                preds[h][idx] = val

    return preds


# =============================================================================
# 5. METRICS
# =============================================================================
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def max_err(y_true, y_pred):
    return float(np.max(np.abs(y_true - y_pred)))


def evaluate_standard(pressure:    np.ndarray,
                       preds_total: dict[int, np.ndarray],
                       split_name:  str) -> pd.DataFrame:
    """
    MAE / RMSE / MaxErr + persistence skill score.
    Persistence baseline: P_hat(t+h) = P(t)  — predict no change.
    Fix: P(t) is correctly built by shifting the target array back h steps.
    """
    rows = []
    n    = len(pressure)

    for h in sorted(preds_total):
        yp   = preds_total[h]
        mask = ~np.isnan(yp)

        yp_m = yp[mask]
        yt_m = pressure[mask]              # actual P(t+h)

        # P(t) aligned to each t+h position then masked
        P_t      = np.full(n, np.nan)
        P_t[h:]  = pressure[:n - h]        # P(t) at position t+h
        P_t_m    = P_t[mask]

        # drop first h rows where P_t is NaN
        valid    = ~np.isnan(P_t_m)
        yp_m     = yp_m[valid]
        yt_m     = yt_m[valid]
        P_t_m    = P_t_m[valid]

        if len(yt_m) == 0:
            continue

        mae_persist = mean_absolute_error(yt_m, P_t_m)
        mae_xgb     = mean_absolute_error(yt_m, yp_m)
        skill       = 1.0 - (mae_xgb / mae_persist) if mae_persist > 0 else np.nan

        rows.append({
            "Split"      : split_name,
            "Horizon"    : f"+{h}h",
            "n"          : len(yt_m),
            "MAE"        : round(mae_xgb, 2),
            "RMSE"       : round(rmse(yt_m, yp_m), 2),
            "MaxErr"     : round(max_err(yt_m, yp_m), 2),
            "MAE_persist": round(mae_persist, 2),
            "Skill"      : round(skill, 3),
        })
    return pd.DataFrame(rows)


def evaluate_threshold(pressure:    np.ndarray,
                        preds_total: dict[int, np.ndarray],
                        split_name:  str,
                        threshold:   float = PRESSURE_THRESHOLD,
                        alert_psi:   float = PRESSURE_ALERT) -> pd.DataFrame:
    rows = []
    for h in sorted(preds_total):
        yp   = preds_total[h]
        mask = ~np.isnan(yp)
        yp_m, yt_m = yp[mask], pressure[mask]

        if len(yt_m) == 0:
            continue

        pred_above   = yp_m >= threshold
        actual_above = yt_m >= threshold

        tp = int(np.sum( pred_above &  actual_above))
        fp = int(np.sum( pred_above & ~actual_above))
        fn = int(np.sum(~pred_above &  actual_above))
        tn = int(np.sum(~pred_above & ~actual_above))

        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall    = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1        = (2 * precision * recall / (precision + recall)
                     if not (np.isnan(precision) or np.isnan(recall))
                     and (precision + recall) > 0 else np.nan)

        alert_mask = (yt_m >= alert_psi) & (yt_m < threshold)
        mae_alert  = (mean_absolute_error(yt_m[alert_mask], yp_m[alert_mask])
                      if alert_mask.sum() > 0 else np.nan)

        rows.append({
            "Split"    : split_name,
            "Horizon"  : f"+{h}h",
            "TP"       : tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": round(precision, 3) if not np.isnan(precision) else "—",
            "Recall"   : round(recall, 3)    if not np.isnan(recall)    else "—",
            "F1"       : round(f1, 3)        if not np.isnan(f1)        else "—",
            "MAE_alert": round(mae_alert, 2) if not np.isnan(mae_alert) else "—",
        })
    return pd.DataFrame(rows)


# =============================================================================
# 6. PLOTS
# =============================================================================
def _save(fname: str):
    """Save to VIZ_DIR, creating it if needed."""
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    path = VIZ_DIR / fname
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")


def plot_timeline(train_df:  pd.DataFrame,
                  test_df:   pd.DataFrame,
                  val_df:    pd.DataFrame,
                  all_preds: dict,
                  horizons:  list[int]):

    full_df    = pd.concat([train_df, test_df, val_df])
    full_dates = full_df.index
    full_pres  = full_df[PRESSURE_COL].values

    shading = [
        (train_df.index[0], train_df.index[-1], "#AED6F1", "Train",    0.25, None),
        (test_df.index[0],  test_df.index[-1],  "#A9DFBF", "Test",     0.30, None),
        (val_df.index[0],   val_df.index[-1],   "#F9E79F", "Validate", 0.35, "//"),
    ]
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(horizons)))

    fig, axes = plt.subplots(len(horizons), 1,
                             figsize=(16, 3.5 * len(horizons)),
                             sharex=True)

    for ax, h, color in zip(axes, horizons, colors):
        full_pred = np.concatenate([
            all_preds["Train"][h],
            all_preds["Test"][h],
            all_preds["Validate"][h],
        ])
        for t0, t1, fc, lbl, alpha, hatch in shading:
            ax.axvspan(t0, t1, facecolor=fc, alpha=alpha,
                       hatch=hatch, label=lbl, zorder=0)
        ax.plot(full_dates, full_pres,
                color="steelblue", lw=0.9, label="Observed", alpha=0.85, zorder=2)
        ax.plot(full_dates, full_pred,
                color=color, lw=1.1, ls="--",
                label=f"XGB +{h}h", zorder=3)
        ax.axhline(PRESSURE_THRESHOLD, color="red",    lw=1.2, ls="--", alpha=0.7)
        ax.axhline(PRESSURE_ALERT,     color="orange", lw=0.8, ls=":",  alpha=0.7)
        ax.fill_between(full_dates, PRESSURE_ALERT, PRESSURE_THRESHOLD,
                        alpha=0.06, color="orange", zorder=1)
        for xval in [test_df.index[0], val_df.index[0]]:
            ax.axvline(xval, color="dimgray", lw=1.2, ls=":", zorder=4)
        ax.set_ylabel("Pressure (psi)")
        ax.set_title(f"+{h}h Forecast")
        ax.legend(loc="upper left", fontsize=7, ncol=4)
        ax.grid(True, alpha=0.25)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=25, fontsize=8)
    fig.suptitle("XGBoost – Pressure Forecasts", fontsize=13)
    plt.tight_layout()
    _save("xgb_timeline.png")
    plt.show()


def plot_error_distributions(train_df:  pd.DataFrame,
                              test_df:   pd.DataFrame,
                              val_df:    pd.DataFrame,
                              all_preds: dict,
                              horizons:  list[int]):
    from scipy.stats import gaussian_kde

    splits = [
        ("Train",    train_df,  "#AED6F1", "steelblue"),
        ("Test",     test_df,   "#A9DFBF", "seagreen"),
        ("Validate", val_df,    "#F9E79F", "goldenrod"),
    ]
    n_h, n_s = len(horizons), len(splits)
    fig, axes = plt.subplots(n_s, n_h,
                             figsize=(4 * n_h, 3.5 * n_s),
                             sharex=False)
    fig.suptitle("XGBoost — Error Distributions (Observed − Predicted)", fontsize=12)

    for row, (split_name, split_df, fc, color) in enumerate(splits):
        pressure = split_df[PRESSURE_COL].values
        for col, h in enumerate(horizons):
            ax   = axes[row, col]
            yp   = all_preds[split_name][h]
            mask = ~np.isnan(yp)
            yp_m = yp[mask]
            yt_m = pressure[mask]

            if len(yt_m) == 0:
                ax.set_visible(False)
                continue

            residuals = yt_m - yp_m
            ax.hist(residuals, bins=40, color=fc,
                    edgecolor=color, linewidth=0.5,
                    alpha=0.7, density=True, zorder=2)
            if len(residuals) > 5:
                kde_x = np.linspace(residuals.min(), residuals.max(), 300)
                ax.plot(kde_x, gaussian_kde(residuals)(kde_x),
                        color=color, lw=1.5, zorder=3)

            ax.axvline(0,                    color="black",  lw=1.2, ls="--",
                       zorder=4, label="Zero")
            ax.axvline(residuals.mean(),     color="red",    lw=1.2, ls="-",
                       zorder=4, label=f"μ={residuals.mean():.0f}")
            ax.axvline(np.median(residuals), color="orange", lw=1.0, ls=":",
                       zorder=4, label=f"med={np.median(residuals):.0f}")

            textstr = (f"μ={residuals.mean():.1f}\n"
                       f"σ={residuals.std():.1f}\n"
                       f"med={np.median(residuals):.1f}\n"
                       f"p5={np.percentile(residuals, 5):.0f}\n"
                       f"p95={np.percentile(residuals, 95):.0f}")
            ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
                    fontsize=7, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", alpha=0.7))

            if row == 0:
                ax.set_title(f"+{h}h", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{split_name}\nDensity", fontsize=8)
            if row == n_s - 1:
                ax.set_xlabel("Residual (psi)", fontsize=8)
            ax.legend(fontsize=6, loc="upper left")
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    _save("xgb_error_distributions.png")
    plt.show()


def plot_confusion_matrices(train_df:  pd.DataFrame,
                             test_df:   pd.DataFrame,
                             val_df:    pd.DataFrame,
                             all_preds: dict,
                             horizons:  list[int],
                             threshold: float = PRESSURE_THRESHOLD):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    splits   = [("Train", train_df), ("Test", test_df), ("Validate", val_df)]
    n_h, n_s = len(horizons), len(splits)
    fig, axes = plt.subplots(n_s, n_h, figsize=(3.5 * n_h, 3.5 * n_s))
    fig.suptitle(f"XGBoost — Confusion Matrices  Threshold {threshold:.0f} psi",
                 fontsize=11, y=1.02)

    for row, (split_name, split_df) in enumerate(splits):
        pressure = split_df[PRESSURE_COL].values
        for col, h in enumerate(horizons):
            ax   = axes[row, col]
            yp   = all_preds[split_name][h]
            mask = ~np.isnan(yp)
            yp_m, yt_m = yp[mask], pressure[mask]

            if len(yt_m) == 0:
                ax.set_visible(False)
                continue

            y_true_bin = (yt_m >= threshold).astype(int)
            y_pred_bin = (yp_m >= threshold).astype(int)
            cm   = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
            disp = ConfusionMatrixDisplay(cm, display_labels=["Below", "Above"])
            disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")

            tp, fp = cm[1, 1], cm[0, 1]
            fn, tn = cm[1, 0], cm[0, 0]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            ax.text(0.98, 0.02,
                    f"P={prec:.3f}\nR={rec:.3f}\nF1={f1:.3f}",
                    transform=ax.transAxes, fontsize=7,
                    va="bottom", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="lightyellow", alpha=0.85))

            ax.set_title(f"+{h}h" if row == 0 else "",         fontsize=9)
            ax.set_ylabel(f"{split_name}\nActual" if col == 0 else "", fontsize=8)
            ax.set_xlabel("Predicted" if row == n_s - 1 else "", fontsize=8)

    plt.tight_layout()
    _save("xgb_confusion_matrices.png")
    plt.show()


def plot_feature_importance(models:       dict[int, xgb.XGBRegressor],
                             feature_cols: list[str],
                             horizons:     list[int],
                             top_n:        int = 20):
    """Top-N feature importances per horizon."""
    n_h   = len(horizons)
    fig, axes = plt.subplots(1, n_h, figsize=(5 * n_h, 6), sharey=False)

    for ax, h in zip(axes, horizons):
        if h not in models:
            ax.set_visible(False)
            continue

        importance = models[h].feature_importances_
        idx        = np.argsort(importance)[::-1][:top_n]
        top_names  = [feature_cols[i] for i in idx]
        top_vals   = importance[idx]

        ax.barh(range(top_n), top_vals[::-1], color="steelblue", alpha=0.8)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names[::-1], fontsize=7)
        ax.set_title(f"+{h}h Feature Importance", fontsize=9)
        ax.set_xlabel("Gain", fontsize=8)
        ax.grid(True, alpha=0.2, axis="x")

    plt.suptitle("XGBoost — Top Feature Importances per Horizon", fontsize=11)
    plt.tight_layout()
    _save("xgb_feature_importance.png")
    plt.show()


# =============================================================================
# 7. MAIN
# =============================================================================
def main():
    # ── load & prepare ───────────────────────────────────────────────────────
    raw = load_data(CSV_FILE)
    df  = prepare(raw)

    # ── split ────────────────────────────────────────────────────────────────
    train, test, validate = event_split(df, TRAIN_END, TEST_END)
    feature_cols          = get_feature_cols(df)
    print(f"Feature cols ({len(feature_cols)}): {feature_cols}")

    # ── train ─────────────────────────────────────────────────────────────────
    models = train_models(train, feature_cols, HORIZONS, warmup=WARMUP_STEPS)

    # ── predict train + test (tuning loop) ───────────────────────────────────
    all_preds   = {}
    std_metrics = []
    thr_metrics = []

    for split_name, split_df in [("Train", train), ("Test", test)]:
        preds = predict_split(split_df, models, feature_cols,
                              HORIZONS, split_name)
        all_preds[split_name] = preds
        std_metrics.append(evaluate_standard(
            split_df[PRESSURE_COL].values, preds, split_name))
        thr_metrics.append(evaluate_threshold(
            split_df[PRESSURE_COL].values, preds, split_name))

    # ── final evaluation on validate ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL EVALUATION ON VALIDATE  (run once)")
    print("=" * 65)

    val_preds = predict_split(validate, models, feature_cols,
                              HORIZONS, "Validate")
    all_preds["Validate"] = val_preds
    std_metrics.append(evaluate_standard(
        validate[PRESSURE_COL].values, val_preds, "Validate"))
    thr_metrics.append(evaluate_threshold(
        validate[PRESSURE_COL].values, val_preds, "Validate"))

    # ── print metrics ─────────────────────────────────────────────────────────
    std_df = pd.concat(std_metrics, ignore_index=True)
    thr_df = pd.concat(thr_metrics, ignore_index=True)

    print("\n" + "=" * 65)
    print("  STANDARD METRICS (psi)")
    print("=" * 65)
    print(std_df.to_string(index=False))

    print("\n" + "=" * 65)
    print("  THRESHOLD METRICS")
    print("=" * 65)
    print(thr_df.to_string(index=False))
    print("=" * 65)

    # ── plots ─────────────────────────────────────────────────────────────────
    plot_timeline(train, test, validate, all_preds, HORIZONS)
    plot_error_distributions(train, test, validate, all_preds, HORIZONS)
    plot_confusion_matrices(train, test, validate, all_preds, HORIZONS)
    plot_feature_importance(models, feature_cols, HORIZONS)

    return std_df, thr_df, models, all_preds


if __name__ == "__main__":
    std_metrics, thr_metrics, models, all_preds = main()
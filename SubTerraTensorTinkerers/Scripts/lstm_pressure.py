# =============================================================================
# LSTM Pressure Prediction — Baseline Comparison
# Mirrors KF + XGBoost workflow: same split, same features, same metrics
# Sequence-to-point: window of past observations → P(t+h)
# Fixes applied:
#   1. Smaller model (hidden=32, layers=1) — prevents overfitting on small data
#   2. LSTM-specific feature set — removes redundant lag/roll features
#   3. P_raw added as sequence feature — past pressure in sequence
#   4. Stronger regularization (dropout=0.3, weight_decay=1e-3)
#   5. Shorter sequence length (12) — more training sequences
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
np.random.seed(42)
torch.manual_seed(42)

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
DATA_DIR   = Path("../Data/02_cleaned")
CSV_FILE   = DATA_DIR / "FTES_1hour_cleaned.csv"

PRESSURE_COL       = "Injection Pressure"
INJECTION_FLOW_COL = "Net Flow"
PRODUCER_FLOW_COL  = "TN Interval Flow"

# Same split boundaries
TRAIN_END = "2025-01-20 00:00:00"
TEST_END  = "2025-02-01 00:00:00"

# Same anomaly window
ANOMALY_START = "2025-01-11 00:00:00"
# ANOMALY_END   = "2025-01-11 18:00:00"
ANOMALY_END   = "2025-01-11 00:00:00"



# Same operational thresholds
PRESSURE_THRESHOLD = 5000.0
PRESSURE_ALERT     = 4800.0

# Same forecast horizons
HORIZONS = [1, 3, 6, 12, 24]

# Output directory
VIZ_DIR = Path("../Visualizations/LSTM/Pressure")

# ── LSTM architecture — smaller model for small dataset ───────────────────────
SEQUENCE_LEN  = 12       # hours of history — shorter = more sequences
HIDDEN_SIZE   = 32       # was 64 — smaller to prevent overfitting
NUM_LAYERS    = 1        # was 2 — single layer
DROPOUT       = 0.3      # was 0.2 — stronger regularization

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS        = 150
BATCH_SIZE    = 16       # was 32 — more gradient updates per epoch
LEARNING_RATE = 5e-4     # was 1e-3 — slower learning
PATIENCE      = 20
WEIGHT_DECAY  = 1e-3     # was 1e-4 — stronger L2

# ── LSTM-specific feature set ─────────────────────────────────────────────────
# No redundant lag/roll features — LSTM learns temporal patterns from sequence
# P_raw included so model sees past pressure trajectory in the window
LSTM_FEATURE_COLS = [
    "P_raw",               # past pressure values in sequence (most important)
    "Net Flow",            # injection rate
    "net_subsurface",      # reservoir fill rate (injection - production)
    "Q_squared",           # friction non-linearity
    "dQ_dt",               # rate-of-change of injection
    "dP_dt",               # pressure velocity
    "d2P_dt2",             # pressure acceleration
    "cumulative_net",      # cumulative reservoir fill proxy
    "P_trend_12h",         # medium-term pressure trend slope
    "in_alert_band",       # binary: currently in 4800-5000 psi band
]

HORIZON_SEQ_LEN = {
    1:  12,   # 12h history for 1h ahead
    3:  12,   # 12h history for 3h ahead
    6:  24,   # 24h history for 6h ahead
    12: 24,   # 24h history for 12h ahead
    24: 48,   # 48h history for 24h ahead ← needs enough train data
}

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps"  if torch.backends.mps.is_available() else
                       "cpu")

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
    Feature engineering — same physical base as KF/XGB.
    LSTM-specific additions:
      - P_raw: copy of pressure included as sequence feature
      - Redundant lag/roll features omitted (LSTM learns these from sequence)
    """
    cols = [PRESSURE_COL, INJECTION_FLOW_COL, PRODUCER_FLOW_COL]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df[cols].copy()

    # ── physical features (mirrors KF) ───────────────────────────────────────
    out["net_subsurface"] = out[INJECTION_FLOW_COL] - out[PRODUCER_FLOW_COL]
    out["cumulative_net"] = out["net_subsurface"].cumsum()
    out["Q_squared"]      = out[INJECTION_FLOW_COL] ** 2
    out["dQ_dt"]          = out[INJECTION_FLOW_COL].diff().fillna(0)
    out["dP_dt"]          = out[PRESSURE_COL].diff().fillna(0)
    out["d2P_dt2"]        = out["dP_dt"].diff().fillna(0)

    # ── threshold-approach features ───────────────────────────────────────────
    out["in_alert_band"] = (
        (out[PRESSURE_COL] >= PRESSURE_ALERT) &
        (out[PRESSURE_COL] < PRESSURE_THRESHOLD)
    ).astype(int)

    out["P_trend_12h"] = (
        out[PRESSURE_COL]
        .rolling(12)
        .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
        .fillna(0)
    )

    # ── P_raw: past pressure as sequence feature ──────────────────────────────
    # PRESSURE_COL is excluded from features (it's the target),
    # so we copy it here so the LSTM can see pressure history in the window
    out["P_raw"] = out[PRESSURE_COL]

    # ── anomaly flag ──────────────────────────────────────────────────────────
    out["anomaly"] = 0
    out.loc[ANOMALY_START:ANOMALY_END, "anomaly"] = 1

    before = len(out)
    out = out.dropna(subset=[PRESSURE_COL, INJECTION_FLOW_COL, PRODUCER_FLOW_COL])
    print(f"Rows after dropna : {len(out)}  (dropped {before - len(out)})")
    print(f"Anomaly rows      : {out['anomaly'].sum()}")
    return out


def event_split(df: pd.DataFrame, train_end: str, test_end: str):
    train    = df.loc[:train_end].copy()
    test     = df.loc[train_end:test_end].copy()
    validate = df.loc[test_end:].copy()
    test     = test.iloc[1:]
    validate = validate.iloc[1:]
    print(f"\nEvent-based split (mirrors KF + XGB):")
    print(f"  Train    : {len(train):>4} rows  "
          f"[{train.index[0]}  →  {train.index[-1]}]")
    print(f"  Test     : {len(test):>4} rows  "
          f"[{test.index[0]}  →  {test.index[-1]}]")
    print(f"  Validate : {len(validate):>4} rows  "
          f"[{validate.index[0]}  →  {validate.index[-1]}]")
    print(f"  ⚠️  Validate locked until final evaluation.")
    return train, test, validate


# =============================================================================
# 2. SCALERS
# =============================================================================
def fit_scalers(train_df:     pd.DataFrame,
                feature_cols: list[str]
                ) -> tuple[StandardScaler, StandardScaler]:
    """
    Separate scalers for features and target.
    Fit on Train only — applied to all splits.
    """
    feat_scaler   = StandardScaler()
    target_scaler = StandardScaler()

    feat_scaler.fit(train_df[feature_cols].fillna(0).values)
    target_scaler.fit(train_df[[PRESSURE_COL]].values)

    print(f"\nScalers fit on Train:")
    print(f"  Features : {len(feature_cols)} columns")
    print(f"  Pressure : mean={target_scaler.mean_[0]:.1f}  "
          f"std={target_scaler.scale_[0]:.1f} psi")
    print(f"  Threshold scaled : "
          f"{(PRESSURE_THRESHOLD - target_scaler.mean_[0]) / target_scaler.scale_[0]:.3f}")
    print(f"  Train P max scaled : "
          f"{(train_df[PRESSURE_COL].max() - target_scaler.mean_[0]) / target_scaler.scale_[0]:.3f}")
    return feat_scaler, target_scaler


# =============================================================================
# 3. SEQUENCE DATASET BUILDER
# =============================================================================
def build_sequences(df, feature_cols, feat_scaler, target_scaler,
                    horizon, seq_len=None):
    """
    Build (X_seq, y) pairs.

    X_seq : (N, seq_len, n_features)  — scaled feature sequences
    y     : (N,)                      — scaled absolute pressure at t+horizon
    idx   : (N,)                      — t+horizon positions in df for alignment

    Anomaly rows at t+horizon excluded.
    """
    if seq_len is None:
        seq_len = HORIZON_SEQ_LEN.get(horizon, SEQUENCE_LEN)

    n        = len(df)
    feats    = feat_scaler.transform(df[feature_cols].fillna(0).values)
    pressure = df[PRESSURE_COL].values
    anomaly  = df["anomaly"].values

    X_list, y_list, idx_list = [], [], []

    for t in range(seq_len, n - horizon):
        th = t + horizon
        if anomaly[th] == 1 or np.isnan(pressure[th]):
            continue
        seq = feats[t - seq_len: t]
        X_list.append(seq)
        y_list.append(pressure[th])
        idx_list.append(th)

    if len(X_list) == 0:
        raise ValueError(f"No valid sequences for horizon={horizon}")

    X_arr = np.stack(X_list).astype(np.float32)
    y_arr = target_scaler.transform(
        np.array(y_list).reshape(-1, 1)
    ).ravel().astype(np.float32)

    return (torch.from_numpy(X_arr),
            torch.from_numpy(y_arr),
            np.array(idx_list))


# =============================================================================
# 4. LSTM MODEL
# =============================================================================
class PressureLSTM(nn.Module):
    """
    Single-layer LSTM with small head.
    Kept deliberately small to avoid overfitting on ~850 training sequences.
    """
    def __init__(self,
                 input_size:  int,
                 hidden_size: int   = HIDDEN_SIZE,
                 num_layers:  int   = NUM_LAYERS,
                 dropout:     float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        # dropout applied before head even for single layer
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last        = lstm_out[:, -1, :]
        last        = self.drop(last)
        return self.head(last).squeeze(-1)


# =============================================================================
# 5. TRAIN
# =============================================================================
def train_one_model(X_train:    torch.Tensor,
                    y_train:    torch.Tensor,
                    X_val:      torch.Tensor,
                    y_val:      torch.Tensor,
                    n_features: int,
                    horizon:    int
                    ) -> tuple[PressureLSTM, list[float], list[float]]:
    """
    Train with Adam + ReduceLROnPlateau + early stopping + grad clipping.
    Returns best model (restored from checkpoint) + loss histories.
    """
    model     = PressureLSTM(input_size=n_features).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    best_val_loss = np.inf
    best_state    = None
    patience_ctr  = 0
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_ds)

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(DEVICE))
            val_loss = criterion(val_pred, y_val.to(DEVICE)).item()

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # log every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr_now = optimiser.param_groups[0]["lr"]
            print(f"    ep {epoch+1:>4}  train={epoch_loss:.5f}  "
                  f"val={val_loss:.5f}  lr={lr_now:.2e}")

        # ── early stopping ────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"    Early stop ep={epoch + 1}  "
                      f"best_val={best_val_loss:.5f}")
                break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses


def train_models(train_df:      pd.DataFrame,
                 test_df:       pd.DataFrame,
                 feature_cols:  list[str],
                 feat_scaler:   StandardScaler,
                 target_scaler: StandardScaler,
                 horizons:      list[int]
                 ) -> tuple[dict[int, PressureLSTM], dict]:
    """
    Train one LSTM per horizon.
    Test split used as early-stopping validation only —
    weights restored to best checkpoint before evaluation.
    """
    n_features = len(feature_cols)
    models     = {}
    histories  = {}

    print(f"\nTraining LSTM models  (device={DEVICE}):")

    for h in horizons:
        print(f"\n  +{h}h ──────────────────────────────────────")
        X_tr, y_tr, _ = build_sequences(
            train_df, feature_cols, feat_scaler, target_scaler, h)
        X_va, y_va, _ = build_sequences(
            test_df,  feature_cols, feat_scaler, target_scaler, h)

        print(f"    Train seqs : {len(X_tr)}"
              f"  |  Val seqs : {len(X_va)}")

        model, tr_loss, va_loss = train_one_model(
            X_tr, y_tr, X_va, y_va, n_features, h)

        models[h]    = model
        histories[h] = {"train": tr_loss, "val": va_loss}
        print(f"    Best val loss : {min(va_loss):.5f}  "
              f"at epoch {int(np.argmin(va_loss)) + 1}")

    return models, histories


# =============================================================================
# 6. PREDICT
# =============================================================================
def predict_split(split_df:      pd.DataFrame,
                  models:        dict[int, PressureLSTM],
                  feature_cols:  list[str],
                  feat_scaler:   StandardScaler,
                  target_scaler: StandardScaler,
                  horizons:      list[int]
                  ) -> dict[int, np.ndarray]:
    """
    Returns {horizon: array of length len(split_df)} with NaN padding.
    Same convention as KF and XGB scripts.
    """
    n     = len(split_df)
    preds = {h: np.full(n, np.nan) for h in horizons}

    for h in horizons:
        if h not in models:
            continue

        X_seq, _, idx_arr = build_sequences(
            split_df, feature_cols, feat_scaler, target_scaler, h)

        if len(X_seq) == 0:
            continue

        model = models[h]
        model.eval()
        with torch.no_grad():
            raw_scaled = model(X_seq.to(DEVICE)).cpu().numpy()

        abs_pred = target_scaler.inverse_transform(
            raw_scaled.reshape(-1, 1)).ravel()

        for idx, val in zip(idx_arr, abs_pred):
            if idx < n:
                preds[h][idx] = val

    return preds


# =============================================================================
# 7. METRICS  — identical to KF and XGB
# =============================================================================
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def max_err(y_true, y_pred):
    return float(np.max(np.abs(y_true - y_pred)))


def evaluate_standard(pressure:    np.ndarray,
                       preds_total: dict[int, np.ndarray],
                       split_name:  str) -> pd.DataFrame:
    rows = []
    n    = len(pressure)
    for h in sorted(preds_total):
        yp   = preds_total[h]
        mask = ~np.isnan(yp)
        yp_m = yp[mask]
        yt_m = pressure[mask]

        P_t     = np.full(n, np.nan)
        P_t[h:] = pressure[:n - h]
        P_t_m   = P_t[mask]
        valid   = ~np.isnan(P_t_m)
        yp_m, yt_m, P_t_m = yp_m[valid], yt_m[valid], P_t_m[valid]

        if len(yt_m) == 0:
            continue

        mae_persist = mean_absolute_error(yt_m, P_t_m)
        mae_lstm    = mean_absolute_error(yt_m, yp_m)
        skill       = 1.0 - (mae_lstm / mae_persist) if mae_persist > 0 else np.nan

        rows.append({
            "Split"      : split_name,
            "Horizon"    : f"+{h}h",
            "n"          : len(yt_m),
            "MAE"        : round(mae_lstm, 2),
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
# 8. PLOTS
# =============================================================================
def _save(fname: str):
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    path = VIZ_DIR / fname
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")


def plot_training_curves(histories: dict[int, dict]):
    n_h   = len(histories)
    fig, axes = plt.subplots(1, n_h, figsize=(4 * n_h, 4), sharey=False)
    if n_h == 1:
        axes = [axes]

    for ax, (h, hist) in zip(axes, sorted(histories.items())):
        ax.plot(hist["train"], label="Train", color="steelblue", lw=1.2)
        ax.plot(hist["val"],   label="Val",   color="tomato",    lw=1.2)
        best_ep = int(np.argmin(hist["val"]))
        ax.axvline(best_ep, color="gray", ls=":", lw=1.0,
                   label=f"Best ep={best_ep + 1}")
        # train/val gap annotation
        gap = hist["val"][best_ep] / max(hist["train"][best_ep], 1e-9)
        ax.text(0.97, 0.97, f"gap={gap:.1f}x",
                transform=ax.transAxes, fontsize=8,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="lightyellow", alpha=0.8))
        ax.set_title(f"+{h}h", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("MSE Loss (scaled)", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

    fig.suptitle("LSTM — Training / Validation Loss Curves", fontsize=11)
    plt.tight_layout()
    _save("lstm_training_curves.png")
    plt.show()


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
                color="steelblue", lw=0.9, label="Observed",
                alpha=0.85, zorder=2)
        ax.plot(full_dates, full_pred,
                color=color, lw=1.1, ls="--",
                label=f"LSTM +{h}h", zorder=3)
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
    fig.suptitle("LSTM – Pressure Forecasts", fontsize=13)
    plt.tight_layout()
    _save("lstm_timeline.png")
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
    fig.suptitle("LSTM — Error Distributions (Observed − Predicted)",
                 fontsize=12)

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

            ax.axvline(0,                    color="black",  lw=1.2, ls="--", zorder=4)
            ax.axvline(residuals.mean(),     color="red",    lw=1.2, ls="-",  zorder=4)
            ax.axvline(np.median(residuals), color="orange", lw=1.0, ls=":",  zorder=4)

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
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    _save("lstm_error_distributions.png")
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
    fig.suptitle(f"LSTM — Confusion Matrices  Threshold {threshold:.0f} psi",
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

            ax.set_title(f"+{h}h" if row == 0 else "",          fontsize=9)
            ax.set_ylabel(f"{split_name}\nActual" if col == 0 else "", fontsize=8)
            ax.set_xlabel("Predicted" if row == n_s - 1 else "",  fontsize=8)

    plt.tight_layout()
    _save("lstm_confusion_matrices.png")
    plt.show()


# =============================================================================
# 9. MAIN
# =============================================================================
def main():
    print(f"Device : {DEVICE}")

    # ── load & prepare ───────────────────────────────────────────────────────
    raw = load_data(CSV_FILE)
    df  = prepare(raw)

    # ── split ────────────────────────────────────────────────────────────────
    train, test, validate = event_split(df, TRAIN_END, TEST_END)

    # ── validate feature cols exist ──────────────────────────────────────────
    feature_cols = [c for c in LSTM_FEATURE_COLS if c in df.columns]
    missing_fc   = [c for c in LSTM_FEATURE_COLS if c not in df.columns]
    if missing_fc:
        print(f"  ⚠️  Missing feature cols (skipped): {missing_fc}")
    print(f"\nLSTM features ({len(feature_cols)}): {feature_cols}")

    # ── scalers ───────────────────────────────────────────────────────────────
    feat_scaler, target_scaler = fit_scalers(train, feature_cols)

    # ── train ─────────────────────────────────────────────────────────────────
    models, histories = train_models(
        train, test, feature_cols, feat_scaler, target_scaler, HORIZONS)

    # ── predict train + test ──────────────────────────────────────────────────
    all_preds   = {}
    std_metrics = []
    thr_metrics = []

    for split_name, split_df in [("Train", train), ("Test", test)]:
        preds = predict_split(split_df, models, feature_cols,
                              feat_scaler, target_scaler, HORIZONS)
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
                              feat_scaler, target_scaler, HORIZONS)
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
    plot_training_curves(histories)
    plot_timeline(train, test, validate, all_preds, HORIZONS)
    plot_error_distributions(train, test, validate, all_preds, HORIZONS)
    plot_confusion_matrices(train, test, validate, all_preds, HORIZONS)

    return std_df, thr_df, models, all_preds, histories


if __name__ == "__main__":
    std_metrics, thr_metrics, models, all_preds, histories = main()
# =============================================================================
# Kalman Filter v2 – Flow → Pressure Prediction
# Physical decomposition: fast (friction) + slow (reservoir) components
# Split A: Train=rising phase, Test=at threshold, Validate=stabilizing
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

DATA_DIR   = Path("../Data/02_cleaned")
CSV_FILE   = DATA_DIR / "FTES_1hour_cleaned.csv"

PRESSURE_COL       = "Injection Pressure"
INJECTION_FLOW_COL = "Net Flow"
PRODUCER_FLOW_COL  = "TN Interval Flow"

WARMUP_STEPS = 48   # 2 days at 1h resolution — let state converge

# Option 3 — replace in CONFIGURATION section
TRAIN_END = "2025-01-11 00:00:00"   # up to anomaly
TEST_END  = "2025-01-22 00:00:00"   # through first crossing

TRAIN_END = "2025-01-20 00:00:00"   # was Jan 11 — now includes ~3 days above threshold
TEST_END  = "2025-02-01 00:00:00"   # shift accordingly

# And adjust anomaly handling — Train now ENDS at anomaly start
# so anomaly flag in Train will be minimal/zero
# The anomaly itself falls in Test — good, that's what you want to evaluate
ANOMALY_START = "2025-01-11 00:00:00"
ANOMALY_END   = "2025-01-11 18:00:00"   # keep extended buffer

# Operational threshold
PRESSURE_THRESHOLD = 5000.0   # psi
PRESSURE_ALERT     = 4800.0   # psi — early warning line

# Forecast horizons
HORIZONS = [1, 3, 6, 12, 24]

# Alert lead time required (hours) — primary performance metric
REQUIRED_LEAD_HOURS = 6

# Kalman hyper-parameters
Q_FRICTION   = 5.0    # process noise — fast friction component
Q_RESERVOIR  = 0.01   # process noise — slow reservoir component (changes slowly)
Q_TREND      = 0.001  # process noise — reservoir trend
R_MEAS       = 2.0    # measurement noise (psi²)
P0_SCALE     = 100.0  # initial state uncertainty

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
    Select columns, engineer features, flag anomaly window.
    """
    cols = [PRESSURE_COL, INJECTION_FLOW_COL, PRODUCER_FLOW_COL]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df[cols].copy()

    # ── core engineered features ─────────────────────────────────────────────

    # Net subsurface flow (injection − production) → reservoir fill rate
    out["net_subsurface"] = out[INJECTION_FLOW_COL] - out[PRODUCER_FLOW_COL]

    # Cumulative net subsurface flow → reservoir pressure proxy
    # Reset at start so it begins at 0
    out["cumulative_net"] = out["net_subsurface"].cumsum()

    # Injection rate squared → friction is non-linear
    out["Q_squared"] = out[INJECTION_FLOW_COL] ** 2

    # Rate of change of injection flow → detects planned reductions
    out["dQ_dt"] = out[INJECTION_FLOW_COL].diff().fillna(0)

    # Pressure derivatives
    out["dP_dt"]    = out[PRESSURE_COL].diff().fillna(0)
    out["dP_dt_6h"] = (out[PRESSURE_COL].diff(6) / 6).fillna(0)

    # Pressure headroom
    out["headroom"] = PRESSURE_THRESHOLD - out[PRESSURE_COL]

    # ── anomaly flag (Jan 11 injection pause) ────────────────────────────────
    out["anomaly"] = 0
    out.loc[ANOMALY_START:ANOMALY_END, "anomaly"] = 1

    # ── drop rows with NaN in core columns ───────────────────────────────────
    before = len(out)
    out = out.dropna(subset=[PRESSURE_COL,
                              INJECTION_FLOW_COL,
                              PRODUCER_FLOW_COL])
    print(f"Rows after dropna: {len(out)}  (dropped {before - len(out)})")
    print(f"Anomaly rows flagged: {out['anomaly'].sum()}")
    return out


def event_split(df: pd.DataFrame,
                train_end: str,
                test_end:  str):
    """
    Split A:
      Train    : start → first threshold crossing  (rising pressure)
      Test     : first crossing → Jan 29            (at threshold, reductions)
      Validate : Jan 29 → end                       (stabilizing)
    """
    train    = df.loc[:train_end].copy()
    test     = df.loc[train_end:test_end].copy()
    validate = df.loc[test_end:].copy()

    # remove overlap at boundaries
    test     = test.iloc[1:]
    validate = validate.iloc[1:]

    print(f"\nEvent-based split (Split A):")
    print(f"  Train    : {len(train):>4} rows  "
          f"[{train.index[0]}  →  {train.index[-1]}]")
    print(f"  Test     : {len(test):>4} rows  "
          f"[{test.index[0]}  →  {test.index[-1]}]")
    print(f"  Validate : {len(validate):>4} rows  "
          f"[{validate.index[0]}  →  {validate.index[-1]}]")
    print(f"  ⚠️  Validate locked until final evaluation.")
    return train, test, validate

# =============================================================================
# 2. PHYSICAL KALMAN FILTER
# =============================================================================

class PhysicalPressureKF:
    """
    Two-component pressure model:

      P_total = P_friction + P_reservoir

    State vector x (3,):
      x[0] = P_friction   — fast component, driven by Q²
      x[1] = P_reservoir  — slow component, driven by cumulative net flow
      x[2] = dP_res/dt    — reservoir pressure trend

    Transition:
      P_friction_{k+1}  = α·Q²_{k+1}              (direct input mapping)
      P_reservoir_{k+1} = P_reservoir_k + dP_res_k·dt + β·net_flow_k
      dP_res_{k+1}      = dP_res_k                 (trend persists)

    Measurement:
      z = P_total = P_friction + P_reservoir  →  H = [1, 1, 0]

    Control inputs u:
      u[0] = Q²           (injection friction)
      u[1] = net_subsurface flow  (reservoir fill rate)
    """

    def __init__(self,
                 q_friction:  float = 5.0,
                 q_reservoir: float = 0.01,
                 q_trend:     float = 0.001,
                 r_meas:      float = 2.0,
                 p0_scale:    float = 100.0):

        self.p0_scale = p0_scale

        # ── state transition F (3×3) ─────────────────────────────────────────
        # P_friction is overwritten by B·u each step (α·Q²)
        # P_reservoir integrates trend
        # trend is constant
        self.F = np.array([
            [0.0, 0.0, 0.0],   # P_friction  — fully driven by input
            [0.0, 1.0, 1.0],   # P_reservoir = P_res + trend
            [0.0, 0.0, 1.0],   # trend       = trend (random walk)
        ])

        # ── process noise Q (3×3) ────────────────────────────────────────────
        self.Q = np.diag([q_friction, q_reservoir, q_trend])

        # ── measurement matrix H (1×3) ───────────────────────────────────────
        # We observe P_friction + P_reservoir (not the trend directly)
        self.H = np.array([[1.0, 1.0, 0.0]])

        # ── measurement noise R (1×1) ────────────────────────────────────────
        self.R = np.array([[r_meas]])

        # ── control input matrix B (3×2) ─────────────────────────────────────
        # u = [Q², net_subsurface_flow]
        # B[0,0] = α : Q² → P_friction
        # B[1,1] = β : net_flow → P_reservoir increment
        self.B = np.zeros((3, 2))
        # α and β fitted on training data
        self.alpha = 0.0   # friction coefficient: P_friction = α·Q²
        self.beta  = 0.0   # reservoir coefficient

        # ── scaler for inputs ─────────────────────────────────────────────────
        self.scaler = StandardScaler()
        self.fitted = False

        # ── state ─────────────────────────────────────────────────────────────
        self.x = np.zeros(3)
        self.P = np.eye(3) * p0_scale

    # ── initialise ───────────────────────────────────────────────────────────

    def reset(self, initial_pressure: float, initial_Q: float = 0.0):
        p_fric = self.alpha * initial_Q ** 2
        p_res  = initial_pressure - p_fric
        self.x = np.array([p_fric, max(p_res, 0.0), 0.0])  # ← clamp P_res ≥ 0
        self.P = np.eye(3) * self.p0_scale

    # ── fit α and β on training data ─────────────────────────────────────────

    def fit(self,
            pressure:      np.ndarray,
            Q_inj:         np.ndarray,
            net_flow:      np.ndarray,
            cumulative_net: np.ndarray,
            anomaly_mask:  np.ndarray | None = None) -> "PhysicalPressureKF":
        """
        Estimate physical coefficients from training data:
          α : P_friction = α · Q²    (OLS on stable periods)
          β : ΔP_reservoir = β · cumulative_net_flow

        Anomaly rows excluded from fitting.
        """
        mask = np.ones(len(pressure), dtype=bool)
        if anomaly_mask is not None:
            mask &= (anomaly_mask == 0)

        # ── fit α: regress P on Q² at early time ────────────────────────────
        # Use first 20% of training (before reservoir buildup dominates)
        n_early = max(int(len(pressure) * 0.20), 10)
        early   = mask.copy()
        early[n_early:] = False

        Q2_early = (Q_inj[early] ** 2).reshape(-1, 1)
        P_early  = pressure[early]

        # simple OLS: P ≈ α·Q²  (no intercept — physical constraint)
        alpha, _, _, _ = np.linalg.lstsq(Q2_early, P_early, rcond=None)
        self.alpha = float(alpha[0])

        # ── fit β: regress residual P on cumulative net flow ─────────────────
        P_friction_all = self.alpha * Q_inj ** 2
        P_residual     = pressure - P_friction_all   # ≈ P_reservoir

        cum_net = cumulative_net[mask].reshape(-1, 1)
        P_res   = P_residual[mask]

        # OLS: P_reservoir ≈ β·cumulative_net + intercept
        X_ols     = np.column_stack([cum_net,
                                     np.ones(len(cum_net))])
        coeffs, _, _, _ = np.linalg.lstsq(X_ols, P_res, rcond=None)
        self.beta      = float(coeffs[0])
        self._P_res_0  = float(coeffs[1])   # intercept

        # ── update B matrix ───────────────────────────────────────────────────
        self.B[0, 0] = self.alpha   # Q² → P_friction
        self.B[1, 1] = self.beta    # net_flow → P_reservoir increment

        # ── fit scaler on training inputs ────────────────────────────────────
        inputs = np.column_stack([Q_inj ** 2, net_flow])
        self.scaler.fit(inputs[mask])
        self.fitted = True

        print(f"\nFitted physical coefficients:")
        print(f"  α (friction)   : {self.alpha:.4f}  "
              f"→ P_friction = {self.alpha:.4f} × Q²")
        print(f"  β (reservoir)  : {self.beta:.6f}  "
              f"→ ΔP_res per unit cumulative net flow")
        print(f"  P_res intercept: {self._P_res_0:.2f} psi")

        # sanity check: at Q=2.5, P_friction ≈ ?
        print(f"\n  Sanity check:")
        for q in [1.5, 2.0, 2.5]:
            print(f"    Q={q:.1f} → P_friction ≈ "
                  f"{self.alpha * q**2:.1f} psi")
        return self

    # ── Kalman steps ──────────────────────────────────────────────────────────

    def _build_u(self, Q_inj: float, net_flow: float) -> np.ndarray:
        """Build control input vector."""
        return np.array([Q_inj ** 2, net_flow])

    def _predict_step(self,
                      x: np.ndarray,
                      P: np.ndarray,
                      u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_pred = self.F @ x + self.B @ u
        P_pred = self.F @ P @ self.F.T + self.Q
        return x_pred, P_pred

    def _update_step(self,
                    x_pred: np.ndarray,
                    P_pred: np.ndarray,
                    z: float) -> tuple[np.ndarray, np.ndarray]:
        z_vec = np.array([[z]])
        S     = self.H @ P_pred @ self.H.T + self.R
        K     = P_pred @ self.H.T @ np.linalg.inv(S)
        innov = z_vec - self.H @ x_pred          # shape (1,1) - (1,3)@(3,) = (1,)
        x_upd = x_pred + (K @ innov).ravel()
        P_upd = (np.eye(3) - K @ self.H) @ P_pred
        return x_upd, P_upd

    # ── multi-step forecast ───────────────────────────────────────────────────

    def forecast(self,
                 horizon:          int,
                 future_Q:         np.ndarray,
                 future_net_flow:  np.ndarray) -> tuple[float, float]:
        """
        Propagate h steps without new measurements.

        future_Q, future_net_flow: arrays of length >= horizon

        Returns (predicted_total_pressure, predicted_reservoir_component)
        """
        x_f = self.x.copy()
        P_f = self.P.copy()

        for h in range(horizon):
            q   = future_Q[h]        if h < len(future_Q)        else future_Q[-1]
            nf  = future_net_flow[h] if h < len(future_net_flow) else future_net_flow[-1]
            u   = self._build_u(q, nf)
            x_f, P_f = self._predict_step(x_f, P_f, u)

        p_total = max(float(x_f[0] + x_f[1]), 0.0)   # friction + reservoir
        p_res   = float(x_f[1])
        return p_total, p_res

    # ── run filter over dataset ───────────────────────────────────────────────

    def run(self,
            pressure:     np.ndarray,
            Q_inj:        np.ndarray,
            net_flow:     np.ndarray,
            anomaly:      np.ndarray,
            horizons:     list[int] | None = None,
            warmup_steps: int = 0
            ) -> dict[str, dict[int, np.ndarray]]:
        """
        Run Kalman filter over dataset.

        Parameters
        ----------
        pressure     : observed pressure array
        Q_inj        : injection flow rate array
        net_flow     : net subsurface flow array (injection - production)
        anomaly      : binary flag array (1 = skip measurement update)
        horizons     : list of forecast horizons in hours
        warmup_steps : number of steps to run filter before recording predictions
                    (state converges but predictions are not stored)

        Returns
        -------
        dict with keys:
            "total"     : {horizon -> prediction array}  total pressure forecasts
            "reservoir" : {horizon -> prediction array}  reservoir component forecasts
            "state"     : {feature -> array}             internal KF state features for ML
        """
        if horizons is None:
            horizons = [1]

        n = len(pressure)

        # ── output arrays ────────────────────────────────────────────────────────
        preds = {h: np.full(n, np.nan) for h in horizons}
        p_res = {h: np.full(n, np.nan) for h in horizons}

        # internal state arrays for ML feature export
        state_friction  = np.full(n, np.nan)   # x[0]: fast friction component
        state_reservoir = np.full(n, np.nan)   # x[1]: slow reservoir component
        state_trend     = np.full(n, np.nan)   # x[2]: reservoir pressure trend
        state_P_var     = np.full(n, np.nan)   # P[1,1]: reservoir state uncertainty
        innov_arr       = np.full(n, np.nan)   # innovation z - H·x_pred

        # ── initialise state ─────────────────────────────────────────────────────
        self.reset(pressure[0], Q_inj[0])

        # ── warmup phase ─────────────────────────────────────────────────────────
        # Run filter forward without recording — lets state converge
        # before predictions are stored. Avoids startup transient spikes.
        warmup_steps = min(warmup_steps, n - 1)
        if warmup_steps > 0:
            for k in range(warmup_steps):
                u              = self._build_u(Q_inj[k], net_flow[k])
                x_pred, P_pred = self._predict_step(self.x, self.P, u)
                if anomaly[k + 1]:
                    self.x, self.P = x_pred, P_pred
                else:
                    self.x, self.P = self._update_step(
                        x_pred, P_pred, pressure[k + 1])
            print(f"  Warmup consumed {warmup_steps} steps — "
                f"recording from index {warmup_steps} "
                f"({pressure[warmup_steps]:.1f} psi at start of recording)")

        # ── main filter loop ──────────────────────────────────────────────────────
        for k in range(warmup_steps, n):

            # ── record current state (before this step's update) ─────────────────
            state_friction[k]  = float(self.x[0])
            state_reservoir[k] = float(self.x[1])
            state_trend[k]     = float(self.x[2])
            state_P_var[k]     = float(self.P[1, 1])

            # ── forecast h steps ahead from current state ─────────────────────────
            for h in horizons:
                if k + h < n:
                    fut_Q  = Q_inj[k: k + h]
                    fut_nf = net_flow[k: k + h]
                    pt, pr = self.forecast(h, fut_Q, fut_nf)
                    preds[h][k] = pt
                    p_res[h][k] = pr

            # ── advance state one step ────────────────────────────────────────────
            if k < n - 1:
                u              = self._build_u(Q_inj[k], net_flow[k])
                x_pred, P_pred = self._predict_step(self.x, self.P, u)

                # record innovation before update (or skip if anomaly)
                if anomaly[k + 1]:
                    # do not update on anomalous measurement — trust prediction
                    innov_arr[k + 1] = np.nan
                    self.x, self.P   = x_pred, P_pred
                else:
                    innov = pressure[k + 1] - float((self.H @ x_pred)[0])
                    innov_arr[k + 1]   = innov
                    self.x, self.P     = self._update_step(
                        x_pred, P_pred, pressure[k + 1])

        return {
            "total"     : preds,
            "reservoir" : p_res,
            "state"     : {
                "P_friction"  : state_friction,
                "P_reservoir" : state_reservoir,
                "trend"       : state_trend,
                "uncertainty" : state_P_var,
                "innovation"  : innov_arr,
            },
        }

# =============================================================================
# 3. METRICS  — standard + threshold-focused
# =============================================================================
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def max_err(y_true, y_pred):
    return float(np.max(np.abs(y_true - y_pred)))

def evaluate_standard(pressure:    np.ndarray,
                       preds_total: dict[int, np.ndarray],
                       split_name:  str) -> pd.DataFrame:
    """MAE / RMSE / MaxErr / Persistence Skill per horizon."""
    rows = []
    for h in sorted(preds_total):
        n_valid = len(pressure) - h
        if n_valid <= 0:
            continue
        yp   = preds_total[h][:n_valid]
        yt   = pressure[h: h + n_valid]
        mask = ~np.isnan(yp)
        yp, yt = yp[mask], yt[mask]
        if len(yt) == 0:
            continue

        # persistence baseline: predict P(t+h) = P(t)
        yt_now      = pressure[:n_valid][mask]
        mae_persist = mean_absolute_error(yt, yt_now)
        mae_kf      = mean_absolute_error(yt, yp)
        skill       = 1.0 - (mae_kf / mae_persist)

        rows.append({
            "Split"      : split_name,
            "Horizon"    : f"+{h}h",
            "n"          : len(yt),
            "MAE"        : round(mae_kf, 2),
            "RMSE"       : round(rmse(yt, yp), 2),
            "MaxErr"     : round(max_err(yt, yp), 2),
            "MAE_persist": round(mae_persist, 2),
            "Skill"      : round(skill, 3),
        })
    return pd.DataFrame(rows)


def evaluate_threshold(pressure:    np.ndarray,
                        preds_total: dict[int, np.ndarray],
                        dates:       pd.DatetimeIndex,
                        split_name:  str,
                        threshold:   float = PRESSURE_THRESHOLD,
                        alert_psi:   float = PRESSURE_ALERT,
                        lead_hours:  int   = REQUIRED_LEAD_HOURS
                        ) -> pd.DataFrame:
    """
    Threshold-focused metrics:
      - True Positive  : predicted P >= threshold AND actual P >= threshold
      - False Positive : predicted P >= threshold AND actual P <  threshold
      - Miss (FN)      : predicted P <  threshold AND actual P >= threshold
      - Precision, Recall, MAE in alert band
    """
    rows = []
    for h in sorted(preds_total):
        n_valid = len(pressure) - h
        if n_valid <= 0:
            continue
        yp   = preds_total[h][:n_valid]
        yt   = pressure[h: h + n_valid]
        mask = ~np.isnan(yp)
        yp, yt = yp[mask], yt[mask]
        if len(yt) == 0:
            continue

        pred_above   = yp >= threshold
        actual_above = yt >= threshold

        tp = int(np.sum( pred_above &  actual_above))
        fp = int(np.sum( pred_above & ~actual_above))
        fn = int(np.sum(~pred_above &  actual_above))
        tn = int(np.sum(~pred_above & ~actual_above))

        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall    = tp / (tp + fn) if (tp + fn) > 0 else np.nan

        # MAE specifically in the alert band [alert_psi, threshold)
        alert_mask = (yt >= alert_psi) & (yt < threshold)
        mae_alert  = (mean_absolute_error(yt[alert_mask], yp[alert_mask])
                      if alert_mask.sum() > 0 else np.nan)

        rows.append({
            "Split"    : split_name,
            "Horizon"  : f"+{h}h",
            "TP"       : tp,
            "FP"       : fp,
            "FN"       : fn,
            "TN"       : tn,
            "Precision": round(precision, 3) if not np.isnan(precision) else "—",
            "Recall"   : round(recall, 3)    if not np.isnan(recall)    else "—",
            "MAE_alert": round(mae_alert, 2) if not np.isnan(mae_alert) else "—",
        })
    return pd.DataFrame(rows)


# =============================================================================
# 4. EXPORT
# =============================================================================
def export_kf_features(train:     pd.DataFrame,
                        test:      pd.DataFrame,
                        validate:  pd.DataFrame,
                        all_preds: dict) -> pd.DataFrame:
    """
    Attach KF state features to dataframe for ML ingestion.
    Exports one CSV containing all splits with a 'split' column label.

    Columns added:
        kf_P_friction   — fast friction component of state
        kf_P_reservoir  — slow reservoir component of state
        kf_trend        — reservoir pressure trend (dP_res/dt)
        kf_uncertainty  — P[1,1] reservoir state covariance
        kf_innovation   — measurement residual z - H·x_pred
        kf_forecast_Xh  — X-step ahead total pressure forecast
    """
    frames = []
    for split_name, split_df in [("Train",    train),
                                  ("Test",     test),
                                  ("Validate", validate)]:
        out   = split_df.copy()
        state = all_preds[split_name]["state"]

        # internal state features
        out["kf_P_friction"]  = state["P_friction"]
        out["kf_P_reservoir"] = state["P_reservoir"]
        out["kf_trend"]       = state["trend"]
        out["kf_uncertainty"] = state["uncertainty"]
        out["kf_innovation"]  = state["innovation"]

        # h-step ahead forecasts as features
        for h in HORIZONS:
            out[f"kf_forecast_{h}h"] = all_preds[split_name]["total"][h]

        out["split"] = split_name
        frames.append(out)

    full     = pd.concat(frames)
    out_path = Path("../Data/03_features/kf_features.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_path)

    kf_cols = [c for c in full.columns if c.startswith("kf_")]
    print(f"\nKF features exported → {out_path}")
    print(f"  Shape    : {full.shape}")
    print(f"  KF cols  : {kf_cols}")
    print(f"  NaN rows : {full[kf_cols].isna().any(axis=1).sum()} "
          f"(warmup + horizon tail — expected)")
    return full

# =============================================================================
# 4. PLOTS
# =============================================================================

def plot_full_timeline(train_df:   pd.DataFrame,
                       test_df:    pd.DataFrame,
                       val_df:     pd.DataFrame,
                       all_preds:  dict[str, dict[str, dict[int, np.ndarray]]],
                       horizons:   list[int],
                       title:      str = "KF v2 – Physical Pressure Model"):
    """
    Full timeline: observed pressure + forecasts, shaded splits,
    threshold lines, and reservoir component overlay.
    """
    full_df    = pd.concat([train_df, test_df, val_df])
    full_dates = full_df.index
    full_pres  = full_df[PRESSURE_COL].values

    # stitch predictions
    full_total = {}
    full_res   = {}
    for h in horizons:
        full_total[h] = np.concatenate([
            all_preds["Train"]["total"][h],
            all_preds["Test"]["total"][h],
            all_preds["Validate"]["total"][h],
        ])
        full_res[h] = np.concatenate([
            all_preds["Train"]["reservoir"][h],
            all_preds["Test"]["reservoir"][h],
            all_preds["Validate"]["reservoir"][h],
        ])

    shading = [
        (train_df.index[0], train_df.index[-1], "#AED6F1", "Train",    0.25, None),
        (test_df.index[0],  test_df.index[-1],  "#A9DFBF", "Test",     0.30, None),
        (val_df.index[0],   val_df.index[-1],   "#F9E79F", "Validate", 0.35, None),
    ]

    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(horizons)))
    n      = len(horizons)
    fig, axes = plt.subplots(n + 1, 1,
                             figsize=(16, 3.5 * (n + 1)),
                             sharex=True)

    # ── top panel: reservoir component ───────────────────────────────────────
    ax = axes[0]
    for t0, t1, fc, lbl, alpha, hatch in shading:
        ax.axvspan(t0, t1, facecolor=fc, alpha=alpha,
                   hatch=hatch, label=f"{lbl}", zorder=0)
    ax.plot(full_dates, full_pres,
            color="steelblue", lw=0.9, label="Observed P", zorder=2)
    ax.plot(full_dates[1:], full_res[1][:-1],
            color="darkgreen", lw=1.0, ls="-.",
            label="Reservoir component (+1h)", alpha=0.8, zorder=3)
    ax.axhline(PRESSURE_THRESHOLD, color="red",
               lw=1.5, ls="--", label=f"Threshold {PRESSURE_THRESHOLD}")
    ax.axhline(PRESSURE_ALERT, color="orange",
               lw=1.0, ls=":", label=f"Alert {PRESSURE_ALERT}")

    # mark anomaly
    ax.axvspan(pd.Timestamp(ANOMALY_START),
               pd.Timestamp(ANOMALY_END),
               color="gray", alpha=0.4, label="Anomaly (excluded)")
    ax.set_ylabel("Pressure (psi)")
    ax.set_title("Pressure + Reservoir Component")
    ax.legend(loc="upper left", fontsize=7, ncol=4)
    ax.grid(True, alpha=0.25)

    # ── per-horizon forecast panels ───────────────────────────────────────────
    for ax, h, color in zip(axes[1:], horizons, colors):
        for t0, t1, fc, lbl, alpha, hatch in shading:
            ax.axvspan(t0, t1, facecolor=fc, alpha=alpha,
                       hatch=hatch, zorder=0)

        n_full    = len(full_pres)
        n_valid   = n_full - h
        pred_vals = full_total[h][:n_valid]
        pred_dates = full_dates[h: h + n_valid]

        ax.plot(full_dates, full_pres,
                color="steelblue", lw=0.9, label="Observed", alpha=0.85, zorder=2)
        ax.plot(pred_dates, pred_vals,
                color=color, lw=1.1, ls="--",
                label=f"KF +{h}h", zorder=3)
        ax.axhline(PRESSURE_THRESHOLD, color="red",
                   lw=1.2, ls="--", alpha=0.7)
        ax.axhline(PRESSURE_ALERT, color="orange",
                   lw=0.8, ls=":", alpha=0.7)

        # shade alert zone
        ax.fill_between(pred_dates, PRESSURE_ALERT, PRESSURE_THRESHOLD,
                        alpha=0.06, color="orange", zorder=1)

        # split lines
        for xval, lbl in [(test_df.index[0],  "Test"),
                          (val_df.index[0],   "Validate")]:
            ax.axvline(xval, color="dimgray", lw=1.2, ls=":", zorder=4)
            ax.text(xval, ax.get_ylim()[1] if ax.get_ylim()[1] != 1 else 1,
                    f" {lbl}", fontsize=7, color="dimgray",
                    va="top", ha="left")

        ax.set_ylabel("Pressure (psi)")
        ax.set_title(f"+{h}h Forecast")
        ax.legend(loc="upper left", fontsize=7, ncol=3)
        ax.grid(True, alpha=0.25)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=25, fontsize=8)
    fig.suptitle(title, fontsize=13, y=1.002)
    plt.tight_layout()
    plt.savefig("../Visualizations/KalmanFilter/Pressure/kf_v2_timeline.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → kf_v2_timeline.png")


def plot_threshold_analysis(train_df:  pd.DataFrame,
                             test_df:   pd.DataFrame,
                             val_df:    pd.DataFrame,
                             all_preds: dict):
    """
    Focus plot: pressure headroom + alert flags per horizon.
    Shows whether model provides sufficient lead time before threshold.
    """
    full_df    = pd.concat([train_df, test_df, val_df])
    full_pres  = full_df[PRESSURE_COL].values
    full_dates = full_df.index

    horizons_plot = [1, 6, 12, 24]
    colors        = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    # ── ax0: pressure + threshold + per-horizon alerts ────────────────────────
    ax = axes[0]
    ax.plot(full_dates, full_pres,
            color="steelblue", lw=1.0, label="Observed", zorder=3)
    ax.axhline(PRESSURE_THRESHOLD, color="red",
               lw=1.5, ls="--", label=f"Threshold {PRESSURE_THRESHOLD} psi")
    ax.axhline(PRESSURE_ALERT, color="orange",
               lw=1.0, ls=":", label=f"Alert {PRESSURE_ALERT} psi")

    for h, color in zip(horizons_plot, colors):
        full_pred = np.concatenate([
            all_preds["Train"]["total"][h],
            all_preds["Test"]["total"][h],
            all_preds["Validate"]["total"][h],
        ])
        n_valid    = len(full_pres) - h
        pred_vals  = full_pred[:n_valid]
        pred_dates = full_dates[h: h + n_valid]

        # only show where prediction exceeds alert
        alert_flag = pred_vals >= PRESSURE_ALERT
        if alert_flag.any():
            ax.scatter(pred_dates[alert_flag],
                       pred_vals[alert_flag],
                       s=4, color=color, alpha=0.5,
                       label=f"+{h}h pred ≥ {PRESSURE_ALERT}", zorder=4)

    ax.set_ylabel("Pressure (psi)")
    ax.set_title("Threshold Approach — Forecast Alerts Overlay")
    ax.legend(loc="upper left", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.25)

    # ── ax1: headroom over time ───────────────────────────────────────────────
    ax = axes[1]
    headroom = PRESSURE_THRESHOLD - full_pres
    ax.plot(full_dates, headroom,
            color="steelblue", lw=1.0, label="Actual headroom")
    ax.fill_between(full_dates, headroom, 0,
                    where=headroom < 0,
                    color="red", alpha=0.3, label="Over threshold")
    ax.fill_between(full_dates, headroom, 0,
                    where=(headroom >= 0) & (headroom < 200),
                    color="orange", alpha=0.3, label="< 200 psi headroom")
    ax.axhline(0, color="red", lw=1.5, ls="--")
    ax.axhline(200, color="orange", lw=1.0, ls=":")
    ax.set_ylabel("Headroom (psi)")
    ax.set_xlabel("Date")
    ax.set_title("Pressure Headroom to Threshold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=25, fontsize=8)
    plt.tight_layout()
    plt.savefig("../Visualizations/KalmanFilter/Pressure/kf_v2_threshold.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → kf_v2_threshold.png")


def plot_physical_components(train_df: pd.DataFrame,
                              test_df:  pd.DataFrame,
                              val_df:   pd.DataFrame):
    """
    Plot the physical features used to drive the model.
    Validates that engineered features make physical sense.
    """
    full_df    = pd.concat([train_df, test_df, val_df])

    fig, axes  = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Physical Features Driving the Model", fontsize=12)

    panels = [
        (PRESSURE_COL,      "Injection Pressure (psi)",       "steelblue"),
        (INJECTION_FLOW_COL,"Injection Flow Rate",             "navy"),
        ("net_subsurface",  "Net Subsurface Flow (Inj−Prod)",  "green"),
        ("cumulative_net",  "Cumulative Net Flow (reservoir proxy)", "darkgreen"),
    ]

    for ax, (col, ylabel, color) in zip(axes, panels):
        ax.plot(full_df.index, full_df[col], color=color, lw=0.9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.25)

        if col == PRESSURE_COL:
            ax.axhline(PRESSURE_THRESHOLD, color="red",
                       lw=1.2, ls="--", alpha=0.7)
            ax.axhline(PRESSURE_ALERT, color="orange",
                       lw=0.8, ls=":", alpha=0.7)

        # shade splits
        for t0, t1, fc, alpha in [
            (train_df.index[0], train_df.index[-1], "#AED6F1", 0.2),
            (test_df.index[0],  test_df.index[-1],  "#A9DFBF", 0.2),
            (val_df.index[0],   val_df.index[-1],   "#F9E79F", 0.2),
        ]:
            ax.axvspan(t0, t1, facecolor=fc, alpha=alpha, zorder=0)

        # anomaly
        ax.axvspan(pd.Timestamp(ANOMALY_START),
                   pd.Timestamp(ANOMALY_END),
                   color="gray", alpha=0.4)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=25, fontsize=8)
    plt.tight_layout()
    plt.savefig("../Visualizations/KalmanFilter/Pressure/kf_v2_features.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → kf_v2_features.png")

def plot_error_distributions(train_df:   pd.DataFrame,
                              test_df:    pd.DataFrame,
                              val_df:     pd.DataFrame,
                              all_preds:  dict,
                              horizons:   list[int]):
    """
    Error distribution (residuals) per horizon, per split.
    Shows: histogram + KDE + mean/std annotations.
    """
    splits = [
        ("Train",    train_df,  "#AED6F1", "steelblue"),
        ("Test",     test_df,   "#A9DFBF", "seagreen"),
        ("Validate", val_df,    "#F9E79F", "goldenrod"),
    ]

    n_horiz = len(horizons)
    n_splits = len(splits)

    fig, axes = plt.subplots(n_splits, n_horiz,
                             figsize=(4 * n_horiz, 3.5 * n_splits),
                             sharex=False, sharey=False)
    fig.suptitle("Forecast Error Distributions (Residuals = Observed − Predicted)",
                 fontsize=12, y=1.01)

    for row, (split_name, split_df, fc, color) in enumerate(splits):
        pressure = split_df[PRESSURE_COL].values
        preds    = all_preds[split_name]["total"]

        for col, h in enumerate(horizons):
            ax = axes[row, col]

            n_valid = len(pressure) - h
            if n_valid <= 0:
                ax.set_visible(False)
                continue

            yp   = preds[h][:n_valid]
            yt   = pressure[h: h + n_valid]
            mask = ~np.isnan(yp)
            yp, yt = yp[mask], yt[mask]

            if len(yt) == 0:
                ax.set_visible(False)
                continue

            residuals = yt - yp   # positive = under-prediction

            # ── histogram ────────────────────────────────────────────────────
            ax.hist(residuals, bins=40, color=fc,
                    edgecolor=color, linewidth=0.5,
                    alpha=0.7, density=True, zorder=2)

            # ── KDE overlay ──────────────────────────────────────────────────
            from scipy.stats import gaussian_kde
            if len(residuals) > 5:
                kde_x = np.linspace(residuals.min(), residuals.max(), 300)
                kde   = gaussian_kde(residuals)
                ax.plot(kde_x, kde(kde_x), color=color, lw=1.5, zorder=3)

            # ── reference lines ───────────────────────────────────────────────
            ax.axvline(0,                  color="black",  lw=1.2, ls="--",
                       zorder=4, label="Zero error")
            ax.axvline(residuals.mean(),   color="red",    lw=1.2, ls="-",
                       zorder=4, label=f"Mean={residuals.mean():.1f}")
            ax.axvline(np.median(residuals), color="orange", lw=1.0, ls=":",
                zorder=4, label=f"Med={np.median(residuals):.1f}")

            # ── annotations ──────────────────────────────────────────────────
            textstr = (f"μ={residuals.mean():.1f} psi\n"
                       f"σ={residuals.std():.1f} psi\n"
                       f"p5={np.percentile(residuals, 5):.0f}\n"
                       f"p95={np.percentile(residuals, 95):.0f}")
            ax.text(0.97, 0.97, textstr,
                    transform=ax.transAxes,
                    fontsize=7, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", alpha=0.7))

            # ── labels ───────────────────────────────────────────────────────
            if row == 0:
                ax.set_title(f"+{h}h Forecast", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{split_name}\nDensity", fontsize=8)
            if row == n_splits - 1:
                ax.set_xlabel("Residual (psi)", fontsize=8)

            ax.legend(fontsize=6, loc="upper left")
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("../Visualizations/KalmanFilter/Pressure/kf_v2_error_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → kf_v2_error_distributions.png")


def plot_confusion_matrices(train_df:   pd.DataFrame,
                             test_df:    pd.DataFrame,
                             val_df:     pd.DataFrame,
                             all_preds:  dict,
                             horizons:   list[int],
                             threshold:  float = PRESSURE_THRESHOLD):
    """
    Confusion matrix for threshold exceedance at each forecast horizon.
    Rows = splits, Cols = horizons.
    Also prints precision / recall / F1 per cell.
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    splits = [
        ("Train",    train_df),
        ("Test",     test_df),
        ("Validate", val_df),
    ]

    n_horiz  = len(horizons)
    n_splits = len(splits)

    fig, axes = plt.subplots(n_splits, n_horiz,
                             figsize=(3.5 * n_horiz, 3.5 * n_splits))
    fig.suptitle(f"Confusion Matrices — Threshold {threshold:.0f} psi\n"
                 f"(Positive = predicted/actual exceeds threshold)",
                 fontsize=11, y=1.02)

    for row, (split_name, split_df) in enumerate(splits):
        pressure = split_df[PRESSURE_COL].values
        preds    = all_preds[split_name]["total"]

        for col, h in enumerate(horizons):
            ax = axes[row, col]

            n_valid = len(pressure) - h
            if n_valid <= 0:
                ax.set_visible(False)
                continue

            yp   = preds[h][:n_valid]
            yt   = pressure[h: h + n_valid]
            mask = ~np.isnan(yp)
            yp, yt = yp[mask], yt[mask]

            if len(yt) == 0:
                ax.set_visible(False)
                continue

            y_true_bin = (yt >= threshold).astype(int)
            y_pred_bin = (yp >= threshold).astype(int)

            # ── confusion matrix ──────────────────────────────────────────────
            cm   = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Below", "Above"]
            )
            disp.plot(ax=ax, colorbar=False, cmap="Blues",
                      values_format="d")

            # ── precision / recall / F1 ───────────────────────────────────────
            tp = cm[1, 1]
            fp = cm[0, 1]
            fn = cm[1, 0]
            tn = cm[0, 0]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)

            textstr = (f"P={precision:.3f}\n"
                       f"R={recall:.3f}\n"
                       f"F1={f1:.3f}")
            ax.text(0.98, 0.02, textstr,
                    transform=ax.transAxes,
                    fontsize=7, va="bottom", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="lightyellow", alpha=0.85))

            # ── titles / labels ───────────────────────────────────────────────
            if row == 0:
                ax.set_title(f"+{h}h", fontsize=9)
            else:
                ax.set_title("")

            if col == 0:
                ax.set_ylabel(f"{split_name}\nActual", fontsize=8)
            else:
                ax.set_ylabel("")

            if row == n_splits - 1:
                ax.set_xlabel("Predicted", fontsize=8)
            else:
                ax.set_xlabel("")

    plt.tight_layout()
    plt.savefig("../Visualizations/KalmanFilter/Pressure/kf_v2_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → kf_v2_confusion_matrices.png")

# =============================================================================
# 5. MAIN
# =============================================================================

def main():
    # ── load & prepare ───────────────────────────────────────────────────────
    raw = load_data(CSV_FILE)
    df  = prepare(raw)

    # ── split ────────────────────────────────────────────────────────────────
    train, test, validate = event_split(df, TRAIN_END, TEST_END)

    # ── build & fit model on TRAIN ───────────────────────────────────────────
    kf = PhysicalPressureKF(
        q_friction  = Q_FRICTION,
        q_reservoir = Q_RESERVOIR,
        q_trend     = Q_TREND,
        r_meas      = R_MEAS,
        p0_scale    = P0_SCALE,
    )
    kf.fit(
        pressure       = train[PRESSURE_COL].values,
        Q_inj          = train[INJECTION_FLOW_COL].values,
        net_flow       = train["net_subsurface"].values,
        cumulative_net = train["cumulative_net"].values,
        anomaly_mask   = train["anomaly"].values,
    )

    # ── evaluate train + test (tuning loop) ──────────────────────────────────
    all_preds   = {}
    std_metrics = []
    thr_metrics = []

    for split_name, split_df in [("Train", train), ("Test", test)]:
        preds = kf.run(
            pressure     = split_df[PRESSURE_COL].values,
            Q_inj        = split_df[INJECTION_FLOW_COL].values,
            net_flow     = split_df["net_subsurface"].values,
            anomaly      = split_df["anomaly"].values,
            horizons     = HORIZONS,
            warmup_steps = WARMUP_STEPS if split_name == "Train" else 0,
        )
        all_preds[split_name] = preds
        std_metrics.append(evaluate_standard(
            split_df[PRESSURE_COL].values, preds["total"], split_name))
        thr_metrics.append(evaluate_threshold(
            split_df[PRESSURE_COL].values, preds["total"],
            split_df.index, split_name))

    # ── FINAL evaluation on VALIDATE ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL EVALUATION ON VALIDATE  (run once)")
    print("=" * 65)

    val_preds = kf.run(
        pressure  = validate[PRESSURE_COL].values,
        Q_inj     = validate[INJECTION_FLOW_COL].values,
        net_flow  = validate["net_subsurface"].values,
        anomaly   = validate["anomaly"].values,
        horizons  = HORIZONS,
    )
    all_preds["Validate"] = val_preds
    std_metrics.append(evaluate_standard(
        validate[PRESSURE_COL].values, val_preds["total"], "Validate"))
    thr_metrics.append(evaluate_threshold(
        validate[PRESSURE_COL].values, val_preds["total"],
        validate.index, "Validate"))

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

    # ── export KF features for ML ─────────────────────────────────────────────
    kf_features = export_kf_features(train, test, validate, all_preds)

    # ── plots ─────────────────────────────────────────────────────────────────
    plot_physical_components(train, test, validate)
    plot_full_timeline(train, test, validate, all_preds, HORIZONS)
    plot_threshold_analysis(train, test, validate, all_preds)
    plot_error_distributions(train, test, validate, all_preds, HORIZONS)  
    plot_confusion_matrices(train, test, validate, all_preds, HORIZONS) 

    return std_df, thr_df, kf, kf_features


if __name__ == "__main__":
    std_metrics, thr_metrics, kf, kf_features = main() 
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

################# Configuration #################

MODEL = "xgb"
PREDICT_DELTA = False  # Train on y(t+1)-y(t) instead of y(t+1) to anchor predictions to persistence
# Narrow down to a few key targets for interpretability. Valid interval metrics were selected
VIZ_TARGETS = ["Injection Pressure", "TC Interval Pressure", "TL Interval Flow", "TN Interval Flow", "TN-TEC-INT-L", "TN-TEC-INT-U"]

################# Paths #################

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if PREDICT_DELTA:
    SAVE_DIR = Path(r"../Models") / f"{MODEL}_delta"
else:
    SAVE_DIR = Path(r"../Models") / f"{MODEL}_original"

EVAL_META = SAVE_DIR / "eval_output"
PRED_FILE = EVAL_META / f"{MODEL}_test_predictions.csv"
METRICS_FILE = EVAL_META / f"{MODEL}_test_metrics.csv"

if PREDICT_DELTA:
    VIZ_DIR = Path(r"../Visualizations") / f"{MODEL}_delta"
else:
    VIZ_DIR = Path(r"../Visualizations") / f"{MODEL}_original"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

###########################################################

def rmse(y_true, y_pred):
	return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def pick_targets(pred_df):
	actual_cols = [c for c in pred_df.columns if c.endswith("__actual")]
	targets = [c.replace("__actual", "") for c in actual_cols]
	return [t for t in targets if t in VIZ_TARGETS]


def plot_target_time_series(pred_df, target, out_path):
	actual = pred_df[f"{target}__actual"]
	pred = pred_df[f"{target}__pred"]
	baseline = pred_df[f"{target}__baseline_pred"] if f"{target}__baseline_pred" in pred_df.columns else None

	fig, ax = plt.subplots(figsize=(15, 5))
	ax.plot(actual.index, actual.values, label="Actual", linewidth=1.2, color="darkblue")
	if baseline is not None:
		ax.plot(baseline.index, baseline.values, label="Baseline", linewidth=1.2, color="green")
	ax.plot(pred.index, pred.values, label="XGB", linewidth=1, color="red")
	ax.set_title(f"{target} - Actual vs Predicted")
	ax.set_ylabel(target)
	ax.grid(alpha=0.3)
	ax.legend()
	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)


def plot_actual_vs_pred_scatter(pred_df, target, out_path):
	actual = pred_df[f"{target}__actual"].values
	pred = pred_df[f"{target}__pred"].values

	fig, ax = plt.subplots(figsize=(6, 6))
	ax.scatter(actual, pred, s=14, alpha=0.5)
	ax.set_title(f"{target} - Actual vs Predicted")
	ax.set_xlabel("Actual")
	ax.set_ylabel("Predicted")
	ax.grid(alpha=0.3)
	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)


def plot_rolling_errors(pred_df, target, out_path, window=24):
	actual = pred_df[f"{target}__actual"]
	pred = pred_df[f"{target}__pred"]

	abs_err = (actual - pred).abs()
	roll_mae = abs_err.rolling(window=window, min_periods=max(4, window // 3)).mean()

	err2 = (actual - pred) ** 2
	roll_rmse = np.sqrt(err2.rolling(window=window, min_periods=max(4, window // 3)).mean())

	fig, (ax_mae, ax_rmse) = plt.subplots(2, 1, figsize=(15, 7), sharex=True)

	ax_mae.plot(roll_mae.index, roll_mae.values, linewidth=1.3)
	ax_mae.set_title(f"{target} - Rolling MAE (smoothed {window}h sliding window)")
	ax_mae.set_ylabel("MAE")
	ax_mae.grid(alpha=0.3)

	ax_rmse.plot(roll_rmse.index, roll_rmse.values, linewidth=1.3, color="tab:orange")
	ax_rmse.set_title(f"{target} - Rolling RMSE (smoothed {window}h sliding window)")
	ax_rmse.set_ylabel("RMSE")
	ax_rmse.grid(alpha=0.3)

	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)


def main():

	if not PRED_FILE.exists():
		raise FileNotFoundError(f"Missing predictions file: {PRED_FILE}")
	if not METRICS_FILE.exists():
		raise FileNotFoundError(f"Missing metrics file: {METRICS_FILE}")

	pred_df = pd.read_csv(PRED_FILE, parse_dates=["Time"], index_col="Time")
	metrics_df = pd.read_csv(METRICS_FILE)

	selected_targets = pick_targets(pred_df)
	for target in selected_targets:
		safe_name = target.replace(" ", "_").replace("/", "_")
		plot_target_time_series(pred_df, target, VIZ_DIR / f"timeseries_{safe_name}.png")
		plot_actual_vs_pred_scatter(pred_df, target, VIZ_DIR / f"scatter_{safe_name}.png")
		plot_rolling_errors(pred_df, target, VIZ_DIR / f"rolling_errors_{safe_name}.png", window=24)

	# Save a compact summary table to pair plots with key metrics.
	summary_cols = [
		c for c in [
			"target",
			"mae",
			"baseline_mae",
			"mae_improvement_pct",
			"rmse",
			"baseline_rmse",
			"rmse_improvement_pct",
		] if c in metrics_df.columns
	]

	print(f"Interpretability plots saved -> {VIZ_DIR}")
	print(f"Generated detailed plots for {len(selected_targets)} targets: {selected_targets}")


if __name__ == "__main__":
	main()

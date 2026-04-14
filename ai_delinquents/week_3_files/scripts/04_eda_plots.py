"""
04_eda_plots.py
Exploratory data analysis plots for the ESSD AI Competition water supply
forecasting project (AI Delinquents).

Reads from data/clean/, writes PNGs to plots/.

Usage:
    pixi run python scripts/04_eda_plots.py

Reproducibility:
    No stochastic operations; output is deterministic given the same input CSVs.
    Figures are saved at FIGURE_DPI with a fixed style and color palette.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CLEAN_DIR = Path("data/clean")
PLOTS_DIR = Path("plots")
FIGURE_DPI = 150
STYLE = "seaborn-v0_8-whitegrid"
PALETTE = "Blues_d"

plt.style.use(STYLE)
PLOTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
feat = pd.read_csv(CLEAN_DIR / "feature_matrix.csv")
flow_monthly = pd.read_csv(CLEAN_DIR / "natural_flow_monthly.csv")
climate = pd.read_csv(CLEAN_DIR / "climate_indices_monthly.csv")
snotel_stations = pd.read_csv(CLEAN_DIR / "snotel_apr1_by_station.csv", index_col=0)
target = pd.read_csv(CLEAN_DIR / "target_apr_sep_volume.csv")

# Month order for the seasonal hydrograph
MONTH_ORDER = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def save(fig: plt.Figure, name: str) -> None:
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# 1. Target distribution
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(feat["target_volume"], bins=10, color="#2171b5", edgecolor="white",
        alpha=0.85, density=True, label="Observed")
x_range = np.linspace(feat["target_volume"].min() - 2000, feat["target_volume"].max() + 2000, 200)
kde = stats.gaussian_kde(feat["target_volume"])
ax.plot(x_range, kde(x_range), color="#08306b", lw=2, label="KDE")
ax.axvline(feat["target_volume"].mean(), color="tomato", ls="--", lw=1.5, label=f"Mean ({feat['target_volume'].mean():,.0f})")
ax.axvline(feat["target_volume"].median(), color="darkorange", ls=":", lw=1.5, label=f"Median ({feat['target_volume'].median():,.0f})")
ax.set_xlabel("Apr-Sep Naturalized Flow Volume (kcfs-days)")
ax.set_ylabel("Density")
ax.set_title("Distribution of Target Variable\n(Apr-Sep Volume at The Dalles, WY 1985-2018)")
ax.legend(fontsize=9)
save(fig, "01_target_distribution.png")

# ---------------------------------------------------------------------------
# 2. Target time series with trend line
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(feat["water_year"], feat["target_volume"], color="#6baed6", edgecolor="white", width=0.8)
slope, intercept, r, p, _ = stats.linregress(feat["water_year"], feat["target_volume"])
trend_y = slope * feat["water_year"] + intercept
ax.plot(feat["water_year"], trend_y, color="#08306b", lw=2,
        label=f"Trend: {slope:+.0f} kcfs-days/yr  (p={p:.2f})")
ax.axhline(feat["target_volume"].mean(), color="tomato", ls="--", lw=1.2, alpha=0.7, label="Mean")
ax.set_xlabel("Water Year")
ax.set_ylabel("Volume (kcfs-days)")
ax.set_title("Annual Apr-Sep Naturalized Flow Volume at The Dalles")
ax.legend(fontsize=9)
ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
save(fig, "02_target_timeseries.png")

# ---------------------------------------------------------------------------
# 3. Feature correlation heatmap
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "target_volume": "Apr-Sep Volume\n(target)",
    "apr1_swe_inches": "Apr 1 SWE (in)",
    "apr1_swe_anomaly_pct": "Apr 1 SWE\nAnomaly (%)",
    "djf_pdo": "DJF PDO",
    "djf_nino34": "DJF Nino3.4",
    "djf_pna": "DJF PNA",
    "jan_mar_mean_q_cfs": "Jan-Mar Mean Q\n(cfs)",
    "oct_mar_volume_kcfs_days": "Oct-Mar Volume\n(kcfs-days)",
}
corr_df = feat[list(FEATURE_LABELS.keys())].rename(columns=FEATURE_LABELS).corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
sns.heatmap(
    corr_df, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
    vmin=-1, vmax=1, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8}
)
ax.set_title("Pearson Correlation Matrix — Feature Matrix\n(lower triangle)")
fig.tight_layout()
save(fig, "03_feature_correlations.png")

# ---------------------------------------------------------------------------
# 4. SWE vs target scatter, colored by ENSO phase
# ---------------------------------------------------------------------------
def enso_phase(val):
    if val > 0.5:
        return "El Nino (>0.5)"
    elif val < -0.5:
        return "La Nina (<-0.5)"
    else:
        return "Neutral"

feat["enso_phase"] = feat["djf_nino34"].apply(enso_phase)
phase_colors = {"El Nino (>0.5)": "#d62728", "Neutral": "#7f7f7f", "La Nina (<-0.5)": "#1f77b4"}

fig, ax = plt.subplots(figsize=(7, 5))
for phase, grp in feat.groupby("enso_phase"):
    ax.scatter(grp["apr1_swe_inches"], grp["target_volume"],
               color=phase_colors[phase], label=phase, s=60, alpha=0.85, edgecolors="white")
# Fit line over all points
slope, intercept, r, p, _ = stats.linregress(feat["apr1_swe_inches"], feat["target_volume"])
x_fit = np.linspace(feat["apr1_swe_inches"].min(), feat["apr1_swe_inches"].max(), 100)
ax.plot(x_fit, slope * x_fit + intercept, color="black", lw=1.5, ls="--",
        label=f"OLS  r={r:.2f}, p={p:.3f}")
ax.set_xlabel("Apr 1 Basin-Average SWE (inches)")
ax.set_ylabel("Apr-Sep Volume (kcfs-days)")
ax.set_title("Apr 1 SWE vs Apr-Sep Flow Volume\n(colored by DJF ENSO phase)")
ax.legend(fontsize=9)
save(fig, "04_swe_vs_target_enso.png")

# ---------------------------------------------------------------------------
# 5. Monthly hydrograph — box plot
# ---------------------------------------------------------------------------
flow_monthly["month"] = pd.Categorical(flow_monthly["month"], categories=MONTH_ORDER, ordered=True)
flow_sorted = flow_monthly.sort_values("month")

fig, ax = plt.subplots(figsize=(10, 4.5))
months = MONTH_ORDER
bp_data = [flow_sorted.loc[flow_sorted["month"] == m, "mean_flow_cfs"].values for m in months]
bp = ax.boxplot(bp_data, patch_artist=True, notch=False, widths=0.6,
                medianprops=dict(color="#08306b", lw=2))
cmap = plt.cm.Blues
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(cmap(0.35 + 0.45 * i / len(months)))
    patch.set_edgecolor("white")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(months)
ax.set_xlabel("Month (water year order)")
ax.set_ylabel("Monthly Mean Naturalized Flow (cfs)")
ax.set_title("Seasonal Hydrograph — Monthly Naturalized Flow at The Dalles\n(WY 1929-2019, box = IQR, whiskers = 1.5×IQR)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
save(fig, "05_monthly_hydrograph.png")

# ---------------------------------------------------------------------------
# 6. Climate indices time series (PDO, Nino3.4, PNA)
# ---------------------------------------------------------------------------
climate["date"] = pd.to_datetime(
    climate[["year", "month"]].rename(columns={"year": "year", "month": "month"})
    .assign(day=1)
)
climate_long = climate.melt(id_vars="date", value_vars=["pdo", "nino34", "pna"],
                             var_name="index", value_name="value").dropna()

index_colors = {"pdo": "#1f77b4", "nino34": "#d62728", "pna": "#2ca02c"}
index_labels = {"pdo": "PDO", "nino34": "Nino3.4", "pna": "PNA"}

fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
for ax, idx in zip(axes, ["pdo", "nino34", "pna"]):
    sub = climate_long[climate_long["index"] == idx]
    ax.fill_between(sub["date"], sub["value"], 0,
                    where=sub["value"] >= 0, color=index_colors[idx], alpha=0.7)
    ax.fill_between(sub["date"], sub["value"], 0,
                    where=sub["value"] < 0, color=index_colors[idx], alpha=0.3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(index_labels[idx], fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
axes[0].set_title("Monthly Climate Teleconnection Indices")
axes[-1].set_xlabel("Date")
fig.tight_layout()
save(fig, "06_climate_indices_timeseries.png")

# ---------------------------------------------------------------------------
# 7. SNOTEL station SWE heatmap (years × stations)
# ---------------------------------------------------------------------------
# Columns are station IDs; rows are years
station_ids = snotel_stations.columns.astype(str).tolist()
snotel_plot = snotel_stations.copy()
snotel_plot.index.name = "Water Year"

fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(
    snotel_plot.T,  # stations as rows, years as columns
    cmap="Blues", ax=ax, linewidths=0,
    cbar_kws={"label": "Apr 1 SWE (inches)", "shrink": 0.8},
    yticklabels=True
)
ax.set_xlabel("Water Year")
ax.set_ylabel("SNOTEL Station ID")
ax.set_title("Apr 1 SWE by SNOTEL Station and Water Year\n(white = missing)")
# Only show every 5th year on x-axis to avoid crowding
n_years = snotel_plot.shape[0]
year_labels = snotel_plot.index.tolist()
tick_positions = list(range(0, n_years, 5))
ax.set_xticks([p + 0.5 for p in tick_positions])
ax.set_xticklabels([year_labels[p] for p in tick_positions], rotation=45, ha="right")
fig.tight_layout()
save(fig, "07_snotel_station_heatmap.png")

# ---------------------------------------------------------------------------
# 8. Scatter matrix: top predictors vs target
# ---------------------------------------------------------------------------
SCATTER_COLS = ["target_volume", "apr1_swe_inches", "oct_mar_volume_kcfs_days",
                "jan_mar_mean_q_cfs", "djf_nino34"]
scatter_labels = {
    "target_volume": "Apr-Sep\nVolume",
    "apr1_swe_inches": "Apr 1\nSWE (in)",
    "oct_mar_volume_kcfs_days": "Oct-Mar\nVolume",
    "jan_mar_mean_q_cfs": "Jan-Mar\nMean Q",
    "djf_nino34": "DJF\nNino3.4",
}
plot_data = feat[SCATTER_COLS].rename(columns=scatter_labels)
g = sns.PairGrid(plot_data, diag_sharey=False)
g.map_upper(sns.scatterplot, s=40, alpha=0.8, color="#2171b5", edgecolor="white")
g.map_lower(sns.kdeplot, fill=True, cmap="Blues", levels=5)
g.map_diag(sns.histplot, color="#2171b5", bins=10)
g.figure.suptitle("Scatter Matrix: Key Predictors vs Target", y=1.01, fontsize=13)
save(g.figure, "08_predictor_scatter_matrix.png")

print("\nAll plots written to:", PLOTS_DIR.resolve())

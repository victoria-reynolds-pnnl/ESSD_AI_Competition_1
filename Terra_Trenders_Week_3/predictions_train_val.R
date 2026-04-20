"""
interpretability.py — Week 3 Cold Extremes: Model Interpretability

Produces the following artefacts (run AFTER train.py):

    Visualizations/coef_irr_plot.png    — IRR forest plot for extreme_type
                                          and c_year:extreme_type terms only
                                          (pure Python/matplotlib)
    Visualizations/re_plot.png          — Year-level random intercepts
                                          (generated via Rscript + sjPlot;
                                           skipped if Rscript not on PATH)
    Output/fixed_effects_table.csv      — Full coefficient table: Estimate,
                                          SE, z, p, IRR, 95 % CI

Interpretability approach
-------------------------
Because glmmTMB does not support SHAP natively, we use coefficient-based
interpretability, which is the standard approach for GLMMs:

  * Fixed effects (IRR): how each cold extreme type and its time trend
    interaction affects expected annual count relative to the reference.
    NERC_ID fixed effects are included in full CSV but excluded from the
    forest plot for clarity (16 region-level intercepts).
  * Random intercepts (BLUPs): year-specific rate multipliers capturing
    systematic above- or below-average cold event frequency by year.

Requirements
------------
  Python : pandas, numpy, matplotlib  (see requirements.txt)
  R      : glmmTMB, sjPlot (optional, only for re_plot.png)
"""

import pickle
import pathlib
import subprocess
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Locate Rscript (handles non-standard install paths on Windows)
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import r_config
try:
    RSCRIPT = r_config.get_rscript_path()
except Exception:
    RSCRIPT = None  # Rscript optional; only needed for re_plot.png

ROOT   = pathlib.Path(__file__).resolve().parents[1]
VIZ    = ROOT / "Visualizations"
OUTPUT = ROOT / "Output"
VIZ.mkdir(exist_ok=True)


# ── 1. Load model wrapper ─────────────────────────────────────────────────────
pkl_path = ROOT / "Models" / "model_1.pkl"
if not pkl_path.exists():
    raise FileNotFoundError(
        f"{pkl_path} not found. Run Scripts/train.py first."
    )

with open(pkl_path, "rb") as fh:
    model = pickle.load(fh)

print(f"Loaded: {model.formula}  [{model.family}]")
print(f"NB dispersion parameter (size): {model.dispersion:.4f}\n")


# ── 2. Build enriched coefficient table ───────────────────────────────────────
coef_df = model.coef_df.copy()
coef_df.index.name = "term"

coef_df["IRR"]   = np.exp(coef_df["Estimate"])
coef_df["CI_lo"] = np.exp(coef_df["Estimate"] - 1.96 * coef_df["Std. Error"])
coef_df["CI_hi"] = np.exp(coef_df["Estimate"] + 1.96 * coef_df["Std. Error"])

out_coef = OUTPUT / "fixed_effects_table.csv"
coef_df.to_csv(out_coef)
print(f"Coefficient table saved → {out_coef.relative_to(ROOT)}")


# ── 3. IRR forest plot (matplotlib, no R required) ────────────────────────────
# Show only extreme_type and c_year:extreme_type terms — exclude NERC fixed
# effects (16 region intercepts) and c_year alone to keep the plot readable.
exclude_prefixes = ("as.factor(NERC_ID)", "(Intercept)")
plot_df = coef_df[
    ~coef_df.index.str.startswith(exclude_prefixes)
].sort_values("IRR")

fig, ax = plt.subplots(figsize=(7, max(4, len(plot_df) * 0.4 + 1.5)))
y_pos = range(len(plot_df))

ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8, zorder=0)

xerr_lo = (plot_df["IRR"] - plot_df["CI_lo"]).clip(lower=0)
xerr_hi = (plot_df["CI_hi"] - plot_df["IRR"]).clip(lower=0)

ax.errorbar(
    plot_df["IRR"],
    list(y_pos),
    xerr=[xerr_lo, xerr_hi],
    fmt="o",
    color="steelblue",
    ecolor="steelblue",
    capsize=3,
    linewidth=1.2,
    markersize=5,
    zorder=2,
)

# Annotate IRR values
for i, (_, row) in enumerate(plot_df.iterrows()):
    ax.text(
        row["IRR"] + max(xerr_hi) * 0.05,
        i,
        f"{row['IRR']:.2f}",
        va="center",
        fontsize=7,
        color="steelblue",
    )

ax.set_yticks(list(y_pos))
ax.set_yticklabels(
    [t.replace("cold", "").replace("extreme_type", "type: ") for t in plot_df.index],
    fontsize=8,
)
ax.set_xlabel("Incidence Rate Ratio (IRR)  [95 % CI]", fontsize=9)
ax.set_title(
    "Fixed-Effect IRRs — Negative Binomial Model\n"
    "Cold Extreme Frequency by Type  (reference = baseline cold type)",
    fontsize=10,
)
ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
plt.tight_layout()

irr_path = VIZ / "coef_irr_plot.png"
fig.savefig(irr_path, dpi=150)
plt.close(fig)
print(f"IRR forest plot saved → {irr_path.relative_to(ROOT)}")


# ── 4. Year random-effects plot via Rscript + sjPlot ─────────────────────────
# m5.4 uses as.factor(NERC_ID) as fixed effects; (1 | start_yr) is the
# only random effect — so we plot year-level BLUPs here.
rds_path = ROOT / "Models" / "fit_m54tv.rds"

if not rds_path.exists():
    print(
        f"\n[SKIP] {rds_path.name} not found — random-effects plot requires "
        "train.py to have been run."
    )
elif RSCRIPT is None:
    print(
        "\n[SKIP] Rscript not found — skipping random-effects plot.\n"
        "  Ensure R/bin is on PATH or set R_HOME_OVERRIDE in Scripts/r_config.py."
    )
else:
    re_path  = VIZ / "re_plot.png"
    blup_out = OUTPUT / "year_random_effects.csv"

    r_code = f"""
library(glmmTMB); library(sjPlot); library(ggplot2); library(tidyverse)
fit_m54tv <- readRDS("{rds_path.as_posix()}")
pr <- plot_model(fit_m54tv, type = "re",
                 sort.est = TRUE, show.values = TRUE) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey50") +
  theme_minimal() +
  xlab("Year-Specific Rate Multiplier") +
  ylab("Year") +
  ggtitle("Year Random Effects (1980-2018)")
ggsave("{re_path.as_posix()}", pr, height = 8, width = 5, dpi = 150)
blup <- as.data.frame(ranef(fit_m54tv)$cond$start_yr)
blup$start_yr <- rownames(blup)
blup$rate_multiplier <- exp(blup[[1]])
write.csv(blup, "{blup_out.as_posix()}", row.names = FALSE)
cat("Random-effects plot and BLUP table saved.\\n")
"""
    try:
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            suffix=".R", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            tmp.write(r_code)
            tmp_path = tmp.name
        subprocess.run([RSCRIPT, "--vanilla", tmp_path], check=True)
        os.unlink(tmp_path)
        print(f"Random-effects plot saved → {re_path.relative_to(ROOT)}")
        print(f"BLUP table saved          → {blup_out.relative_to(ROOT)}")
    except Exception as exc:
        print(f"\n[WARNING] Random-effects plot step failed: {exc}")


# ── 5. Print coefficient summary ──────────────────────────────────────────────
print("\n" + "─" * 60)
print("FIXED EFFECTS SUMMARY (IRR scale)")
print("─" * 60)
display_cols = ["Estimate", "Std. Error", "IRR", "CI_lo", "CI_hi", "Pr(>|z|)"]
available = [c for c in display_cols if c in coef_df.columns]
print(coef_df[available].round(4).to_string())
print("\nInterpretation: IRR > 1 → that cold type occurs at a higher rate than")
print("the reference category; IRR < 1 → lower rate.  All values control for")
print("NERC_ID fixed effects and year-level random intercepts (start_yr).")
print("\nInterpretability complete.")

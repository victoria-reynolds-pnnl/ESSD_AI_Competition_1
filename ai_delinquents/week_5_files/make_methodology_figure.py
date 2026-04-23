"""Render methodology flowchart as a PNG for the Week 5 presentation."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = "presentation_figures/methodology_flowchart.png"

# ── Palette ──────────────────────────────────────────────────────────────────
BLUE_DARK  = "#0d4c73"
BLUE_MID   = "#1a7ab5"
BLUE_LIGHT = "#cde8f7"
TEAL       = "#4fb3e8"
ORANGE     = "#d4622a"
DARK_NAVY  = "#062d4a"
GREEN      = "#2a7a4b"
WHITE      = "#ffffff"
BG         = "#f4f9fd"

# ── Canvas ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 14))
ax.set_xlim(0, 9)
ax.set_ylim(0, 14)
ax.axis("off")
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

CX = 4.5   # centre x for the main spine

# ── Helpers ──────────────────────────────────────────────────────────────────
def box(x, y, w, h, label, fc, tc=WHITE, fs=10):
    p = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=fc, edgecolor=BLUE_DARK, linewidth=1.8, zorder=3)
    ax.add_patch(p)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold",
            zorder=4, multialignment="center", linespacing=1.4)

def diamond(x, y, w, h, label, fc=TEAL, tc=WHITE, fs=10):
    hw, hh = w/2, h/2
    pts = [(x, y+hh), (x+hw, y), (x, y-hh), (x-hw, y)]
    poly = mpatches.Polygon(pts, closed=True,
                            facecolor=fc, edgecolor=BLUE_DARK,
                            linewidth=1.8, zorder=3)
    ax.add_patch(poly)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold", zorder=4,
            multialignment="center")

def varrow(x, y1, y2, color=BLUE_DARK, lw=2.0):
    """Straight vertical arrow from (x,y1) down to (x,y2)."""
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=16),
                zorder=2)

def harrow(x1, x2, y, color=BLUE_DARK, lw=2.0):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=16),
                zorder=2)

def elbow(x1, y1, x2, y2, color=BLUE_DARK, lw=2.0, label="", label_side="left"):
    """L-shaped connector: go horizontal then vertical."""
    ax.plot([x1, x2], [y1, y1], color=color, lw=lw, zorder=2)
    ax.annotate("", xy=(x2, y2), xytext=(x2, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=16),
                zorder=2)
    if label:
        lx = (x1 + x2) / 2
        ax.text(lx, y1 + 0.12, label, ha="center", va="bottom",
                fontsize=8.5, color=color, style="italic")

# ═════════════════════════════════════════════════════════════════════════════
# Layout — y increases upward; top of chart ≈ 13.4
# ═════════════════════════════════════════════════════════════════════════════
BW, BH = 3.2, 0.70   # standard box width / height

# Step 1 — Raw Data
Y1 = 13.0
box(CX, Y1, BW, BH, "Raw Data  (4 sources)", BLUE_MID)

# Step 2 — Clean
Y2 = 11.4
box(CX, Y2, BW, BH, "Clean & QA/QC", BLUE_MID)
varrow(CX, Y1 - BH/2, Y2 + BH/2)

# Step 3 — Feature Engineering
Y3 = 9.8
box(CX, Y3, BW, BH, "Feature Engineering\n(7 predictors)", BLUE_MID)
varrow(CX, Y2 - BH/2, Y3 + BH/2)

# Decision — Temporal Split
Y4 = 8.2
diamond(CX, Y4, 2.8, 1.0, "Temporal Split", TEAL)
varrow(CX, Y3 - BH/2, Y4 + 0.5)

# ── Left branch: Train ───────────────────────────────────────────────────────
LX = 2.0   # left branch x

Y5L = 6.7
box(LX, Y5L, 2.8, BH, "Train / Val\nWY 1985–2012  (28 yr)", BLUE_MID, fs=9)
elbow(CX - 1.4, Y4, LX, Y5L + BH/2, color=BLUE_MID, label="train")

Y6L = 5.2
box(LX, Y6L, 2.8, BH, "LOYO CV\n(Leave-One-Year-Out)", BLUE_LIGHT, tc=BLUE_DARK, fs=9)
varrow(LX, Y5L - BH/2, Y6L + BH/2, color=BLUE_MID)

Y7L = 3.7
box(LX, Y7L, 2.8, BH, "Hyperparameter Tuning\n(Optuna, 50 trials)", BLUE_LIGHT, tc=BLUE_DARK, fs=9)
varrow(LX, Y6L - BH/2, Y7L + BH/2, color=BLUE_MID)

# Models — side by side under tuning
MBW = 1.2
Y8 = 2.3
MLRX  = LX - 0.8
XGBX  = LX + 0.8
box(MLRX, Y8, MBW, BH, "MLR", DARK_NAVY, fs=9)
box(XGBX, Y8, MBW, BH, "XGBoost\nq10/q50/q90", DARK_NAVY, fs=9)

# arrows from tuning box to each model
ax.annotate("", xy=(MLRX, Y8 + BH/2), xytext=(MLRX, Y7L - BH/2),
            arrowprops=dict(arrowstyle="-|>", color=DARK_NAVY, lw=1.8,
                            mutation_scale=14), zorder=2)
ax.annotate("", xy=(XGBX, Y8 + BH/2), xytext=(XGBX, Y7L - BH/2),
            arrowprops=dict(arrowstyle="-|>", color=DARK_NAVY, lw=1.8,
                            mutation_scale=14), zorder=2)
# fan out lines from bottom of tuning box
ax.plot([LX, MLRX], [Y7L - BH/2, Y7L - BH/2], color=DARK_NAVY, lw=1.8, zorder=2)
ax.plot([LX, XGBX], [Y7L - BH/2, Y7L - BH/2], color=DARK_NAVY, lw=1.8, zorder=2)

# ── Right branch: Test ───────────────────────────────────────────────────────
RX = 7.0

Y5R = 6.7
box(RX, Y5R, 2.8, BH, "Test  (held out)\nWY 2013–2018  (6 yr)", ORANGE, fs=9)
elbow(CX + 1.4, Y4, RX, Y5R + BH/2, color=ORANGE, label="test", label_side="right")

# ── Evaluate (bottom centre) ─────────────────────────────────────────────────
EVALX = CX
EVALY = 0.9
box(EVALX, EVALY, 3.4, BH, "Evaluate\nNSE · RMSE · Skill score", GREEN)

# MLR → Evaluate (elbow right then down)
ax.plot([MLRX, EVALX - 0.5], [Y8 - BH/2, Y8 - BH/2], color=DARK_NAVY, lw=1.8, zorder=2)
ax.plot([EVALX - 0.5, EVALX - 0.5], [Y8 - BH/2, EVALY + BH/2 + 0.05], color=DARK_NAVY, lw=1.8, zorder=2)
ax.annotate("", xy=(EVALX - 0.02, EVALY + BH/2),
            xytext=(EVALX - 0.5, EVALY + BH/2),
            arrowprops=dict(arrowstyle="-|>", color=DARK_NAVY, lw=1.8,
                            mutation_scale=14), zorder=2)

# XGBoost → Evaluate (elbow right then down)
ax.plot([XGBX, EVALX + 0.5], [Y8 - BH/2, Y8 - BH/2], color=DARK_NAVY, lw=1.8, zorder=2)
ax.plot([EVALX + 0.5, EVALX + 0.5], [Y8 - BH/2, EVALY + BH/2 + 0.05], color=DARK_NAVY, lw=1.8, zorder=2)
ax.annotate("", xy=(EVALX + 0.02, EVALY + BH/2),
            xytext=(EVALX + 0.5, EVALY + BH/2),
            arrowprops=dict(arrowstyle="-|>", color=DARK_NAVY, lw=1.8,
                            mutation_scale=14), zorder=2)

# Test → Evaluate (elbow: down from test box, then left to evaluate)
TEST_BOTTOM_Y = Y5R - BH/2
ROUTE_Y = EVALY + BH/2 + 0.15
ax.plot([RX, RX], [TEST_BOTTOM_Y, ROUTE_Y], color=ORANGE, lw=1.8, zorder=2)
ax.plot([RX, EVALX + 1.7], [ROUTE_Y, ROUTE_Y], color=ORANGE, lw=1.8, zorder=2)
ax.annotate("", xy=(EVALX + 1.7, EVALY + BH/2),
            xytext=(EVALX + 1.7, ROUTE_Y),
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.8,
                            mutation_scale=14), zorder=2)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(CX, 13.7, "Modeling Pipeline", ha="center", va="center",
        fontsize=14, color=BLUE_DARK, fontweight="bold")

# ── Legend (right side, between test box and evaluate) ───────────────────────
legend_items = [
    (BLUE_MID,   WHITE,      "Data processing & training set"),
    (TEAL,       WHITE,      "Decision / split point"),
    (BLUE_LIGHT, BLUE_DARK,  "Cross-validation & tuning"),
    (DARK_NAVY,  WHITE,      "Fitted models"),
    (ORANGE,     WHITE,      "Held-out test set"),
    (GREEN,      WHITE,      "Final evaluation"),
]
LGX = 5.5   # left edge of legend block
LGY = 5.2   # y of first item

# legend background box
bg = mpatches.FancyBboxPatch((LGX - 0.15, LGY - len(legend_items)*0.45 + 0.05),
                              3.45, len(legend_items)*0.45 + 0.45,
                              boxstyle="round,pad=0.1",
                              facecolor=WHITE, edgecolor=BLUE_DARK,
                              linewidth=1.2, alpha=0.85, zorder=3)
ax.add_patch(bg)
ax.text(LGX + 1.55, LGY + 0.28, "Legend", fontsize=9.5, color=BLUE_DARK,
        fontweight="bold", va="center", ha="center", zorder=5)

for i, (fc, tc, label) in enumerate(legend_items):
    yi = LGY - i * 0.45
    rect = mpatches.FancyBboxPatch((LGX, yi - 0.16), 0.38, 0.32,
                                    boxstyle="round,pad=0.03",
                                    facecolor=fc, edgecolor=BLUE_DARK,
                                    linewidth=1.0, zorder=4)
    ax.add_patch(rect)
    ax.text(LGX + 0.55, yi, label, fontsize=8, color=BLUE_DARK,
            va="center", zorder=5)

plt.tight_layout(pad=0.4)
plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved: {OUT}")

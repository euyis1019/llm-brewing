"""Figure 3 — Cross-model comparison (computing task).

Panel (a): KDE with peak spotlight
Panel (b): Brewing Coverage bar chart

Reads all data from fig3_data.json.
Transparent background.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path as P
from scipy.stats import gaussian_kde

# ── Load data ─────────────────────────────────────────────────────────
DATA_PATH = P(__file__).parent / "fig3_data.json"
with open(DATA_PATH) as f:
    DATA = json.load(f)

GROUPS = ["scaling", "training", "architecture"]

# Extract model info
MODELS = {}
for grp in GROUPS:
    MODELS[grp] = [(m["label"], m["layers"]) for m in DATA["models"][grp]]

# Colors
COLORS = DATA["colors"]
GROUP_COLORS = {
    "scaling": COLORS["scaling"],
    "training": COLORS["training"],
    "architecture": COLORS["architecture"],
}
BG_COLORS = {
    "scaling": COLORS["bg_scaling"],
    "training": COLORS["bg_training"],
    "architecture": COLORS["bg_architecture"],
}

# Panel A data: per-sample FJC normalized
PANEL_A = DATA["panel_a_fjc_normalized"]
# Panel B data: FJC exist rates
PANEL_B = DATA["panel_b_fjc_exist_rate"]

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 7,
    "axes.linewidth": 0.4,
})

fig, (ax_kde, ax_bar) = plt.subplots(2, 1, figsize=(5.5, 4.8),
                                      gridspec_kw={"height_ratios": [2, 0.8]})
fig.patch.set_alpha(0.0)
for a in [ax_kde, ax_bar]:
    a.set_facecolor("none")
fig.subplots_adjust(hspace=0.35)

# =====================================================================
# Panel (a): KDE with peak spotlight
# =====================================================================
ax = ax_kde
x_grid = np.linspace(0, 1, 500)

LINESTYLES = {"scaling": "-", "training": "--", "architecture": "-."}


def plot_kde_spotlight(ax, group):
    linestyle = LINESTYLES[group]
    colors = GROUP_COLORS[group]
    models = MODELS[group]
    fjc_data = PANEL_A[group]

    for (label, nl), color in zip(models, colors):
        fjcs = np.array(fjc_data[label])
        if len(fjcs) < 5:
            continue

        # Adaptive bandwidth: smoother for smaller samples
        bw = 0.15 if len(fjcs) > 400 else 0.20 if len(fjcs) > 200 else 0.25
        kde = gaussian_kde(fjcs, bw_method=bw)
        density = kde(x_grid)
        peak_idx = np.argmax(density)
        peak_x = x_grid[peak_idx]
        peak_y = density[peak_idx]
        sigma = 0.08
        spotlight = np.exp(-0.5 * ((x_grid - peak_x) / sigma) ** 2)

        # Faded full curve
        ax.plot(x_grid, density, color=color, linewidth=0.6,
                linestyle=linestyle, alpha=0.2, zorder=2)

        # Spotlight segments
        for j in range(len(x_grid) - 1):
            seg_alpha = 0.15 + 0.85 * spotlight[j]
            seg_lw = 0.8 + 1.5 * spotlight[j]
            ax.plot(x_grid[j:j+2], density[j:j+2], color=color,
                    linewidth=seg_lw, alpha=seg_alpha, linestyle=linestyle,
                    zorder=3, solid_capstyle="round")

        # Peak fill
        peak_mask = spotlight > 0.3
        ax.fill_between(x_grid, density, where=peak_mask,
                         color=color, alpha=0.15, zorder=1)

        # Vertical drop line
        ax.plot([peak_x, peak_x], [0, peak_y], color=color,
                linewidth=1.2, alpha=0.6, linestyle=":", zorder=1)

        # Peak marker + label
        ax.scatter(peak_x, peak_y, s=25, color=color, edgecolors="black",
                   linewidths=0.4, zorder=5)
        ax.annotate(label, (peak_x, peak_y),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=4.5, color=color, fontweight="medium", zorder=6)


for grp in GROUPS:
    plot_kde_spotlight(ax, grp)

ax.set_xlim(0, 1)
ax.set_ylim(0, 4.0)
ax.set_xlabel("Normalized Layer Depth (FJC / L)", fontsize=7, color="#555")
ax.set_ylabel("Density", fontsize=7, color="#555")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=6, colors="#777")

# Legend
legend_items = []
for grp in GROUPS:
    legend_items.append(Line2D([0], [0], color="none",
                               label=grp.capitalize()))
    for (label, _), c in zip(MODELS[grp], GROUP_COLORS[grp]):
        legend_items.append(Line2D([0], [0], color=c, lw=1.5,
                                    linestyle=LINESTYLES[grp],
                                    label=f"  {label}"))

ax.legend(handles=legend_items, fontsize=4.5, loc="upper left",
          frameon=False, handlelength=1.8, labelspacing=0.3,
          bbox_to_anchor=(0.02, 0.88))

ax.text(0.02, 0.97, "(a) Normalized FJC Distribution (FJC>0)", ha="left", va="top",
        fontsize=7.5, fontweight="bold", color="#333", transform=ax.transAxes)

# =====================================================================
# Panel (b): Brewing Coverage bar chart
# =====================================================================
ax = ax_bar

bars = []
x = 0
group_ranges = {}
group_points = {}

for grp in GROUPS:
    x_start = x
    group_points[grp] = []
    rates = PANEL_B[grp]
    colors = GROUP_COLORS[grp]
    models = MODELS[grp]
    for (label, _), color in zip(models, colors):
        rate = rates[label]
        bars.append((x, rate, color, label))
        group_points[grp].append((x, rate))
        x += 1
    group_ranges[grp] = (x_start - 0.5, x - 0.5)
    x += 0.8

# Background bands
for grp in GROUPS:
    lo, hi = group_ranges[grp]
    rect = mpatches.FancyBboxPatch(
        (lo, 0), hi - lo, 0.82,
        boxstyle=mpatches.BoxStyle.Round(pad=0.02),
        facecolor=BG_COLORS[grp], alpha=0.2, edgecolor=BG_COLORS[grp],
        linewidth=0.5, zorder=0)
    ax.add_patch(rect)

# Bars
bar_w = 0.42
for xpos, rate, color, label in bars:
    ax.bar(xpos, rate, width=bar_w, color=color, edgecolor="black",
           linewidth=0.5, alpha=0.85, zorder=2)
    ax.text(xpos, rate + 0.015, f"{rate:.0%}", ha="center", va="bottom",
            fontsize=5, color="#333", fontweight="bold", zorder=3)

# Trend lines
TREND_COLORS = {"scaling": "#0D47A1", "training": "#BF360C", "architecture": "#1B5E20"}
for grp in GROUPS:
    pts = group_points[grp]
    if len(pts) >= 2:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=TREND_COLORS[grp], linewidth=1.4, linestyle="--",
                alpha=0.5, zorder=5, marker="D", markersize=3.5,
                markerfacecolor="white", markeredgecolor=TREND_COLORS[grp],
                markeredgewidth=0.8)

# X labels
xticks = [v[0] for v in bars]
xlabels = [v[3] for v in bars]
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, fontsize=5, rotation=35, ha="right", color="#444")

# Light gridlines
for v in [0.2, 0.4, 0.6, 0.8]:
    ax.axhline(v, color="#eee", linewidth=0.4, zorder=0)

ax.set_ylabel("Brewing Coverage", fontsize=7, color="#555")
ax.set_xlim(-0.8, max(xticks) + 0.8)
ax.set_ylim(0, 0.82)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.3)
ax.spines["bottom"].set_linewidth(0.3)
ax.tick_params(axis="y", labelsize=6, colors="#777")

ax.text(0.02, 0.97, "(b) Brewing Coverage", ha="left", va="top",
        fontsize=7.5, fontweight="bold", color="#333", transform=ax.transAxes)

# ── Save ──────────────────────────────────────────────────────────────
OUT_DIR = P(__file__).parent.parent.parent.parent / "output" / "section4" / "fig3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

out_pdf = OUT_DIR / "fig3_cross_model.pdf"
out_png = OUT_DIR / "fig3_cross_model.png"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", transparent=True)
fig.savefig(out_png, dpi=300, bbox_inches="tight", transparent=True)
print(f"Saved → {out_pdf}")
print(f"Saved → {out_png}")

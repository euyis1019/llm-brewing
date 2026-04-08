"""Figure 2 compound — Sankey (flipped) + Dumbbell, shared Y-axis.

Panel (a): Outcomes (left) → Tasks (right)  [Sankey flipped]
Panel (b): Dumbbell (easy→hard Res%) sharing task Y-axis with panel (a)

Reads all data from fig2_data.json.
Transparent background.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import numpy as np
from pathlib import Path as P

# ── Load data ─────────────────────────────────────────────────────────
DATA_PATH = P(__file__).parent / "fig2_data.json"
with open(DATA_PATH) as f:
    DATA = json.load(f)

TASKS = DATA["tasks"]
TASK_LABELS = DATA["task_labels"]
TASK_SHORT = DATA["task_short"]
OUTCOME_NAMES = DATA["outcome_names"]
FLOWS = DATA["panel_a_flows"]
RES = DATA["panel_b_resolved_by_difficulty"]

# ── Colors from JSON ──────────────────────────────────────────────────
OUTCOME_COLORS = DATA["colors"]["outcomes"]
TASK_COLORS = {k: v["fill"] for k, v in DATA["colors"]["tasks"].items()
               if isinstance(v, dict)}

RIBBON_ALPHA = 0.35

# ── Layout ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 7,
    "axes.linewidth": 0.4,
})

fig = plt.figure(figsize=(5.5, 4.0))
fig.patch.set_alpha(0.0)

gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1], wspace=0.01)
ax_sankey = fig.add_subplot(gs[0])
ax_dumb = fig.add_subplot(gs[1])

for a in [ax_sankey, ax_dumb]:
    a.set_facecolor("none")

# =====================================================================
# Panel (a): Sankey — FLIPPED (outcomes left, tasks right)
# =====================================================================
ax = ax_sankey
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.04, 1.04)
ax.axis("off")

NODE_W = 0.065
GAP_TASK = 10.0
GAP_OUTCOME = 10.0


def compute_positions(totals, gap, x_center):
    total_h = sum(totals) + gap * (len(totals) - 1)
    scale = 1.0 / total_h
    positions = []
    y = 1.0
    for t in totals:
        h = t * scale
        y -= h
        positions.append((x_center, y, h))
        y -= gap * scale
    return positions


# Outcome nodes on LEFT
outcome_totals = [0.0] * 4
for t in TASKS:
    for k in range(4):
        outcome_totals[k] += FLOWS[t][k]
outcome_pos = compute_positions(outcome_totals, GAP_OUTCOME, 0.10)

# Task nodes on RIGHT
task_totals = [100.0] * 6
task_pos = compute_positions(task_totals, GAP_TASK, 0.90)


def draw_node(ax, x, y_bot, h, w, color, label, side, show_label=True):
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y_bot), w, h,
        boxstyle=mpatches.BoxStyle.Round(pad=0.003),
        facecolor=color, edgecolor="black", linewidth=0.6, alpha=0.92, zorder=5,
    )
    ax.add_patch(rect)
    if not show_label:
        return
    if side == "left":
        tx, ha = x - w / 2 - 0.015, "right"
    else:
        tx, ha = x + w / 2 + 0.015, "left"
    ty = y_bot + h / 2
    ax.text(tx, ty, label, ha=ha, va="center", fontsize=6.5,
            fontweight="medium", color="#333", zorder=6)


def draw_ribbon(ax, x0, y0_bot, h0, x1, y1_bot, h1, color, alpha):
    top_left = (x0, y0_bot + h0)
    bot_left = (x0, y0_bot)
    top_right = (x1, y1_bot + h1)
    bot_right = (x1, y1_bot)
    mid_x = (x0 + x1) / 2
    verts = [
        top_left,
        (mid_x, top_left[1]), (mid_x, top_right[1]), top_right,
        bot_right,
        (mid_x, bot_right[1]), (mid_x, bot_left[1]), bot_left,
        top_left,
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    path = Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor=color, edgecolor="none",
                                alpha=alpha, zorder=2)
    ax.add_patch(patch)


# Cursors
outcome_cursor = [pos[1] + pos[2] for pos in outcome_pos]
task_cursor = [pos[1] + pos[2] for pos in task_pos]
task_scales = [pos[2] / 100.0 for pos in task_pos]
outcome_scales = [pos[2] / tot if tot > 0 else 0
                  for pos, tot in zip(outcome_pos, outcome_totals)]

# Draw ribbons: from outcome (left) to task (right)
for k in range(4):
    for i, t in enumerate(TASKS):
        val = FLOWS[t][k]
        if val < 0.5:
            continue
        h_src = val * outcome_scales[k]
        h_dst = val * task_scales[i]
        src_top = outcome_cursor[k]
        dst_top = task_cursor[i]
        src_x = 0.10 + NODE_W / 2
        dst_x = 0.90 - NODE_W / 2
        draw_ribbon(ax,
                    src_x, src_top - h_src, h_src,
                    dst_x, dst_top - h_dst, h_dst,
                    color=OUTCOME_COLORS[OUTCOME_NAMES[k]],
                    alpha=RIBBON_ALPHA)
        outcome_cursor[k] -= h_src
        task_cursor[i] -= h_dst

# Draw outcome nodes (left)
for k, name in enumerate(OUTCOME_NAMES):
    x, y, h = outcome_pos[k]
    draw_node(ax, x, y, h, NODE_W, OUTCOME_COLORS[name], name, side="left")

# Draw task nodes (right) — record y-centers for dumbbell alignment
task_y_centers = []
for i, t in enumerate(TASKS):
    x, y, h = task_pos[i]
    draw_node(ax, x, y, h, NODE_W, TASK_COLORS[t], TASK_LABELS[i],
              side="right", show_label=False)
    task_y_centers.append(y + h / 2)

ax.text(0.5, 0.98, "(a) Task \u2192 Outcome Flow", ha="center", va="top",
        fontsize=8, fontweight="bold", color="#333", transform=ax.transAxes)

# =====================================================================
# Panel (b): Dumbbell — shared Y positions with Sankey task nodes
# =====================================================================
ax = ax_dumb
ax.set_facecolor("none")

ax.set_ylim(-0.04, 1.04)
ax.set_xlim(-5, 100)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_yticks(task_y_centers)
ax.set_yticklabels(TASK_SHORT, fontsize=5, color="#444", fontweight="medium")
ax.tick_params(left=False, pad=1)
ax.tick_params(axis="x", labelsize=5.5, colors="#777")
ax.spines["bottom"].set_linewidth(0.3)
ax.set_xlabel("Resolved %", fontsize=6.5, color="#555")

for i, t in enumerate(TASKS):
    yc = task_y_centers[i]
    easy, hard = RES[t][0], RES[t][2]
    c = TASK_COLORS[t]

    # Connecting bar
    ax.plot([hard, easy], [yc, yc], color=c, linewidth=2.2,
            alpha=0.4, zorder=2, solid_capstyle="round")

    # Arrow
    ax.annotate("", xy=(hard, yc), xytext=(easy, yc),
                arrowprops=dict(arrowstyle="-|>", color=c,
                                lw=1.2, mutation_scale=8), zorder=3)

    # Dots
    ax.scatter(easy, yc, s=80, color=c, edgecolors="black",
               linewidths=0.6, zorder=4)
    ax.scatter(hard, yc, s=60, color=c, edgecolors="black",
               linewidths=0.6, zorder=4)

    # Delta label
    delta = hard - easy
    mid = max((easy + hard) / 2, hard + 3)
    offset = 0.014
    ax.text(mid, yc + offset, f"{delta:+.0f}%", ha="center", va="bottom",
            fontsize=4.5, color="#777", fontweight="medium")

# Light vertical gridlines
for v in [0, 25, 50, 75, 100]:
    ax.axvline(v, color="#eee", linewidth=0.3, zorder=0)

ax.set_xticks([0, 25, 50, 75, 100])

ax.text(0.5, 0.98, "(b) Difficulty Response", ha="center", va="top",
        fontsize=8, fontweight="bold", color="#333", transform=ax.transAxes)

# ── Save ──────────────────────────────────────────────────────────────
OUT_DIR = P(__file__).parent.parent.parent.parent / "output" / "section4" / "fig2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

out_pdf = OUT_DIR / "fig2_compound.pdf"
out_png = OUT_DIR / "fig2_compound.png"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", transparent=True)
fig.savefig(out_png, dpi=200, bbox_inches="tight", transparent=True)
print(f"Saved → {out_pdf}")
print(f"Saved → {out_png}")

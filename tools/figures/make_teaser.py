"""Teaser figure for the README: concept, measured curves, what the readouts buy, where they break.

Data: docs/figures/teaser_curves.csv (per-layer content-level 4/4 accuracy of the probe and CSD
readouts, computed from the October caches) and numbers from
docs/logs/depth_intervention_RESULTS.md and gen_depth_RESULTS.md (sections 3 and 16).
Run: python tools/figures/make_teaser.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

NAVY = "#1b2743"
INK = "#3a3f4a"
GRAY = "#9a968e"
BLUE = "#2f6fd0"
BLUE_L = "#a9c4ee"
ORANGE = "#e0603c"
ORANGE_D = "#a8321a"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "docs", "figures", "teaser.png")
CURVES = os.path.join(ROOT, "docs", "figures", "teaser_curves.csv")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.edgecolor": GRAY, "axes.linewidth": 0.9, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "text.color": INK, "axes.facecolor": "white", "figure.facecolor": "white",
})


def title(ax, text, size=11.5):
    ax.set_title(text, fontsize=size, fontweight="bold", color=NAVY, pad=14, loc="left")


def bare(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------- (a) concept
def draw_concept(ax):
    bare(ax)
    L, emit = 36, 24
    ax.set_xlim(-4, L + 4); ax.set_ylim(-1.9, 3.0)

    ax.annotate("", xy=(L + 2.5, 0), xytext=(-2.5, 0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
    ax.text(-2.5, -0.5, "input", ha="left", fontsize=8.5, color=INK)
    ax.text(L + 2.5, -0.5, "layer", ha="right", fontsize=8.5, color=INK)

    ax.add_patch(FancyBboxPatch((17.5, -0.26), emit - 17.5, 0.52,
                                boxstyle="round,pad=0.02,rounding_size=0.5",
                                fc=BLUE_L, ec="none", alpha=0.55))
    ax.text((17.5 + emit) / 2, -1.05, "readable, not yet decodable", ha="center", fontsize=8.3, color=INK)

    ax.plot([emit, emit], [-0.35, 2.2], color=NAVY, lw=1.4, ls=(0, (4, 2.5)))
    ax.text(emit + 0.9, 2.3, "emission layer", fontsize=9.3, color=NAVY, fontweight="bold")
    ax.plot([emit], [0], marker="o", ms=8, mfc=NAVY, mec="white", mew=1.2, zorder=5)

    ax.annotate("", xy=(emit - 0.5, 1.5), xytext=(8.0, 1.5),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6))
    ax.text(8.2, 1.64, "probe reads it", ha="left", fontsize=8.6, color=BLUE)

    ax.annotate("", xy=(emit + 0.5, 0.9), xytext=(L + 1.8, 0.9),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6))
    ax.text(L + 2.0, 1.05, "the model emits it", ha="right", fontsize=8.6, color=ORANGE)
    title(ax, "Concept")


# ---------------------------------------------------------------- (b) curves
def draw_curves(ax):
    rows = [r for r in csv.DictReader(open(CURVES)) if r["run"] == "Qwen3-4B-Base"]
    layer = np.array([int(r["layer"]) for r in rows])
    pr = np.array([float(r["probe"]) for r in rows])
    cs = np.array([float(r["csd"]) for r in rows])

    ax.fill_between(layer, cs, pr, color=BLUE_L, alpha=0.45, lw=0)
    ax.plot(layer, pr, color=BLUE, lw=2.0, label="probe (external)")
    ax.plot(layer, cs, color=ORANGE, lw=2.0, label="self-decoding (CSD)")
    ax.plot([layer[-1]], [0.474], marker="*", ms=12, color=NAVY, zorder=6, clip_on=False)
    ax.annotate("output\n47.4%", xy=(layer[-1] + 0.6, 0.474), xytext=(31.5, 0.30),
                fontsize=8.3, color=NAVY, ha="center",
                arrowprops=dict(arrowstyle="-", color=NAVY, lw=0.7))
    ax.axvline(24, color=NAVY, lw=1.1, ls=(0, (4, 2.5)))
    ax.text(24.7, 0.035, "L24", fontsize=8.3, color=NAVY)
    ax.set_xlabel("layer")
    ax.set_ylabel("questions answered,\nall 4 permutations")
    ax.set_ylim(0, 0.62); ax.set_xlim(0, 36)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_yticklabels(["0", "20%", "40%", "60%"])
    ax.grid(color="#ecebe7", lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=8.4, handlelength=1.6)
    title(ax, "Measured, Qwen3-4B-Base")


# ---------------------------------------------------------------- (c) exit
def draw_exit(ax):
    bare(ax)
    ax.set_xlim(21, 39); ax.set_ylim(-0.7, 3.9)
    items = [("full depth", 36, GRAY),
             ("confidence only", 31.4, BLUE_L),
             ("label-free readouts", 29.0, BLUE),
             ("oracle exit", 24.5, NAVY)]
    for i, (name, x, c) in enumerate(items):
        y = 3.1 - i * 0.95
        ax.plot([x, 36.6], [y, y], color="#e2dfd9", lw=1.2, zorder=1, solid_capstyle="round")
        ax.plot([x], [y], marker="o", ms=9, mfc=c, mec="white", mew=1.2, zorder=3)
        ax.text(37.4, y, f"{x:.1f}".rstrip("0").rstrip("."), fontsize=8.4, color=INK, va="center")
        ax.text(x, y + 0.33, name, ha="center", va="bottom", fontsize=8.6, color=NAVY)
    title(ax, "Exit depth, label-free\naccuracy held within 1 point", size=11)


# ---------------------------------------------------------------- (d) gen
def draw_gen(ax):
    names = ["base", "exit,\ncopied K/V", "exit,\ntrue K/V"]
    vals = [66.4, 27.9, 48.6]
    cols = [GRAY, ORANGE, ORANGE_D]
    x = np.arange(3)
    ax.bar(x, vals, width=0.56, color=cols, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 2.0, f"{v:.1f}", ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.0)
    ax.set_ylim(0, 84); ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_ylabel("MATH-500 accuracy (%)")
    ax.grid(axis="y", color="#ecebe7", lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.annotate("", xy=(0.95, 34), xytext=(0.15, 62),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE_D, lw=1.3,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(1.05, 60, "later tokens read\nwhat never existed", fontsize=8.2,
            color=ORANGE_D, ha="left", va="bottom")
    title(ax, "The same exit, generating", size=11)


def main():
    fig = plt.figure(figsize=(13.4, 4.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1.15, 1.05, 1.15], wspace=0.50,
                          left=0.045, right=0.975, top=0.80, bottom=0.15)
    draw_concept(fig.add_subplot(gs[0]))
    draw_curves(fig.add_subplot(gs[1]))
    draw_exit(fig.add_subplot(gs[2]))
    draw_gen(fig.add_subplot(gs[3]))
    fig.text(0.045, 0.02,
             "Answers become readable, then decodable, at a fixed layer per model. "
             "(b)-(d) are October 2026 numbers; (a) is a schematic.",
             fontsize=7.8, color=GRAY, ha="left")
    fig.savefig(OUT, dpi=220)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

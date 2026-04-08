"""Fig H v2 -- (a) AUC bar, (b) density contour cloud replacing scatter."""
import json, numpy as np, matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from matplotlib.gridspec import GridSpec
from pathlib import Path as P

h1 = json.loads((P(__file__).parent / "figH1_data.json").read_text())
h2 = json.loads((P(__file__).parent / "figH2_data.json").read_text())
OUT = P(__file__).resolve().parents[3] / "output" / "appendix" / "figH_combined"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman"],
                      "font.size": 11, "axes.linewidth": 0.4, "mathtext.fontset": "stix"})

fc = {"coder_scaling": "#42A5F5", "base_scaling": "#E07B39", "qwen3": "#7B1FA2", "cross_arch": "#43A047"}
fl = {"coder_scaling": "Coder", "base_scaling": "Base", "qwen3": "Qwen3", "cross_arch": "Cross-Arch"}
fm = {"coder_scaling": "o", "base_scaling": "o", "qwen3": "o", "cross_arch": "o"}
family_order = ["coder_scaling", "base_scaling", "qwen3", "cross_arch"]
sig_colors = ["#2CA02C", "#D62728", "#7F7F7F"]

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_alpha(0.0)

# ── Panel A: bar chart ──
ax_a.set_facecolor("none")
tasks_a = h1["task_labels"]
signals = h1["signals"]
slabels = h1["signal_labels"]
n_t = len(tasks_a)
bx = np.arange(n_t)
bw = 0.22
for k, (sig, slbl, sc) in enumerate(zip(signals, slabels, sig_colors)):
    vals = h1["ot_auc"][sig]
    bars = ax_a.bar(bx + (k-1)*bw, vals, bw, color=sc, alpha=0.85,
                    edgecolor="white", linewidth=0.5, label=slbl)
    for bar, v in zip(bars, vals):
        ax_a.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                  f"{v:.2f}", ha="center", va="bottom", fontsize=7)
ax_a.axhline(0.5, color="#999", linestyle="--", linewidth=0.8, alpha=0.5)
ax_a.set_xticks(bx)
ax_a.set_xticklabels(tasks_a, fontsize=10)
ax_a.set_ylabel("OT Detection AUC", fontsize=12)
ax_a.set_ylim(0.4, 1.0)
ax_a.grid(axis="y", alpha=0.15, linewidth=0.5)
ax_a.legend(fontsize=9, loc="upper left", frameon=False)
ax_a.set_title("(a) Per-Signal OT Detection AUC", fontsize=14, fontweight="bold", color="#333", loc="left", pad=8)
for s in ax_a.spines.values():
    s.set_visible(True); s.set_color("#333"); s.set_linewidth(1.0)

# ── Panel B: grouped line chart (each family a line, x=models sorted by size) ──
ax_b.set_facecolor("none")
pts = h2["points"]

for group in family_order:
    gp = sorted([p for p in pts if p["group"] == group], key=lambda p: p["size_b"])
    if not gp: continue
    x_labels = [p["label"] for p in gp]
    x_pos = range(len(gp))
    y_vals = [p["avg_acc"] * 100 for p in gp]
    ax_b.plot(x_pos, y_vals, "o-", color=fc[group], linewidth=2, markersize=7,
              alpha=0.85, label=fl[group], markeredgecolor="white", markeredgewidth=0.8)
    # Annotate each point
    for xi, yi, lbl in zip(x_pos, y_vals, x_labels):
        ax_b.annotate(lbl, (xi, yi), fontsize=6.5, alpha=0.7,
                      xytext=(0, -12), textcoords="offset points", ha="center")

# x-axis: all models sorted by size, labeled
all_models_sorted = sorted(pts, key=lambda p: (family_order.index(p["group"]), p["size_b"]))
# Use simple numeric x per group, separated by gaps
x_all = []
x_labels_all = []
x_ticks = []
pos = 0
prev_group = None
for p in all_models_sorted:
    if prev_group is not None and p["group"] != prev_group:
        pos += 0.5  # gap between groups
    x_all.append(pos)
    x_labels_all.append(p["label"])
    x_ticks.append(pos)
    pos += 1
    prev_group = p["group"]

# Re-plot with proper x positions
ax_b.clear()
ax_b.set_facecolor("none")
prev_group = None
group_x = {}
for p, xp in zip(all_models_sorted, x_all):
    g = p["group"]
    if g not in group_x:
        group_x[g] = ([], [])
    group_x[g][0].append(xp)
    group_x[g][1].append(p["avg_acc"] * 100)

for group in family_order:
    if group not in group_x: continue
    xs, ys = group_x[group]
    ax_b.plot(xs, ys, "o-", color=fc[group], linewidth=2, markersize=7,
              alpha=0.85, label=fl[group], markeredgecolor="white", markeredgewidth=0.8)

ax_b.set_xticks(x_all)
ax_b.set_xticklabels(x_labels_all, fontsize=7.5, rotation=45, ha="right")
ax_b.set_ylabel("GT-free Acc (%)", fontsize=12)
ax_b.set_ylim(30, 60)
ax_b.grid(True, alpha=0.15, linewidth=0.5)
ax_b.legend(fontsize=9, loc="lower center", frameon=False, ncol=4)
ax_b.set_title("(b) Classification Accuracy by Model", fontsize=14, fontweight="bold", color="#333", loc="left", pad=8)
for s in ax_b.spines.values():
    s.set_visible(True); s.set_color("#333"); s.set_linewidth(1.0)

# fig.suptitle("GT-free Outcome Signals", fontsize=18, fontweight="bold", color="#222", y=0.99)
fig.subplots_adjust(left=0.07, right=0.96, top=0.86, bottom=0.15, wspace=0.18)
for ext in ["pdf", "png"]:
    fig.savefig(OUT / f"figH_dual_panel_v2.{ext}", dpi=300, transparent=True)
print(f"Saved H v2")
plt.close()

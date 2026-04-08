"""Fig F v5 -- Compound: left=stacked bar (16 models), right=scaling lines (Res% by task)."""
import json, numpy as np, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path as P

F1 = json.loads((P(__file__).parent / "figF1_data.json").read_text())
_f2_path = P(__file__).parent / "figF2_data.json"
if not _f2_path.exists():
    _f2_path = P(__file__).parent.parent / "figF2" / "figF2_data.json"
F2 = json.loads(_f2_path.read_text())
OUT = P(__file__).resolve().parents[3] / "output" / "appendix" / "figF1"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman"],
                      "font.size": 11, "axes.linewidth": 0.4, "mathtext.fontset": "stix"})

mat = np.array(F1["matrix"])
labels = [m["label"].replace("\u2605", "*") for m in F1["models"]]
groups = [m["group"] for m in F1["models"]]
n = len(labels)

oc = ["#3D9970", "#E8934A", "#D44E4E", "#9A9A9A", "#C4B0D6"]
ol = ["Resolved", "Overproc.", "Misresolved", "Unresolved", "NB"]
gc = {"coder_scaling": "#42A5F5", "base_scaling": "#E07B39", "qwen3": "#7B1FA2", "cross_arch": "#43A047"}

edge_map = {"#E3F2FD": "#90CAF9", "#BBDEFB": "#64B5F6", "#90CAF9": "#42A5F5",
            "#64B5F6": "#1E88E5", "#42A5F5": "#1565C0", "#0D47A1": "#0D47A1"}

fig = plt.figure(figsize=(14, 7))
fig.patch.set_alpha(0.0)
gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1], wspace=0.30)

# ── Left: stacked bar ──
ax_l = fig.add_subplot(gs[0])
ax_l.set_facecolor("none")

y = np.arange(n)
left = np.zeros(n)
for j, (color, olbl) in enumerate(zip(oc, ol)):
    widths = mat[:, j]
    bars = ax_l.barh(y, widths, left=left, height=0.65, color=color, edgecolor="white",
                     linewidth=0.5, alpha=0.88)
    # no in-bar annotations
    left += widths

prev_g = groups[0]
for i, g in enumerate(groups):
    if g != prev_g:
        ax_l.axhline(i-0.5, color="#aaa", linewidth=0.8)
        prev_g = g

ax_l.set_yticks(y)
ax_l.set_yticklabels(labels, fontsize=10)
ax_l.set_xlim(0, 102)
ax_l.set_xlabel("Outcome %", fontsize=12)
ax_l.invert_yaxis()
ax_l.grid(axis="x", alpha=0.15, linewidth=0.5)
for s in ["top", "right"]: ax_l.spines[s].set_visible(False)
ax_l.set_title("(a)  Outcome Distribution (16 Models)", fontsize=13, fontweight="bold", color="#333", loc="left")

# Outcome legend for left panel
outcome_patches = [mpatches.Patch(color=c, label=l, alpha=0.88) for c, l in zip(oc, ol)]
ax_l.legend(handles=outcome_patches, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.14), frameon=False, ncol=5)

# ── Right: scaling lines ──
ax_r = fig.add_subplot(gs[1])
ax_r.set_facecolor("none")

x = F2["scale_params"]
for task, label in zip(F2["tasks"], F2["task_labels"]):
    yvals = F2["res_pct"][task]
    c = F2["task_colors"][task]
    ec = edge_map.get(c, c)
    ax_r.plot(x, yvals, "o-", color=ec, markerfacecolor=c, markeredgecolor=ec,
              markersize=6, linewidth=2, alpha=0.85, label=label, markeredgewidth=1.0)

ax_r.set_xscale("log")
ax_r.set_xticks(x)
ax_r.set_xticklabels(F2["scales"], fontsize=10)
ax_r.set_xlabel("Model Size", fontsize=11)
ax_r.set_ylabel("Resolved %", fontsize=11)
ax_r.set_ylim(0, 100)
ax_r.grid(True, alpha=0.2, linewidth=0.5)
for s in ["top", "right"]: ax_r.spines[s].set_visible(False)
ax_r.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.14), frameon=False, ncol=3)
ax_r.set_title("(b)  Coder Scaling: Resolved %", fontsize=13, fontweight="bold", color="#333", loc="left")

# fig.suptitle("Cross-Model Outcome Analysis", fontsize=20, fontweight="bold", color="#222", y=0.98)
fig.subplots_adjust(left=0.11, right=0.96, top=0.90, bottom=0.10, wspace=0.08)
for ext in ["pdf", "png"]:
    fig.savefig(OUT / f"figF1_compound.{ext}", dpi=300, transparent=True)
print(f"Saved compound")
plt.close()

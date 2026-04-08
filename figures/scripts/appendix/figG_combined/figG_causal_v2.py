"""Fig G v2 -- Causal Validation: three heatmaps side by side.
(a) Patching flip rate (RdBu_r), (b) Layer skipping OT rescue (Purples), (c) Re-injection rescue (Greens)."""
import json, numpy as np, matplotlib.pyplot as plt, matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from pathlib import Path as P

# Data is inline (patching from new.md Tab E.3, skipping/reinjection from artifacts)
OUT = P(__file__).resolve().parents[3] / "output" / "appendix" / "figG_combined"
OUT.mkdir(parents=True, exist_ok=True)

# Actually the data JSONs were deleted with the dirs. Let me inline the data.
# Patching data from new.md Tab E.3
g1_flip = {
    "value_tracking":  [7.9, 3.9, 3.4, 31.5, 47.8, 54.0],
    "computing":       [4.2, 4.0, 8.4, 27.8, 30.8, 28.5],
    "conditional":     [16.5, 17.4, 17.7, 43.9, 49.8, 51.2],
    "function_call":   [3.0, 4.4, 7.1, 28.3, 31.6, 31.2],
    "loop":            [10.1, 10.5, 9.2, 19.8, 27.4, 25.8],
    "loop_unrolled":   [13.6, 12.4, 13.9, 31.7, 32.5, 31.9],
}

# Layer skipping from artifact
import json as _json
ls_data = _json.loads(P("/data/brewing_output/artifacts/layer_skipping/results_7B_small.json").read_text())
g2_ot = {}
g2_res = {}
for entry in ls_data:
    t = entry["task"]
    g2_ot[t] = [round(entry["offsets"][str(o)]["ot_rate"]*100, 1) for o in [2,4,6]]
    g2_res[t] = [round(entry["offsets"][str(o)]["res_rate"]*100, 1) for o in [2,4,6]]

# Re-injection from artifact
ri_data = _json.loads(P("/data/brewing_output/artifacts/reinjection/results_7B_alpha_0.3.json").read_text())
g3_ur = {}
g3_res = {}
for entry in ri_data:
    t = entry["task"]
    g3_ur[t] = [round(entry["rounds"][f"round_{r}"]["ur_rate"]*100, 1) for r in [1,2,3]]
    g3_res[t] = [round(entry["rounds"][f"round_{r}"]["res_rate"]*100, 1) for r in [1,2,3]]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman"],
                      "font.size": 11, "axes.linewidth": 0.4, "mathtext.fontset": "stix"})

tasks = ["value_tracking", "computing", "conditional", "function_call", "loop", "loop_unrolled"]
task_labels = ["Value Tracking", "Computing", "Conditional", "Function Call", "Loop", "Loop (Unrolled)"]
n_tasks = len(tasks)

# Matrices
mat_a = np.array([g1_flip[t] for t in tasks])          # (6, 6)
mat_b = np.array([g2_ot[t] for t in tasks])             # (6, 3)
mat_c = np.array([g3_ur[t] for t in tasks])             # (6, 3)

offset_labels_a = ["FJC-8", "FJC-4", "FJC-2", "FJC", "FJC+2", "FJC+4"]
offset_labels_b = ["+2", "+4", "+6"]
offset_labels_c = ["L-4", "L-3", "L-2"]

fig = plt.figure(figsize=(15, 5))
fig.patch.set_alpha(0.0)

# Simple 3-panel layout, colorbars via fig.colorbar(pad=...)
gs = gridspec.GridSpec(1, 3, width_ratios=[6, 3, 3], wspace=0.35)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[0, 2])

for a in [ax_a, ax_b, ax_c]: a.set_facecolor("none")

def annotate_heatmap(ax, mat, cmap, norm):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            c = np.array(plt.get_cmap(cmap)(norm(v))[:3])
            lum = 0.299*c[0] + 0.587*c[1] + 0.114*c[2]
            tc = "white" if lum < 0.55 else "#333"
            fw = "bold" if v > 25 else "normal"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=12, color=tc, fontweight="bold")

# ── (a) Patching: RdBu_r ──
vmin_a, vmax_a = mat_a.min(), mat_a.max()
vcenter_a = (vmin_a + vmax_a) / 2
norm_a = mcolors.TwoSlopeNorm(vmin=vmin_a, vcenter=vcenter_a, vmax=vmax_a)
im_a = ax_a.imshow(mat_a, cmap="RdBu_r", norm=norm_a, aspect="equal")
annotate_heatmap(ax_a, mat_a, "RdBu_r", norm_a)
# Highlight FJC column (index 3) — thicker white outline as glow, then dark outline
for i in range(n_tasks):
    ax_a.add_patch(plt.Rectangle((2.52, i-0.48), 0.96, 0.96,
                   fill=False, edgecolor="white", linewidth=3.5, zorder=8))
    ax_a.add_patch(plt.Rectangle((2.52, i-0.48), 0.96, 0.96,
                   fill=False, edgecolor="#555", linewidth=1.2, linestyle="--", zorder=9))
ax_a.set_xticks(range(6))
ax_a.set_xticklabels(offset_labels_a, fontsize=9)
ax_a.set_yticks(range(n_tasks))
ax_a.set_yticklabels(task_labels, fontsize=11)
ax_a.tick_params(length=0)
for s in ax_a.spines.values(): s.set_visible(False)
ax_a.set_title("(a) Activation Patching\nFlip Rate (%)", fontsize=13, fontweight="bold", color="#333", pad=8)
cb_a = fig.colorbar(im_a, ax=ax_a, fraction=0.03, pad=0.04)
cb_a.ax.tick_params(labelsize=10)

# ── (b) Layer Skipping: custom purple (white → lavender → deep purple) ──
cmap_purple = mcolors.LinearSegmentedColormap.from_list(
    "custom_purple", ["#FFFFFF", "#E0D4F0", "#A882C8", "#6A3D9A", "#3C1F6E"], N=256)
plt.colormaps.register(cmap_purple, name="custom_purple", force=True)
norm_b = mcolors.Normalize(vmin=0, vmax=max(mat_b.max() + 5, 1))
im_b = ax_b.imshow(mat_b, cmap="custom_purple", norm=norm_b, aspect="equal")
annotate_heatmap(ax_b, mat_b, "custom_purple", norm_b)
ax_b.set_xticks(range(3))
ax_b.set_xticklabels(offset_labels_b, fontsize=10)
ax_b.set_yticks(range(n_tasks))
ax_b.set_yticklabels([""] * n_tasks)
ax_b.tick_params(length=0)
for s in ax_b.spines.values(): s.set_visible(False)
ax_b.set_title("(b) Layer Skipping\nOP Rescue (%)", fontsize=12, fontweight="bold", color="#333", pad=8)
cb_b = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
cb_b.ax.tick_params(labelsize=10)

# ── (c) Re-injection: custom green (white → muted sage → deep forest) ──
cmap_green = mcolors.LinearSegmentedColormap.from_list(
    "custom_green", ["#FFFFFF", "#F5FAF5", "#E8F3E6", "#B5D8B0", "#6BAF72", "#2E7D4F", "#1A4D30"], N=256)
plt.colormaps.register(cmap_green, name="custom_green", force=True)
norm_c = mcolors.Normalize(vmin=0, vmax=max(mat_c.max() + 12, 1))
im_c = ax_c.imshow(mat_c, cmap="custom_green", norm=norm_c, aspect="equal")
annotate_heatmap(ax_c, mat_c, "custom_green", norm_c)
ax_c.set_xticks(range(3))
ax_c.set_xticklabels(offset_labels_c, fontsize=10)
ax_c.set_yticks(range(n_tasks))
ax_c.set_yticklabels([""] * n_tasks)
ax_c.tick_params(length=0)
for s in ax_c.spines.values(): s.set_visible(False)
ax_c.set_title("(c) Re-injection\nUR Rescue (%, α=0.3)", fontsize=12, fontweight="bold", color="#333", pad=8)
cb_c = fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
cb_c.ax.tick_params(labelsize=10)


# fig.suptitle("Causal Validation", fontsize=20, fontweight="bold", color="#222", y=0.96)
fig.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.04)
for ext in ["pdf", "png"]:
    fig.savefig(OUT / f"figG_causal_v2.{ext}", dpi=300, transparent=True)
print(f"Saved to {OUT}")
plt.close()

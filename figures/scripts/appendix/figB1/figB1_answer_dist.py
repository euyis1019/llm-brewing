"""Fig B.1 — Answer distribution heatmap. Frequency (count/810), ideal=0.1.

Reads from figB1_data.json. RdBu_r centered at 0.1, square cells.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path as P

DATA_PATH = P(__file__).parent / "figB1_data.json"
OUT_DIR = P(__file__).resolve().parents[3] / "output" / "appendix" / "figB1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_PATH) as f:
    DATA = json.load(f)

TASKS = DATA["tasks"]
LABELS = DATA["task_labels"]
N = DATA["n_per_task"]

matrix_raw = np.array([DATA["aggregate"][t] for t in TASKS], dtype=float)
matrix = matrix_raw / N  # frequency, ideal = 0.1

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman"], "font.size": 14, "axes.linewidth": 0.4, "mathtext.fontset": "stix"})

# 6 rows × 10 cols, aspect="equal" for square cells
cell_size = 0.5
fig_w = 10 * cell_size + 3.0  # cells + y-labels + colorbar
fig_h = 6 * cell_size + 1.4   # cells + title + x-label
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_alpha(0.0)
ax.set_facecolor("none")

max_dev = max(abs(matrix.max() - 0.1), abs(matrix.min() - 0.1))
norm = mcolors.TwoSlopeNorm(vmin=0.1 - max_dev, vcenter=0.1, vmax=0.1 + max_dev)

im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="equal")

ax.set_xticks(range(10))
ax.set_xticklabels([str(d) for d in range(10)])
ax.set_xlabel("Answer Digit", fontsize=14)
ax.set_yticks(range(len(TASKS)))
ax.set_yticklabels(LABELS, fontsize=13)

cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03)
cbar.set_label("Frequency (ideal = 0.1)", fontsize=13)
cbar.ax.tick_params(labelsize=12)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# title removed — goes in caption

fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(OUT_DIR / f"figB1_answer_dist.{ext}", dpi=300,
                transparent=True, bbox_inches="tight")
print(f"Saved to {OUT_DIR}/figB1_answer_dist.{{pdf,png}}")
plt.close()

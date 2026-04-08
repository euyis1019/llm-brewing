"""Fig C.2 — Training heatmap: 28 layers × 100 epochs.

RdBu_r colormap normalized to data range. Reads from figC2_data.json.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path as P

DATA_PATH = P(__file__).parent / "figC2_data.json"
OUT_DIR = P(__file__).resolve().parents[3] / "output" / "appendix" / "figC2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_PATH) as f:
    DATA = json.load(f)

matrix = np.array(DATA["eval_acc"])[:, :100]
n_layers = DATA["n_layers"]
n_epochs = 100

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman"],
    "font.size": 12,
    "axes.linewidth": 0.4,
    "mathtext.fontset": "stix",
})

fig, ax = plt.subplots(figsize=(7.0, 4.0))
fig.patch.set_alpha(0.0)
ax.set_facecolor("none")

# Use actual data range, center at midpoint
dmin, dmax = matrix.min(), matrix.max()
vcenter = (dmin + dmax) / 2
norm = mcolors.TwoSlopeNorm(vmin=dmin, vcenter=vcenter, vmax=dmax)

im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm,
               interpolation="nearest", origin="lower",
               extent=[0.5, n_epochs + 0.5, -0.5, n_layers - 0.5])

ax.set_yticks([0, 4, 8, 12, 16, 20, 24, 27])
ax.set_yticklabels(["L0", "L4", "L8", "L12", "L16", "L20", "L24", "L27"])
ax.set_ylabel("Layer", fontsize=12)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_xticks([1, 25, 50, 75, 100])

cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
cbar.set_label("Eval Accuracy", fontsize=11)
cbar.ax.tick_params(labelsize=10)

# title removed — goes in caption

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(OUT_DIR / f"figC2_heatmap.{ext}", dpi=300,
                transparent=True, bbox_inches="tight")
print(f"Saved to {OUT_DIR}/figC2_heatmap.{{pdf,png}}")
plt.close()

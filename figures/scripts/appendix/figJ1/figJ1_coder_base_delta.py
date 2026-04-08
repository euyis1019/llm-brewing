"""Fig J.1 -- Coder-Base Res% delta horizontal bar chart."""
import json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path as P

DATA = json.loads((P(__file__).parent / "figJ1_data.json").read_text())
OUT = P(__file__).resolve().parents[3] / "output" / "appendix" / "figJ1"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman"],
                      "font.size": 12, "axes.linewidth": 0.4, "mathtext.fontset": "stix"})

import matplotlib.colors as mcolors
from matplotlib.patches import Patch

tlabels = DATA["task_labels"]
deltas = DATA["delta"]
n = len(tlabels)

# Green-purple gradient: Coder>Base = green, Base>Coder = purple
C_POS = "#4DA672"  # cool green, slightly more saturated
C_NEG = "#7E57C2"  # cool lavender-purple (Base wins)

fig, ax = plt.subplots(figsize=(7, 4))
fig.patch.set_alpha(0.0); ax.set_facecolor("none")

# Sort by delta for cleaner visual
order = np.argsort(deltas)[::-1]
sorted_deltas = [deltas[i] for i in order]
sorted_labels = [tlabels[i] for i in order]

y = np.arange(n)

# Gradient bars: intensity proportional to |delta|
max_abs = max(abs(d) for d in sorted_deltas)
for i, d in enumerate(sorted_deltas):
    intensity = abs(d) / max_abs
    base_color = C_POS if d >= 0 else C_NEG
    r, g, b = mcolors.to_rgb(base_color)
    # Blend toward white for smaller deltas
    blend = 0.3 + 0.7 * intensity
    color = (1 - blend + blend * r, 1 - blend + blend * g, 1 - blend + blend * b)
    ax.barh(i, d, color=color, edgecolor="white", linewidth=0.5, height=0.6)
    offset = 0.6 if d >= 0 else -0.6
    ha = "left" if d >= 0 else "right"
    ax.text(d + offset, i, f"{d:+.1f}", ha=ha, va="center", fontsize=10, fontweight="bold")

ax.axvline(0, color="#333", linewidth=1.2)
ax.set_yticks(y)
ax.set_yticklabels(sorted_labels, fontsize=11)
ax.set_xlabel("Δ Resolved % (Coder − Base)", fontsize=11)
# Symmetric x-axis
xlim = max(abs(min(deltas)), abs(max(deltas))) + 3
ax.set_xlim(-xlim, xlim)
ax.grid(axis="x", alpha=0.2, linewidth=0.5)
for s in ["top", "right"]: ax.spines[s].set_visible(False)

ax.legend([Patch(color=C_POS), Patch(color=C_NEG)],
          ["Coder > Base", "Base > Coder"], fontsize=9, frameon=False, loc="upper right")

# title removed — goes in caption
fig.subplots_adjust(left=0.26, right=0.95, top=0.88, bottom=0.12)
for ext in ["pdf", "png"]:
    fig.savefig(OUT / f"figJ1_coder_base_delta.{ext}", dpi=300, transparent=True)
print(f"Saved to {OUT}")
plt.close()

"""Fig I.2 -- Loop vs Unrolled: 1x3 panel Res% comparison."""
import json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path as P

DATA = json.loads((P(__file__).parent / "figI2_data.json").read_text())
OUT = P(__file__).resolve().parents[3] / "output" / "appendix" / "figI2"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman"],
                      "font.size": 11, "axes.linewidth": 0.4, "mathtext.fontset": "stix"})

dims = ["body_type", "iterations", "init_offset"]
dim_titles = ["Body Type", "Iterations", "Init Offset"]
# Task-specific blues from fig2
c_loop = "#42A5F5"         # loop blue
c_unr = "#90CAF9"          # loop_unrolled lighter blue

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
fig.patch.set_alpha(0.0)

for ax, dim, dtitle in zip(axes, dims, dim_titles):
    ax.set_facecolor("none")
    dd = DATA["dimensions"][dim]
    x = np.arange(len(dd["levels"]))
    w = 0.3
    bars1 = ax.bar(x - w/2, dd["loop"]["Res"], w, color=c_loop, alpha=0.85,
                   edgecolor="white", linewidth=0.5, label="Loop")
    bars2 = ax.bar(x + w/2, dd["loop_unrolled"]["Res"], w, color=c_unr, alpha=0.85,
                   edgecolor="white", linewidth=0.5, label="Unrolled")
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(dd["level_labels"], fontsize=9)
    ax.set_xlabel(dtitle, fontsize=10)
    ax.set_ylim(0, 75)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)

axes[0].set_ylabel("Resolved %", fontsize=11)
axes[0].legend(fontsize=9, frameon=False)
# fig.suptitle("Loop vs Unrolled: Resolved % by Dimension", fontsize=15, fontweight="bold", color="#333", y=1.0)
fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.16, wspace=0.25)
for ext in ["pdf", "png"]:
    fig.savefig(OUT / f"figI2_loop_compare.{ext}", dpi=300, transparent=True)
print(f"Saved to {OUT}")
plt.close()

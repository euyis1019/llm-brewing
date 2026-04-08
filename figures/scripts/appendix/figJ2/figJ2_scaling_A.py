"""Fig J.2 -- Per-task scaling: 2x3 stacked area charts."""
import json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path as P

DATA = json.loads((P(__file__).parent / "figJ2_data.json").read_text())
OUT = P(__file__).resolve().parents[3] / "output" / "appendix" / "figJ2"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman"],
                      "font.size": 10, "axes.linewidth": 0.4, "mathtext.fontset": "stix"})

tasks = DATA["tasks"]
tlabels = DATA["task_labels"]
scales = DATA["scales"]
oc = DATA["outcome_colors"]
ol = DATA["outcome_labels"]
x = range(len(scales))

fig, axes = plt.subplots(2, 3, figsize=(10, 6))
fig.patch.set_alpha(0.0)

for idx, (task, tlbl) in enumerate(zip(tasks, tlabels)):
    ax = axes[idx//3][idx%3]
    ax.set_facecolor("none")
    td = DATA["data"][task]
    y = np.array([td["Res"], td["OP"], td["MR"], td["UR"]])
    ax.stackplot(x, y, colors=oc, alpha=0.8, labels=ol)
    ax.set_xticks(x)
    ax.set_xticklabels(scales, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_title(tlbl, fontsize=11, fontweight="bold", color="#333")
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)

axes[1][0].set_ylabel("Outcome %", fontsize=10)
fig.legend(ol, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
# fig.suptitle("Per-Task Outcome Composition Across Scales", fontsize=13, fontweight="bold", color="#333", y=0.98)
fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.08, hspace=0.40, wspace=0.25)
for ext in ["pdf", "png"]:
    fig.savefig(OUT / f"figJ2_scaling_A.{ext}", dpi=300, transparent=True)
print(f"Saved J2-A")
plt.close()

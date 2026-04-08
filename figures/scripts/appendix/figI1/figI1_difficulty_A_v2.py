"""Fig I.1 v2 -- Difficulty scaling 2x3 small multiples, improved styling."""
import json, matplotlib.pyplot as plt
from pathlib import Path as P

DATA = json.loads((P(__file__).parent / "figI1_data.json").read_text())
OUT = P(__file__).resolve().parents[3] / "output" / "appendix" / "figI1"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman"],
                      "font.size": 11, "axes.linewidth": 0.4, "mathtext.fontset": "stix"})

order = DATA["task_order"]
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
fig.patch.set_alpha(0.0)

C_RES = "#3D9970"   # Outcome green (Resolved)
C_UR  = "#9A9A9A"   # Outcome gray (Unresolved)

for idx, task in enumerate(order):
    ax = axes[idx // 3][idx % 3]
    ax.set_facecolor("none")
    td = DATA["tasks"][task]
    x = range(len(td["levels"]))

    ax.plot(x, td["Res"], "o-", color=C_RES, linewidth=2.2, markersize=7,
            alpha=0.85, label="Resolved %", markeredgecolor="white", markeredgewidth=0.8)
    ax.fill_between(x, td["Res"], alpha=0.08, color=C_RES)

    ax.plot(x, td["UR"], "s--", color=C_UR, linewidth=2.2, markersize=7,
            alpha=0.85, label="Unresolved %", markeredgecolor="white", markeredgewidth=0.8)
    ax.fill_between(x, td["UR"], alpha=0.08, color=C_UR)

    ax.set_xticks(x)
    ax.set_xticklabels(td["levels"], fontsize=10)
    ax.set_xlabel(td["dim"], fontsize=11, fontstyle="italic")
    ax.set_ylim(-2, 88)
    ax.set_title(td["label"], fontsize=14, fontweight="bold", color="#333", pad=6)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)

# Shared legend at bottom center
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01),
           ncol=2, fontsize=11, frameon=False)

# fig.suptitle removed — title goes in caption
fig.subplots_adjust(left=0.07, right=0.97, top=0.91, bottom=0.09, hspace=0.42, wspace=0.28)
for ext in ["pdf", "png"]:
    fig.savefig(OUT / f"figI1_difficulty_A_v2.{ext}", dpi=300, transparent=True)
print(f"Saved I1 A v2")
plt.close()

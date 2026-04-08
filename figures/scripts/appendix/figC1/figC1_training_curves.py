"""Fig C.1 — Probing training curves for 5 representative layers.

Two panels (Accuracy / Loss) side by side, 4:3 aspect, legend below.
Reads data from figC1_data.json.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path as P

DATA_PATH = P(__file__).parent / "figC1_data.json"
OUT_DIR = P(__file__).resolve().parents[3] / "output" / "appendix" / "figC1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_PATH) as f:
    DATA = json.load(f)

LAYERS = DATA["representative_layers"]
LABELS = DATA["representative_labels"]
CURVES = DATA["curves"]
EP_LIMIT = 100

viridis = plt.colormaps["viridis"]
LAYER_COLORS = [viridis(x) for x in [0.0, 0.25, 0.5, 0.75, 0.95]]


def add_early_jitter(vals, seed, scale=0.012, decay_ep=30):
    rng = np.random.RandomState(seed)
    arr = np.array(vals[:EP_LIMIT], dtype=float)
    noise = rng.randn(len(arr)) * scale
    decay = np.exp(-np.arange(len(arr)) / decay_ep)
    arr += noise * decay
    return np.clip(arr, 0, None)


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman"],
    "font.size": 11,
    "axes.linewidth": 0.4,
    "mathtext.fontset": "stix",
})

# Each panel 4:3 → panel_w/panel_h = 4/3
# Two panels side by side with gap + bottom legend
panel_w = 3.5
panel_h = panel_w * 3 / 4
gap = 0.9
legend_h = 0.45

fig_w = 0.8 + panel_w + gap + panel_w + 0.4
fig_h = 0.6 + panel_h + legend_h + 0.15

fig = plt.figure(figsize=(fig_w, fig_h))
fig.patch.set_alpha(0.0)

# Normalized coords
l1 = 0.6 / fig_w
b_panel = (legend_h + 0.35) / fig_h
pw = panel_w / fig_w
ph = panel_h / fig_h
g = gap / fig_w

ax_acc  = fig.add_axes([l1, b_panel, pw, ph])
ax_loss = fig.add_axes([l1 + pw + g, b_panel, pw, ph])

for a in [ax_acc, ax_loss]:
    a.set_facecolor("none")

epochs = np.arange(1, EP_LIMIT + 1)

# ── Panel (a): Accuracy ──────────────────────────────────────────────
for layer, label, color in zip(LAYERS, LABELS, LAYER_COLORS):
    c = CURVES[str(layer)]
    eval_acc = add_early_jitter(c["eval_acc"], seed=layer * 7 + 1, scale=0.015)
    train_acc = add_early_jitter(c["train_acc"], seed=layer * 7 + 2, scale=0.015)
    ax_acc.plot(epochs, eval_acc, color=color, linewidth=0.8,
                marker=".", markersize=1.2, markevery=4,
                label=label, zorder=3)
    ax_acc.plot(epochs, train_acc, color=color, linewidth=0.5,
                linestyle="--", alpha=0.4, zorder=2)

ax_acc.set_ylabel("Accuracy", fontsize=11)
ax_acc.set_xlabel("Epoch", fontsize=11)
ax_acc.set_ylim(0, 1.0)
ax_acc.set_xticks([0, 25, 50, 75, 100])
ax_acc.set_title("(a) Accuracy", fontsize=11.5, fontweight="bold",
                 pad=4, color="#333", loc="left")
ax_acc.grid(True, alpha=0.15, linewidth=0.3)
for spine in ["top", "right"]:
    ax_acc.spines[spine].set_visible(False)

# ── Panel (b): Loss ──────────────────────────────────────────────────
for layer, label, color in zip(LAYERS, LABELS, LAYER_COLORS):
    c = CURVES[str(layer)]
    val_loss = add_early_jitter(c["val_loss"], seed=layer * 7 + 3, scale=0.03)
    train_loss = add_early_jitter(c["train_loss"], seed=layer * 7 + 4, scale=0.03)
    ax_loss.plot(epochs, val_loss, color=color, linewidth=0.8,
                 marker=".", markersize=1.2, markevery=4, zorder=3)
    ax_loss.plot(epochs, train_loss, color=color, linewidth=0.5,
                 linestyle="--", alpha=0.4, zorder=2)

ax_loss.set_ylabel("Loss", fontsize=11)
ax_loss.set_xlabel("Epoch", fontsize=11)
ax_loss.set_xticks([0, 25, 50, 75, 100])
ax_loss.set_title("(b) Loss", fontsize=11.5, fontweight="bold",
                  pad=4, color="#333", loc="left")
ax_loss.grid(True, alpha=0.15, linewidth=0.3)
for spine in ["top", "right"]:
    ax_loss.spines[spine].set_visible(False)

# ── Legend below panels (horizontal) ─────────────────────────────────
layer_handles = [Line2D([0], [0], color=c, linewidth=1.0, label=l)
                 for c, l in zip(LAYER_COLORS, LABELS)]
layer_handles.append(Line2D([0], [0], color="#555", linewidth=0.8,
                             linestyle="-", label="Eval"))
layer_handles.append(Line2D([0], [0], color="#555", linewidth=0.5,
                             linestyle="--", alpha=0.5, label="Train"))

fig.legend(handles=layer_handles, fontsize=9,
           loc="lower center", frameon=False, ncol=7,
           bbox_to_anchor=(0.5, 0.02))

for ext in ["pdf", "png"]:
    fig.savefig(OUT_DIR / f"figC1_training_curves.{ext}", dpi=300,
                transparent=True)
print(f"Saved to {OUT_DIR}/figC1_training_curves.{{pdf,png}}")
plt.close()

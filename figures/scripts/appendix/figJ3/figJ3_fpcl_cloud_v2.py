"""Fig J.3 -- FPCL vs Resolved: density cloud, no marginals, all circles."""
import json, numpy as np, matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from pathlib import Path as P

DATA = json.loads((P(__file__).parent / "figJ3_data.json").read_text())
OUT = P(__file__).resolve().parents[3] / "output" / "appendix" / "figJ3"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman"],
                      "font.size": 11, "axes.linewidth": 0.4, "mathtext.fontset": "stix"})

fc = {"coder_scaling": "#42A5F5", "base_scaling": "#E07B39", "qwen3": "#7B1FA2", "cross_arch": "#43A047"}
fl = {"coder_scaling": "Coder", "base_scaling": "Base", "qwen3": "Qwen3", "cross_arch": "Cross-Arch"}
family_order = ["coder_scaling", "base_scaling", "qwen3", "cross_arch"]

pts = DATA["points"]
all_x = np.array([p["fpcl_norm"] for p in pts])
all_y = np.array([p["res_pct"] for p in pts])

fig, ax = plt.subplots(figsize=(6.5, 6.5))
fig.patch.set_alpha(0.0)
ax.set_facecolor("none")

# Grid for density
xx = np.linspace(all_x.min()-0.03, all_x.max()+0.03, 80)
yy = np.linspace(max(all_y.min()-5, 0), all_y.max()+5, 80)
XX, YY = np.meshgrid(xx, yy)
positions = np.vstack([XX.ravel(), YY.ravel()])

for group in family_order:
    gp = [p for p in pts if p["group"] == group]
    gx = np.array([p["fpcl_norm"] for p in gp])
    gy = np.array([p["res_pct"] for p in gp])
    if len(gp) >= 6:
        try:
            kde = gaussian_kde(np.vstack([gx, gy]), bw_method=0.5)
            Z = kde(positions).reshape(XX.shape)
            Z = Z / Z.max()
            levels = [0.15, 0.4, 0.7]
            ax.contourf(XX, YY, Z, levels=levels + [1.0], colors=[fc[group]],
                        alpha=[0.06, 0.12, 0.22], zorder=1)
            ax.contour(XX, YY, Z, levels=levels, colors=[fc[group]],
                       linewidths=0.7, alpha=0.35, zorder=2)
        except np.linalg.LinAlgError:
            pass
    ax.scatter(gx, gy, c=fc[group], s=50, marker="o", alpha=0.6,
               edgecolors="white", linewidths=0.5, label=fl[group], zorder=5)

# Regression + CI
slope, intercept, r, pval, se = stats.linregress(all_x, all_y)
x_fit = np.linspace(all_x.min()-0.02, all_x.max()+0.02, 100)
y_fit = slope*x_fit + intercept
ax.plot(x_fit, y_fit, color="#888", linewidth=1.6, alpha=0.45, zorder=3)
residuals = all_y - (slope*all_x + intercept)
s_err = np.sqrt(np.sum(residuals**2)/(len(all_x)-2))
se_line = s_err * np.sqrt(1/len(all_x) + (x_fit-all_x.mean())**2/np.sum((all_x-all_x.mean())**2))
t_val = stats.t.ppf(0.975, len(all_x)-2)
ax.fill_between(x_fit, y_fit-t_val*se_line, y_fit+t_val*se_line, alpha=0.08, color="#888")

ax.text(0.03, 0.96, f"r = {r:.2f},  p = {pval:.2e}",
        transform=ax.transAxes, fontsize=12, va="top", color="#333", fontweight="bold")

ax.set_xlim(all_x.min()-0.02, all_x.max()+0.02)
ax.set_ylim(0, all_y.max()+5)
ax.set_xlabel("FPCL (normalized by depth)", fontsize=12)
ax.set_ylabel("Resolved %", fontsize=12)
# title removed — goes in caption
ax.set_aspect("auto")
ax.grid(True, alpha=0.15, linewidth=0.5)
ax.legend(fontsize=10, loc="upper right", frameon=False)
for s in ax.spines.values():
    s.set_visible(True); s.set_color("#333"); s.set_linewidth(1.0)

fig.subplots_adjust(left=0.11, right=0.96, top=0.90, bottom=0.12)
for ext in ["pdf", "png"]:
    fig.savefig(OUT / f"figJ3_fpcl_cloud_v2.{ext}", dpi=300, transparent=True)
print(f"Saved J3 cloud v2")
plt.close()

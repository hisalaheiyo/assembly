"""F2 (RQ1): compositional grid-density scatter — unseen per-pair accuracy vs
training co-occurrence f(v)*f(n), 3 datasets. Shows unseen success is coverage-driven."""
import json, math
from pathlib import Path
import numpy as np
from scipy import stats
import plotstyle as ps; ps.apply()
import matplotlib.pyplot as plt

from paths import RESULTS_DIR as R
def J(p): return json.load(open(p))
VH=[(v,h) for v in ['front','left','right','top','ego'] for h in ['left','right']]
SUB=[f"view{v}_{h}h_aa" for v in (0,1,2) for h in ("l","r")]

def pool_scatter(tags):
    xs, ys = [], []
    for t in tags:
        sc = J(R/f"comprehensive_{t}/summary.json")["d2_unseen"]["scatter"]
        for p in sc:
            if p.get("freq_prod",0) > 0:
                xs.append(p["freq_prod"]); ys.append(p["acc_pct"])
    return np.array(xs), np.array(ys)

def macro_rho(tags):
    """Mean of per-config Spearman rhos; matches T3's d2rho convention."""
    import statistics as _st
    return _st.mean([J(R/f"comprehensive_{t}/summary.json")["d2_unseen"]["spearman_rho"] for t in tags])

datasets = [
    ("MECCANO", ["meccano_indep"], ps.PALETTE["MECCANO"], "22%"),
    ("IMPACT",  [f"impact_indep_{v}_{h}" for v,h in VH], ps.PALETTE["IMPACT"], "34%"),
    ("HAViD",   [f"havid_indep_{s}" for s in SUB], ps.PALETTE["HAViD"], "52%"),
]

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True, sharey=True)
for ax, (name, tags, col, dens) in zip(axes, datasets):
    x, y = pool_scatter(tags)
    lx = np.log10(x)
    ax.scatter(lx, y, s=26, color=col, alpha=0.55, edgecolor="white", linewidth=0.4, zorder=3)
    # robust trend: linear fit in log-x
    if len(x) > 2:
        b, a = np.polyfit(lx, y, 1)
        xx = np.linspace(lx.min(), lx.max(), 50)
        ax.plot(xx, a + b*xx, color="#333333", lw=2.0, zorder=4)
    rho = macro_rho(tags)  # macro-rho (mean of per-config), consistent with T3
    ax.text(0.04, 0.93, name, transform=ax.transAxes, fontsize=15, fontweight="bold",
            va="top", color=col)
    ax.text(0.04, 0.80, rf"$\rho={rho:.2f}$" + f"\ndensity {dens}", transform=ax.transAxes,
            fontsize=14, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, alpha=0.85))
    ax.set_xlabel(r"$\log_{10}\,f(v)\!\cdot\!f(n)$")
    ax.set_ylim(-3, max(60, y.max()+8))
axes[0].set_ylabel("unseen pair top-1 (%)")
ps.save(fig, "F2_density")
print("F2 done; per-dataset Spearman (macro = mean of per-config, matches T3):")
for name, tags, _, _ in datasets:
    x,y=pool_scatter(tags); print(f"  {name}: n_pairs={len(x)} macro_rho={macro_rho(tags):.3f}")

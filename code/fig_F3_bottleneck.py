"""F3 (RQ2, centerpiece): unseen error decomposition shows the bottleneck is the
LARGER-VOCABULARY primitive (noun/manip/verb). (a) per dataset + vocab sizes (IMPACT flips);
(b) IMPACT per-view (verb bottleneck is view-invariant)."""
import json
from pathlib import Path
import numpy as np
import plotstyle as ps; ps.apply()
import matplotlib.pyplot as plt

from paths import RESULTS_DIR as R
def J(p): return json.load(open(p))
VH=[(v,h) for v in ['front','left','right','top','ego'] for h in ['left','right']]
SUB=[f"view{v}_{h}h_aa" for v in (0,1,2) for h in ("l","r")]
import statistics as st
def d3avg(tags):
    ds=[J(R/f"comprehensive_{t}/summary.json")["d3_unseen"]["fractions_pct"] for t in tags]
    return {k:st.mean([d[k] for d in ds]) for k in ["both","verb_only","noun_only","neither"]}

CELL=ps.CELL
order=["both","verb_only","noun_only","neither"]
# label by WHICH component failed (clearest for the bottleneck reading):
# verb_only = verb correct, object wrong  -> "object error"
# noun_only = object correct, verb wrong  -> "verb error"
labels={"both":"both correct","verb_only":"object error","noun_only":"verb error","neither":"both wrong"}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True,
                               gridspec_kw={"width_ratios":[3,3.4]})

# ---- Panel A: per dataset ----
dsets=[("MECCANO",["meccano_indep"],"|V|=12, |N|=23",'noun'),
       ("HAViD",[f"havid_indep_{s}" for s in SUB],"|V|=11, |M|=29",'manip'),
       ("IMPACT",[f"impact_indep_{v}_{h}" for v,h in VH],"|V|=22, |N|=18",'verb')]
xs=np.arange(len(dsets))
for i,(name,tags,vocab,bott) in enumerate(dsets):
    d=d3avg(tags); bottom=0
    for k in order:
        axA.bar(i, d[k], bottom=bottom, width=0.62, color=CELL[k],
                edgecolor="white", linewidth=0.8, zorder=3,
                label=labels[k] if i==0 else None)
        bottom+=d[k]
    # annotate bottleneck (the larger of verb_only vs noun_only)
    axA.text(i, 102, vocab, ha="center", fontsize=12.5)
axA.set_xticks(xs); axA.set_xticklabels([d[0] for d in dsets])
axA.set_ylabel("unseen predictions (%)"); axA.set_ylim(0,112)
axA.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5,-0.30), fontsize=12.5, columnspacing=1.2, handlelength=1.3)

# ---- Panel B: IMPACT per view ----
views=['front','left','right','top','ego']
for j,v in enumerate(views):
    d=d3avg([f"impact_indep_{v}_{h}" for h in ['left','right']]); bottom=0
    for k in order:
        axB.bar(j, d[k], bottom=bottom, width=0.66, color=CELL[k],
                edgecolor="white", linewidth=0.8, zorder=3)
        bottom+=d[k]
axB.set_xticks(range(len(views))); axB.set_xticklabels(views)
axB.set_ylim(0,112); axB.set_ylabel("unseen predictions (%)")
# subtle panel tags
axA.text(-0.13,1.06,"(a)",transform=axA.transAxes,fontsize=15,fontweight="bold")
axB.text(-0.10,1.06,"(b)",transform=axB.transAxes,fontsize=15,fontweight="bold")
ps.save(fig,"F3_bottleneck")
print("F3 done.")
for name,tags,_,_ in dsets:
    d=d3avg(tags); print(f"  {name}: verb_only(noun-fail)={d['verb_only']:.1f} noun_only(verb-fail)={d['noun_only']:.1f}")

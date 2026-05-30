"""F6: unseen top-K recovery. Atomic = 0 for all K (structural ceiling);
decomposition ranks the correct unseen composition within top-5 (~38-51%) though
top-1 is low (8-23%) -> the failure is re-ranking (co-occurrence prior wins top-1),
not representation. Three panels: MECCANO, HAViD, IMPACT (all datasets).
HAViD top-K from havid_topk_F6.json (eval_havid_topk_F6.py, reproducibility gate passed)."""
import json, statistics as st
from pathlib import Path
import numpy as np
import plotstyle as ps; ps.apply()
import matplotlib.pyplot as plt

from paths import RESULTS_DIR as R
def J(p): return json.load(open(p))
VH=[(v,h) for v in ['front','left','right','top','ego'] for h in ['left','right']]
Ks=[1,3,5,10]
def imp_topk(fam):
    rows=[J(R/f"{fam}_{v}_{h}/comp_eval.json")["unseen"] for v,h in VH]
    return [st.mean([r[f"action_top{k}"] for r in rows]) for k in Ks]
def mec_topk(path,node):
    n=J(R/path)[node]; return [n[f"action_top{k}"] for k in Ks]
def hav_topk(cfg):  # from gate-passed eval_havid_topk_F6.py aggregate
    n=J(R/"havid_topk_F6.json")["aggregate"][cfg]["unseen"]; return [n[f"action_top{k}"] for k in Ks]

data = {
 "MECCANO": {"shared": mec_topk("meccano_comp_shared/final_eval.json","val_unseen"),
             "independent": mec_topk("meccano_comp_indep_combined/final_eval.json","val_unseen"),
             "color": ps.PALETTE["MECCANO"]},
 "HAViD":   {"shared": hav_topk("shared"),
             "independent": hav_topk("independent"),
             "color": ps.PALETTE["HAViD"]},
 "IMPACT":  {"shared": imp_topk("impact_comp"),
             "independent": imp_topk("impact_indep"),
             "color": ps.PALETTE["IMPACT"]},
}

fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.7), constrained_layout=True, sharey=True)
for ax,(name,d) in zip(axes, data.items()):
    c=d["color"]
    ax.plot(Ks,[0,0,0,0],"--",color=ps.PALETTE["atomic"],lw=2,marker="o",ms=6,
            markerfacecolor="white",label="atomic",zorder=3)
    ax.plot(Ks,d["shared"],"--",color=c,lw=2,marker="s",ms=7,
            markerfacecolor="white",label="shared",zorder=4)
    ax.plot(Ks,d["independent"],"-",color=c,lw=2.4,marker="D",ms=7,label="independent",zorder=5)
    # annotate the top-1 vs top-5 gap for independent
    t1,t5=d["independent"][0],d["independent"][2]
    ax.annotate("", xy=(5,t5), xytext=(5,t1),
                arrowprops=dict(arrowstyle="<->",color=c,lw=1.3,alpha=0.8))
    ax.text(5.3,(t1+t5)/2, f"top-1: {t1:.0f}%\ntop-5: {t5:.0f}%", fontsize=12, color=c, va="center")
    ax.text(0.04,0.93,name,transform=ax.transAxes,fontsize=15,fontweight="bold",color=c,va="top")
    ax.set_xticks(Ks); ax.set_xlabel("$K$ (top-$K$)")
    ax.set_xlim(0.5,10.7); ax.set_ylim(-3,75)
axes[0].set_ylabel("unseen action top-$K$ (%)")
axes[-1].legend(loc="lower right", fontsize=12.5)
ps.save(fig,"F6_topk")
print("F6 done.")
for name,d in data.items():
    print(f"  {name}: atomic=0 | shared {[round(x,1) for x in d['shared']]} | indep {[round(x,1) for x in d['independent']]}")

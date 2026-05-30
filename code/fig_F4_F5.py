"""F4 (RQ4): granularity Pareto (HAViD k=1..4) + MECCANO/IMPACT k1->k2.
F5 (interventions): unseen HM for shared / independent / C2C-shared / C2C-indep x 3 datasets."""
import json
from pathlib import Path
import numpy as np, statistics as st
import plotstyle as ps; ps.apply()
import matplotlib.pyplot as plt

from paths import RESULTS_DIR as R
def J(p): return json.load(open(p))
VH=[(v,h) for v in ['front','left','right','top','ego'] for h in ['left','right']]
SUB=[f"view{v}_{h}h_aa" for v in (0,1,2) for h in ("l","r")]
def havg(fam):
    ds=[J(R/f"{fam}_{s}/comp_eval.json") for s in SUB]
    return st.mean([d['seen']['joint_top1'] for d in ds]), st.mean([d['unseen']['joint_top1'] for d in ds])
def iavg(fam):
    ds=[J(R/f"{fam}_{v}_{h}/comp_eval.json") for v,h in VH]
    return st.mean([d['seen']['action_top1'] for d in ds]), st.mean([d['unseen']['action_top1'] for d in ds]), st.mean([d['HM_action_top1'] for d in ds])
def hm(s,u): return 2*s*u/(s+u) if s+u>0 else 0

# ===================== F4 =====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True,
                               gridspec_kw={"width_ratios":[3.2,2.8]})
# (a) HAViD k=1..4
# k=2 uses unified argmax-pair from havid_topk_F6.json (matches T2/T5 HAV Shared = 51.1/7.9/13.7).
# k=1,3,4 use per-head joint_top1 (no argmax-pair saved for k=3,4); k=1 is single action head (unaffected).
ks=[1,2,3,4]
def pt_k(k):
    if k == 2:
        h=J(R/"havid_topk_F6.json")["aggregate"]["shared"]
        return (h["seen"]["action_top1"], h["unseen"]["action_top1"])
    return havg(f"havid_gran_k{k}")
pts=[pt_k(k) for k in ks]
xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
def hk_hm(k):
    if k == 2:
        h=J(R/"havid_topk_F6.json")["aggregate"]["shared"]
        return hm(h["seen"]["action_top1"], h["unseen"]["action_top1"])
    ds=[J(R/f"havid_gran_k{k}_{s}/comp_eval.json") for s in SUB]
    return st.mean([d.get("HM_joint", hm(d["seen"]["joint_top1"], d["unseen"]["joint_top1"])) for d in ds])
hms=[hk_hm(k) for k in ks]
mk=["o","s","^","D"]; cols=[ps.PALETTE["atomic"],ps.PALETTE["shared"],ps.PALETTE["accent"],ps.PALETTE["noun"]]
# per-point label offsets to avoid overlap (k=2,k=3 are close)
off={1:(12,-4),2:(8,10),3:(-22,-22),4:(12,4)}
for k,x,y,h,m,c in zip(ks,xs,ys,hms,mk,cols):
    ax1.scatter(x,y,s=180,marker=m,color=c,edgecolor="white",linewidth=1.2,zorder=4)
    ax1.annotate(f"k={k} (HM={h:.1f})", (x,y), textcoords="offset points",
                 xytext=off[k], fontsize=12.5, va="bottom")
ax1.set_xlabel("seen joint top-1 (%)"); ax1.set_ylabel("unseen joint top-1 (%)")
ax1.set_ylim(-0.8, max(ys)+2.6); ax1.set_xlim(min(xs)-0.8, max(xs)+1.6)

# (b) MECCANO / IMPACT  k1(atomic, unseen=0) -> k2(decomp)
mk_s=J(R/"meccano_comp_shared/final_eval.json"); m_k2=(mk_s["val_seen"]["action_top1"],mk_s["val_unseen"]["action_top1"])
m_k1=(J(R/"meccano_comp_atomic/comp_eval.json")["seen_top1"],0.0)
i_k2=iavg("impact_comp")[:2]; i_k1=(iavg("impact_comp")[0],0.0)  # atomic unseen structurally 0; seen ~ shared seen proxy
for (k1,k2,col,name) in [(m_k1,m_k2,ps.PALETTE["MECCANO"],"MECCANO"),(i_k1,i_k2,ps.PALETTE["IMPACT"],"IMPACT")]:
    ax2.annotate("", xy=k2, xytext=k1, arrowprops=dict(arrowstyle="-|>",color=col,lw=2.2))
    ax2.scatter(*k1,s=90,marker="o",color="white",edgecolor=col,linewidth=2,zorder=4)
    ax2.scatter(*k2,s=130,marker="s",color=col,edgecolor="white",linewidth=1.2,zorder=4)
    ax2.text(k2[0],k2[1]+0.4,name,color=col,fontsize=13,ha="center",fontweight="bold")
ax2.set_xlabel("seen action top-1 (%)"); ax2.set_ylabel("unseen action top-1 (%)")
ax2.set_ylim(-0.6, 9)
ax1.text(-0.12,1.06,"(a)",transform=ax1.transAxes,fontsize=15,fontweight="bold")
ax2.text(-0.14,1.06,"(b)",transform=ax2.transAxes,fontsize=15,fontweight="bold")
ps.save(fig,"F4_granularity")
print("F4 done. HAViD (mean-of-HMs):",[(k,round(h,1)) for k,h in zip(ks,hms)])

# ===================== F5 =====================
fig2, ax = plt.subplots(figsize=(8.2,4.0), constrained_layout=True)
configs=["shared","independent","C2C-shared","C2C-indep"]
ccol=[ps.PALETTE["shared"],ps.PALETTE["indep"],"#B58CC2",ps.PALETTE["accent"]]
def hms(ds):
    """Unseen HM under unified argmax-pair scoring (matches T2 anchor exactly)."""
    if ds=="MECCANO":
        # Shared = single-head action (unaffected by scoring choice)
        sh=hm(*( (lambda d:(d["val_seen"]["action_top1"],d["val_unseen"]["action_top1"]))(J(R/"meccano_comp_shared/final_eval.json")) ))
        # Independent: argmax-pair from meccano_comp_indep_combined (was: comprehensive both-correct)
        ij=J(R/"meccano_comp_indep_combined/final_eval.json")
        ind=hm(ij["val_seen"]["action_top1"], ij["val_unseen"]["action_top1"])
        cs=J(R/"c2c_shared_meccano/comp_eval.json")["HM_action_top1"]  # already argmax-pair
        ci=J(R/"c2c_indep_meccano/comp_eval.json")["HM_action_top1"]
        return [sh,ind,cs,ci]
    if ds=="HAViD":
        # Shared (k=2): argmax-pair from havid_topk_F6 (was: joint_top1 mean-of-HMs)
        hsh=J(R/"havid_topk_F6.json")["aggregate"]["shared"]
        sh=hm(hsh["seen"]["action_top1"], hsh["unseen"]["action_top1"])
        # Independent: argmax-pair from havid_topk_F6 (was: comprehensive both-correct)
        hin=J(R/"havid_topk_F6.json")["aggregate"]["independent"]
        ind=hm(hin["seen"]["action_top1"], hin["unseen"]["action_top1"])
        cs=st.mean([J(R/f"c2c_shared_havid_{x}/comp_eval.json")["HM_action_top1"] for x in SUB])
        ci=st.mean([J(R/f"c2c_indep_havid_{x}/comp_eval.json")["HM_action_top1"] for x in SUB])
        return [sh,ind,cs,ci]
    if ds=="IMPACT":
        sh=iavg("impact_comp")[2]  # single-head action (unaffected)
        # Independent: argmax-pair (was: comprehensive both-correct)
        ind=st.mean([J(R/f"impact_indep_{v}_{h}/comp_eval.json")["HM_action_top1"] for v,h in VH])
        cs=iavg("c2c_shared_impact")[2]; ci=iavg("c2c_indep_impact")[2]
        return [sh,ind,cs,ci]
dss=["MECCANO","HAViD","IMPACT"]
W=0.2; xbase=np.arange(len(dss))
for j,cfg in enumerate(configs):
    vals=[hms(d)[j] for d in dss]
    ax.bar(xbase+(j-1.5)*W, vals, width=W, color=ccol[j], edgecolor="white", linewidth=0.7, label=cfg, zorder=3)
    for x,v in zip(xbase+(j-1.5)*W, vals):
        ax.text(x, v+0.4, f"{v:.0f}", ha="center", fontsize=11)
ax.set_xticks(xbase); ax.set_xticklabels(dss)
ax.set_ylabel("unseen HM (%)"); ax.set_ylim(0, 36)
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5,1.12), fontsize=12.5, columnspacing=1.3, handlelength=1.2)
ps.save(fig2,"F5_interventions")
print("F5 done.")
for d in dss: print(f"  {d}: shared/indep/C2Csh/C2Cin HM = {[round(x,1) for x in hms(d)]}")

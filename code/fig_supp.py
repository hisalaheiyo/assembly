"""Supplementary figures SF1-SF3 (only where a plot beats a table).
SF1 failure-mode stacked bars (I7 visual; backs S5).
SF2 calibration reliability diagrams (I6; bin-level, a table of ECE cannot show this).
SF3 C2C objective-ablation bars (MECCANO; backs S8a).
Per-pair D2 -> table S4b; seed variance -> table S8b (figures would be too thin)."""
import json, statistics as st
from pathlib import Path
import numpy as np
import plotstyle as ps; ps.apply()
import matplotlib.pyplot as plt

from paths import RESULTS_DIR as R
def J(p): return json.load(open(p))
VH=[(v,h) for v in ['front','left','right','top','ego'] for h in ['left','right']]
SUB=[f"view{v}_{h}h_aa" for v in (0,1,2) for h in ("l","r")]
DS_TAGS={"MECCANO":["meccano_indep"],
         "HAViD":[f"havid_indep_{s}" for s in SUB],
         "IMPACT":[f"impact_indep_{v}_{h}" for v,h in VH]}
def m(xs): xs=[x for x in xs if x is not None]; return st.mean(xs) if xs else 0.0

# ===================================================== SF1 failure-mode stacked bars
FM=[("seen pair","both_wrong_pred_is_train_pair","co"),
    ("seen $n$-partner","same_v_pred_n_is_train_partner","co"),
    ("seen $v$-partner","same_n_pred_v_is_train_partner","co"),
    ("novel $n$","same_v_pred_n_not_train_partner","nv"),
    ("novel $v$","same_n_pred_v_not_train_partner","nv"),
    ("off-grid","both_wrong_pred_off_grid","og")]
COCOL=["#B23A48","#DD8452","#E8B04B"]      # co-occurrence fallbacks (warm gradient)
NVCOL=["#4C72B0","#7BA6D9"]  # genuine novel attempts (cool)
OGCOL="#9AA0A6"
def fm_frac(tags):
    fr=[J(R/f"comprehensive_{t}/summary.json")["failure_mode_unseen"]["fractions_pct"] for t in tags]
    return {k:m([d[k] for d in fr]) for _,k,_ in FM}
def sf1():
    dss=list(DS_TAGS); fig,ax=plt.subplots(figsize=(8.6,3.2),constrained_layout=True)
    cols=COCOL+NVCOL+[OGCOL]
    y=np.arange(len(dss))[::-1]
    data={ds:fm_frac(DS_TAGS[ds]) for ds in dss}
    for ds,yy in zip(dss,y):
        left=0; seen_tot=0
        for i,(lbl,key,grp) in enumerate(FM):
            w=data[ds][key]
            ax.barh(yy,w,left=left,color=cols[i],edgecolor="white",lw=0.6,
                    label=lbl if ds==dss[0] else None,zorder=3)
            if grp=="co": seen_tot+=w
            left+=w
        ax.text(seen_tot+1.0,yy,f"seen-partner: {seen_tot:.0f}%",va="center",ha="left",
                fontsize=11.5,fontweight="bold",color="#3A3A3A")
        ax.plot([seen_tot,seen_tot],[yy-0.42,yy+0.42],color="#222",lw=1.4,ls=(0,(2,1.5)),zorder=5)
    ax.set_yticks(y); ax.set_yticklabels(dss,fontsize=13)
    ax.set_xlabel("share of unseen errors (%)"); ax.set_xlim(0,100)
    ax.legend(ncol=6,loc="upper center",bbox_to_anchor=(0.5,1.16),fontsize=10.2,
              columnspacing=0.9,handlelength=1.1,handletextpad=0.4,frameon=False)
    ps.save(fig,"SF1_failuremode")
    print("SF1:",{ds:round(sum(data[ds][k] for _,k,g in FM if g=="co"),1) for ds in dss})

# ===================================================== SF2 calibration reliability
def load_ps(tags,split):
    recs=[]
    for t in tags:
        recs+=J(R/f"comprehensive_{t}/per_sample_{split}.json")
    return recs
def reliability(recs,comp,nb=10):
    conf=np.array([r[f"prob_{comp}_max"] for r in recs])
    corr=np.array([1.0 if r[f"pred_{comp}"]==r[f"gt_{comp}"] else 0.0 for r in recs])
    edges=np.linspace(0,1,nb+1); xs=[];ys=[]
    for i in range(nb):
        m_=(conf>=edges[i])&(conf<edges[i+1] if i<nb-1 else conf<=edges[i+1])
        if m_.sum()>=10: xs.append(conf[m_].mean()); ys.append(corr[m_].mean()*100)
    return np.array(xs),np.array(ys)
def sf2():
    dss=list(DS_TAGS); fig,axes=plt.subplots(1,3,figsize=(10.5,3.5),constrained_layout=True,sharey=True)
    for ax,ds in zip(axes,dss):
        tags=DS_TAGS[ds]
        ax.plot([0,100],[0,100],color="#BBB",lw=1.2,ls="--",zorder=1)
        for comp,col,nm in [("v",ps.PALETTE["verb"],"verb"),("n",ps.PALETTE["noun"],"object")]:
            for split,style,mk,a in [("seen","--","o",0.55),("unseen","-","D",1.0)]:
                x,y=reliability(load_ps(tags,split),comp)
                if len(x): ax.plot(x*100,y,style,color=col,lw=2.0 if split=="unseen" else 1.4,
                                   marker=mk,ms=5,alpha=a,zorder=3,
                                   label=f"{nm} ({split})" if ax is axes[-1] else None)
        ax.text(0.04,0.94,ds,transform=ax.transAxes,fontsize=14,fontweight="bold",
                color=ps.PALETTE[ds],va="top")
        ax.set_xlim(0,100); ax.set_ylim(0,100); ax.set_xlabel("confidence (%)")
    axes[0].set_ylabel("accuracy (%)")
    axes[-1].legend(loc="lower right",fontsize=10,frameon=True,framealpha=0.9)
    fig.text(0.5,-0.02,"below diagonal $=$ overconfident",ha="center",fontsize=10.5,style="italic",color="#555")
    ps.save(fig,"SF2_calibration")
    print("SF2 done.")

# ===================================================== SF3 C2C ablation bars
def sf3():
    def cev(d):
        j=J(R/f"{d}/comp_eval.json"); s,u=j["seen"]["action_top1"],j["unseen"]["action_top1"]
        return u, j.get("HM_action_top1", 2*s*u/(s+u) if s+u>0 else 0)
    variants=[("full","c2c_shared_meccano"),("$-$compose","c2c_shared_meccano_abl_nocomp"),
              ("$-$HSIC","c2c_shared_meccano_abl_nohsic"),("$-$contrast","c2c_shared_meccano_abl_nocon"),
              ("$-$new-pair","c2c_shared_meccano_abl_nonew")]
    lbls=[l for l,_ in variants]; U=[];H=[]
    for _,d in variants:
        u,h=cev(d); U.append(u); H.append(h)
    fig,ax=plt.subplots(figsize=(7.4,3.4),constrained_layout=True)
    x=np.arange(len(lbls)); w=0.38
    b1=ax.bar(x-w/2,U,w,color=ps.PALETTE["MECCANO"],edgecolor="white",lw=0.7,label="unseen top-1",zorder=3)
    b2=ax.bar(x+w/2,H,w,color=ps.PALETTE["accent"],edgecolor="white",lw=0.7,label="HM",zorder=3)
    for b in list(b1)+list(b2):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.25,f"{b.get_height():.1f}",
                ha="center",fontsize=10.5)
    ax.axhline(U[0],color=ps.PALETTE["MECCANO"],lw=1,ls=":",alpha=0.6,zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(lbls,fontsize=12)
    ax.set_ylabel("MECCANO (%)"); ax.set_ylim(0,max(H+U)+2.5)
    ax.legend(ncol=2,loc="upper center",bbox_to_anchor=(0.5,1.14),fontsize=11,
              frameon=False,columnspacing=1.6,handlelength=1.3)
    ps.save(fig,"SF3_c2cablation")
    print("SF3: unseen",[round(u,1) for u in U],"HM",[round(h,1) for h in H])

sf1(); sf2(); sf3()
print("Supp figures done.")

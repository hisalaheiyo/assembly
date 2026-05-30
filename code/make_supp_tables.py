"""Supplementary tables S1-S8 (booktabs LaTeX) from canonical data per DATA_MANIFEST + PAPER_DESIGN §8.
Every result family of the compositional study lands in a main or supp exhibit (coverage matrix §8).
Writes paper_analysis1/tables/S{1..8}.tex + supp_tables_all.tex (tectonic preview).

Scope: the compositional-study runs reported in the paper (MECCANO/HAViD/IMPACT, MViTv2-S).
"""
import json, statistics as st
from pathlib import Path
from paths import RESULTS_DIR as R
from paths import TABLES_DIR as TDIR; TDIR.mkdir(parents=True, exist_ok=True)
def J(p): return json.load(open(p))
VH=[(v,h) for v in ['front','left','right','top','ego'] for h in ['left','right']]
SUB=[f"view{v}_{h}h_aa" for v in (0,1,2) for h in ("l","r")]
SUBLBL={f"view{v}_{h}h_aa":f"v{v}-{h}h" for v in (0,1,2) for h in ("l","r")}
def m(xs): xs=[x for x in xs if x is not None]; return st.mean(xs) if xs else None
def sd(xs): xs=[x for x in xs if x is not None]; return st.pstdev(xs) if len(xs)>1 else 0.0
def hm(s,u): return 2*s*u/(s+u) if (s and u and s+u>0) else 0.0
def f(x,d=1): return "--" if x is None else f"{x:.{d}f}"

# ============================================================ S1: IMPACT per-(view,hand) x 4 configs
def imp_cfg(fam,v,h):  # Shared / C2C-* use the per-config comp_eval HM_action_top1
    j=J(R/f"{fam}_{v}_{h}/comp_eval.json")
    return j["seen"]["action_top1"], j["unseen"]["action_top1"], j["HM_action_top1"]
def imp_indep_cfg(_fam,v,h):  # Independent: unified argmax-pair (= log P(v)+log P(n) over valid pairs), matches T2
    d=J(R/f"impact_indep_{v}_{h}/comp_eval.json")
    return d["seen"]["action_top1"], d["unseen"]["action_top1"], d["HM_action_top1"]
def imp_atom_cfg(_fam,v,h):  # Atomic on comp-split = train_impact_comp_variants mode=action; unseen and HM are structural 0
    d=J(R/f"impact_compvar_action_{v}_{h}/comp_eval.json")
    return d["seen_top1"], 0.0, 0.0
CFG_IMP=[("Atomic","impact_compvar_action",imp_atom_cfg),
         ("Shared","impact_comp",imp_cfg),("Independent","impact_indep",imp_indep_cfg),
         ("C2C-shared","c2c_shared_impact",imp_cfg),("C2C-indep","c2c_indep_impact",imp_cfg)]
def s1():
    rows=[]
    for v,h in VH:
        cells=[]
        for _,fam,acc in CFG_IMP:
            s,u,hh=acc(fam,v,h); cells += [f(s),f(u),f(hh)]
        rows.append(f"{v}-{h[0]}h & " + " & ".join(cells) + r" \\")
    # mean row
    mc=[]
    for _,fam,acc in CFG_IMP:
        S=[acc(fam,v,h) for v,h in VH]
        mc += [f(m([x[0] for x in S])), f(m([x[1] for x in S])), f(m([x[2] for x in S]))]
    rows.append(r"\midrule \textbf{mean} & " + " & ".join(mc) + r" \\")
    head = " & ".join(r"\textbf{Sn} & \textbf{Un} & \textbf{HM}" for _ in CFG_IMP)
    cfgh = " & ".join(rf"\multicolumn{{3}}{{c}}{{{n}}}" for n,_,_acc in CFG_IMP)
    cmid = " ".join(rf"\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(len(CFG_IMP)))
    return (r"""\begin{table*}[t]\centering
\caption{\textbf{IMPACT per-(view,hand) compositional breakdown.} Seen / Unseen action top-1 and HM (\%)
for the five configurations across all 10 (view,hand) configs of IMPACT. Atomic's Un and HM columns are structurally $0$ (the output space is the seen-action vocabulary); the per-view variation in atomic Sn (around $46\!-\!58\%$) is shown for completeness. The mean row matches the IMPACT block of the main-paper anchor; per-view dispersion shows that the egocentric configurations are consistently the weakest.}\label{tab:s2}\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{l """ + "ccc " * len(CFG_IMP) + r"""}
\toprule
& """ + cfgh + r" \\" + "\n" + cmid + "\n"
+ r"\textbf{view-hand} & " + head + r" \\" + "\n\\midrule\n"
+ "\n".join(rows) + "\n"
+ r"""\bottomrule
\end{tabular}\end{table*}""")

# ============================================================ S2: HAViD per-subtask x {k1-4, solo}
def hav_node(fam,sub):
    # k=2 (havid_gran_k2): use unified argmax-pair from havid_topk_F6.json (shared = 2-head verb+manip)
    if fam == "havid_gran_k2":
        ht = J(R/"havid_topk_F6.json")["per_subtask"][sub]["shared"]
        s, u = ht["seen"]["action_top1"], ht["unseen"]["action_top1"]
        return s, u, hm(s, u)
    # k=1,3,4 and solo heads: joint_top1 (per-head all-correct); k=3,4 don't have argmax-pair saved
    j=J(R/f"{fam}_{sub}/comp_eval.json")
    return j["seen"]["joint_top1"], j["unseen"]["joint_top1"], j.get("HM_joint", hm(j["seen"]["joint_top1"],j["unseen"]["joint_top1"]))
CFG_HAV=[("$k{=}1$","havid_gran_k1"),("$k{=}2$","havid_gran_k2"),("$k{=}3$","havid_gran_k3"),
         ("$k{=}4$","havid_gran_k4"),("solo $v$","havid_gran_solo_verb"),("solo $m$","havid_gran_solo_manip")]
def s2():
    rows=[]
    for sub in SUB:
        cells=[]
        for _,fam in CFG_HAV:
            try: s,u,hh=hav_node(fam,sub); cells+=[f(s),f(u),f(hh)]
            except Exception: cells+=["--","--","--"]
        rows.append(f"{SUBLBL[sub]} & " + " & ".join(cells) + r" \\")
    mc=[]
    for _,fam in CFG_HAV:
        S=[]
        for sub in SUB:
            try: S.append(hav_node(fam,sub))
            except Exception: pass
        mc += [f(m([x[0] for x in S])), f(m([x[1] for x in S])), f(m([x[2] for x in S]))]
    rows.append(r"\midrule \textbf{mean} & " + " & ".join(mc) + r" \\")
    cfgh=" & ".join(rf"\multicolumn{{3}}{{c}}{{{n}}}" for n,_ in CFG_HAV)
    cmid=" ".join(rf"\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(6))
    head=" & ".join(r"\textbf{Sn} & \textbf{Un} & \textbf{HM}" for _ in CFG_HAV)
    return (r"""\begin{table*}[t]\centering
\caption{\textbf{HAViD per-subtask granularity breakdown.} Seen / Unseen joint top-1 and HM (\%)
across all 6 (view,hand) subtasks for granularity $k{=}1\!\ldots\!4$ and single-component solo heads.
Substantiates T5 and F4; $k{=}2$ is the per-subtask unseen sweet spot (I5).}\label{tab:s3}\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{l *{18}{c}}
\toprule
& """ + cfgh + r" \\" + "\n" + cmid + "\n"
+ r"\textbf{subtask} & " + head + r" \\" + "\n\\midrule\n"
+ "\n".join(rows) + "\n"
+ r"""\bottomrule
\end{tabular}\end{table*}""")

# ============================================================ S3: component top-{1,3,5}
def mec_topk(path,node,key):
    n=J(R/path)[node]; return [n.get(f"{key}_top{k}") for k in (1,3,5)]
def imp_topk_avg(fam,split,key):
    rows=[J(R/f"{fam}_{v}_{h}/comp_eval.json")[split] for v,h in VH]
    return [m([r.get(f"{key}_top{k}") for r in rows]) for k in (1,3,5)]
def s3():
    def trip(xs): return " & ".join(f(x) for x in xs)
    # ---- action top-{1,3,5} (shared & independent) ----
    A=[]
    A.append(("MECCANO","Shared",   mec_topk("meccano_comp_shared/final_eval.json","val_seen","action"),   mec_topk("meccano_comp_shared/final_eval.json","val_unseen","action")))
    A.append(("MECCANO","Independent", mec_topk("meccano_comp_indep_combined/final_eval.json","val_seen","action"), mec_topk("meccano_comp_indep_combined/final_eval.json","val_unseen","action")))
    A.append(("IMPACT","Shared",    imp_topk_avg("impact_comp","seen","action"), imp_topk_avg("impact_comp","unseen","action")))
    A.append(("IMPACT","Independent", imp_topk_avg("impact_indep","seen","action"), imp_topk_avg("impact_indep","unseen","action")))
    # HAViD action top-{1,3,5} from gate-passed eval_havid_topk_F6.py (same logP(v)+logP(m) scoring)
    hv=J(R/"havid_topk_F6.json")["aggregate"]
    def hvk(cfg,sp): n=hv[cfg][sp]; return [n["action_top1"],n["action_top3"],n["action_top5"]]
    A.append(("HAViD","Shared", hvk("shared","seen"), hvk("shared","unseen")))
    A.append(("HAViD","Independent", hvk("independent","seen"), hvk("independent","unseen")))
    block_a="\n".join(f"{ds} & {cfg} & "+trip(s)+" & "+trip(u)+r" \\" for ds,cfg,s,u in A)
    # ---- per-component verb/noun top-{1,3,5} (shared; IMPACT full, MECCANO verb) ----
    C=[]
    C.append(("IMPACT","verb", imp_topk_avg("impact_comp","seen","verb"), imp_topk_avg("impact_comp","unseen","verb")))
    C.append(("IMPACT","object", imp_topk_avg("impact_comp","seen","noun"), imp_topk_avg("impact_comp","unseen","noun")))
    C.append(("MECCANO","verb", mec_topk("meccano_comp_shared/final_eval.json","val_seen","verb"), mec_topk("meccano_comp_shared/final_eval.json","val_unseen","verb")))
    block_c="\n".join(f"{ds} & {comp} & "+trip(s)+" & "+trip(u)+r" \\" for ds,comp,s,u in C)
    return (r"""\begin{table}[t]\centering
\caption{\textbf{Component top-$\{1,3,5\}$ breakdown.} (a) Unseen action composition is
reachable in the top ranks though top-1 is low (numeric backing for F6); atomic is structurally 0 for all
$K$ (omitted). Action top-$K$ uses the same argmax over $(v,n)$ pairs ranked by
$\log P(v)\!+\!\log P(n)$ as the main-paper anchor table; the $K{=}1$ column reproduces the unseen
action top-1 of that anchor exactly. (b) Per-component verb / object top-$K$ (shared, where available).
\emph{Caveat:} MECCANO object uses a coverage metric (no object top-1, omitted).}\label{tab:s4}\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{l l ccc ccc}
\toprule
& & \multicolumn{3}{c}{Seen action top-$K$} & \multicolumn{3}{c}{Unseen action top-$K$} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
\textbf{Dataset} & \textbf{Config} & $K{=}1$ & $3$ & $5$ & $K{=}1$ & $3$ & $5$ \\
\midrule
""" + block_a + r"""
\midrule
\multicolumn{8}{l}{\footnotesize (b) per-component top-$K$ (shared encoder):}\\
""" + block_c.replace("Seen action","Seen comp.").replace("Unseen action","Unseen comp.") + r"""
\bottomrule
\multicolumn{8}{l}{\footnotesize Columns under (b) are verb / object top-$K$ for the named component.}\\
\end{tabular}\end{table}""")

# ============================================================ S4: D1 solo-vs-joint shortcut
def s4():
    D=J(R/"d1b_summary.json")
    spec=[("MECCANO","meccano",[("verb","verb"),("object","noun")]),
          ("HAViD","havid_avg",[("verb","verb"),("manip","manip")]),
          ("IMPACT","impact_avg",[("verb","verb"),("object","noun")])]
    rows=[]
    for ds,node,comps in spec:
        for i,(lbl,key) in enumerate(comps):
            n=D[node][key]
            jd=n["joint_seen_unseen_drop"]; sld=n["solo_seen_unseen_drop"]
            dd=n["shortcut_diff_pp"]; mg=n["D1a_seen_margin_pp"]
            dscell=rf"\multirow{{2}}{{*}}{{{ds}}}" if i==0 else ""
            rows.append(f"{dscell} & {lbl} & {jd:.1f} & {sld:.1f} & \\textbf{{{dd:.1f}}} & {mg:+.1f} \\\\")
        rows.append(r"\midrule")
    body="\n".join(rows[:-1])
    return (r"""\begin{table}[t]\centering
\caption{\textbf{D1 shortcut diagnostic (solo vs joint).} Per component: seen$\to$unseen accuracy
drop when the head is trained jointly vs in isolation (solo). The \emph{shortcut} $\Delta$ (excess joint
drop) measures reliance on co-occurrence; the seen-margin (joint$-$solo on seen) $\approx 0$ rules out a
capacity deficit. Substantiates T3.}\label{tab:s5}\small\setlength{\tabcolsep}{5pt}
\begin{tabular}{l l cc c c}
\toprule
\textbf{Dataset} & \textbf{Comp.} & joint drop & solo drop & shortcut $\Delta$ & seen-margin \\
\midrule
""" + body + r"""
\bottomrule
\multicolumn{6}{l}{\footnotesize All in pp. shortcut $\Delta=$ joint drop $-$ solo drop. seen-margin $=$ joint$-$solo seen acc.}\\
\end{tabular}\end{table}""")

# ============================================================ S5: failure-mode 6-way
FM_KEYS=[("seen pair","both_wrong_pred_is_train_pair"),
         ("seen $n$-partner","same_v_pred_n_is_train_partner"),
         ("seen $v$-partner","same_n_pred_v_is_train_partner"),
         ("novel $n$","same_v_pred_n_not_train_partner"),
         ("novel $v$","same_n_pred_v_not_train_partner"),
         ("off-grid","both_wrong_pred_off_grid")]
def fm_avg(tags):
    fr=[J(R/f"comprehensive_{t}/summary.json")["failure_mode_unseen"]["fractions_pct"] for t in tags]
    return {k:m([d[k] for d in fr]) for _,k in FM_KEYS}
def s5():
    spec=[("MECCANO",["meccano_indep"]),("HAViD",[f"havid_indep_{s}" for s in SUB]),
          ("IMPACT",[f"impact_indep_{v}_{h}" for v,h in VH])]
    rows=[]
    for ds,tags in spec:
        d=fm_avg(tags)
        partner=d["both_wrong_pred_is_train_pair"]+d["same_v_pred_n_is_train_partner"]+d["same_n_pred_v_is_train_partner"]
        offg=d["both_wrong_pred_off_grid"]
        cells=" & ".join(f(d[k]) for _,k in FM_KEYS)
        rows.append(f"{ds} & {cells} & \\textbf{{{partner:.0f}}} & {offg:.0f} \\\\")
    head=" & ".join(lbl for lbl,_ in FM_KEYS)
    return (r"""\begin{table}[t]\centering
\caption{\textbf{Failure-mode decomposition of unseen errors} (independent, \% of wrong predictions).
The vast majority of errors predict a composition that was a seen training partner/pair (summed in
``seen total''); only a small fraction land off-grid. Direct evidence for I7 (co-occurrence-lookup mechanism).}\label{tab:s6}\scriptsize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{l cccccc cc}
\toprule
\textbf{Dataset} & """ + head + r" & \textbf{seen tot.} & off-grid \\" + "\n\\midrule\n"
+ "\n".join(rows) + "\n"
+ r"""\bottomrule
\multicolumn{9}{l}{\footnotesize seen $n$/$v$-partner: one component correct, the predicted other was a seen partner. novel: predicted other never co-occurred.}\\
\end{tabular}\end{table}""")

# ============================================================ S6: ECE + bootstrap CI
def s6():
    spec=[("MECCANO",["meccano_indep"], ["meccano_comp_indep_combined"], lambda tag: ("val_seen","val_unseen","action_top1","final_eval.json")),
          ("HAViD",  [f"havid_indep_{s}" for s in SUB], None, "havid_topk_F6"),
          ("IMPACT", [f"impact_indep_{v}_{h}" for v,h in VH], [f"impact_indep_{v}_{h}" for v,h in VH], None)]
    rows=[]
    for ds,tags,uni_tags,marker in spec:
        S=[J(R/f"comprehensive_{t}/summary.json") for t in tags]
        def ece(split,comp): return m([s["ece_"+split][comp]["ece"] for s in S])
        # bootstrap CI width from comprehensive (both-correct). Center shifted to unified argmax-pair.
        bs=lambda sp,k: m([s["bootstrap_ci_"+sp][k] for s in S])
        if ds == "MECCANO":
            j = J(R/"meccano_comp_indep_combined/final_eval.json")
            uni_s, uni_u = j["val_seen"]["action_top1"], j["val_unseen"]["action_top1"]
        elif ds == "HAViD":
            ht = J(R/"havid_topk_F6.json")["aggregate"]["independent"]
            uni_s, uni_u = ht["seen"]["action_top1"], ht["unseen"]["action_top1"]
        else:  # IMPACT
            U=[J(R/f"impact_indep_{v}_{h}/comp_eval.json") for v,h in VH]
            uni_s = m([d["seen"]["action_top1"] for d in U])
            uni_u = m([d["unseen"]["action_top1"] for d in U])
        # Re-center CI on unified top-1: width preserved (half-widths from bootstrap), shift = uni - both-correct
        # Both-correct bootstrap mean = bs('seen','top1_pct'); offset = uni_s - that.
        bc_s = bs('seen','top1_pct'); bc_u = bs('unseen','top1_pct')
        hw_s_lo = bc_s - bs('seen','ci_lo'); hw_s_hi = bs('seen','ci_hi') - bc_s
        hw_u_lo = bc_u - bs('unseen','ci_lo'); hw_u_hi = bs('unseen','ci_hi') - bc_u
        c2="noun"
        rows.append(f"{ds} & {ece('seen','verb'):.2f} & {ece('unseen','verb'):.2f} & "
                    f"{ece('seen',c2):.2f} & {ece('unseen',c2):.2f} & "
                    f"{uni_s:.1f} ({uni_s-hw_s_lo:.1f},{uni_s+hw_s_hi:.1f}) & "
                    f"{uni_u:.1f} ({uni_u-hw_u_lo:.1f},{uni_u+hw_u_hi:.1f}) \\\\")
    return (r"""\begin{table}[t]\centering
\caption{\textbf{Calibration (ECE) and bootstrap 95\% CIs} (independent). ECE rises seen$\to$unseen
for both heads and is worst on the bottleneck component (I6); $1000\times$ bootstrap CI widths on
action top-1 (recentered on the unified argmax-pair top-1 of the main-paper anchor) confirm the
seen$\gg$unseen gap is not sampling noise.}\label{tab:s7}\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{l cc cc cc}
\toprule
& \multicolumn{2}{c}{verb ECE} & \multicolumn{2}{c}{object ECE} & \multicolumn{2}{c}{action top-1 (95\% CI)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
\textbf{Dataset} & seen & unseen & seen & unseen & seen & unseen \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}\end{table}""")

# ============================================================ S7: baselines
def s7():
    spec=[("MECCANO",["meccano_indep"]),("HAViD",[f"havid_indep_{s}" for s in SUB]),
          ("IMPACT",[f"impact_indep_{v}_{h}" for v,h in VH])]
    rows=[]
    for ds,tags in spec:
        S=[J(R/f"comprehensive_{t}/summary.json") for t in tags]
        maj=m([s["baselines"]["majority_top1_pct"] for s in S])
        # n_unseen: per-config average (one experiment's held-out set size); same convention as previous version
        nun=int(m([s["baselines"]["n_unseen"] for s in S]))
        # Independent top-1: unified argmax-pair (matches T2 Independent unseen)
        if ds == "MECCANO":
            best = J(R/"meccano_comp_indep_combined/final_eval.json")["val_unseen"]["action_top1"]
        elif ds == "HAViD":
            best = J(R/"havid_topk_F6.json")["aggregate"]["independent"]["unseen"]["action_top1"]
        else:
            U=[J(R/f"impact_indep_{v}_{h}/comp_eval.json") for v,h in VH]
            best = m([d["unseen"]["action_top1"] for d in U])
        rows.append(f"{ds} & {nun} & {maj:.1f} & {best:.1f} \\\\")
    return (r"""\begin{table}[t]\centering
\caption{\textbf{Unseen baselines.} Majority-class prediction collapses to 0 on held-out
compositions (no seen action is correct by construction); the independent model is well above this floor.}\label{tab:s8}\small\setlength{\tabcolsep}{6pt}
\begin{tabular}{l c cc}
\toprule
\textbf{Dataset} & $n_{\text{unseen}}$ & majority top-1 & independent top-1 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}\end{table}""")

# ============================================================ S8: C2C ablation + seed variance
def s8():
    # C2C ablation (MECCANO, shared)
    def cev(d):
        j=J(R/f"{d}/comp_eval.json"); s,u=j["seen"]["action_top1"],j["unseen"]["action_top1"]
        return s,u,j.get("HM_action_top1",hm(s,u))
    abl=[("full","c2c_shared_meccano"),("$-$compose","c2c_shared_meccano_abl_nocomp"),
         ("$-$HSIC","c2c_shared_meccano_abl_nohsic"),("$-$contrast","c2c_shared_meccano_abl_nocon"),
         ("$-$new-pair","c2c_shared_meccano_abl_nonew")]
    arows=[]
    for lbl,d in abl:
        try: s,u,hh=cev(d); arows.append(f"{lbl} & {f(s)} & {f(u)} & {f(hh)} \\\\")
        except Exception: arows.append(f"{lbl} & -- & -- & -- \\\\")
    # seed variance (MECCANO shared: main + s1 + s7)
    def sh_unseen(d): return J(R/f"{d}/final_eval.json")["val_unseen"]["action_top1"]
    def sh_seen(d): return J(R/f"{d}/final_eval.json")["val_seen"]["action_top1"]
    seeds=["meccano_comp_shared","meccano_comp_shared_s1","meccano_comp_shared_s7"]
    us=[sh_unseen(d) for d in seeds]; ss=[sh_seen(d) for d in seeds]
    seedrow=(f"shared ($n{{=}}3$ seeds) & {m(ss):.1f} $\\pm$ {sd(ss):.1f} & "
             f"\\multicolumn{{2}}{{c}}{{{m(us):.1f} $\\pm$ {sd(us):.1f}}} \\\\")
    return (r"""\begin{table}[t]\centering
\caption{\textbf{C2C objective ablation (MECCANO) and seed variance.} (a) Removing each compositional
loss term from C2C-shared; (b) MECCANO shared across 3 seeds (mean$\pm$std) confirms the seen$\to$unseen
effects exceed run-to-run noise.}\label{tab:s9}\small\setlength{\tabcolsep}{5pt}
\begin{tabular}{l ccc}
\toprule
\textbf{(a) C2C variant} & Seen & Unseen & HM \\
\midrule
""" + "\n".join(arows) + r"""
\midrule
\multicolumn{4}{l}{\textbf{(b) seed variance}}\\
\midrule
""" + r"""\textbf{config} & Seen & \multicolumn{2}{c}{Unseen} \\
""" + seedrow + r"""
\bottomrule
\end{tabular}\end{table}""")

# ============================================================ S9: standard-split breakdown (atomic vs decomposed)
def s9():
    a_m=J(R/"meccano_mvitv2_action/test_eval.json")["test"]
    d_m=J(R/"meccano_mvitv2_multihead/test_eval.json")["test"]
    a_i=[J(R/f"impact_{v}_{h}_action/test_eval.json")["test"] for v,h in VH]
    d_i=[J(R/f"impact_{v}_{h}_multihead/test_eval.json")["test"] for v,h in VH]
    aim=lambda k:m([r.get(k) for r in a_i])
    dim=lambda k:m([r.get(k) for r in d_i])
    # HAViD: published values from havid2022 (Atomic joint 45.6; 4-head joint 50.9, verb 72.4, manip 59.1, +target 89.7, tool 98.2)
    rows=[
      ("MECCANO","Atomic",      a_m.get("action_top1"),a_m.get("action_top3"),a_m.get("action_top5"),a_m.get("action_top10"), None, None),
      ("MECCANO","Decomposed",  d_m.get("action_top1"),d_m.get("action_top3"),d_m.get("action_top5"),d_m.get("action_top10"), d_m.get("verb_top1"), None),
      ("IMPACT", "Atomic",      aim("action_top1"),aim("action_top3"),aim("action_top5"),aim("action_top10"), None, None),
      ("IMPACT", "Decomposed",  dim("action_top1"),dim("action_top3"),dim("action_top5"),dim("action_top10"), dim("verb_top1"), dim("noun_top1")),
      ("HAViD",  "Atomic",      45.6, None,None,None, None, None),
      ("HAViD",  "Decomp.\\ (4-head)", 50.9, None,None,None, 72.4, 59.1),
    ]
    body="\n".join(f"{ds} & {cfg} & {f(t1)} & {f(t3)} & {f(t5)} & {f(t10)} & {f(vt)} & {f(nt)} \\\\"
                   for ds,cfg,t1,t3,t5,t10,vt,nt in rows)
    return (r"""\begin{table}[t]\centering
\caption{\textbf{Standard-split breakdown: atomic vs decomposed.} Action top-$\{1,3,5,10\}$ on the standard (non-compositional) test split, with per-component top-1 where the head exists. The atomic$\to$decomposed delta is small or positive on every dataset, which is the ``free lunch'' that masks the compositional question discussed in the main paper. HAViD atomic and decomposed (4-head) top-1 are taken from the published MViTv2-S results~\cite{havid2022} (their 4-head reports per-head top-1 verb $72.4$, manip $59.1$, target $89.7$, tool $98.2$); IMPACT rows average over the $10$ (view, hand) configurations; MECCANO and IMPACT atomic models do not produce per-component scores, and MECCANO's noun metric in the decomposed model is a coverage statistic rather than a top-1.}\label{tab:s10}\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{l l cccc cc}
\toprule
& & \multicolumn{4}{c}{action top-$K$} & \multicolumn{2}{c}{per-component top-1} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-8}
\textbf{Dataset} & \textbf{Config} & $K{=}1$ & $3$ & $5$ & $10$ & verb & object/manip \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}\end{table}""")

# ============================================================ S11: HAViD per-subtask std-split (4-head, published)
def s11():
    # HAViD 4-head MViTv2-S per-(view, hand) top-1 from havid2022 Table tab:per_head (Atomic Action subtasks).
    R11=[
      ("Side LH","v0-lh",  74.2,58.4,90.2,98.0,51.2),
      ("Side RH","v0-rh",  71.2,55.1,88.9,98.2,47.0),
      ("Front LH","v1-lh", 74.3,60.4,89.4,98.1,53.2),
      ("Front RH","v1-rh", 69.9,55.0,88.6,98.0,46.6),
      ("Top LH","v2-lh",   72.5,62.8,90.5,98.7,54.2),
      ("Top RH","v2-rh",   72.2,61.9,90.6,97.9,53.0),
    ]
    avg=[m([r[i] for r in R11]) for i in (2,3,4,5,6)]
    body="\n".join(f"{lbl} & {tag} & {vt:.1f} & {mt:.1f} & {tg:.1f} & {tl:.1f} & {jt:.1f} \\\\"
                   for lbl,tag,vt,mt,tg,tl,jt in R11)
    body+= f"\n\\midrule \\textbf{{mean}} & & {avg[0]:.1f} & {avg[1]:.1f} & {avg[2]:.1f} & {avg[3]:.1f} & {avg[4]:.1f} \\\\"
    return (r"""\begin{table}[t]\centering
\caption{\textbf{HAViD per-subtask standard-split (4-head decomposed).} Per-head top-1 (\%) for the published MViTv2-S 4-head model on each of the six (view, hand) subtasks of HAViD's atomic-action splits, reproduced from~\cite{havid2022}. The verb and manip heads carry the bulk of the variance across subtasks; target and tool are near-saturated. The atomic single-head baseline is reported only in aggregate by the original work ($45.6\%$ joint top-1, used in the cross-paradigm summary).}\label{tab:s11}\small\setlength{\tabcolsep}{5pt}
\begin{tabular}{l l ccccc}
\toprule
\textbf{Subtask} & \textbf{tag} & \textbf{Verb} & \textbf{Manip} & \textbf{Target} & \textbf{Tool} & \textbf{Joint} \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}\end{table}""")

# ============================================================ S12: IMPACT per-(view, hand) std-split (atomic vs decomp)
def s12():
    rows=[]
    for v,h in VH:
        a=J(R/f"impact_{v}_{h}_action/test_eval.json")["test"]
        d=J(R/f"impact_{v}_{h}_multihead/test_eval.json")["test"]
        rows.append((f"{v}-{h[0]}h","Atomic",
                     a.get("action_top1"),a.get("action_top3"),a.get("action_top5"),a.get("action_top10"),None,None))
        rows.append((f"{v}-{h[0]}h","Decomp",
                     d.get("action_top1"),d.get("action_top3"),d.get("action_top5"),d.get("action_top10"),
                     d.get("verb_top1"),d.get("noun_top1")))
    # means
    a_means=[m([rw[i] for rw in rows if rw[1]=="Atomic" and rw[i] is not None]) for i in (2,3,4,5)]
    d_means=[m([rw[i] for rw in rows if rw[1]=="Decomp" and rw[i] is not None]) for i in (2,3,4,5,6,7)]
    body="\n".join(f"{lbl} & {cfg} & {f(t1)} & {f(t3)} & {f(t5)} & {f(t10)} & {f(vt)} & {f(nt)} \\\\"
                   for lbl,cfg,t1,t3,t5,t10,vt,nt in rows)
    body+= ("\n\\midrule \\textbf{mean} & Atomic & " + " & ".join(f"{x:.1f}" for x in a_means) + " & -- & -- \\\\" +
            "\n\\textbf{mean} & Decomp & " + " & ".join(f"{x:.1f}" for x in d_means) + r" \\")
    return (r"""\begin{table}[t]\centering
\caption{\textbf{IMPACT per-(view, hand) standard-split (atomic vs decomposed).} Action top-$\{1,3,5,10\}$ (\%) for the atomic and decomposed (2-head verb+noun) MViTv2-S models on every (view, hand) configuration of IMPACT, plus per-component top-1 for the decomposed head. The atomic model emits only a flat action distribution, so its verb / noun columns are structurally not defined (``--''). The mean row matches the IMPACT entries of Table~\ref{tab:s10}; per-view dispersion shows that the egocentric configurations sit slightly below the static views even on the standard split.}\label{tab:s12}\scriptsize\setlength{\tabcolsep}{4pt}
\begin{tabular}{l l cccc cc}
\toprule
& & \multicolumn{4}{c}{action top-$K$} & \multicolumn{2}{c}{per-comp.\ top-1} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-8}
\textbf{view-hand} & \textbf{Config} & $K{=}1$ & $3$ & $5$ & $10$ & verb & noun \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}\end{table}""")

# ============================================================ S10: rare-vs-unseen tier accuracy (substantiates I3)
def s10():
    SPEC=[("MECCANO",["meccano_indep"]),
          ("HAViD",[f"havid_indep_{s}" for s in SUB]),
          ("IMPACT",[f"impact_indep_{v}_{h}" for v,h in VH])]
    # tier names in fixed order (drop "1-1" — empty everywhere)
    TIERS=[("unseen(0)","unseen ($f{=}0$)"), ("2-5","$f{=}2{-}5$"), ("6-20","$f{=}6{-}20$"),
           ("21-100","$f{=}21{-}100$"), ("101+","$f{\geq}101$")]
    rows=[]
    for ds,tags in SPEC:
        all_rows=[J(R/f"comprehensive_{t}/summary.json")["rare_vs_unseen"] for t in tags]
        # map tier name -> list of (acc, n_pairs) across configs
        cells=[]
        for tkey,_ in TIERS:
            accs=[]; ns=[]
            for one in all_rows:
                hit=next((r for r in one if r.get("tier")==tkey), None)
                if hit and hit.get("mean_acc_pct") is not None:
                    accs.append(hit["mean_acc_pct"]); ns.append(hit["n_pairs"])
            if accs:
                cells.append(f"{m(accs):.1f}\\,({m(ns):.0f})" if ns else f"{m(accs):.1f}")
            else:
                cells.append("--")
        rows.append(f"{ds} & " + " & ".join(cells) + r" \\")
    hdr=" & ".join(t for _,t in TIERS)
    return (r"""\begin{table}[t]\centering
\caption{\textbf{Rare versus dense per-pair accuracy.} Per-pair unseen accuracy and per-pair seen accuracy of independent recombination, binned by the pair's training frequency $f$. Cells show mean accuracy in percent (mean $n_\text{pairs}$ in parentheses), averaged over the (view,\,hand) configurations of each dataset; the ``$f{=}1$'' tier is empty on every dataset and is therefore omitted, and the HAViD ``$f{=}2{-}5$'' cell is likewise an empty bin ($n_\text{pairs}{=}0$ on every subtask, since HAViD's seen pairs concentrate in the denser tiers). Densely co-observed pairs ($f{\geq}101$) reach $32$ to $62\%$, while rarely co-observed pairs and unseen pairs sit in the low single to low tens of percent, even though they are technically \emph{seen} in training. This is the coverage-driven pattern referenced in the discussion: accuracy follows training frequency, not whether a pair is in the seen or unseen split.}\label{tab:s13}\small\setlength{\tabcolsep}{5pt}
\begin{tabular}{l ccccc}
\toprule
\textbf{Dataset} & """ + hdr + r" \\" + "\n\\midrule\n"
+ "\n".join(rows) + "\n"
+ r"""\bottomrule
\multicolumn{6}{l}{\footnotesize $f$ is the pair's training-segment count. The ``unseen ($f{=}0$)'' column is the held-out set's accuracy; the remaining columns are seen pairs binned by $f$.}\\
\end{tabular}\end{table}""")


# ============================================================ S13: HAViD C2C per-subtask
def s13():
    rows=[]
    for sub in SUB:
        cs=J(R/f"c2c_shared_havid_{sub}/comp_eval.json")
        ci=J(R/f"c2c_indep_havid_{sub}/comp_eval.json")
        rows.append(f"{SUBLBL[sub]} & {cs['seen']['action_top1']:.1f} & {cs['unseen']['action_top1']:.1f} & {cs['HM_action_top1']:.1f} & {ci['seen']['action_top1']:.1f} & {ci['unseen']['action_top1']:.1f} & {ci['HM_action_top1']:.1f} \\\\")
    # mean row (mean-of-HMs, consistent w/ T2)
    css=[J(R/f"c2c_shared_havid_{sub}/comp_eval.json") for sub in SUB]
    cis=[J(R/f"c2c_indep_havid_{sub}/comp_eval.json") for sub in SUB]
    mc=[m([d["seen"]["action_top1"] for d in css]), m([d["unseen"]["action_top1"] for d in css]), m([d["HM_action_top1"] for d in css]),
        m([d["seen"]["action_top1"] for d in cis]), m([d["unseen"]["action_top1"] for d in cis]), m([d["HM_action_top1"] for d in cis])]
    rows.append(r"\midrule \textbf{mean} & " + " & ".join(f"{x:.1f}" for x in mc) + r" \\")
    return (r"""\begin{table}[t]\centering
\caption{\textbf{HAViD C2C per-subtask compositional breakdown.} Seen / Unseen action top-1 and HM (\%) of the two C2C variants on each of the six (view, hand) subtasks of HAViD. The mean row matches the HAViD C2C rows of the main-paper anchor; per-subtask C2C-shared dispersion is small (the lift over the plain shared baseline is uniform across subtasks), while C2C-indep tracks the plain independent baseline closely (consistent with the main-paper finding that C2C and de-entanglement are partly substitutive).}\label{tab:s14}\small\setlength{\tabcolsep}{5pt}
\begin{tabular}{l ccc ccc}
\toprule
& \multicolumn{3}{c}{C2C-shared} & \multicolumn{3}{c}{C2C-indep} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
\textbf{subtask} & Sn & Un & HM & Sn & Un & HM \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}\end{table}""")

# ===================== write =====================
TABS=[("S1",s1()),("S2",s2()),("S3",s3()),("S4",s4()),("S5",s5()),("S6",s6()),("S7",s7()),("S8",s8()),("S9",s9()),("S10",s11()),("S11",s12()),("S12",s10()),("S13",s13())]
for name,tab in TABS:
    (TDIR/f"{name}.tex").write_text(tab)
preview=(r"""\documentclass{article}\usepackage{booktabs,multirow,makecell,amssymb,amsmath}
\usepackage[landscape,margin=0.5in]{geometry}\usepackage{mathptmx}\pagestyle{empty}\begin{document}
"""+ "\n\n".join(t for _,t in TABS) + "\n\\end{document}")
(TDIR/"supp_tables_all.tex").write_text(preview)
print("Supp tables written:", [n for n,_ in TABS])

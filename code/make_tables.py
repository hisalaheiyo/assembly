"""Generate T1-T5 LaTeX (booktabs) from canonical data per DATA_MANIFEST sourcing rules.
Writes paper_analysis1/tables/T{1..5}.tex + tables_all.tex (compile preview w/ tectonic).

Sourcing: HAViD standard top-1 = published 4-head baseline (45.6 atomic / 50.9 decomposed).
independent = comprehensive_*_indep (uniform harness, all 3 datasets); C2C = c2c_*.
"""
import json, statistics as st
from pathlib import Path
from paths import RESULTS_DIR as R
from paths import TABLES_DIR as TDIR; TDIR.mkdir(parents=True, exist_ok=True)
def J(p): return json.load(open(p))
VH=[(v,h) for v in ['front','left','right','top','ego'] for h in ['left','right']]
SUB=[f"view{v}_{h}h_aa" for v in (0,1,2) for h in ("l","r")]
def m(xs): return st.mean([x for x in xs if x is not None])
def hm(s,u): return 2*s*u/(s+u) if (s and u and s+u>0) else 0.0

# ---------- data accessors ----------
def teval_t1(d):
    j=J(R/f"{d}/test_eval.json"); t=j.get("test",j)
    return t.get("action_top1") or t.get("action_acc1") or t.get("top1")
def comp_imp(d):  # IMPACT comp_eval / C2C
    j=J(R/f"{d}/comp_eval.json"); return j["seen"]["action_top1"], j["unseen"]["action_top1"], j["HM_action_top1"]
def comp_hav(d):  # HAViD gran (joint_top1) or C2C (action_top1) — schema-robust
    j=J(R/f"{d}/comp_eval.json")
    def g(node): return node.get("joint_top1", node.get("action_top1"))
    return g(j["seen"]), g(j["unseen"]), j.get("HM_joint", j.get("HM_action_top1"))
def indep_agg(tags):  # DEPRECATED — both-correct scoring; kept for reference
    a=[J(R/f"comprehensive_{t}/summary.json")["aggregate"] for t in tags]
    return m([x["seen_top1_pct"] for x in a]), m([x["unseen_top1_pct"] for x in a]), m([x["hm"] for x in a])

def hm(s,u): return 2*s*u/(s+u) if (s and u and s+u>0) else 0.0

def indep_argmax_pair(ds, tags):
    """Independent action top-1 under unified argmax-pair (= log P(v)+log P(n) over valid pairs).
    MECCANO/IMPACT: read comp_eval.json of impact_indep / meccano_comp_indep_combined.
    HAViD: read havid_topk_F6.json aggregate (already argmax-pair via eval_havid_topk_F6.py)."""
    if ds == "MECCANO":
        j = J(R/"meccano_comp_indep_combined/final_eval.json")
        s, u = j["val_seen"]["action_top1"], j["val_unseen"]["action_top1"]
        return s, u, hm(s, u)
    if ds == "HAViD":
        h = J(R/"havid_topk_F6.json")["aggregate"]["independent"]
        s, u = h["seen"]["action_top1"], h["unseen"]["action_top1"]
        return s, u, hm(s, u)
    if ds == "IMPACT":
        rows = [J(R/f"impact_indep_{v}_{h}/comp_eval.json") for v,h in VH]
        s = m([r["seen"]["action_top1"] for r in rows])
        u = m([r["unseen"]["action_top1"] for r in rows])
        hms = m([r["HM_action_top1"] for r in rows])
        return s, u, hms

def shared_argmax_pair_hav():
    """HAViD shared (gran_k2) under unified argmax-pair (from havid_topk_F6.json)."""
    h = J(R/"havid_topk_F6.json")["aggregate"]["shared"]
    s, u = h["seen"]["action_top1"], h["unseen"]["action_top1"]
    return s, u, hm(s, u)
def imp_avg(fam):
    rows=[comp_imp(f"{fam}_{v}_{h}") for v,h in VH]; return m([r[0] for r in rows]),m([r[1] for r in rows]),m([r[2] for r in rows])
def hav_avg(fam):
    rows=[comp_hav(f"{fam}_{s}") for s in SUB]; return m([r[0] for r in rows]),m([r[1] for r in rows]),m([r[2] for r in rows])
def d1(): return J(R/"d1b_summary.json")
def d2rho(tags): return m([J(R/f"comprehensive_{t}/summary.json")["d2_unseen"]["spearman_rho"] for t in tags])
def d3avg(tags):
    ds=[J(R/f"comprehensive_{t}/summary.json")["d3_unseen"]["fractions_pct"] for t in tags]
    return {k:m([d[k] for d in ds]) for k in ["both","verb_only","noun_only","neither"]}

def f(x, d=1): return "--" if x is None else f"{x:.{d}f}"

# ===================== T1 datasets =====================
T1 = r"""\begin{table}[t]\centering
\caption{\textbf{The three assembly datasets}, chosen to span the structural axes over which decomposition's adequacy could vary. Sparsity is the observed fraction of the $|\mathcal{V}|\!\times\!|\mathcal{N}|$ grid. The larger-vocabulary component (``larger vocab'') predicts the unseen bottleneck of RQ2; IMPACT's verb-heavy vocabulary is the flip case.}\label{tab:datasets}\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{l ccc c c l}
\toprule
\textbf{Dataset} & $|\mathcal{V}|$ & $|\mathcal{N}|$ & grid & obs.\ (sparsity) & larger vocab & viewpoints / hands \\
\midrule
MECCANO & 12 & 23 & 276 & 61 (22.1\%) & noun ($1.9\times$) & egocentric, single-hand \\
HAViD   & 11 & 29 & 319 & 165 (51.7\%) & object ($2.6\times$) & 3 static, bimanual \\
IMPACT  & 22 & 18 & 396 & 136 (34.3\%) & \textbf{verb} ($1.2\times$) & 5 views (incl.\ ego), bimanual \\
\bottomrule
\end{tabular}\end{table}"""

# ===================== T2 master landscape =====================
# standard T1
std={"MECCANO":(teval_t1("meccano_mvitv2_action"),teval_t1("meccano_mvitv2_multihead")),
     "IMPACT":(m([teval_t1(f"impact_{v}_{h}_action") for v,h in VH]), m([teval_t1(f"impact_{v}_{h}_multihead") for v,h in VH])),
     "HAViD":(45.6,50.9)}
# comp per config: (seen,unseen,hm)
def comp_cfg(ds):
    """All configs use unified argmax-pair scoring (action top-1 = argmax over valid (v,n) pairs by log P(v)+log P(n)).
    Atomic / single-head Shared are unaffected (single argmax)."""
    if ds=="MECCANO":
        at=J(R/"meccano_comp_atomic/comp_eval.json"); atom=(at["seen_top1"],0.0,0.0)
        sh=J(R/"meccano_comp_shared/final_eval.json"); s,u=sh["val_seen"]["action_top1"],sh["val_unseen"]["action_top1"]; shared=(s,u,hm(s,u))
        ind=indep_argmax_pair("MECCANO", None)  # unified argmax-pair
        cs=comp_imp("c2c_shared_meccano"); ci=comp_imp("c2c_indep_meccano")
        return atom,shared,ind,cs,ci
    if ds=="HAViD":
        atom=(hav_avg("havid_gran_k1")[0],0.0,0.0)
        shared=shared_argmax_pair_hav()  # unified argmax-pair (was joint_top1 = both-correct)
        ind=indep_argmax_pair("HAViD", None)  # unified argmax-pair (was comprehensive both-correct)
        cs=hav_avg("c2c_shared_havid"); ci=hav_avg("c2c_indep_havid")  # already argmax-pair
        return atom,shared,ind,cs,ci
    if ds=="IMPACT":
        # Atomic on comp-split = train_impact_comp_variants.py with mode=action
        seen_a=m([J(R/f"impact_compvar_action_{v}_{h}/comp_eval.json")["seen_top1"] for v,h in VH])
        atom=(seen_a,0.0,0.0)
        sh=imp_avg("impact_comp"); shared=(sh[0],sh[1],sh[2])  # single-head, unaffected
        ind=indep_argmax_pair("IMPACT", None)  # unified argmax-pair (was comprehensive both-correct)
        cs=imp_avg("c2c_shared_impact"); ci=imp_avg("c2c_indep_impact")  # already argmax-pair
        return atom,shared,ind,(cs[0],cs[1],cs[2]),(ci[0],ci[1],ci[2])

rows_cfg=["Atomic","Shared","Independent","C2C-shared","C2C-indep"]
def t2_block(ds):
    cfgs=comp_cfg(ds); lines=[]
    for i,(name,(s,u,h)) in enumerate(zip(rows_cfg,cfgs)):
        lines.append(f"  & {name} & {f(s)} & {f(u)} & {f(h)} \\\\")
    return "\n".join(lines)
T2 = (r"""\begin{table}[t]\centering
\caption{\textbf{Compositional-split anchor: the map for all that follows.} Seen\,/\,unseen action top-1 and their harmonic mean for the five configurations across three datasets, on the compositional split. Action top-1 is argmax over valid $(v,n)$ pairs by $\log P(v)\!+\!\log P(n)$ (single-head Atomic and Shared on MECCANO/IMPACT reduce to a single argmax over the action vocabulary). The supplementary top-$K$ table and figure use the same scoring. Atomic unseen is structurally $0$ (the output space is the seen action vocabulary). Independent recombination dominates the shared default everywhere, by up to $6.0\times$ in HM. Every subsequent analysis reads into a cell or gap of this table; the standard-split comparison of atomic against decomposed (no seen / unseen concept there) is reported separately in the supplementary material.}\label{tab:master}\small\setlength{\tabcolsep}{6pt}
\begin{tabular}{l l ccc}
\toprule
\textbf{Dataset} & \textbf{Config} & \textbf{Seen} & \textbf{Unseen} & \textbf{HM} \\
\midrule
"""
+ "\\multirow{5}{*}{MECCANO}\n" + t2_block("MECCANO") + "\n\\midrule\n"
+ "\\multirow{5}{*}{HAViD}\n" + t2_block("HAViD") + "\n\\midrule\n"
+ "\\multirow{5}{*}{IMPACT}\n" + t2_block("IMPACT") + "\n"
+ r"""\bottomrule
\end{tabular}\end{table}""")

# ===================== T3 separability + factorizability =====================
D=d1()
def t3row(ds, vkey, nkey, rho, dens):
    if ds=="MECCANO": node=D["meccano"]
    elif ds=="IMPACT": node=D["impact_avg"]
    else: node=D["havid_avg"]
    vΔ=node[vkey]["shortcut_diff_pp"]; nΔ=node[nkey]["shortcut_diff_pp"]
    vm=node[vkey]["D1a_seen_margin_pp"]; nm=node[nkey]["D1a_seen_margin_pp"]
    return f"{ds} & {vΔ:.1f} & {nΔ:.1f} & {rho:.2f} & {dens} & ({vm:+.1f}, {nm:+.1f}) \\\\"
T3 = (r"""\begin{table}[t]\centering
\caption{\textbf{Separability and factorizability (RQ1, RQ3).} Per dataset: the verb / object shortcut $\Delta$ (D1, the excess seen$\to$unseen drop of the jointly trained head over its solo counterpart), the grid-density correlation $\rho$ (D2), grid density, and the seen-margin (joint$-$solo accuracy on seen, $\approx0$, which rules out a capacity deficit as the cause of the shared encoder's shortfall).}\label{tab:sep}\small\setlength{\tabcolsep}{5pt}
\begin{tabular}{l cc c c c}
\toprule
& \multicolumn{2}{c}{shortcut $\Delta$ (pp)} & & grid & seen-margin \\
\cmidrule(lr){2-3}
\textbf{Dataset} & verb & object & $\rho$ (D2) & density & (v, o) pp \\
\midrule
"""
+ t3row("MECCANO","verb","noun", d2rho(["meccano_indep"]), "22.1\\%") + "\n"
+ t3row("HAViD","verb","manip", d2rho([f"havid_indep_{s}" for s in SUB]), "51.7\\%") + "\n"
+ t3row("IMPACT","verb","noun", d2rho([f"impact_indep_{v}_{h}" for v,h in VH]), "34.3\\%") + "\n"
+ r"""\bottomrule
\multicolumn{6}{l}{\footnotesize $\Delta=$ excess seen$\to$unseen drop of joint over solo (shortcut). seen-margin $\approx 0$ rules out capacity.}\\
\end{tabular}\end{table}""")

# ===================== T4 bottleneck =====================
VOC={"MECCANO":(12,23),"HAViD":(11,29),"IMPACT":(22,18)}
def t4row(ds,tags):
    d=d3avg(tags); v,n=VOC[ds]; bott="noun/obj" if d["verb_only"]>d["noun_only"] else "verb"
    return f"{ds} & {v} & {n} & {d['both']:.1f} & {d['verb_only']:.1f} & {d['noun_only']:.1f} & {d['neither']:.1f} & {bott} \\\\"
def t4view(v):
    d=d3avg([f"impact_indep_{v}_{h}" for h in ['left','right']])
    return f"\\quad {v} & & & {d['both']:.1f} & {d['verb_only']:.1f} & {d['noun_only']:.1f} & {d['neither']:.1f} & verb \\\\"
T4 = (r"""\begin{table}[t]\centering
\caption{\textbf{Component bottleneck (RQ2, the centerpiece).} Per dataset, and per IMPACT view, the unseen error decomposition: $\pi_v$ (verb correct, object wrong $\Rightarrow$ object fails), $\pi_n$ (object correct, verb wrong $\Rightarrow$ verb fails). Classification is per-head ($v_\text{pred}{=}\operatorname{argmax}P(v)$, $n_\text{pred}{=}\operatorname{argmax}P(n)$) since the question is which head's argmax failed; the ``both'' column equals the joint-argmax action top-1 of Table~\ref{tab:master} within $\sim\!1.3$\,pp. The bottleneck falls on the larger-vocabulary component; IMPACT's verb-heavy vocabulary flips it to the verb, and the flip is invariant across all five views.}\label{tab:bottleneck}\small\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{l cc cccc c}
\toprule
\textbf{Dataset} & $|\mathcal{V}|$ & $|\mathcal{N}|$ & both & \makecell{obj\\err ($\pi_v$)} & \makecell{verb\\err ($\pi_n$)} & \makecell{both\\wrong} & bottleneck \\
\midrule
"""
+ t4row("MECCANO",["meccano_indep"]) + "\n"
+ t4row("HAViD",[f"havid_indep_{s}" for s in SUB]) + "\n"
+ t4row("IMPACT",[f"impact_indep_{v}_{h}" for v,h in VH]) + "\n"
+ "\\midrule\n\\multicolumn{8}{l}{\\footnotesize IMPACT per view (verb bottleneck is view-invariant):}\\\\\n"
+ "\n".join(t4view(v) for v in ['front','left','right','top','ego']) + "\n"
+ r"""\bottomrule
\multicolumn{8}{l}{\footnotesize $\pi_v$ (verb correct, object wrong) $\Rightarrow$ object fails; $\pi_n$ $\Rightarrow$ verb fails. Bottleneck $=$ larger-vocab primitive.}\\
\end{tabular}\end{table}""")

# ===================== T5 granularity =====================
def t5hav(k):
    # k=2: switch to unified argmax-pair (havid_topk_F6 shared) — matches T2 HAV Shared row
    if k == 2:
        return shared_argmax_pair_hav()
    # k=1,3,4: joint_top1 (no argmax-pair saved for k=3,4 — ≤0.3pp lower than argmax-pair would be)
    s,u,h=hav_avg(f"havid_gran_k{k}")
    return s,u,h
mec_atom=J(R/"meccano_comp_atomic/comp_eval.json"); mec_sh=J(R/"meccano_comp_shared/final_eval.json")
T5 = (r"""\begin{table}[t]\centering
\caption{\textbf{Granularity (RQ4).} HAViD $k{=}1\!\ldots\!4$ and the MECCANO\,/\,IMPACT $k{=}1$ vs $k{=}2$ cross-check, as seen\,/\,unseen\,/\,HM. $k{=}2$ (verb, object) is the unseen sweet spot; finer factoring recovers seen accuracy but not unseen, and coarser ($k{=}1$, atomic) is structurally $0$ on unseen.}\label{tab:granularity}\small\setlength{\tabcolsep}{5pt}
\begin{tabular}{l l cc c}
\toprule
\textbf{Dataset} & \textbf{$k$ (heads)} & \textbf{Seen} & \textbf{Unseen} & \textbf{HM} \\
\midrule
"""
+ "\\multirow{4}{*}{HAViD}\n"
+ "\n".join(f"  & $k={k}$ {desc} & {f(s)} & {f(u)} & {f(h)} \\\\"
            for k,desc in [(1,"(atomic)"),(2,"($v,m$)"),(3,"($+t$)"),(4,"($+t,l$)")]
            for (s,u,h) in [t5hav(k)])
+ "\n\\midrule\n"
+ f"\\multirow{{2}}{{*}}{{MECCANO}}\n  & $k=1$ (atomic) & {mec_atom['seen_top1']:.1f} & 0.0 & 0.0 \\\\\n"
+ f"  & $k=2$ ($v,n$) & {mec_sh['val_seen']['action_top1']:.1f} & {mec_sh['val_unseen']['action_top1']:.1f} & {hm(mec_sh['val_seen']['action_top1'],mec_sh['val_unseen']['action_top1']):.1f} \\\\\n"
+ "\\midrule\n"
+ (lambda sn: f"\\multirow{{2}}{{*}}{{IMPACT}}\n  & $k=1$ (atomic) & {sn:.1f} & 0.0 & 0.0 \\\\\n")(m([J(R/f"impact_compvar_action_{v}_{h}/comp_eval.json")["seen_top1"] for v,h in VH]))
+ (lambda s,u,h: f"  & $k=2$ ($v,n$) & {s:.1f} & {u:.1f} & {h:.1f} \\\\\n")(*( (lambda r:(r[0],r[1],r[2]))(imp_avg("impact_comp")) ))
+ r"""\bottomrule
\multicolumn{5}{l}{\footnotesize HAViD action $=(v,m)$; $k{=}3,4$ add target/tool as auxiliary heads. $k{=}2$ is the unseen sweet spot.}\\
\end{tabular}\end{table}""")

for name,tab in [("T1",T1),("T2",T2),("T3",T3),("T4",T4),("T5",T5)]:
    (TDIR/f"{name}.tex").write_text(tab)
# combined preview
preview = (r"""\documentclass{article}\usepackage{booktabs,multirow,makecell,amssymb,amsmath,geometry}
\geometry{margin=1in}\usepackage{mathptmx}\pagestyle{empty}\begin{document}
"""+ "\n\n".join([T1,T2,T3,T4,T5]) + "\n\\end{document}")
(TDIR/"tables_all.tex").write_text(preview)
print("Tables written to", TDIR)
print("\n=== T2 sanity (HM) ===")
for ds in ["MECCANO","HAViD","IMPACT"]:
    for name,(s,u,h) in zip(rows_cfg,comp_cfg(ds)):
        print(f"  {ds:8s} {name:12s}: seen={f(s)} unseen={f(u)} HM={f(h)}")

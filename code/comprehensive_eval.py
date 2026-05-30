"""
Comprehensive per-(dataset,config) eval for the analysis paper. Extends
compute_d2d3.py with all P0 analyses:

  1. D3 4-cell error decomposition (seen + unseen)
  2. D2 grid-density correlation (per-pair unseen acc vs f(v)*f(n)) + scatter
  3. PER-PAIR seen accuracy (for "rare-seen vs unseen" comparison)
  4. RARE-VS-UNSEEN: per-pair seen acc stratified by train frequency tier
     vs unseen acc — addresses "is it grid interpolation?" directly
  5. Per-VERB and per-NOUN unseen accuracy distribution
  6. Failure mode: when wrong, what does the model predict (same-verb? same-noun?
     train pair? completely off?) and pred-pair frequency-in-train distribution
  7. ECE (Expected Calibration Error) for verb head, noun head, and action
  8. Bootstrap 95% CI for top-1 (seen, unseen, HM)
  9. Random + most-frequent-pair baselines

All on a single INDEPENDENT-architecture decomp model (the cleanest decomp test
case across all 3 datasets — chosen because shared has the MECCANO multiplicative
collapse making D2 degenerate there).

Outputs: results/comprehensive_{tag}/{summary.json, per_sample_seen.json,
                                       per_sample_unseen.json}

Usage:
  python comprehensive_eval.py --dataset meccano
  python comprehensive_eval.py --dataset impact --view front --hand left
  python comprehensive_eval.py --dataset havid --subtask view0_lh_aa
"""

import argparse, csv, json, math, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from paths import PROJECT_DIR  # see code/paths.py for layout / env-var overrides


# ============== loaders (same independent backbones as compute_d2d3) ==============
def load_meccano():
    from train_meccano_comp import MeccanoCompDataset, IndepHeadModel, COMP_DIR, N_VERB, N_NOUN
    seen = MeccanoCompDataset("comp_val_seen", is_train=False)
    unseen = MeccanoCompDataset("comp_val_unseen", is_train=False)
    train_vn = []
    with open(COMP_DIR / "comp_train.csv") as f:
        r = csv.DictReader(f)
        ncols = [c for c in r.fieldnames if c.startswith("noun_")]
        for row in r:
            v = int(row["verb_id"])
            for i, c in enumerate(ncols):
                if int(row[c]) == 1:
                    train_vn.append((v, i))
    verb_m = IndepHeadModel("verb", dropout=0.0, pretrained=False)
    noun_m = IndepHeadModel("noun", dropout=0.0, pretrained=False)
    cv = torch.load(PROJECT_DIR / "checkpoints/meccano_comp_indep_verb/best.pth",
                    map_location="cpu", weights_only=False)
    cn = torch.load(PROJECT_DIR / "checkpoints/meccano_comp_indep_noun/best.pth",
                    map_location="cpu", weights_only=False)
    verb_m.load_state_dict(cv["model"]); noun_m.load_state_dict(cn["model"])

    class MecWrap(torch.utils.data.Dataset):
        def __init__(self, ds): self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            clip, action, verb, nv = self.ds[i]
            nset = [j for j, x in enumerate(nv.tolist()) if x > 0.5]
            n = nset[0] if nset else 0
            return clip, int(verb), int(n)

    return {"tag": "meccano_indep", "n_verb": N_VERB, "n_noun": N_NOUN,
            "verb_m": verb_m, "noun_m": noun_m,
            "seen_ds": MecWrap(seen), "unseen_ds": MecWrap(unseen),
            "train_vn": train_vn}


def load_impact(view, hand):
    from train_impact_mvitv2 import N_VERB as I_NV, N_NOUN as I_NN
    from train_impact_comp import IMPACTCompDataset, COMP_DIR as IMP_COMP
    from train_impact_comp_variants import SingleHeadModel
    train_vn = []
    for line in open(IMP_COMP / hand / "comp_train.txt"):
        line = line.strip()
        if not line: continue
        _, _, _, v, n, a = line.split(",")
        train_vn.append((int(v), int(n)))
    verb_m = SingleHeadModel(I_NV, dropout=0.0, pretrained=False)
    noun_m = SingleHeadModel(I_NN, dropout=0.0, pretrained=False)
    cv = torch.load(PROJECT_DIR / f"checkpoints/impact_compvar_verb_only_{view}_{hand}/best.pth",
                    map_location="cpu", weights_only=False)
    cn = torch.load(PROJECT_DIR / f"checkpoints/impact_compvar_noun_only_{view}_{hand}/best.pth",
                    map_location="cpu", weights_only=False)
    verb_m.load_state_dict(cv["model"]); noun_m.load_state_dict(cn["model"])
    seen = IMPACTCompDataset(view, hand, "comp_val_seen", is_train=False)
    unseen = IMPACTCompDataset(view, hand, "comp_val_unseen", is_train=False)

    class ImpWrap(torch.utils.data.Dataset):
        def __init__(self, ds): self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            clip, action, verb, noun = self.ds[i]
            return clip, int(verb), int(noun)

    return {"tag": f"impact_indep_{view}_{hand}", "n_verb": I_NV, "n_noun": I_NN,
            "verb_m": verb_m, "noun_m": noun_m,
            "seen_ds": ImpWrap(seen), "unseen_ds": ImpWrap(unseen),
            "train_vn": train_vn}


def load_havid(subtask):
    from train_mvitv2_comp import HAViDDataset, MViTv2MultiHead, HEAD_CLASSES, DATA_BASE
    n_verb, n_manip = HEAD_CLASSES[0], HEAD_CLASSES[1]
    dr = DATA_BASE / subtask
    train_vid = dr / "train" / "videos_train"
    val_vid = dr / "val" / "videos_val"
    train_vn = []
    with open(dr / f"{subtask}_comp_train.txt") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10: continue
            v = parts[6].index("1") if "1" in parts[6] else -1
            m = parts[7].index("1") if "1" in parts[7] else -1
            if v >= 0 and m >= 0: train_vn.append((v, m))
    verb_m = MViTv2MultiHead([n_verb], dropout=0.0, pretrained=False)
    manip_m = MViTv2MultiHead([n_manip], dropout=0.0, pretrained=False)
    cv = torch.load(PROJECT_DIR / f"checkpoints/havid_gran_solo_verb_{subtask}/best.pth",
                    map_location="cpu", weights_only=False)
    cm = torch.load(PROJECT_DIR / f"checkpoints/havid_gran_solo_manip_{subtask}/best.pth",
                    map_location="cpu", weights_only=False)
    verb_m.load_state_dict(cv["model"]); manip_m.load_state_dict(cm["model"])
    seen = HAViDDataset(dr / f"{subtask}_comp_val_seen.txt",
                        [train_vid, val_vid], is_train=False)
    unseen = HAViDDataset(dr / f"{subtask}_comp_val_unseen.txt",
                          [train_vid, val_vid], is_train=False)

    class HavWrap(torch.utils.data.Dataset):
        def __init__(self, ds): self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            clip, labels = self.ds[i]
            return clip, int(labels[0]), int(labels[1])

    return {"tag": f"havid_indep_{subtask}", "n_verb": n_verb, "n_noun": n_manip,
            "verb_m": verb_m, "noun_m": manip_m,
            "seen_ds": HavWrap(seen), "unseen_ds": HavWrap(unseen),
            "train_vn": train_vn}


# ============== per-sample eval ==============
@torch.no_grad()
def per_sample_eval(verb_m, noun_m, loader, device):
    """Returns list of dicts: gt_v, gt_n, pred_v, pred_n, prob_v_max, prob_n_max."""
    verb_m.eval(); noun_m.eval()
    out = []
    for clip, gv, gn in tqdm(loader, desc="eval", leave=False):
        clip = clip.to(device, non_blocking=True)
        v_lg = verb_m(clip); n_lg = noun_m(clip)
        if isinstance(v_lg, list): v_lg = v_lg[0]
        if isinstance(n_lg, list): n_lg = n_lg[0]
        v_prob = F.softmax(v_lg, dim=1); n_prob = F.softmax(n_lg, dim=1)
        v_top, v_pred = v_prob.max(dim=1)
        n_top, n_pred = n_prob.max(dim=1)
        gv = gv.tolist() if hasattr(gv, "tolist") else list(gv)
        gn = gn.tolist() if hasattr(gn, "tolist") else list(gn)
        for i in range(len(clip)):
            out.append({"gt_v": int(gv[i]), "gt_n": int(gn[i]),
                        "pred_v": int(v_pred[i]), "pred_n": int(n_pred[i]),
                        "prob_v_max": float(v_top[i]), "prob_n_max": float(n_top[i])})
    return out


# ============== metric computations ==============
def d3(per_sample):
    cell = {"both": 0, "verb_only": 0, "noun_only": 0, "neither": 0}
    for s in per_sample:
        v_ok = s["pred_v"] == s["gt_v"]; n_ok = s["pred_n"] == s["gt_n"]
        if v_ok and n_ok: cell["both"] += 1
        elif v_ok: cell["verb_only"] += 1
        elif n_ok: cell["noun_only"] += 1
        else: cell["neither"] += 1
    N = max(1, len(per_sample))
    return {"counts": cell, "fractions_pct": {k: cell[k]/N*100 for k in cell}, "n": N}


def per_pair_acc(per_sample, min_n=1):
    """Returns {(v,n): {n_samples, n_correct, acc_pct}}."""
    tot = defaultdict(int); cor = defaultdict(int)
    for s in per_sample:
        key = (s["gt_v"], s["gt_n"])
        tot[key] += 1
        if s["pred_v"] == s["gt_v"] and s["pred_n"] == s["gt_n"]:
            cor[key] += 1
    return {f"{v}_{n}": {"v": v, "n": n, "n_samples": tot[(v,n)],
                        "n_correct": cor[(v,n)], "acc_pct": cor[(v,n)]/tot[(v,n)]*100}
            for (v, n) in tot if tot[(v,n)] >= min_n}


def d2_correlation(per_pair, train_vn, min_per_pair=10):
    f_v = Counter(v for v, _ in train_vn)
    f_n = Counter(n for _, n in train_vn)
    scatter = []
    for key, d in per_pair.items():
        if d["n_samples"] < min_per_pair: continue
        prod = f_v.get(d["v"], 0) * f_n.get(d["n"], 0)
        scatter.append({**d, "f_v": f_v.get(d["v"], 0), "f_n": f_n.get(d["n"], 0),
                        "freq_prod": prod})
    rho, pval = spearman([s["acc_pct"] for s in scatter],
                         [s["freq_prod"] for s in scatter])
    return {"n_pairs_eligible": len(scatter), "min_per_pair": min_per_pair,
            "spearman_rho": rho, "spearman_p_approx": pval, "scatter": scatter}


def rare_vs_unseen(seen_per_pair, unseen_per_pair, train_vn, tiers=(1, 5, 20, 100)):
    """Per-pair seen accuracy stratified by training frequency tier, contrasted
    with unseen accuracy. Tiers cover SEEN pairs only (f>=1); unseen (f=0) is
    reported as a separate prepended row."""
    pair_freq = Counter((v, n) for v, n in train_vn)
    # SEEN-only bins (f >= 1). Aligned 1-to-1 with bin_labels.
    bins = [(1, tiers[0]), (tiers[0]+1, tiers[1]), (tiers[1]+1, tiers[2]),
            (tiers[2]+1, tiers[3]), (tiers[3]+1, float("inf"))]
    bin_labels = [f"1-{tiers[0]}", f"{tiers[0]+1}-{tiers[1]}",
                  f"{tiers[1]+1}-{tiers[2]}", f"{tiers[2]+1}-{tiers[3]}",
                  f"{tiers[3]+1}+"]
    seen_buckets = [[] for _ in bins]
    n_seen_no_train = 0
    for key, d in seen_per_pair.items():
        if d["n_samples"] < 5: continue
        f = pair_freq.get((d["v"], d["n"]), 0)
        if f == 0:
            n_seen_no_train += 1  # should be zero unless data inconsistency
            continue
        for i, (lo, hi) in enumerate(bins):
            if lo <= f <= hi:
                seen_buckets[i].append(d["acc_pct"])
                break
    unseen_accs = [d["acc_pct"] for d in unseen_per_pair.values() if d["n_samples"] >= 5]
    out = [{"tier": "unseen(0)", "n_pairs": len(unseen_accs),
            "mean_acc_pct": float(np.mean(unseen_accs)) if unseen_accs else None}]
    for label, bucket in zip(bin_labels, seen_buckets):
        out.append({"tier": label, "n_pairs": len(bucket),
                    "mean_acc_pct": float(np.mean(bucket)) if bucket else None})
    if n_seen_no_train > 0:
        out.append({"_warn": f"{n_seen_no_train} 'seen' pairs had no entries in train_vn"})
    return out


def per_primitive_unseen(per_sample, n_verb, n_noun):
    """Marginal per-verb and per-noun unseen accuracy."""
    v_tot, v_cor = Counter(), Counter()
    n_tot, n_cor = Counter(), Counter()
    for s in per_sample:
        v_tot[s["gt_v"]] += 1; n_tot[s["gt_n"]] += 1
        if s["pred_v"] == s["gt_v"]: v_cor[s["gt_v"]] += 1
        if s["pred_n"] == s["gt_n"]: n_cor[s["gt_n"]] += 1
    per_v = [{"verb_id": v, "n_samples": v_tot[v], "acc_pct": v_cor[v]/v_tot[v]*100}
             for v in sorted(v_tot)]
    per_n = [{"noun_id": n, "n_samples": n_tot[n], "acc_pct": n_cor[n]/n_tot[n]*100}
             for n in sorted(n_tot)]
    return {"per_verb": per_v, "per_noun": per_n}


def failure_mode(per_sample, train_vn):
    """For each WRONG prediction, what did the model predict?
    Categories:
      - same_v_diff_n: verb correct, noun confused with a SEEN partner of this verb
      - same_v_diff_n_unseen: verb correct, noun is from a not-this-verb training partner
      - same_n_diff_v: symmetric
      - both_wrong_pred_train_pair: predicted (v',n') IS a train pair
      - both_wrong_pred_off_grid: predicted (v',n') NOT a train pair
    """
    train_pairs = set(train_vn)
    train_v_partners = defaultdict(set)  # v -> set of n
    train_n_partners = defaultdict(set)
    for v, n in train_vn:
        train_v_partners[v].add(n); train_n_partners[n].add(v)
    cats = Counter()
    for s in per_sample:
        gv, gn, pv, pn = s["gt_v"], s["gt_n"], s["pred_v"], s["pred_n"]
        if pv == gv and pn == gn: continue   # correct
        if pv == gv:    # verb correct, noun wrong
            if pn in train_v_partners[gv]:
                cats["same_v_pred_n_is_train_partner"] += 1
            else:
                cats["same_v_pred_n_not_train_partner"] += 1
        elif pn == gn:  # noun correct
            if pv in train_n_partners[gn]:
                cats["same_n_pred_v_is_train_partner"] += 1
            else:
                cats["same_n_pred_v_not_train_partner"] += 1
        else:           # both wrong
            if (pv, pn) in train_pairs:
                cats["both_wrong_pred_is_train_pair"] += 1
            else:
                cats["both_wrong_pred_off_grid"] += 1
    total_wrong = sum(cats.values())
    return {"n_wrong": total_wrong, "counts": dict(cats),
            "fractions_pct": {k: v/total_wrong*100 if total_wrong else 0
                              for k, v in cats.items()}}


def ece(per_sample, head, n_bins=15):
    """Expected Calibration Error for a single head ('verb' or 'noun')."""
    probs = [s[f"prob_{head[0]}_max"] for s in per_sample]
    correct = [int(s[f"pred_{head[0]}"] == s[f"gt_{head[0]}"]) for s in per_sample]
    bins = np.linspace(0, 1, n_bins + 1)
    ece_sum = 0.0; N = len(per_sample)
    bin_stats = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        in_bin = [(p, c) for p, c in zip(probs, correct) if lo <= p < hi or (i == n_bins-1 and p == hi)]
        if not in_bin:
            bin_stats.append({"range": [lo, hi], "n": 0})
            continue
        n_bin = len(in_bin)
        avg_conf = sum(p for p, _ in in_bin) / n_bin
        accuracy = sum(c for _, c in in_bin) / n_bin
        ece_sum += (n_bin / N) * abs(avg_conf - accuracy)
        bin_stats.append({"range": [lo, hi], "n": n_bin,
                          "avg_conf": avg_conf, "accuracy": accuracy})
    return {"ece": ece_sum, "n_bins": n_bins, "bins": bin_stats}


def bootstrap_ci(per_sample, n_iters=1000, ci=0.95):
    """Bootstrap CI for top-1 accuracy."""
    rng = np.random.default_rng(42)
    arr = np.array([1 if s["pred_v"]==s["gt_v"] and s["pred_n"]==s["gt_n"] else 0
                    for s in per_sample], dtype=np.float32)
    if len(arr) == 0:
        return {"top1_pct": None, "ci_lo": None, "ci_hi": None}
    means = []
    N = len(arr)
    for _ in range(n_iters):
        idx = rng.integers(0, N, N)
        means.append(arr[idx].mean() * 100)
    means.sort()
    lo_i = int((1-ci)/2 * n_iters); hi_i = int((1+ci)/2 * n_iters)
    return {"top1_pct": float(arr.mean()*100),
            "ci_lo": float(means[lo_i]), "ci_hi": float(means[hi_i])}


def baselines(per_sample_unseen, train_vn):
    """Random and majority-class baselines on unseen."""
    # Random: P(correct) = 1 / |distinct (v,n) pairs the model could output|
    #   For independent decomp, output space is |V|*|N| (all combos). Random accuracy = 1/(|V|*|N|).
    #   For atomic, output space is |A_train| → 0 on unseen.
    # Majority: predict the most-frequent training pair for every unseen sample
    most_freq = Counter(train_vn).most_common(1)[0][0] if train_vn else (0, 0)
    correct = sum(1 for s in per_sample_unseen
                  if s["gt_v"] == most_freq[0] and s["gt_n"] == most_freq[1])
    return {"majority_pred_pair": list(most_freq),
            "majority_top1_pct": correct/len(per_sample_unseen)*100 if per_sample_unseen else None,
            "n_unseen": len(per_sample_unseen)}


# ============== spearman + normal helpers ==============
def spearman(x, y):
    n = len(x)
    if n < 3: return None, None
    rx = _rank(x); ry = _rank(y)
    mx, my = sum(rx)/n, sum(ry)/n
    num = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a-mx)**2 for a in rx))
    dy = math.sqrt(sum((b-my)**2 for b in ry))
    if dx == 0 or dy == 0: return 0.0, None
    rho = num / (dx*dy)
    if abs(rho) >= 1.0: return rho, 0.0
    t = rho * math.sqrt((n-2)/(1-rho**2))
    return rho, 2*(1-_normal_cdf(abs(t)))


def _rank(xs):
    idx = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0]*len(xs); i = 0
    while i < len(xs):
        j = i
        while j+1 < len(xs) and xs[idx[j+1]] == xs[idx[i]]: j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j+1): r[idx[k]] = avg
        i = j + 1
    return r


def _normal_cdf(z): return 0.5*(1+math.erf(z/math.sqrt(2)))


# ============== main ==============
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["meccano","impact","havid"])
    p.add_argument("--view", default=None); p.add_argument("--hand", default=None)
    p.add_argument("--subtask", default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == "meccano":
        cfg = load_meccano()
    elif args.dataset == "impact":
        assert args.view and args.hand
        cfg = load_impact(args.view, args.hand)
    else:
        assert args.subtask
        cfg = load_havid(args.subtask)

    out_dir = PROJECT_DIR / "results" / f"comprehensive_{cfg['tag']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[task] {cfg['tag']}  seen={len(cfg['seen_ds'])} unseen={len(cfg['unseen_ds'])} "
          f"train_vn={len(cfg['train_vn'])}", flush=True)

    verb_m = cfg["verb_m"].to(device); noun_m = cfg["noun_m"].to(device)

    # Eval seen + unseen
    seen_loader = DataLoader(cfg["seen_ds"], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    unseen_loader = DataLoader(cfg["unseen_ds"], batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)
    t0 = time.time()
    ps_seen = per_sample_eval(verb_m, noun_m, seen_loader, device)
    ps_unseen = per_sample_eval(verb_m, noun_m, unseen_loader, device)
    print(f"[eval] seen={len(ps_seen)} unseen={len(ps_unseen)} in {time.time()-t0:.0f}s", flush=True)

    # Compute all metrics
    pp_seen = per_pair_acc(ps_seen, min_n=1)
    pp_unseen = per_pair_acc(ps_unseen, min_n=1)
    summary = {
        "tag": cfg["tag"],
        "n_seen": len(ps_seen), "n_unseen": len(ps_unseen),
        "n_train_vn": len(cfg["train_vn"]),
        "n_verb": cfg["n_verb"], "n_noun": cfg["n_noun"],
        "aggregate": {
            "seen_top1_pct": sum(1 for s in ps_seen
                                  if s["pred_v"]==s["gt_v"] and s["pred_n"]==s["gt_n"]) /
                              max(1, len(ps_seen)) * 100,
            "unseen_top1_pct": sum(1 for s in ps_unseen
                                    if s["pred_v"]==s["gt_v"] and s["pred_n"]==s["gt_n"]) /
                                max(1, len(ps_unseen)) * 100,
        },
        "d3_seen": d3(ps_seen),
        "d3_unseen": d3(ps_unseen),
        "d2_unseen": d2_correlation(pp_unseen, cfg["train_vn"], min_per_pair=10),
        "rare_vs_unseen": rare_vs_unseen(pp_seen, pp_unseen, cfg["train_vn"]),
        "per_primitive_unseen": per_primitive_unseen(ps_unseen, cfg["n_verb"], cfg["n_noun"]),
        "failure_mode_unseen": failure_mode(ps_unseen, cfg["train_vn"]),
        "ece_seen": {h: ece(ps_seen, h) for h in ["verb", "noun"]},
        "ece_unseen": {h: ece(ps_unseen, h) for h in ["verb", "noun"]},
        "bootstrap_ci_seen": bootstrap_ci(ps_seen),
        "bootstrap_ci_unseen": bootstrap_ci(ps_unseen),
        "baselines": baselines(ps_unseen, cfg["train_vn"]),
    }
    # HM with CI
    sm = summary["aggregate"]["seen_top1_pct"]; um = summary["aggregate"]["unseen_top1_pct"]
    summary["aggregate"]["hm"] = 2*sm*um/(sm+um) if (sm+um) > 0 else 0

    # Save
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    with open(out_dir / "per_sample_seen.json", "w") as f:
        json.dump(ps_seen, f)
    with open(out_dir / "per_sample_unseen.json", "w") as f:
        json.dump(ps_unseen, f)
    with open(out_dir / "per_pair_seen.json", "w") as f:
        json.dump(pp_seen, f, indent=2)
    with open(out_dir / "per_pair_unseen.json", "w") as f:
        json.dump(pp_unseen, f, indent=2)

    print(f"\n=== {cfg['tag']} ===")
    print(f"seen_t1={sm:.2f}% unseen_t1={um:.2f}% HM={summary['aggregate']['hm']:.2f}")
    print(f"D2 rho={summary['d2_unseen']['spearman_rho']}")
    print(f"D3 unseen: {summary['d3_unseen']['fractions_pct']}")
    print(f"Bootstrap unseen 95% CI: [{summary['bootstrap_ci_unseen']['ci_lo']:.2f}, "
          f"{summary['bootstrap_ci_unseen']['ci_hi']:.2f}]")
    print(f"Baselines: majority {summary['baselines']['majority_top1_pct']:.2f}%")
    print(f"ECE unseen verb: {summary['ece_unseen']['verb']['ece']:.4f}  "
          f"noun: {summary['ece_unseen']['noun']['ece']:.4f}")
    print(f"Failure mode: {summary['failure_mode_unseen']['fractions_pct']}")
    print(f"Rare-vs-unseen: {[t['mean_acc_pct'] for t in summary['rare_vs_unseen']]}")
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()

"""
Unified D2 (grid-density correlation) and D3 (per-component error decomposition)
post-processing for MECCANO / IMPACT / HAViD comp-split INDEPENDENT models.

Independent architecture is chosen as the canonical decomp model for D2/D3
because it has nontrivial unseen accuracy on all 3 datasets (shared has the
multiplicative collapse on MECCANO, making D2 degenerate there).

Per-sample dump: gt_v, gt_n, pred_v, pred_n, action top-1 correctness.
D2: Spearman correlation of per-pair unseen top-1 accuracy vs f(v)*f(n) where
    f(v), f(n) are counts of comp_train segments containing each primitive.
D3: 4-cell counts (both correct, only verb correct, only noun correct, neither).

Reads only the existing trained backbones; writes nothing that overlaps with
any prior result. Outputs to results/d2d3_{dataset}_{tag}/{per_sample,d2,d3}.json

Usage:
  python compute_d2d3.py --dataset meccano
  python compute_d2d3.py --dataset impact --view front --hand left
  python compute_d2d3.py --dataset havid --subtask view0_lh_aa
"""

import argparse, csv, json, math, sys
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


# ============== D2 / D3 computation (shared across datasets) ==============
def compute_d3(per_sample):
    """Per-component error decomposition. per_sample: list of (gv, gn, pv, pn)."""
    cell = {"both": 0, "verb_only": 0, "noun_only": 0, "neither": 0}
    for gv, gn, pv, pn in per_sample:
        v_ok = (pv == gv); n_ok = (pn == gn)
        if v_ok and n_ok:     cell["both"] += 1
        elif v_ok and not n_ok: cell["verb_only"] += 1
        elif not v_ok and n_ok: cell["noun_only"] += 1
        else:                  cell["neither"] += 1
    N = max(1, len(per_sample))
    return {"n": N, "counts": cell,
            "fractions": {k: cell[k] / N * 100 for k in cell}}


def compute_d2(per_sample, train_vn, min_samples_per_pair=10):
    """Grid-density correlation. train_vn: list of (v,n) from comp_train."""
    # Per-component frequencies
    f_v = Counter(v for v, _ in train_vn)
    f_n = Counter(n for _, n in train_vn)
    # Per-pair aggregation
    pair_total = defaultdict(int)
    pair_correct = defaultdict(int)
    for gv, gn, pv, pn in per_sample:
        pair = (gv, gn)
        pair_total[pair] += 1
        if pv == gv and pn == gn:
            pair_correct[pair] += 1
    # Build scatter: (log_freq_product, accuracy) for pairs with enough samples
    scatter = []
    for pair, n in pair_total.items():
        if n < min_samples_per_pair:
            continue
        acc = pair_correct[pair] / n * 100
        gv, gn = pair
        freq_prod = f_v.get(gv, 0) * f_n.get(gn, 0)
        scatter.append({"pair": [int(gv), int(gn)], "n": n, "acc_top1": acc,
                        "f_v": f_v.get(gv, 0), "f_n": f_n.get(gn, 0),
                        "freq_prod": freq_prod})
    # Spearman correlation
    if len(scatter) < 5:
        rho = None; pval = None
    else:
        accs = [s["acc_top1"] for s in scatter]
        prods = [s["freq_prod"] for s in scatter]
        rho, pval = spearman(accs, prods)
    return {"n_pairs_eligible": len(scatter), "min_per_pair": min_samples_per_pair,
            "spearman_rho": rho, "spearman_p": pval, "scatter": scatter}


def spearman(x, y):
    """Manual Spearman correlation (scipy may not be available in the env)."""
    n = len(x)
    if n < 3: return None, None
    rx = _rank(x); ry = _rank(y)
    mx, my = sum(rx)/n, sum(ry)/n
    num = sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    dx = math.sqrt(sum((a-mx)**2 for a in rx))
    dy = math.sqrt(sum((b-my)**2 for b in ry))
    if dx == 0 or dy == 0: return 0.0, None
    rho = num / (dx*dy)
    # Approximate two-sided p-value via t-distribution
    if abs(rho) >= 1.0:
        pval = 0.0
    else:
        t = rho * math.sqrt((n-2)/(1-rho**2))
        # rough p via normal approx (good enough for our use)
        pval = 2 * (1 - _normal_cdf(abs(t)))
    return rho, pval


def _rank(xs):
    """Average rank for ties."""
    sorted_idx = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j+1 < len(xs) and xs[sorted_idx[j+1]] == xs[sorted_idx[i]]:
            j += 1
        avg = (i + j + 2) / 2.0  # 1-indexed
        for k in range(i, j+1):
            ranks[sorted_idx[k]] = avg
        i = j + 1
    return ranks


def _normal_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ============== Dataset loaders (per-dataset) ==============
def load_meccano():
    from train_meccano_comp import (
        MeccanoCompDataset, IndepHeadModel, COMP_DIR, FEATURE_DIM, N_VERB, N_NOUN,
    )
    # MECCANO comp_val_unseen is 100% single-noun -> well-defined (v,n) per sample.
    # noun for D2/D3 == the single noun id (extracted from noun_vec)
    unseen = MeccanoCompDataset("comp_val_unseen", is_train=False)
    # Build train (v,n) list from comp_train CSV directly (single-noun only -> matches unseen-side noun semantics)
    train_vn = []
    with open(COMP_DIR / "comp_train.csv") as f:
        r = csv.DictReader(f)
        ncols = [c for c in r.fieldnames if c.startswith("noun_")]
        for row in r:
            v = int(row["verb_id"])
            for i, c in enumerate(ncols):
                if int(row[c]) == 1:
                    train_vn.append((v, i))   # contribute pair per (v, single noun) — multi-noun expanded
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
            # comp_val_unseen is single-noun → first set noun is "the" noun
            nset = [j for j, x in enumerate(nv.tolist()) if x > 0.5]
            n = nset[0] if len(nset) == 1 else nset[0]   # fallback first
            return clip, int(verb), int(n)

    return {"tag": "meccano_indep", "n_verb": N_VERB, "n_noun": N_NOUN,
            "verb_m": verb_m, "noun_m": noun_m,
            "unseen_ds": MecWrap(unseen), "train_vn": train_vn}


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
    unseen = IMPACTCompDataset(view, hand, "comp_val_unseen", is_train=False)

    class ImpWrap(torch.utils.data.Dataset):
        def __init__(self, ds): self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            clip, action, verb, noun = self.ds[i]
            return clip, int(verb), int(noun)

    return {"tag": f"impact_indep_{view}_{hand}", "n_verb": I_NV, "n_noun": I_NN,
            "verb_m": verb_m, "noun_m": noun_m,
            "unseen_ds": ImpWrap(unseen), "train_vn": train_vn}


def load_havid(subtask):
    """HAViD INDEPENDENT 2-component (verb + manip) using solo backbones."""
    from train_mvitv2_comp import (
        HAViDDataset, MViTv2MultiHead, HEAD_CLASSES, DATA_BASE,
    )
    N_VERB, N_MANIP = HEAD_CLASSES[0], HEAD_CLASSES[1]
    dr = DATA_BASE / subtask
    train_vid = dr / "train" / "videos_train"
    val_vid   = dr / "val"   / "videos_val"

    # Build train_vn from {subtask}_comp_train.txt (verb-onehot col 6, manip-onehot col 7)
    train_vn = []
    with open(dr / f"{subtask}_comp_train.txt") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10: continue
            v = parts[6].index("1") if "1" in parts[6] else -1
            m = parts[7].index("1") if "1" in parts[7] else -1
            if v >= 0 and m >= 0: train_vn.append((v, m))

    # Load solo verb + solo manip backbones (each is MViTv2MultiHead with one head)
    verb_m = MViTv2MultiHead([N_VERB], dropout=0.0, pretrained=False)
    manip_m = MViTv2MultiHead([N_MANIP], dropout=0.0, pretrained=False)
    cv = torch.load(PROJECT_DIR / f"checkpoints/havid_gran_solo_verb_{subtask}/best.pth",
                    map_location="cpu", weights_only=False)
    cm = torch.load(PROJECT_DIR / f"checkpoints/havid_gran_solo_manip_{subtask}/best.pth",
                    map_location="cpu", weights_only=False)
    verb_m.load_state_dict(cv["model"]); manip_m.load_state_dict(cm["model"])

    unseen = HAViDDataset(dr / f"{subtask}_comp_val_unseen.txt",
                          [train_vid, val_vid], is_train=False)

    class HavWrap(torch.utils.data.Dataset):
        def __init__(self, ds): self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            clip, labels = self.ds[i]   # labels = tensor [v, m, t, l]
            return clip, int(labels[0]), int(labels[1])

    return {"tag": f"havid_indep_{subtask}", "n_verb": N_VERB, "n_noun": N_MANIP,
            "verb_m": verb_m, "noun_m": manip_m,
            "unseen_ds": HavWrap(unseen), "train_vn": train_vn}


# ============== main eval loop ==============
@torch.no_grad()
def collect_per_sample(verb_m, noun_m, loader, device):
    verb_m.eval(); noun_m.eval()
    out = []
    for batch in tqdm(loader, desc="eval", leave=False):
        clip, gv, gn = batch
        clip = clip.to(device, non_blocking=True)
        v_lg = verb_m(clip); n_lg = noun_m(clip)
        # Models may return a list (multi-head with k=1) or a tensor (single-head)
        if isinstance(v_lg, list): v_lg = v_lg[0]
        if isinstance(n_lg, list): n_lg = n_lg[0]
        pv = v_lg.argmax(1).cpu().tolist()
        pn = n_lg.argmax(1).cpu().tolist()
        gv = gv.tolist() if hasattr(gv, "tolist") else list(gv)
        gn = gn.tolist() if hasattr(gn, "tolist") else list(gn)
        for a, b, c, d in zip(gv, gn, pv, pn):
            out.append((int(a), int(b), int(c), int(d)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["meccano", "impact", "havid"])
    p.add_argument("--view", default=None)
    p.add_argument("--hand", default=None)
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

    out_dir = PROJECT_DIR / "results" / f"d2d3_{cfg['tag']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[task] {cfg['tag']}  n_verb={cfg['n_verb']} n_noun={cfg['n_noun']} "
          f"unseen={len(cfg['unseen_ds'])} train_vn={len(cfg['train_vn'])}", flush=True)

    verb_m = cfg["verb_m"].to(device); noun_m = cfg["noun_m"].to(device)
    loader = DataLoader(cfg["unseen_ds"], batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    per_sample = collect_per_sample(verb_m, noun_m, loader, device)
    print(f"[done] collected {len(per_sample)} per-sample preds", flush=True)

    d3 = compute_d3(per_sample)
    d2 = compute_d2(per_sample, cfg["train_vn"], min_samples_per_pair=10)
    print(f"D3 (% of unseen): {d3['fractions']}")
    print(f"D2: rho={d2['spearman_rho']:.4f}  eligible pairs={d2['n_pairs_eligible']}")

    with open(out_dir / "per_sample.json", "w") as f:
        json.dump([{"gt_v":a,"gt_n":b,"pred_v":c,"pred_n":d} for a,b,c,d in per_sample], f)
    with open(out_dir / "d3.json", "w") as f:
        json.dump(d3, f, indent=2)
    with open(out_dir / "d2.json", "w") as f:
        json.dump(d2, f, indent=2)
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()

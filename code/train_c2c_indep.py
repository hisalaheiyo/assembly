"""
independent + C2C-Full (NO CutMix) for MECCANO and IMPACT compositional splits.

Mirrors HAViD's 4-Indep+C2C-Full-nocm pipeline, reduced to 2 components (verb+noun):
  1. load the frozen independent verb-backbone and noun-backbone (already trained)
  2. pre-extract CLS features once
  3. train a lightweight 2-component C2C composition module on features
  4. recombine -> action; report seen / unseen / HM

C2C losses: L_com + alpha*L_comp + beta*L_hsic + gamma*L_con   (CutMix L_new omitted,
following the HAViD 4-Indep+C2C-nocm design -- CutMix needs the image-space pipeline).

MECCANO noun = categorical noun-SET token (Solution B): each segment's noun-set
(deterministic, <=2 nouns, consistent per action) is one class. Uses ALL data, fair.

Backbones consumed (frozen, NOT modified):
  MECCANO: checkpoints/meccano_comp_indep_{verb,noun}/best.pth
  IMPACT : checkpoints/impact_compvar_{verb_only,noun_only}_{view}_{hand}/best.pth

Output: checkpoints/c2c_indep_{tag}/best.pth ; results/c2c_indep_{tag}/comp_eval.json
  tag = meccano   or   impact_{view}_{hand}

Usage:
  python train_c2c_indep.py --dataset meccano
  python train_c2c_indep.py --dataset impact --view front --hand left
"""

import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

FEATURE_DIM = 768
EMB_DIM = 300
COSINE_SCALE = 20.0
from paths import PROJECT_DIR  # see code/paths.py for layout / env-var overrides


# ===================== shared C2C pieces =====================
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, out_dim))
    def forward(self, x): return self.net(x)


class C2CModule(nn.Module):
    """2-component (verb, noun) C2C composition module, operates on features."""
    def __init__(self, n_verb, n_noun, train_pairs, dropout=0.3):
        super().__init__()
        self.n_verb, self.n_noun = n_verb, n_noun
        h = 512
        self.verb_proj1 = MLP(FEATURE_DIM, h, EMB_DIM, dropout)
        self.noun_proj1 = MLP(FEATURE_DIM, h, EMB_DIM, dropout)
        self.verb_proj2 = MLP(FEATURE_DIM, h, EMB_DIM, dropout)
        self.noun_proj2 = MLP(FEATURE_DIM, h, EMB_DIM, dropout)
        self.verb_emb = nn.Embedding(n_verb, EMB_DIM)
        self.noun_emb = nn.Embedding(n_noun, EMB_DIM)
        nn.init.xavier_uniform_(self.verb_emb.weight)
        nn.init.xavier_uniform_(self.noun_emb.weight)
        self.fuse_v2n = nn.Linear(2 * EMB_DIM, EMB_DIM)
        self.fuse_n2v = nn.Linear(2 * EMB_DIM, EMB_DIM)
        self.register_buffer("pair_v", torch.tensor([p[0] for p in train_pairs]))
        self.register_buffer("pair_n", torch.tensor([p[1] for p in train_pairs]))
        self.pair_to_idx = {(v, n): i for i, (v, n) in enumerate(train_pairs)}

    def forward(self, vf, nf):
        B = vf.size(0)
        f_v = self.verb_proj1(vf); f_n = self.noun_proj1(nf)
        v_e_n = F.normalize(self.verb_emb.weight, dim=1)
        n_e_n = F.normalize(self.noun_emb.weight, dim=1)
        p_v = (F.normalize(f_v, dim=1) @ v_e_n.T) * 0.5 + 0.5     # [B, V]
        p_n = (F.normalize(f_n, dim=1) @ n_e_n.T) * 0.5 + 0.5     # [B, N]

        f_v2 = self.verb_proj2(vf); f_n2 = self.noun_proj2(nf)
        # given verb -> predict noun
        f_n2_e = f_n2.unsqueeze(1).expand(B, self.n_verb, EMB_DIM)
        v_e_e = self.verb_emb.weight.unsqueeze(0).expand(B, self.n_verb, EMB_DIM)
        fused = self.fuse_v2n(torch.cat([f_n2_e, v_e_e], dim=2))
        p_n_given_v = torch.einsum("bvc,nc->bvn", F.normalize(fused, dim=2), n_e_n) * 0.5 + 0.5
        # given noun -> predict verb
        f_v2_e = f_v2.unsqueeze(1).expand(B, self.n_noun, EMB_DIM)
        n_e_e = self.noun_emb.weight.unsqueeze(0).expand(B, self.n_noun, EMB_DIM)
        fused2 = self.fuse_n2v(torch.cat([f_v2_e, n_e_e], dim=2))
        p_v_given_n = torch.einsum("bnc,vc->bnv", F.normalize(fused2, dim=2), v_e_n) * 0.5 + 0.5

        ps1 = p_v.unsqueeze(2) * p_n_given_v
        ps2 = p_n.unsqueeze(1) * p_v_given_n.permute(0, 2, 1)
        pair_scores = ps1 + ps2                                  # [B, V, N]
        return {"p_v": p_v, "p_n": p_n, "pair_scores": pair_scores,
                "p_n_given_v": p_n_given_v, "p_v_given_n": p_v_given_n,
                "f_v": f_v, "f_n": f_n}


def rbf_kernel(X, bw=20.0):
    XXT = X @ X.T
    d = XXT.diag().unsqueeze(1)
    return torch.exp(-(d + d.T - 2 * XXT) / (2 * bw ** 2))


def hsic(X, Y, bw=20.0):
    n = X.size(0)
    if n < 4: return torch.tensor(0.0, device=X.device)
    K = rbf_kernel(X, bw); L = rbf_kernel(Y, bw)
    H = torch.eye(n, device=X.device) - 1.0 / n
    return (K @ H @ L @ H).trace() / ((n - 1) ** 2)


def compute_loss(out, v_lab, n_lab, module, co_v, co_n, alpha=0.2, beta=0.1, gamma=0.1):
    p_v, p_n = out["p_v"], out["p_n"]
    L_comp = (F.cross_entropy(p_v * COSINE_SCALE, v_lab)
              + F.cross_entropy(p_n * COSINE_SCALE, n_lab))
    # L_com: composition CE over training pairs
    pair_pred = out["pair_scores"][:, module.pair_v, module.pair_n]    # [B, n_pairs]
    tgt, valid = [], []
    for i in range(v_lab.size(0)):
        idx = module.pair_to_idx.get((v_lab[i].item(), n_lab[i].item()), -1)
        if idx >= 0:
            tgt.append(idx); valid.append(i)
    if valid:
        pt = torch.tensor(tgt, dtype=torch.long, device=v_lab.device)
        L_com = F.cross_entropy(pair_pred[valid] * COSINE_SCALE, pt)
    else:
        L_com = torch.tensor(0.0, device=v_lab.device)
    half = EMB_DIM // 2
    L_hsic = hsic(out["f_v"][:, :half], out["f_n"][:, :half])
    # L_con: align mean conditionals to empirical co-occurrence
    p_nv = out["p_n_given_v"].mean(0)
    p_nv = p_nv / p_nv.sum(1, keepdim=True).clamp(min=1e-6)
    L_con_v = -(co_v * torch.log(p_nv.clamp(min=1e-6))).sum(1).mean()
    p_vn = out["p_v_given_n"].mean(0)
    p_vn = p_vn / p_vn.sum(1, keepdim=True).clamp(min=1e-6)
    L_con_n = -(co_n * torch.log(p_vn.clamp(min=1e-6))).sum(1).mean()
    L_con = L_con_v + L_con_n
    total = L_com + alpha * L_comp + beta * L_hsic + gamma * L_con
    return total, {"L_com": L_com.item(), "L_comp": L_comp.item(),
                   "L_hsic": L_hsic.item(), "L_con": L_con.item()}


def hm(a, b):
    return 2 * a * b / (a + b) if (a + b) > 0 else 0.0


# ===================== dataset adapters =====================
def load_meccano():
    """Returns dict with: feature-extraction datasets, n_verb, n_noun(=noun-set),
    verb/noun backbones, action<->(v,n) maps."""
    from train_meccano_comp import (MeccanoCompDataset, IndepHeadModel, COMP_DIR as MEC_COMP)

    # ---- build noun-set vocabulary from all comp splits ----
    nset_to_id = {}
    action_to_vn = {}     # action_id -> (verb_id, nset_id)
    for split in ["comp_train", "comp_val_seen", "comp_val_unseen"]:
        with open(MEC_COMP / f"{split}.csv") as f:
            r = csv.DictReader(f)
            ncols = [c for c in r.fieldnames if c.startswith("noun_")]
            for row in r:
                nset = tuple(i for i, c in enumerate(ncols) if int(row[c]) == 1)
                if nset not in nset_to_id:
                    nset_to_id[nset] = len(nset_to_id)
                a, v = int(row["action_id"]), int(row["verb_id"])
                action_to_vn[a] = (v, nset_to_id[nset])
    n_verb, n_noun = 12, len(nset_to_id)

    class MecWrap(torch.utils.data.Dataset):
        """Yields (clip, verb_id, nset_id, action_id)."""
        def __init__(self, split):
            self.ds = MeccanoCompDataset(split, is_train=False)
            self.ncols = self.ds.noun_cols
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            clip, action, verb, noun_vec = self.ds[i]
            nset = tuple(j for j, x in enumerate(noun_vec.tolist()) if x > 0.5)
            return clip, verb, torch.tensor(nset_to_id[nset]), action

    verb_bb = IndepHeadModel("verb", dropout=0.0, pretrained=False)
    cv = torch.load(PROJECT_DIR / "checkpoints/meccano_comp_indep_verb/best.pth",
                    map_location="cpu", weights_only=False)
    verb_bb.load_state_dict(cv["model"])
    noun_bb = IndepHeadModel("noun", dropout=0.0, pretrained=False)
    cn = torch.load(PROJECT_DIR / "checkpoints/meccano_comp_indep_noun/best.pth",
                    map_location="cpu", weights_only=False)
    noun_bb.load_state_dict(cn["model"])

    return {"n_verb": n_verb, "n_noun": n_noun, "action_to_vn": action_to_vn,
            "verb_bb": verb_bb, "noun_bb": noun_bb, "tag": "meccano",
            "make_ds": lambda sp: MecWrap(sp)}


def load_impact(view, hand):
    from train_impact_mvitv2 import N_VERB as I_NV, N_NOUN as I_NN
    from train_impact_comp import IMPACTCompDataset
    from train_impact_comp_variants import SingleHeadModel, build_seen_action_map  # noqa

    # action -> (verb, noun): IMPACT bijective; build from all comp splits
    from train_impact_comp import COMP_DIR as IMP_COMP
    action_to_vn = {}
    for split in ["comp_train", "comp_val_seen", "comp_val_unseen"]:
        for line in open(IMP_COMP / hand / f"{split}.txt"):
            line = line.strip()
            if not line: continue
            _, _, _, v, n, a = line.split(",")
            action_to_vn[int(a)] = (int(v), int(n))

    verb_bb = SingleHeadModel(I_NV, dropout=0.0, pretrained=False)
    cv = torch.load(PROJECT_DIR / f"checkpoints/impact_compvar_verb_only_{view}_{hand}/best.pth",
                    map_location="cpu", weights_only=False)
    verb_bb.load_state_dict(cv["model"])
    noun_bb = SingleHeadModel(I_NN, dropout=0.0, pretrained=False)
    cn = torch.load(PROJECT_DIR / f"checkpoints/impact_compvar_noun_only_{view}_{hand}/best.pth",
                    map_location="cpu", weights_only=False)
    noun_bb.load_state_dict(cn["model"])

    class ImpWrap(torch.utils.data.Dataset):
        def __init__(self, split):
            self.ds = IMPACTCompDataset(view, hand, split, is_train=False)
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            clip, action, verb, noun = self.ds[i]
            return clip, verb, noun, action

    return {"n_verb": I_NV, "n_noun": I_NN, "action_to_vn": action_to_vn,
            "verb_bb": verb_bb, "noun_bb": noun_bb, "tag": f"impact_{view}_{hand}",
            "make_ds": lambda sp: ImpWrap(sp)}


# ---- HAViD 2-component (verb, manip) ----
class _HAViDFeatExtractor(nn.Module):
    """Adapter: wraps HAViD's MViTv2MultiHead so .backbone(x) returns the pooled
    CLS feature (drops the linear head). Used to feed C2CModule."""
    def __init__(self, mh_model):
        super().__init__()
        self.conv_proj = mh_model.conv_proj
        self.pos_encoding = mh_model.pos_encoding
        self.blocks = mh_model.blocks
        self.norm = mh_model.norm

    def backbone(self, x):
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_encoding(x)
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        for block in self.blocks:
            x, thw = block(x, thw)
        x = self.norm(x)
        return x[:, 0]


def load_havid(subtask):
    """HAViD INDEPENDENT 2-component (verb, manip) on the comp split. Reuses the
    pretrained havid_gran_solo_verb / solo_manip backbones (each a MViTv2-S with
    a single head, trained on the same comp_train as everything else)."""
    from train_mvitv2_comp import (
        HAViDDataset, MViTv2MultiHead, HEAD_CLASSES, DATA_BASE,
    )
    n_verb, n_manip = HEAD_CLASSES[0], HEAD_CLASSES[1]
    dr = DATA_BASE / subtask
    train_vid = dr / "train" / "videos_train"
    val_vid   = dr / "val"   / "videos_val"

    # action_to_vn: enumerate unique (v, m) pairs across all 3 comp splits;
    # "action id" is just a dense index over those pairs. Eval action top-1
    # therefore equals (v, m) joint top-1.
    pair_to_id = {}
    action_to_vn = {}
    for split in ["comp_train", "comp_val_seen", "comp_val_unseen"]:
        with open(dr / f"{subtask}_{split}.txt") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 10: continue
                v = parts[6].index("1") if "1" in parts[6] else -1
                m = parts[7].index("1") if "1" in parts[7] else -1
                if v < 0 or m < 0: continue
                key = (v, m)
                if key not in pair_to_id:
                    aid = len(pair_to_id)
                    pair_to_id[key] = aid
                    action_to_vn[aid] = key

    verb_mh = MViTv2MultiHead([n_verb], dropout=0.0, pretrained=False)
    manip_mh = MViTv2MultiHead([n_manip], dropout=0.0, pretrained=False)
    cv = torch.load(PROJECT_DIR / f"checkpoints/havid_gran_solo_verb_{subtask}/best.pth",
                    map_location="cpu", weights_only=False)
    cm = torch.load(PROJECT_DIR / f"checkpoints/havid_gran_solo_manip_{subtask}/best.pth",
                    map_location="cpu", weights_only=False)
    verb_mh.load_state_dict(cv["model"]); manip_mh.load_state_dict(cm["model"])
    verb_bb = _HAViDFeatExtractor(verb_mh)
    manip_bb = _HAViDFeatExtractor(manip_mh)

    class HavWrap(torch.utils.data.Dataset):
        """Yields (clip, verb_id, manip_id, action_id)."""
        def __init__(self, split):
            self.ds = HAViDDataset(dr / f"{subtask}_{split}.txt",
                                   [train_vid, val_vid], is_train=False)
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            clip, labels = self.ds[i]    # labels = [v, m, t, l]
            v, m = int(labels[0]), int(labels[1])
            aid = pair_to_id.get((v, m), -1)
            # For unseen pairs not in pair_to_id (shouldn't happen since we
            # enumerate all 3 splits), fall back to -1 (action_topk will give -inf)
            return clip, v, m, aid if aid >= 0 else 0

    return {"n_verb": n_verb, "n_noun": n_manip, "action_to_vn": action_to_vn,
            "verb_bb": verb_bb, "noun_bb": manip_bb,
            "tag": f"havid_{subtask}",
            "make_ds": lambda sp: HavWrap(sp)}


@torch.no_grad()
def extract(verb_bb, noun_bb, loader, device):
    verb_bb.eval(); noun_bb.eval()
    vf, nf, V, N, A = [], [], [], [], []
    for clip, verb, noun, action in tqdm(loader, desc="extract", leave=False):
        clip = clip.to(device, non_blocking=True)
        vf.append(verb_bb.backbone(clip).cpu())
        nf.append(noun_bb.backbone(clip).cpu())
        V.append(verb); N.append(noun); A.append(action)
    return (torch.cat(vf), torch.cat(nf),
            torch.cat(V), torch.cat(N), torch.cat(A))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["meccano", "impact", "havid"])
    p.add_argument("--view", default=None)
    p.add_argument("--hand", default=None)
    p.add_argument("--subtask", default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--feat_batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "meccano":
        cfg = load_meccano()
    elif args.dataset == "impact":
        assert args.view and args.hand, "--view/--hand required for impact"
        cfg = load_impact(args.view, args.hand)
    else:  # havid
        assert args.subtask, "--subtask required for havid (e.g. view0_lh_aa)"
        cfg = load_havid(args.subtask)
    tag = cfg["tag"]
    n_verb, n_noun = cfg["n_verb"], cfg["n_noun"]
    print(f"[task] c2c_indep {tag}  n_verb={n_verb} n_noun={n_noun}")

    ckpt_dir = PROJECT_DIR / "checkpoints" / f"c2c_indep_{tag}"
    result_dir = PROJECT_DIR / "results" / f"c2c_indep_{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    verb_bb = cfg["verb_bb"].to(device)
    noun_bb = cfg["noun_bb"].to(device)
    for m in (verb_bb, noun_bb):
        for prm in m.parameters(): prm.requires_grad = False

    # ---- extract features for the 3 splits ----
    feats = {}
    for sp in ["comp_train", "comp_val_seen", "comp_val_unseen"]:
        ds = cfg["make_ds"](sp)
        loader = DataLoader(ds, batch_size=args.feat_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        feats[sp] = extract(verb_bb, noun_bb, loader, device)
        print(f"  {sp}: {feats[sp][0].size(0)} samples")
    del verb_bb, noun_bb; torch.cuda.empty_cache()

    # ---- train pairs + co-occurrence (from comp_train) ----
    tv, tn = feats["comp_train"][2], feats["comp_train"][3]
    train_pairs = sorted(set((int(a), int(b)) for a, b in zip(tv.tolist(), tn.tolist())))
    counts = torch.zeros(n_verb, n_noun)
    for a, b in zip(tv.tolist(), tn.tolist()):
        counts[a, b] += 1.0
    co_v = (counts / counts.sum(1, keepdim=True).clamp(min=1.0)).to(device)
    co_n = (counts / counts.sum(0, keepdim=True).clamp(min=1.0)).T.to(device)
    print(f"  {len(train_pairs)} train (verb,noun) pairs")

    module = C2CModule(n_verb, n_noun, train_pairs, dropout=0.3).to(device)
    opt = optim.AdamW(module.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    # action recombination lookup
    a2vn = cfg["action_to_vn"]
    max_a = max(a2vn.keys()) + 1
    act_v = torch.full((max_a,), -1, dtype=torch.long)
    act_n = torch.full((max_a,), -1, dtype=torch.long)
    for a, (v, n) in a2vn.items():
        act_v[a] = v; act_n[a] = n
    act_v, act_n = act_v.to(device), act_n.to(device)
    valid_a = (act_v >= 0)

    def action_topk(p_v, p_n, action, ks=(1, 3, 5, 10)):
        lpv = F.log_softmax(p_v * COSINE_SCALE, dim=1)
        lpn = F.log_softmax(p_n * COSINE_SCALE, dim=1)
        scores = torch.full((p_v.size(0), max_a), float("-inf"), device=p_v.device)
        idx = valid_a.nonzero(as_tuple=True)[0]
        scores[:, idx] = lpv[:, act_v[idx]] + lpn[:, act_n[idx]]
        maxk = min(max(ks), max_a)
        _, pred = scores.topk(maxk, dim=1)
        corr = pred.eq(action.unsqueeze(1))
        return {k: corr[:, :min(k, maxk)].any(1).float().sum().item() for k in ks}

    def make_loader(sp, shuffle):
        vf, nf, V, N, A = feats[sp]
        return DataLoader(TensorDataset(vf, nf, V, N, A),
                          batch_size=args.batch_size, shuffle=shuffle, drop_last=shuffle)

    train_loader = make_loader("comp_train", True)

    @torch.no_grad()
    def evaluate(sp):
        module.eval()
        loader = make_loader(sp, False)
        n = 0; ac = {1: 0, 3: 0, 5: 0, 10: 0}
        cell = {"both": 0, "verb_only": 0, "noun_only": 0, "neither": 0}
        for vf, nf, V, N, A in loader:
            vf, nf = vf.to(device), nf.to(device)
            V, N, A = V.to(device), N.to(device), A.to(device)
            out = module(vf, nf)
            tk = action_topk(out["p_v"], out["p_n"], A)
            for k in ac: ac[k] += tk[k]
            pv = (out["p_v"] * COSINE_SCALE).argmax(1)
            pn = (out["p_n"] * COSINE_SCALE).argmax(1)
            v_ok = (pv == V); n_ok = (pn == N)
            cell["both"] += int((v_ok & n_ok).sum())
            cell["verb_only"] += int((v_ok & ~n_ok).sum())
            cell["noun_only"] += int((~v_ok & n_ok).sum())
            cell["neither"] += int((~v_ok & ~n_ok).sum())
            n += vf.size(0)
        return {"n": n, **{f"action_top{k}": ac[k] / n * 100 for k in ac},
                "err_decomp": {k: cell[k] / n * 100 for k in cell}}

    best_hm = 0.0
    for epoch in range(args.epochs):
        module.train()
        t0 = time.time(); tot = 0.0; nn_ = 0
        for vf, nf, V, N, A in train_loader:
            vf, nf, V, N = vf.to(device), nf.to(device), V.to(device), N.to(device)
            out = module(vf, nf)
            loss, _ = compute_loss(out, V, N, module, co_v, co_n,
                                   args.alpha, args.beta, args.gamma)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(module.parameters(), max_norm=5.0)
            opt.step()
            tot += loss.item() * vf.size(0); nn_ += vf.size(0)
        sched.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            seen = evaluate("comp_val_seen")
            unseen = evaluate("comp_val_unseen")
            cur_hm = hm(seen["action_top1"], unseen["action_top1"])
            print(f"E{epoch:03d}/{args.epochs} loss={tot/nn_:.4f} "
                  f"seen_t1={seen['action_top1']:.2f}% unseen_t1={unseen['action_top1']:.2f}% "
                  f"HM={cur_hm:.2f} ({time.time()-t0:.0f}s)")
            if cur_hm > best_hm:
                best_hm = cur_hm
                torch.save({"epoch": epoch, "module": module.state_dict(),
                            "train_pairs": train_pairs, "best_hm": best_hm},
                           ckpt_dir / "best.pth")

    # final eval
    c = torch.load(ckpt_dir / "best.pth", map_location=device, weights_only=False)
    module.load_state_dict(c["module"])
    seen = evaluate("comp_val_seen")
    unseen = evaluate("comp_val_unseen")
    final_hm = hm(seen["action_top1"], unseen["action_top1"])
    print(f"\n=== FINAL c2c_indep {tag} (epoch {c['epoch']}) ===")
    print(f"SEEN  : {seen}")
    print(f"UNSEEN: {unseen}")
    print(f"HM(action_top1) = {final_hm:.2f}")
    with open(result_dir / "comp_eval.json", "w") as f:
        json.dump({"dataset": args.dataset, "tag": tag, "arch": "indep+C2C-full(noCutMix)",
                   "best_epoch": c["epoch"], "seen": seen, "unseen": unseen,
                   "HM_action_top1": final_hm}, f, indent=2)
    print(f"[done] -> {result_dir/'comp_eval.json'}")


if __name__ == "__main__":
    main()

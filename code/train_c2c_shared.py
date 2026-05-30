"""
shared + C2C-Full (WITH CutMix) for MECCANO and IMPACT compositional splits.

Faithful 2-component (verb+noun) port of HAViD's MViTv2C2CFull (train_mvitv2_c2c_full.py):
  - single shared MViTv2-S backbone, fully fine-tuned
  - C2C composition head (verb/noun projections + embeddings + bidirectional
    conditional prediction), reused from train_c2c_indep.C2CModule
  - losses: L_com + alpha*L_comp + beta*L_hsic + gamma*L_con + delta*L_new
  - L_new uses CutMix (image-space clip mixing) -> kept here (shared pipeline supports it)

Hyperparameters match the HAViD C2C-Full script: bb_lr=5e-5, c2c_lr=2.5e-4,
alpha=0.2, beta=0.1, gamma=0.1, delta=0.1, cutmix_prob=0.5, 100 epochs, warmup 20.

MECCANO noun = categorical noun-SET token (Solution B; 26 classes), same as
train_c2c_indep. Action top-1 (recombined) is the comparable metric across configs.

Output: checkpoints/c2c_shared_{tag}/{best,latest}.pth ; results/c2c_shared_{tag}/comp_eval.json

Usage:
  python train_c2c_shared.py --dataset meccano
  python train_c2c_shared.py --dataset impact --view front --hand left
"""

import argparse, csv, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from train_c2c_indep import (
    C2CModule, hsic, hm, compute_loss,
    FEATURE_DIM, EMB_DIM, COSINE_SCALE, PROJECT_DIR,
)


# ===================== model =====================
class MViTv2C2CSharedVN(nn.Module):
    """Shared MViTv2-S backbone + 2-component C2C head (verb, noun)."""
    def __init__(self, n_verb, n_noun, train_pairs, dropout=0.5, pretrained=True):
        super().__init__()
        weights = MViT_V2_S_Weights.DEFAULT if pretrained else None
        base = mvit_v2_s(weights=weights)
        self.conv_proj = base.conv_proj
        self.pos_encoding = base.pos_encoding
        self.blocks = base.blocks
        self.norm = base.norm
        self.bb_dropout = nn.Dropout(p=dropout)
        self.c2c = C2CModule(n_verb, n_noun, train_pairs, dropout=0.3)

    def extract_backbone(self, x):
        x = self.conv_proj(x); x = x.flatten(2).transpose(1, 2)
        x = self.pos_encoding(x)
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        for blk in self.blocks:
            x, thw = blk(x, thw)
        return self.norm(x)[:, 0]

    def forward(self, x):
        feat = self.bb_dropout(self.extract_backbone(x))
        return self.c2c(feat, feat)        # shared: verb & noun see the same feature

    def backbone_params(self):
        return [p for n, p in self.named_parameters() if n.startswith(
            ("conv_proj", "pos_encoding", "blocks", "norm"))]

    def c2c_params(self):
        return [p for n, p in self.named_parameters() if n.startswith("c2c.")]


# ===================== CutMix =====================
def rand_bbox(H, W, lam):
    cut_rat = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_rat), int(W * cut_rat)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = np.clip(cy - cut_h // 2, 0, H), np.clip(cy + cut_h // 2, 0, H)
    x1, x2 = np.clip(cx - cut_w // 2, 0, W), np.clip(cx + cut_w // 2, 0, W)
    return y1, x1, y2, x2


def apply_cutmix(frames, labels):
    """frames [B,3,T,H,W], labels [B,2] (verb,noun). Returns mixed, lab_a, lab_b, eff_lam."""
    B, C, T, H, W = frames.shape
    lam = np.random.beta(1.0, 1.0)
    perm = torch.randperm(B, device=frames.device)
    y1, x1, y2, x2 = rand_bbox(H, W, lam)
    mixed = frames.clone()
    mixed[:, :, :, y1:y2, x1:x2] = frames[perm][:, :, :, y1:y2, x1:x2]
    eff_lam = 1.0 - ((y2 - y1) * (x2 - x1) / float(H * W))
    return mixed, labels, labels[perm], eff_lam


def l_new_loss(pair_scores, lab_a, lab_b, n_noun):
    """Novel-composition CutMix loss: encourage (v_a,n_b) and (v_b,n_a)."""
    bs = pair_scores.size(0)
    flat = pair_scores.view(bs, -1)                  # [B, V*N]
    v_a, n_a = lab_a[:, 0], lab_a[:, 1]
    v_b, n_b = lab_b[:, 0], lab_b[:, 1]
    tgt_va_nb = v_a * n_noun + n_b
    tgt_vb_na = v_b * n_noun + n_a
    return (F.cross_entropy(flat * COSINE_SCALE, tgt_va_nb)
            + F.cross_entropy(flat * COSINE_SCALE, tgt_vb_na)) / 2


def build_scheduler(opt, warmup, total, mr=1/600):
    def f(e):
        if e < warmup: return (e + 1) / warmup
        prog = (e - warmup) / (total - warmup)
        return mr + (1 - mr) * 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


# ===================== dataset config =====================
def get_config(dataset, view, hand):
    """Returns n_verb, n_noun, action_to_vn, tag, the 3 video datasets yielding
    (clip, verb_id, noun_id, action_id), and train_vn = list of (verb,noun) for
    every comp_train segment (built from annotations, no video decoding)."""
    if dataset == "meccano":
        from train_meccano_comp import MeccanoCompDataset, COMP_DIR as MEC_COMP
        nset_to_id, action_to_vn = {}, {}
        train_vn = []
        for split in ["comp_train", "comp_val_seen", "comp_val_unseen"]:
            with open(MEC_COMP / f"{split}.csv") as f:
                r = csv.DictReader(f)
                ncols = [c for c in r.fieldnames if c.startswith("noun_")]
                for row in r:
                    nset = tuple(i for i, c in enumerate(ncols) if int(row[c]) == 1)
                    if nset not in nset_to_id:
                        nset_to_id[nset] = len(nset_to_id)
                    v, nid = int(row["verb_id"]), nset_to_id[nset]
                    action_to_vn[int(row["action_id"])] = (v, nid)
                    if split == "comp_train":
                        train_vn.append((v, nid))
        n_verb, n_noun = 12, len(nset_to_id)

        class MecWrap(torch.utils.data.Dataset):
            def __init__(self, split, is_train):
                self.ds = MeccanoCompDataset(split, is_train=is_train)
            def __len__(self): return len(self.ds)
            def __getitem__(self, i):
                clip, action, verb, noun_vec = self.ds[i]
                nset = tuple(j for j, x in enumerate(noun_vec.tolist()) if x > 0.5)
                return clip, verb, torch.tensor(nset_to_id[nset]), action

        return (n_verb, n_noun, action_to_vn, "meccano",
                MecWrap("comp_train", True), MecWrap("comp_val_seen", False),
                MecWrap("comp_val_unseen", False), train_vn)

    elif dataset == "impact":
        from train_impact_mvitv2 import N_VERB as I_NV, N_NOUN as I_NN
        from train_impact_comp import IMPACTCompDataset, COMP_DIR as IMP_COMP
        action_to_vn = {}
        train_vn = []
        for split in ["comp_train", "comp_val_seen", "comp_val_unseen"]:
            for line in open(IMP_COMP / hand / f"{split}.txt"):
                line = line.strip()
                if not line: continue
                _, _, _, v, n, a = line.split(",")
                action_to_vn[int(a)] = (int(v), int(n))
                if split == "comp_train":
                    train_vn.append((int(v), int(n)))

        class ImpWrap(torch.utils.data.Dataset):
            def __init__(self, split, is_train):
                self.ds = IMPACTCompDataset(view, hand, split, is_train=is_train)
            def __len__(self): return len(self.ds)
            def __getitem__(self, i):
                clip, action, verb, noun = self.ds[i]
                return clip, verb, noun, action

        return (I_NV, I_NN, action_to_vn, f"impact_{view}_{hand}",
                ImpWrap("comp_train", True), ImpWrap("comp_val_seen", False),
                ImpWrap("comp_val_unseen", False), train_vn)

    # HAViD 2-component (verb, manip) — fixed-action == (v,m) pair
    if dataset == "havid":
        from train_mvitv2_comp import HAViDDataset, HEAD_CLASSES, DATA_BASE
        assert view is not None, "for havid, pass subtask via --view (reused arg slot)"
        # We re-purpose --view as the subtask name (e.g. view0_lh_aa) for HAViD
        subtask = view
        n_verb, n_manip = HEAD_CLASSES[0], HEAD_CLASSES[1]
        dr = DATA_BASE / subtask
        train_vid = dr / "train" / "videos_train"
        val_vid = dr / "val" / "videos_val"
        # action_to_vn = (v, m) pair → dense id; also collect train_vn for co-occ
        pair_to_id = {}; action_to_vn = {}; train_vn = []
        for split in ["comp_train", "comp_val_seen", "comp_val_unseen"]:
            with open(dr / f"{subtask}_{split}.txt") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 10: continue
                    v = parts[6].index("1") if "1" in parts[6] else -1
                    m = parts[7].index("1") if "1" in parts[7] else -1
                    if v < 0 or m < 0: continue
                    if (v, m) not in pair_to_id:
                        aid = len(pair_to_id)
                        pair_to_id[(v, m)] = aid
                        action_to_vn[aid] = (v, m)
                    if split == "comp_train":
                        train_vn.append((v, m))

        class HavWrap(torch.utils.data.Dataset):
            def __init__(self, split, is_train):
                self.ds = HAViDDataset(dr / f"{subtask}_{split}.txt",
                                       [train_vid, val_vid], is_train=is_train)
            def __len__(self): return len(self.ds)
            def __getitem__(self, i):
                clip, labels = self.ds[i]   # [v, m, t, l]
                v, m = int(labels[0]), int(labels[1])
                aid = pair_to_id.get((v, m), 0)
                return clip, torch.tensor(v), torch.tensor(m), torch.tensor(aid)

        return (n_verb, n_manip, action_to_vn, f"havid_{subtask}",
                HavWrap("comp_train", True), HavWrap("comp_val_seen", False),
                HavWrap("comp_val_unseen", False), train_vn)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["meccano", "impact", "havid"])
    p.add_argument("--view", default=None)
    p.add_argument("--hand", default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--bb_lr", type=float, default=5e-5)
    p.add_argument("--c2c_lr", type=float, default=2.5e-4)
    p.add_argument("--warmup_epochs", type=int, default=20)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--cutmix_prob", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag_suffix", type=str, default="",
                   help="Append to ckpt/result dir name; used for ablation runs.")
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "impact":
        assert args.view and args.hand, "--view/--hand required for impact"
    elif args.dataset == "havid":
        assert args.view, "for havid, pass subtask via --view (e.g. view0_lh_aa)"
    n_verb, n_noun, action_to_vn, tag, train_ds, seen_ds, unseen_ds, train_vn = \
        get_config(args.dataset, args.view, args.hand)
    print(f"[task] c2c_shared {tag}  n_verb={n_verb} n_noun={n_noun}")

    ckpt_dir = PROJECT_DIR / "checkpoints" / f"c2c_shared_{tag}{args.tag_suffix}"
    result_dir = PROJECT_DIR / "results" / f"c2c_shared_{tag}{args.tag_suffix}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.num_workers, pin_memory=True,
                                   drop_last=sh, persistent_workers=False, prefetch_factor=2)
    train_loader = dl(train_ds, True)
    seen_loader = dl(seen_ds, False)
    unseen_loader = dl(unseen_ds, False)
    print(f"[data] train={len(train_ds)} seen={len(seen_ds)} unseen={len(unseen_ds)}")

    # train pairs + co-occurrence, built from comp_train annotations (no video decode)
    counts = torch.zeros(n_verb, n_noun)
    for v, n in train_vn:
        counts[v, n] += 1.0
    train_pairs = sorted(set(train_vn))
    co_v = (counts / counts.sum(1, keepdim=True).clamp(min=1.0)).to(device)
    co_n = (counts / counts.sum(0, keepdim=True).clamp(min=1.0)).T.to(device)
    print(f"[pairs] {len(train_pairs)} train (verb,noun) pairs")
    print(f"[loss] a={args.alpha} b={args.beta} g={args.gamma} d={args.delta} "
          f"cutmix_prob={args.cutmix_prob}")

    model = MViTv2C2CSharedVN(n_verb, n_noun, train_pairs,
                              dropout=args.dropout, pretrained=True).to(device)
    opt = optim.AdamW([
        {"params": model.backbone_params(), "lr": args.bb_lr,  "weight_decay": args.weight_decay},
        {"params": model.c2c_params(),      "lr": args.c2c_lr, "weight_decay": 5e-5},
    ], betas=(0.9, 0.999))
    sched = build_scheduler(opt, args.warmup_epochs, args.epochs)

    # action recombination lookup
    max_a = max(action_to_vn.keys()) + 1
    act_v = torch.full((max_a,), -1, dtype=torch.long)
    act_n = torch.full((max_a,), -1, dtype=torch.long)
    for a, (v, n) in action_to_vn.items():
        act_v[a], act_n[a] = v, n
    act_v, act_n = act_v.to(device), act_n.to(device)
    valid_idx = (act_v >= 0).nonzero(as_tuple=True)[0]

    def action_topk(p_v, p_n, action, ks=(1, 3, 5, 10)):
        lpv = F.log_softmax(p_v * COSINE_SCALE, dim=1)
        lpn = F.log_softmax(p_n * COSINE_SCALE, dim=1)
        scores = torch.full((p_v.size(0), max_a), float("-inf"), device=p_v.device)
        scores[:, valid_idx] = lpv[:, act_v[valid_idx]] + lpn[:, act_n[valid_idx]]
        maxk = min(max(ks), max_a)
        _, pred = scores.topk(maxk, dim=1)
        corr = pred.eq(action.unsqueeze(1))
        return {k: corr[:, :min(k, maxk)].any(1).float().sum().item() for k in ks}

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        n = 0; ac = {1: 0, 3: 0, 5: 0, 10: 0}
        cell = {"both": 0, "verb_only": 0, "noun_only": 0, "neither": 0}
        for clip, verb, noun, action in tqdm(loader, desc="eval", leave=False):
            clip = clip.to(device); verb = verb.to(device)
            noun = noun.to(device); action = action.to(device)
            out = model(clip)
            tk = action_topk(out["p_v"], out["p_n"], action)
            for k in ac: ac[k] += tk[k]
            pv = (out["p_v"] * COSINE_SCALE).argmax(1)
            pn = (out["p_n"] * COSINE_SCALE).argmax(1)
            v_ok, n_ok = (pv == verb), (pn == noun)
            cell["both"] += int((v_ok & n_ok).sum())
            cell["verb_only"] += int((v_ok & ~n_ok).sum())
            cell["noun_only"] += int((~v_ok & n_ok).sum())
            cell["neither"] += int((~v_ok & ~n_ok).sum())
            n += clip.size(0)
        return {"n": n, **{f"action_top{k}": ac[k]/n*100 for k in ac},
                "err_decomp": {k: cell[k]/n*100 for k in cell}}

    # Robust resume: prefer latest.pth, fall back to best.pth if latest.pth is
    # missing or corrupt (e.g. torch.save interrupted
    # SIGKILL). Both checkpoints carry the full model/optimizer/scheduler state.
    start_epoch, best_hm = 0, 0.0
    for cand in ["latest.pth", "best.pth"]:
        cpath = ckpt_dir / cand
        if not cpath.exists():
            continue
        try:
            c = torch.load(cpath, map_location=device, weights_only=False)
            model.load_state_dict(c["model"]); opt.load_state_dict(c["optimizer"])
            sched.load_state_dict(c["scheduler"]); start_epoch = c["epoch"] + 1
            best_hm = c.get("best_hm", 0.0)
            print(f"[resume] from {cand}: epoch={c['epoch']} best_hm={best_hm:.4f}")
            break
        except Exception as e:
            print(f"[resume] {cand} unreadable ({e}); trying fallback")

    writer = SummaryWriter(log_dir=str(result_dir / "tensorboard"))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time(); tot = 0.0; nn_ = 0
        for clip, verb, noun, action in tqdm(train_loader, desc=f"E{epoch:03d}", leave=False):
            clip = clip.to(device, non_blocking=True)
            labels = torch.stack([verb, noun], dim=1).to(device)   # [B,2]
            cm_a = cm_b = None
            if random.random() < args.cutmix_prob:
                clip, cm_a, cm_b, _ = apply_cutmix(clip, labels)
                cur = cm_a
            else:
                cur = labels
            out = model(clip)
            loss, _ = compute_loss(out, cur[:, 0], cur[:, 1], model.c2c,
                                   co_v, co_n, args.alpha, args.beta, args.gamma)
            if cm_a is not None:
                loss = loss + args.delta * l_new_loss(out["pair_scores"], cm_a, cm_b, n_noun)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            bs = clip.size(0); nn_ += bs; tot += loss.item() * bs
        sched.step()

        seen = evaluate(seen_loader)
        unseen = evaluate(unseen_loader)
        cur_hm = hm(seen["action_top1"], unseen["action_top1"])
        print(f"E{epoch:03d}/{args.epochs} loss={tot/nn_:.4f} "
              f"seen_t1={seen['action_top1']:.2f}% unseen_t1={unseen['action_top1']:.2f}% "
              f"HM={cur_hm:.2f} ({time.time()-t0:.0f}s)")
        writer.add_scalar("seen/action_top1", seen["action_top1"], epoch)
        writer.add_scalar("unseen/action_top1", unseen["action_top1"], epoch)
        writer.add_scalar("unseen/HM", cur_hm, epoch)

        c = {"epoch": epoch, "model": model.state_dict(),
             "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
             "best_hm": best_hm}
        # Atomic save: write to .tmp then rename, so a interruption mid-write
        # cannot leave a truncated/corrupt checkpoint.
        tmp = ckpt_dir / "latest.pth.tmp"
        torch.save(c, tmp); tmp.replace(ckpt_dir / "latest.pth")
        if cur_hm > best_hm:
            best_hm = cur_hm
            c["best_hm"] = best_hm
            tmp = ckpt_dir / "best.pth.tmp"
            torch.save(c, tmp); tmp.replace(ckpt_dir / "best.pth")
            print(f"  [*] new best HM = {best_hm:.2f}")

    writer.close()
    c = torch.load(ckpt_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(c["model"])
    seen = evaluate(seen_loader)
    unseen = evaluate(unseen_loader)
    final_hm = hm(seen["action_top1"], unseen["action_top1"])
    print(f"\n=== FINAL c2c_shared {tag} (epoch {c['epoch']}) ===")
    print(f"SEEN  : {seen}")
    print(f"UNSEEN: {unseen}")
    print(f"HM(action_top1) = {final_hm:.2f}")
    with open(result_dir / "comp_eval.json", "w") as f:
        json.dump({"dataset": args.dataset, "tag": tag, "arch": "shared+C2C-full(CutMix)",
                   "best_epoch": c["epoch"], "seen": seen, "unseen": unseen,
                   "HM_action_top1": final_hm}, f, indent=2)
    print(f"[done] -> {result_dir/'comp_eval.json'}")


if __name__ == "__main__":
    main()

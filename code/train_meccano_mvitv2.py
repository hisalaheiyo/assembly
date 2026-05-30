"""
Train MViTv2-S on MECCANO with two heads (verb + noun multi-label).
Uses ORIGINAL train/val/test split (NOT compositional).

At evaluation, action_id is reconstructed from (verb, noun_set) prediction
and compared to ground truth action_id for top-1/top-5 accuracy.

Usage:
  python train_meccano_mvitv2.py --mode multihead   # verb + noun
  python train_meccano_mvitv2.py --mode action      # single 61-class head
"""

import argparse, os, time, math, json, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
import av
from tqdm import tqdm

# ---- Constants ----
N_VERB = 12
N_NOUN = 23
N_ACTION = 61
from paths import MECCANO_DATA as DATA_BASE
from paths import PROJECT_DIR  # see code/paths.py for layout / env-var overrides

# MECCANO clip parameters (videos vary in length, sample 16 frames)
CLIP_LEN = 16
SPATIAL_SIZE = 224
RESIZE_SHORT = 256
FEATURE_DIM = 768


# ---- Helpers ----
def load_action_to_vn():
    """Build {action_id: (verb_id, frozenset(noun_names))}."""
    a2vn = {}
    for sp in ["train", "val", "test"]:
        with open(DATA_BASE / f"{sp}_index_expanded.csv") as f:
            r = csv.DictReader(f)
            ncols = [c for c in r.fieldnames if c.startswith("noun_")]
            for row in r:
                aid = int(row["action_id"])
                if aid in a2vn: continue
                vid = int(row["verb_id"])
                nouns = frozenset(c.replace("noun_","") for c in ncols if row[c]=="1")
                a2vn[aid] = (vid, nouns)
    return a2vn


# Build noun_name <-> noun_idx mapping (alphabetical for reproducibility)
NOUN_NAMES = None
def get_noun_names():
    global NOUN_NAMES
    if NOUN_NAMES is None:
        with open(DATA_BASE / "train_index_expanded.csv") as f:
            r = csv.DictReader(f)
            ncols = [c for c in r.fieldnames if c.startswith("noun_")]
        NOUN_NAMES = sorted([c.replace("noun_","") for c in ncols])
    return NOUN_NAMES


def noun_set_to_indices(noun_set):
    names = get_noun_names()
    return [names.index(n) for n in noun_set if n in names]


# Pre-build action ↔ verb/noun lookup (for fast eval)
_ACTION_TO_VN = None
_ACTION_VERB = None      # tensor [61] of verb_id per action
_ACTION_NOUN_MASK = None # tensor [61, 23] multi-hot noun mask per action

def build_action_lookup(device):
    global _ACTION_TO_VN, _ACTION_VERB, _ACTION_NOUN_MASK
    if _ACTION_VERB is not None: return
    a2vn = load_action_to_vn()
    _ACTION_TO_VN = a2vn
    _ACTION_VERB = torch.zeros(N_ACTION, dtype=torch.long)
    _ACTION_NOUN_MASK = torch.zeros(N_ACTION, N_NOUN)
    for aid, (vid, nouns) in a2vn.items():
        _ACTION_VERB[aid] = vid
        for nidx in noun_set_to_indices(nouns):
            _ACTION_NOUN_MASK[aid, nidx] = 1.0
    _ACTION_VERB = _ACTION_VERB.to(device)
    _ACTION_NOUN_MASK = _ACTION_NOUN_MASK.to(device)


# ---- Dataset ----
class MeccanoDataset(Dataset):
    # CSVs may carry an absolute path prefix from another machine; remap via env vars
    from paths import MECCANO_CSV_OLD_PREFIX as PATH_OLD_PREFIX, MECCANO_CSV_NEW_PREFIX as PATH_NEW_PREFIX
    PATH_NEW_PREFIX = str(DATA_BASE)

    def __init__(self, split, is_train):
        self.is_train = is_train
        self.samples = []
        n_missing = 0
        with open(DATA_BASE / f"{split}_index_expanded.csv") as f:
            r = csv.DictReader(f)
            ncols = [c for c in r.fieldnames if c.startswith("noun_")]
            self.noun_cols = sorted(ncols)  # alphabetical
            for row in r:
                p = row["clip_path"]
                if p.startswith(self.PATH_OLD_PREFIX):
                    p = p.replace(self.PATH_OLD_PREFIX, self.PATH_NEW_PREFIX, 1)
                if not os.path.exists(p):
                    n_missing += 1
                    continue
                self.samples.append({
                    "clip_path": p,
                    "action_id": int(row["action_id"]),
                    "verb_id": int(row["verb_id"]),
                    "noun_vec": [int(row[c]) for c in self.noun_cols],
                })
        if n_missing > 0:
            print(f"  [warn] {split}: skipped {n_missing} missing clips")

    def __len__(self):
        return len(self.samples)

    def _decode(self, path):
        # MECCANO clip files are loaded from MECCANO_DATA (see paths.py)
        # but actual location is in the user's scratch
        c = av.open(path)
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
        c.close()
        return frames

    def _sample_frames(self, frames):
        T = len(frames)
        if T >= CLIP_LEN:
            if self.is_train:
                # Random uniform sampling
                step = T / CLIP_LEN
                offsets = np.random.uniform(0, step, CLIP_LEN)
                indices = [min(T - 1, int(i * step + offsets[i])) for i in range(CLIP_LEN)]
            else:
                # Uniform sampling (deterministic)
                indices = np.linspace(0, T - 1, CLIP_LEN, dtype=int).tolist()
        else:
            indices = np.linspace(0, T - 1, CLIP_LEN, dtype=int).tolist()
        return [frames[i] for i in indices]

    def _transform(self, frames):
        ts = [torch.from_numpy(f).permute(2,0,1).float()/255.0 for f in frames]
        clip = torch.stack(ts)  # [T, 3, H, W]
        h, w = clip.shape[2], clip.shape[3]
        scale = RESIZE_SHORT / min(h, w)
        new_h, new_w = int(h*scale), int(w*scale)
        clip = F.interpolate(clip, size=(new_h, new_w), mode="bilinear", align_corners=False)
        if self.is_train:
            i = np.random.randint(0, new_h - SPATIAL_SIZE + 1)
            j = np.random.randint(0, new_w - SPATIAL_SIZE + 1)
            clip = clip[:, :, i:i+SPATIAL_SIZE, j:j+SPATIAL_SIZE]
            if np.random.rand() > 0.5:
                clip = clip.flip(-1)
        else:
            i = (new_h - SPATIAL_SIZE) // 2
            j = (new_w - SPATIAL_SIZE) // 2
            clip = clip[:, :, i:i+SPATIAL_SIZE, j:j+SPATIAL_SIZE]
        m = torch.tensor([0.45,0.45,0.45]).view(1,3,1,1)
        s = torch.tensor([0.225,0.225,0.225]).view(1,3,1,1)
        return ((clip - m) / s).permute(1,0,2,3)  # [3, T, H, W]

    def __getitem__(self, idx):
        s = self.samples[idx]
        frames = self._decode(s["clip_path"])
        frames = self._sample_frames(frames)
        clip = self._transform(frames)
        return (clip,
                torch.tensor(s["action_id"], dtype=torch.long),
                torch.tensor(s["verb_id"], dtype=torch.long),
                torch.tensor(s["noun_vec"], dtype=torch.float32))  # multi-hot


# ---- Models ----
class MViTv2Backbone(nn.Module):
    def __init__(self, dropout=0.5, pretrained=True):
        super().__init__()
        weights = MViT_V2_S_Weights.DEFAULT if pretrained else None
        base = mvit_v2_s(weights=weights)
        self.conv_proj = base.conv_proj
        self.pos_encoding = base.pos_encoding
        self.blocks = base.blocks
        self.norm = base.norm
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.conv_proj(x); x = x.flatten(2).transpose(1, 2)
        x = self.pos_encoding(x)
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        for blk in self.blocks: x, thw = blk(x, thw)
        x = self.norm(x)
        return self.dropout(x[:, 0])  # [B, 768]


class MeccanoActionModel(nn.Module):
    """Single-head action classifier (61 classes)."""
    def __init__(self, dropout=0.5, pretrained=True):
        super().__init__()
        self.backbone = MViTv2Backbone(dropout, pretrained)
        self.head = nn.Linear(FEATURE_DIM, N_ACTION)
        nn.init.trunc_normal_(self.head.weight, std=0.001)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x):
        return self.head(self.backbone(x))


class MeccanoMultiHeadModel(nn.Module):
    """Shared backbone + verb head (12) + noun head (23 multi-label)."""
    def __init__(self, dropout=0.5, pretrained=True):
        super().__init__()
        self.backbone = MViTv2Backbone(dropout, pretrained)
        self.verb_head = nn.Linear(FEATURE_DIM, N_VERB)
        self.noun_head = nn.Linear(FEATURE_DIM, N_NOUN)
        for h in [self.verb_head, self.noun_head]:
            nn.init.trunc_normal_(h.weight, std=0.001)
            nn.init.constant_(h.bias, 0.0)

    def forward(self, x):
        feat = self.backbone(x)
        return self.verb_head(feat), self.noun_head(feat)


# ---- Action reconstruction from verb + noun (for multi-head eval) ----
@torch.no_grad()
def action_logits_from_vn(verb_logits, noun_logits, device):
    """Combine verb log-probs + noun log-probs into a 61-dim action score.
    score(action_a) = log P(verb=verb_of_a) + sum_{n in nouns_of_a} log P(noun_n=1)
                     + sum_{n NOT in nouns_of_a} log (1 - P(noun_n=1))
    """
    build_action_lookup(device)
    log_pv = F.log_softmax(verb_logits, dim=1)  # [B, N_VERB]
    pn = torch.sigmoid(noun_logits).clamp(1e-6, 1-1e-6)  # [B, N_NOUN]
    log_pn = torch.log(pn)
    log_1mpn = torch.log(1 - pn)

    # action_score[b, a] = log_pv[b, _ACTION_VERB[a]] + sum mask*log_pn + sum (1-mask)*log_1mpn
    verb_score = log_pv[:, _ACTION_VERB]  # [B, N_ACTION]
    # noun contribution: [B, N_ACTION] = log_pn @ mask.T + log_1mpn @ (1-mask).T
    noun_score = log_pn @ _ACTION_NOUN_MASK.T + log_1mpn @ (1 - _ACTION_NOUN_MASK).T
    return verb_score + noun_score  # [B, N_ACTION]


# ---- Metrics ----
@torch.no_grad()
def topk_acc(logits, target, K_list=[1, 5]):
    bs = logits.size(0)
    maxk = min(max(K_list), logits.size(1))
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.eq(target.unsqueeze(1).expand_as(pred))
    out = {}
    for k in K_list:
        kk = min(k, logits.size(1))
        out[k] = correct[:, :kk].any(dim=1).float().sum().item()
    return out, bs


@torch.no_grad()
def topk_coverage(logits, target_vec, K):
    """Multi-label coverage: all GT positives are in top-K."""
    bs = logits.size(0)
    _, top = logits.topk(min(K, logits.size(1)), dim=1)
    correct = 0
    for b in range(bs):
        gt = (target_vec[b] > 0).nonzero(as_tuple=True)[0]
        if len(gt) == 0:
            correct += 1
        else:
            ts = set(top[b].tolist())
            if all(g.item() in ts for g in gt):
                correct += 1
    return correct


# ---- Schedules ----
def build_scheduler(opt, warmup, total, mr=1/600):
    def lr_lambda(epoch):
        if epoch < warmup: return (epoch + 1) / warmup
        prog = (epoch - warmup) / (total - warmup)
        return mr + (1 - mr) * 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# ---- Training ----
def train_one_epoch(mode, model, loader, opt, sched, device, epoch, writer):
    model.train()
    total_loss, n = 0.0, 0
    a_top1, a_top5 = 0, 0
    pos_w_noun = torch.tensor([10.0]*N_NOUN).to(device)

    for clip, action, verb, noun_vec in tqdm(loader, desc=f"Train E{epoch:03d}", leave=False):
        clip = clip.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)
        verb = verb.to(device, non_blocking=True)
        noun_vec = noun_vec.to(device, non_blocking=True)

        if mode == "action":
            logits = model(clip)
            loss = F.cross_entropy(logits, action, label_smoothing=0.1)
        else:  # multihead
            verb_logits, noun_logits = model(clip)
            loss_v = F.cross_entropy(verb_logits, verb, label_smoothing=0.1)
            loss_n = F.binary_cross_entropy_with_logits(noun_logits, noun_vec, pos_weight=pos_w_noun)
            loss = loss_v + loss_n

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        bs = clip.size(0); n += bs; total_loss += loss.item() * bs

        # Compute action accuracy
        with torch.no_grad():
            if mode == "action":
                act_logits = logits
            else:
                act_logits = action_logits_from_vn(verb_logits, noun_logits, device)
            cnts, _ = topk_acc(act_logits, action, [1, 5])
            a_top1 += cnts[1]; a_top5 += cnts[5]

    sched.step()
    if writer:
        writer.add_scalar("train/loss", total_loss/n, epoch)
        writer.add_scalar("train/action_top1", a_top1/n*100, epoch)
        writer.add_scalar("train/action_top5", a_top5/n*100, epoch)
        writer.add_scalar("train/lr", opt.param_groups[0]["lr"], epoch)
    return total_loss/n, a_top1/n, a_top5/n


@torch.no_grad()
def validate(mode, model, loader, device, epoch, writer, split):
    """Returns dict with comprehensive top-K metrics."""
    model.eval()
    n = 0; total_loss = 0.0
    a_K = [1, 3, 5, 10]; v_K = [1, 3, 5]; n_K = [2, 3, 5, 10]
    a_correct = {k: 0 for k in a_K}
    v_correct = {k: 0 for k in v_K}
    n_cov = {k: 0 for k in n_K}
    pos_w = torch.tensor([10.0]*N_NOUN).to(device)
    for clip, action, verb, noun_vec in tqdm(loader, desc=f"Val({split}) E{epoch:03d}", leave=False):
        clip = clip.to(device, non_blocking=True); action = action.to(device, non_blocking=True)
        verb = verb.to(device, non_blocking=True); noun_vec = noun_vec.to(device, non_blocking=True)
        if mode == "action":
            logits = model(clip)
            loss = F.cross_entropy(logits, action)
            act_logits = logits
        else:
            verb_logits, noun_logits = model(clip)
            loss = (F.cross_entropy(verb_logits, verb)
                    + F.binary_cross_entropy_with_logits(noun_logits, noun_vec, pos_weight=pos_w))
            act_logits = action_logits_from_vn(verb_logits, noun_logits, device)
            cv, _ = topk_acc(verb_logits, verb, v_K)
            for k in v_K: v_correct[k] += cv[k]
            for k in n_K: n_cov[k] += topk_coverage(noun_logits, noun_vec, k)
        bs = clip.size(0); n += bs; total_loss += loss.item() * bs
        ca, _ = topk_acc(act_logits, action, a_K)
        for k in a_K: a_correct[k] += ca[k]

    res = {"loss": total_loss/n, "n": n,
           **{f"action_top{k}": a_correct[k]/n*100 for k in a_K}}
    if mode != "action":
        res.update({f"verb_top{k}": v_correct[k]/n*100 for k in v_K})
        res.update({f"noun_cov{k}": n_cov[k]/n*100 for k in n_K})
    if writer:
        writer.add_scalar(f"{split}/loss", total_loss/n, epoch)
        for k in a_K: writer.add_scalar(f"{split}/action_top{k}", a_correct[k]/n*100, epoch)
        if mode != "action":
            for k in v_K: writer.add_scalar(f"{split}/verb_top{k}", v_correct[k]/n*100, epoch)
            for k in n_K: writer.add_scalar(f"{split}/noun_cov{k}", n_cov[k]/n*100, epoch)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["action", "multihead"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--base_lr", type=float, default=0.0096)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--ckpt_dir", type=str, default=None)
    p.add_argument("--result_dir", type=str, default=None)
    p.add_argument("--no_pretrain", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.ckpt_dir is None:
        args.ckpt_dir = str(PROJECT_DIR / "checkpoints" / f"meccano_mvitv2_{args.mode}")
    if args.result_dir is None:
        args.result_dir = str(PROJECT_DIR / "results" / f"meccano_mvitv2_{args.mode}")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[task] MECCANO {args.mode}")
    eff_lr = args.base_lr * args.batch_size / 256
    print(f"[lr] base={args.base_lr}, batch={args.batch_size}, eff={eff_lr:.6f}")

    train_ds = MeccanoDataset("train", is_train=True)
    val_ds = MeccanoDataset("val", is_train=False)
    test_ds = MeccanoDataset("test", is_train=False)
    print(f"[data] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=True)

    if args.mode == "action":
        model = MeccanoActionModel(dropout=args.dropout, pretrained=not args.no_pretrain).to(device)
    else:
        model = MeccanoMultiHeadModel(dropout=args.dropout, pretrained=not args.no_pretrain).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] {args.mode}, {n_params:.1f}M params")

    opt = optim.AdamW(model.parameters(), lr=eff_lr, weight_decay=args.weight_decay)
    sched = build_scheduler(opt, args.warmup_epochs, args.epochs)

    start_epoch, best_val_t1 = 0, 0.0
    resume = os.path.join(args.ckpt_dir, "latest.pth")
    if os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_t1 = ckpt.get("best_val_t1", 0.0)
        print(f"[resume] epoch={ckpt['epoch']}, best_val_t1={best_val_t1*100:.2f}%")

    tb_dir = os.path.join(args.result_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr_loss, tr_t1, tr_t5 = train_one_epoch(args.mode, model, train_loader, opt, sched, device, epoch, writer)
        va = validate(args.mode, model, val_loader, device, epoch, writer, "val")

        elapsed = time.time() - t0
        extra = ""
        if args.mode != "action":
            extra = f" v_t1={va['verb_top1']:.1f}% n_c5={va['noun_cov5']:.1f}%"
        print(f"E{epoch:03d}/{args.epochs}  lr={opt.param_groups[0]['lr']:.2e}  "
              f"train: loss={tr_loss:.4f} t1={tr_t1*100:.2f}% t5={tr_t5*100:.2f}%  |  "
              f"val: loss={va['loss']:.4f} a_t1={va['action_top1']:.2f}% a_t5={va['action_top5']:.2f}%{extra}  ({elapsed:.0f}s)")

        ckpt = {"epoch": epoch, "model": model.state_dict(),
                "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "best_val_t1": best_val_t1}
        os.makedirs(args.ckpt_dir, exist_ok=True)
        torch.save(ckpt, os.path.join(args.ckpt_dir, "latest.pth"))
        if va["action_top1"]/100 > best_val_t1:
            best_val_t1 = va["action_top1"]/100
            ckpt["best_val_t1"] = best_val_t1
            torch.save(ckpt, os.path.join(args.ckpt_dir, "best.pth"))
            print(f"  [*] new best val_action_top1 = {best_val_t1*100:.2f}%")

    writer.close()

    # Final test eval (comprehensive metrics)
    print("\n=== Final test evaluation (using best.pth) ===")
    ckpt = torch.load(os.path.join(args.ckpt_dir, "best.pth"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    te = validate(args.mode, model, test_loader, device, ckpt["epoch"], None, "test")
    va = validate(args.mode, model, val_loader, device, ckpt["epoch"], None, "val_final")
    print(f"VAL  action: t1={va['action_top1']:.2f}% t3={va['action_top3']:.2f}% t5={va['action_top5']:.2f}% t10={va['action_top10']:.2f}%")
    print(f"TEST action: t1={te['action_top1']:.2f}% t3={te['action_top3']:.2f}% t5={te['action_top5']:.2f}% t10={te['action_top10']:.2f}%")
    if args.mode != "action":
        print(f"TEST verb:   t1={te['verb_top1']:.2f}% t3={te['verb_top3']:.2f}% t5={te['verb_top5']:.2f}%")
        print(f"TEST noun:   c2={te['noun_cov2']:.2f}% c3={te['noun_cov3']:.2f}% c5={te['noun_cov5']:.2f}% c10={te['noun_cov10']:.2f}%")
    with open(os.path.join(args.result_dir, "test_eval.json"), "w") as f:
        json.dump({"mode": args.mode, "best_epoch": ckpt["epoch"],
                   "best_val_top1": best_val_t1*100,
                   "val": va, "test": te}, f, indent=2)


if __name__ == "__main__":
    main()

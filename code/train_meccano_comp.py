"""
Train MViTv2-S on MECCANO COMPOSITIONAL split (Plan A: 6 unseen pairs).

Three modes (matching HAViD experiments):
  --arch shared:           1 backbone + verb head + noun head (joint train)
  --arch indep --head verb: 1 backbone + verb head only (independent verb model)
  --arch indep --head noun: 1 backbone + noun head only (independent noun model)

Trains on comp_train, validates on comp_val_seen, final eval on comp_val_unseen.
Evaluation:
  - verb head: top-1/5 accuracy (single-label)
  - noun head: top-K coverage (multi-label)
  - action:    reconstruct from (verb, noun) → 61-class action_id top-1/5
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

N_VERB = 12
N_NOUN = 23
N_ACTION = 61
from paths import MECCANO_DATA as DATA_BASE
COMP_DIR = DATA_BASE / "comp_split"
from paths import PROJECT_DIR  # see code/paths.py for layout / env-var overrides
CLIP_LEN = 16
SPATIAL_SIZE = 224
RESIZE_SHORT = 256
FEATURE_DIM = 768


# ---- Action ↔ (verb, noun_set) lookup ----
def load_action_to_vn():
    """Build {action_id: (verb_id, frozenset(noun_names))} from ALL splits."""
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


def get_noun_names():
    with open(DATA_BASE / "train_index_expanded.csv") as f:
        ncols = [c for c in csv.DictReader(f).fieldnames if c.startswith("noun_")]
    return sorted([c.replace("noun_","") for c in ncols])


_ACTION_VERB = None
_ACTION_NOUN_MASK = None

def build_action_lookup(device):
    global _ACTION_VERB, _ACTION_NOUN_MASK
    if _ACTION_VERB is not None: return
    a2vn = load_action_to_vn()
    names = get_noun_names()
    name_to_idx = {n: i for i, n in enumerate(names)}
    _ACTION_VERB = torch.zeros(N_ACTION, dtype=torch.long)
    _ACTION_NOUN_MASK = torch.zeros(N_ACTION, N_NOUN)
    for aid, (vid, nouns) in a2vn.items():
        _ACTION_VERB[aid] = vid
        for n in nouns:
            if n in name_to_idx:
                _ACTION_NOUN_MASK[aid, name_to_idx[n]] = 1.0
    _ACTION_VERB = _ACTION_VERB.to(device)
    _ACTION_NOUN_MASK = _ACTION_NOUN_MASK.to(device)


# ---- Dataset ----
class MeccanoCompDataset(Dataset):
    from paths import MECCANO_CSV_OLD_PREFIX as PATH_OLD_PREFIX, MECCANO_CSV_NEW_PREFIX as PATH_NEW_PREFIX
    PATH_NEW_PREFIX = str(DATA_BASE)

    def __init__(self, split, is_train):
        """split ∈ {comp_train, comp_val_seen, comp_val_unseen}"""
        self.is_train = is_train
        self.samples = []
        n_missing = 0
        with open(COMP_DIR / f"{split}.csv") as f:
            r = csv.DictReader(f)
            ncols = sorted([c for c in r.fieldnames if c.startswith("noun_")])
            self.noun_cols = [c.replace("noun_","") for c in ncols]  # alphabetical
            for row in r:
                noun_vec = [int(row[f"noun_{n}"]) for n in self.noun_cols]
                p = row["clip_path"]
                if p.startswith(self.PATH_OLD_PREFIX):
                    p = p.replace(self.PATH_OLD_PREFIX, self.PATH_NEW_PREFIX, 1)
                if not os.path.exists(p):
                    n_missing += 1
                    continue
                self.samples.append({
                    "clip_path": p,
                    "action_id": int(row["action_id"]),
                    "verb_id":   int(row["verb_id"]),
                    "noun_vec":  noun_vec,
                })
        if n_missing > 0:
            print(f"  [warn] {split}: skipped {n_missing} missing clips")

    def __len__(self): return len(self.samples)

    def _decode(self, path):
        c = av.open(path)
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
        c.close()
        return frames

    def _sample_frames(self, frames):
        T = len(frames)
        if T >= CLIP_LEN:
            if self.is_train:
                step = T / CLIP_LEN
                offsets = np.random.uniform(0, step, CLIP_LEN)
                indices = [min(T-1, int(i*step + offsets[i])) for i in range(CLIP_LEN)]
            else:
                indices = np.linspace(0, T-1, CLIP_LEN, dtype=int).tolist()
        else:
            indices = np.linspace(0, T-1, CLIP_LEN, dtype=int).tolist()
        return [frames[i] for i in indices]

    def _transform(self, frames):
        ts = [torch.from_numpy(f).permute(2,0,1).float()/255.0 for f in frames]
        clip = torch.stack(ts)
        h, w = clip.shape[2], clip.shape[3]
        scale = RESIZE_SHORT / min(h, w)
        nh, nw = int(h*scale), int(w*scale)
        clip = F.interpolate(clip, size=(nh, nw), mode="bilinear", align_corners=False)
        if self.is_train:
            i = np.random.randint(0, nh - SPATIAL_SIZE + 1)
            j = np.random.randint(0, nw - SPATIAL_SIZE + 1)
            clip = clip[:, :, i:i+SPATIAL_SIZE, j:j+SPATIAL_SIZE]
            if np.random.rand() > 0.5: clip = clip.flip(-1)
        else:
            i = (nh - SPATIAL_SIZE) // 2
            j = (nw - SPATIAL_SIZE) // 2
            clip = clip[:, :, i:i+SPATIAL_SIZE, j:j+SPATIAL_SIZE]
        m = torch.tensor([0.45,0.45,0.45]).view(1,3,1,1)
        s = torch.tensor([0.225,0.225,0.225]).view(1,3,1,1)
        return ((clip - m) / s).permute(1,0,2,3)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frames = self._decode(s["clip_path"])
        frames = self._sample_frames(frames)
        clip = self._transform(frames)
        return (clip,
                torch.tensor(s["action_id"], dtype=torch.long),
                torch.tensor(s["verb_id"],   dtype=torch.long),
                torch.tensor(s["noun_vec"],  dtype=torch.float32))


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
        return self.dropout(self.norm(x)[:, 0])


class SharedModel(nn.Module):
    """1 backbone + verb head + noun head."""
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


class IndepHeadModel(nn.Module):
    """1 backbone + 1 head (verb OR noun)."""
    def __init__(self, head_type, dropout=0.5, pretrained=True):
        super().__init__()
        self.head_type = head_type  # "verb" or "noun"
        self.backbone = MViTv2Backbone(dropout, pretrained)
        n_out = N_VERB if head_type == "verb" else N_NOUN
        self.head = nn.Linear(FEATURE_DIM, n_out)
        nn.init.trunc_normal_(self.head.weight, std=0.001)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x):
        return self.head(self.backbone(x))


# ---- Action reconstruction & metrics ----
@torch.no_grad()
def action_logits_from_vn(verb_logits, noun_logits, device):
    build_action_lookup(device)
    log_pv = F.log_softmax(verb_logits, dim=1)
    pn = torch.sigmoid(noun_logits).clamp(1e-6, 1-1e-6)
    log_pn = torch.log(pn); log_1mpn = torch.log(1 - pn)
    verb_score = log_pv[:, _ACTION_VERB]
    noun_score = log_pn @ _ACTION_NOUN_MASK.T + log_1mpn @ (1 - _ACTION_NOUN_MASK).T
    return verb_score + noun_score


@torch.no_grad()
def topk_acc(logits, target, K_list=[1, 5]):
    maxk = min(max(K_list), logits.size(1))
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.eq(target.unsqueeze(1).expand_as(pred))
    out = {}
    for k in K_list:
        kk = min(k, logits.size(1))
        out[k] = correct[:, :kk].any(dim=1).float().sum().item()
    return out


@torch.no_grad()
def topk_coverage(logits, target_vec, K):
    K_eff = min(K, logits.size(1))
    _, top = logits.topk(K_eff, dim=1)
    correct = 0
    for b in range(logits.size(0)):
        gt = (target_vec[b] > 0).nonzero(as_tuple=True)[0]
        if len(gt) == 0:
            correct += 1
        else:
            ts = set(top[b].tolist())
            if all(g.item() in ts for g in gt): correct += 1
    return correct


# ---- Training loops ----
def build_scheduler(opt, warmup, total, mr=1/600):
    def f(e):
        if e < warmup: return (e+1)/warmup
        prog = (e-warmup)/(total-warmup)
        return mr + (1-mr) * 0.5 * (1 + math.cos(math.pi*prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def train_shared(model, loader, opt, sched, device, epoch, writer, pos_w):
    model.train()
    total_loss, n = 0.0, 0
    a_t1 = a_t5 = v_t1 = nc3 = nc5 = 0
    for clip, action, verb, noun_vec in tqdm(loader, desc=f"Train E{epoch:03d}", leave=False):
        clip, action, verb, noun_vec = (clip.to(device), action.to(device),
                                         verb.to(device), noun_vec.to(device))
        v_lg, n_lg = model(clip)
        loss = (F.cross_entropy(v_lg, verb, label_smoothing=0.1)
                + F.binary_cross_entropy_with_logits(n_lg, noun_vec, pos_weight=pos_w))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0); opt.step()
        bs = clip.size(0); n += bs; total_loss += loss.item()*bs
        with torch.no_grad():
            a_lg = action_logits_from_vn(v_lg, n_lg, device)
            ac = topk_acc(a_lg, action); a_t1 += ac[1]; a_t5 += ac[5]
            vc = topk_acc(v_lg, verb); v_t1 += vc[1]
            nc3 += topk_coverage(n_lg, noun_vec, 3)
            nc5 += topk_coverage(n_lg, noun_vec, 5)
    sched.step()
    if writer:
        writer.add_scalar("train/loss", total_loss/n, epoch)
        writer.add_scalar("train/action_top1", a_t1/n*100, epoch)
        writer.add_scalar("train/verb_top1", v_t1/n*100, epoch)
        writer.add_scalar("train/noun_cov3", nc3/n*100, epoch)
        writer.add_scalar("train/lr", opt.param_groups[0]["lr"], epoch)
    return total_loss/n, a_t1/n, a_t5/n


def train_indep(model, loader, opt, sched, device, epoch, writer, pos_w=None):
    model.train()
    total_loss, n = 0.0, 0
    t1, t5 = 0, 0
    for clip, action, verb, noun_vec in tqdm(loader, desc=f"Train E{epoch:03d}", leave=False):
        clip = clip.to(device)
        if model.head_type == "verb":
            target = verb.to(device)
            logits = model(clip)
            loss = F.cross_entropy(logits, target, label_smoothing=0.1)
        else:
            target = noun_vec.to(device)
            logits = model(clip)
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_w)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0); opt.step()
        bs = clip.size(0); n += bs; total_loss += loss.item()*bs
        with torch.no_grad():
            if model.head_type == "verb":
                ac = topk_acc(logits, target); t1 += ac[1]; t5 += ac[5]
            else:
                t1 += topk_coverage(logits, target, 3)  # cov3 as "t1" placeholder
                t5 += topk_coverage(logits, target, 5)
    sched.step()
    if writer:
        writer.add_scalar("train/loss", total_loss/n, epoch)
        writer.add_scalar(f"train/{model.head_type}_metric_K3", t1/n*100, epoch)
        writer.add_scalar(f"train/{model.head_type}_metric_K5", t5/n*100, epoch)
        writer.add_scalar("train/lr", opt.param_groups[0]["lr"], epoch)
    return total_loss/n, t1/n, t5/n


A_K = [1, 3, 5, 10]
V_K = [1, 3, 5]
N_K = [2, 3, 5, 10]


@torch.no_grad()
def validate_shared(model, loader, device, pos_w):
    model.eval()
    total_loss, n = 0.0, 0
    a_c = {k: 0 for k in A_K}; v_c = {k: 0 for k in V_K}; n_cov = {k: 0 for k in N_K}
    for clip, action, verb, noun_vec in tqdm(loader, desc="Val", leave=False):
        clip, action, verb, noun_vec = (clip.to(device), action.to(device),
                                         verb.to(device), noun_vec.to(device))
        v_lg, n_lg = model(clip)
        loss = (F.cross_entropy(v_lg, verb)
                + F.binary_cross_entropy_with_logits(n_lg, noun_vec, pos_weight=pos_w))
        bs = clip.size(0); n += bs; total_loss += loss.item()*bs
        a_lg = action_logits_from_vn(v_lg, n_lg, device)
        ca = topk_acc(a_lg, action, A_K); cv = topk_acc(v_lg, verb, V_K)
        for k in A_K: a_c[k] += ca[k]
        for k in V_K: v_c[k] += cv[k]
        for k in N_K: n_cov[k] += topk_coverage(n_lg, noun_vec, k)
    return {"loss": total_loss/n, "n": n,
            **{f"action_top{k}": a_c[k]/n*100 for k in A_K},
            **{f"verb_top{k}": v_c[k]/n*100 for k in V_K},
            **{f"noun_cov{k}": n_cov[k]/n*100 for k in N_K}}


@torch.no_grad()
def validate_indep(model, loader, device, pos_w=None):
    model.eval()
    total_loss, n = 0.0, 0
    if model.head_type == "verb":
        v_c = {k: 0 for k in V_K}
        for clip, _, verb, _ in tqdm(loader, desc="Val", leave=False):
            clip = clip.to(device); verb = verb.to(device)
            logits = model(clip)
            loss = F.cross_entropy(logits, verb)
            bs = clip.size(0); n += bs; total_loss += loss.item()*bs
            cv = topk_acc(logits, verb, V_K)
            for k in V_K: v_c[k] += cv[k]
        return {"loss": total_loss/n, "n": n,
                **{f"verb_top{k}": v_c[k]/n*100 for k in V_K}}
    else:
        n_cov = {k: 0 for k in N_K}
        for clip, _, _, noun_vec in tqdm(loader, desc="Val", leave=False):
            clip = clip.to(device); noun_vec = noun_vec.to(device)
            logits = model(clip)
            loss = F.binary_cross_entropy_with_logits(logits, noun_vec, pos_weight=pos_w)
            bs = clip.size(0); n += bs; total_loss += loss.item()*bs
            for k in N_K: n_cov[k] += topk_coverage(logits, noun_vec, k)
        return {"loss": total_loss/n, "n": n,
                **{f"noun_cov{k}": n_cov[k]/n*100 for k in N_K}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=["shared", "indep"])
    p.add_argument("--head", default=None, choices=["verb", "noun"], help="for indep only")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--base_lr", type=float, default=0.0096)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--no_pretrain", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag_suffix", type=str, default="",
                   help="Append to ckpt/result dir name; for multi-seed noise-floor runs.")
    args = p.parse_args()

    if args.arch == "indep" and args.head is None:
        raise ValueError("--head required when --arch=indep")

    name = args.arch if args.arch == "shared" else f"indep_{args.head}"
    ckpt_dir = PROJECT_DIR / "checkpoints" / f"meccano_comp_{name}{args.tag_suffix}"
    result_dir = PROJECT_DIR / "results" / f"meccano_comp_{name}{args.tag_suffix}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[task] MECCANO comp {name}")
    eff_lr = args.base_lr * args.batch_size / 256
    print(f"[lr] eff={eff_lr:.6f}")

    train_ds = MeccanoCompDataset("comp_train",      is_train=True)
    seen_ds  = MeccanoCompDataset("comp_val_seen",   is_train=False)
    unsn_ds  = MeccanoCompDataset("comp_val_unseen", is_train=False)
    print(f"[data] train={len(train_ds)}, val_seen={len(seen_ds)}, val_unseen={len(unsn_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=True)
    seen_loader = DataLoader(seen_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=True)
    unsn_loader = DataLoader(unsn_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=True)

    pos_w = torch.tensor([10.0]*N_NOUN).to(device)

    if args.arch == "shared":
        model = SharedModel(dropout=args.dropout, pretrained=not args.no_pretrain).to(device)
    else:
        model = IndepHeadModel(args.head, dropout=args.dropout,
                               pretrained=not args.no_pretrain).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] {n_params:.1f}M params")

    opt = optim.AdamW(model.parameters(), lr=eff_lr, weight_decay=args.weight_decay)
    sched = build_scheduler(opt, args.warmup_epochs, args.epochs)

    start_epoch, best = 0, 0.0
    resume = ckpt_dir / "latest.pth"
    if resume.exists():
        c = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["optimizer"])
        sched.load_state_dict(c["scheduler"]); start_epoch = c["epoch"]+1
        best = c.get("best", 0.0)
        print(f"[resume] epoch={c['epoch']}, best={best:.4f}")

    writer = SummaryWriter(log_dir=str(result_dir / "tensorboard"))

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        if args.arch == "shared":
            tr_loss, tr_t1, tr_t5 = train_shared(model, train_loader, opt, sched, device, epoch, writer, pos_w)
            va = validate_shared(model, seen_loader, device, pos_w)
            metric_for_best = va["action_top1"]  # use action top-1 to select best
        else:
            tr_loss, tr_t1, tr_t5 = train_indep(model, train_loader, opt, sched, device, epoch, writer, pos_w)
            va = validate_indep(model, seen_loader, device, pos_w)
            metric_for_best = va["verb_top1"] if args.head == "verb" else va["noun_cov3"]

        elapsed = time.time() - t0
        if args.arch == "shared":
            print(f"E{epoch:03d}/{args.epochs}  tr_loss={tr_loss:.4f}  val: action_t1={va['action_top1']:.2f}% "
                  f"verb_t1={va['verb_top1']:.2f}% noun_c3={va['noun_cov3']:.2f}%  ({elapsed:.0f}s)")
        else:
            metric_str = (f"verb_t1={va['verb_top1']:.2f}% verb_t5={va['verb_top5']:.2f}%" if args.head=="verb"
                          else f"noun_c3={va['noun_cov3']:.2f}% noun_c5={va['noun_cov5']:.2f}%")
            print(f"E{epoch:03d}/{args.epochs}  tr_loss={tr_loss:.4f}  val: {metric_str}  ({elapsed:.0f}s)")

        c = {"epoch": epoch, "model": model.state_dict(),
             "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
             "best": best}
        torch.save(c, ckpt_dir / "latest.pth")
        if metric_for_best > best:
            best = metric_for_best
            c["best"] = best
            torch.save(c, ckpt_dir / "best.pth")
            print(f"  [*] new best = {best:.2f}")

    writer.close()

    # Final unseen evaluation
    print("\n=== Final eval on val_seen and val_unseen (using best.pth) ===")
    c = torch.load(ckpt_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(c["model"])

    if args.arch == "shared":
        seen_res = validate_shared(model, seen_loader, device, pos_w)
        unsn_res = validate_shared(model, unsn_loader, device, pos_w)
        print(f"SEEN:   action_t1={seen_res['action_top1']:.2f}%  t5={seen_res['action_top5']:.2f}%  "
              f"verb_t1={seen_res['verb_top1']:.2f}%  noun_c3={seen_res['noun_cov3']:.2f}%")
        print(f"UNSEEN: action_t1={unsn_res['action_top1']:.2f}%  t5={unsn_res['action_top5']:.2f}%  "
              f"verb_t1={unsn_res['verb_top1']:.2f}%  noun_c3={unsn_res['noun_cov3']:.2f}%")
    else:
        seen_res = validate_indep(model, seen_loader, device, pos_w)
        unsn_res = validate_indep(model, unsn_loader, device, pos_w)
        if args.head == "verb":
            print(f"SEEN:   verb_t1={seen_res['verb_top1']:.2f}%  t5={seen_res['verb_top5']:.2f}%")
            print(f"UNSEEN: verb_t1={unsn_res['verb_top1']:.2f}%  t5={unsn_res['verb_top5']:.2f}%")
        else:
            print(f"SEEN:   noun_c3={seen_res['noun_cov3']:.2f}%  c5={seen_res['noun_cov5']:.2f}%  c10={seen_res['noun_cov10']:.2f}%")
            print(f"UNSEEN: noun_c3={unsn_res['noun_cov3']:.2f}%  c5={unsn_res['noun_cov5']:.2f}%  c10={unsn_res['noun_cov10']:.2f}%")

    with open(result_dir / "final_eval.json", "w") as f:
        json.dump({"arch": args.arch, "head": args.head,
                   "best_val_seen_metric": best,
                   "val_seen": seen_res, "val_unseen": unsn_res}, f, indent=2)


if __name__ == "__main__":
    main()

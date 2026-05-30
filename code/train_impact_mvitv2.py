"""
Train MViTv2-S on IMPACT TAS-B for one (view, hand) combination.

Uses splits_v1 (recording-level, hand-stratified, view-agnostic).
Two modes:
  --mode action     : single 138-class action head (CE)
  --mode multihead  : verb (22-cls CE) + noun (19-cls CE) heads with shared backbone
                      action top-K reconstructed from (v, n) outputs

Usage:
  python train_impact_mvitv2.py --view front --hand left --mode action
  python train_impact_mvitv2.py --view front --hand left --mode multihead
"""

import argparse, os, time, math, json
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
N_VERB = 22
N_NOUN = 19
N_ACTION = 138  # ids 0..137 (0 = null)
from paths import IMPACT_DATA as DATA_BASE
SPLITS_DIR = DATA_BASE / "annotations/TAS-B/splits_v1"
from paths import PROJECT_DIR  # see code/paths.py for layout / env-var overrides

CLIP_LEN = 16
SPATIAL_SIZE = 224
RESIZE_SHORT = 256
FEATURE_DIM = 768

VIEWS = ["front", "left", "right", "top", "ego"]
HANDS = ["left", "right"]


# ---- (verb, noun) ↔ action lookup ----
def load_action_to_vn():
    """Return dict {action_id: (verb_id, noun_id)} from any annotation."""
    a2vn = {}
    ann_dir = DATA_BASE / "annotations/TAS-B/front"
    for f in ann_dir.glob("*.json"):
        with open(f) as fp: ann = json.load(fp)
        for s in ann["segments"]:
            v, n, a = s["verb"], s["noun"], s["action_label"]
            if v >= 0 and n >= 0 and a > 0 and a not in a2vn:
                a2vn[a] = (v, n)
    return a2vn


_ACTION_VERB = None
_ACTION_NOUN = None
_VALID_ACTIONS_MASK = None

def build_action_lookup(device):
    global _ACTION_VERB, _ACTION_NOUN, _VALID_ACTIONS_MASK
    if _ACTION_VERB is not None: return
    a2vn = load_action_to_vn()
    _ACTION_VERB = torch.full((N_ACTION,), -1, dtype=torch.long)
    _ACTION_NOUN = torch.full((N_ACTION,), -1, dtype=torch.long)
    _VALID_ACTIONS_MASK = torch.zeros(N_ACTION, dtype=torch.bool)
    for aid, (vid, nid) in a2vn.items():
        _ACTION_VERB[aid] = vid
        _ACTION_NOUN[aid] = nid
        _VALID_ACTIONS_MASK[aid] = True
    _ACTION_VERB = _ACTION_VERB.to(device)
    _ACTION_NOUN = _ACTION_NOUN.to(device)
    _VALID_ACTIONS_MASK = _VALID_ACTIONS_MASK.to(device)


@torch.no_grad()
def action_logits_from_vn(verb_logits, noun_logits, device):
    """score(action_a) = log P(verb=v_a) + log P(noun=n_a). Invalid actions get -inf."""
    build_action_lookup(device)
    log_pv = F.log_softmax(verb_logits, dim=1)
    log_pn = F.log_softmax(noun_logits, dim=1)
    bs = verb_logits.size(0)
    # Initialize with -inf
    scores = torch.full((bs, N_ACTION), float("-inf"), device=device)
    valid_idx = _VALID_ACTIONS_MASK.nonzero(as_tuple=True)[0]
    v_ids = _ACTION_VERB[valid_idx]; n_ids = _ACTION_NOUN[valid_idx]
    scores[:, valid_idx] = log_pv[:, v_ids] + log_pn[:, n_ids]
    return scores


# ---- Dataset ----
class IMPACTDataset(Dataset):
    """Loads (view, hand, split) from splits_v1.
    Each sample: (video_path, start_frame, end_frame, action, verb, noun).
    Skips null segments (verb=-1 or action=0)."""

    def __init__(self, view, hand, split, is_train,
                 skip_null=True):
        self.is_train = is_train
        self.view = view
        self.samples = []

        split_file = SPLITS_DIR / hand / f"{split}.txt"
        rec_ids = [l.strip() for l in open(split_file) if l.strip()]

        ann_dir = DATA_BASE / f"annotations/TAS-B/{view}"
        video_dir = DATA_BASE / f"videos/{view}"

        n_dropped_null = 0
        n_dropped_missing = 0
        for rid in rec_ids:
            ann_file = ann_dir / f"{rid}_{view}.json"
            video_file = video_dir / f"{rid}_{view}.mp4"
            if not ann_file.exists() or not video_file.exists():
                n_dropped_missing += 1
                continue
            with open(ann_file) as fp: ann = json.load(fp)
            fps = ann["meta_data"]["fps"]
            for s in ann["segments"]:
                if s["entity"] != hand: continue
                if skip_null and (s["verb"] < 0 or s["action_label"] == 0):
                    n_dropped_null += 1
                    continue
                self.samples.append({
                    "video": str(video_file),
                    "start": s["start_frame"],
                    "end":   s["end_frame"],
                    "action": s["action_label"],
                    "verb":   s["verb"],
                    "noun":   s["noun"],
                    "fps":    fps,
                })
        print(f"  [{view}/{hand}/{split}] {len(self.samples)} segments  "
              f"(dropped {n_dropped_null} null, {n_dropped_missing} missing files)")

    def __len__(self): return len(self.samples)

    def _decode_segment(self, path, start_f, end_f, fps):
        """Use PyAV seek+decode to grab only frames in [start_f, end_f]."""
        container = av.open(path)
        stream = container.streams.video[0]
        # Seek backward to nearest keyframe
        try:
            ts = int(start_f / fps / stream.time_base)
            container.seek(ts, stream=stream, backward=True)
        except Exception:
            container.seek(0)

        frames = []
        cur_idx = 0
        for frame in container.decode(stream):
            # cur_idx based on frame.pts (pts in stream.time_base units)
            if frame.pts is not None:
                cur_idx = int(frame.pts * stream.time_base * fps)
            if cur_idx > end_f: break
            if cur_idx >= start_f:
                frames.append(frame.to_ndarray(format="rgb24"))
            cur_idx += 1  # fallback if pts None
        container.close()
        return frames

    def _sample_frames(self, frames):
        T = len(frames)
        if T == 0:
            # Should not happen, but defensive
            raise RuntimeError("Empty segment after decode")
        if T >= CLIP_LEN:
            if self.is_train:
                step = T / CLIP_LEN
                offsets = np.random.uniform(0, step, CLIP_LEN)
                indices = [min(T-1, int(i*step + offsets[i])) for i in range(CLIP_LEN)]
            else:
                indices = np.linspace(0, T-1, CLIP_LEN, dtype=int).tolist()
        else:
            # Pad by repeating
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
        try:
            frames = self._decode_segment(s["video"], s["start"], s["end"], s["fps"])
            if not frames:
                # Try whole video as fallback
                container = av.open(s["video"])
                frames = [f.to_ndarray(format="rgb24") for f in container.decode(container.streams.video[0])]
                container.close()
                # Trim to segment
                frames = frames[s["start"]:s["end"]+1] if s["end"]+1 <= len(frames) else frames[s["start"]:]
            frames = self._sample_frames(frames)
            clip = self._transform(frames)
        except Exception as e:
            # Fallback: zero clip
            print(f"[warn] decode error {s['video']} [{s['start']}-{s['end']}]: {e}")
            clip = torch.zeros(3, CLIP_LEN, SPATIAL_SIZE, SPATIAL_SIZE)
        return (clip,
                torch.tensor(s["action"], dtype=torch.long),
                torch.tensor(s["verb"], dtype=torch.long),
                torch.tensor(s["noun"], dtype=torch.long))


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


class IMPACTActionModel(nn.Module):
    def __init__(self, dropout=0.5, pretrained=True):
        super().__init__()
        self.backbone = MViTv2Backbone(dropout, pretrained)
        self.head = nn.Linear(FEATURE_DIM, N_ACTION)
        nn.init.trunc_normal_(self.head.weight, std=0.001)
        nn.init.constant_(self.head.bias, 0.0)
    def forward(self, x): return self.head(self.backbone(x))


class IMPACTMultiHeadModel(nn.Module):
    def __init__(self, dropout=0.5, pretrained=True):
        super().__init__()
        self.backbone = MViTv2Backbone(dropout, pretrained)
        self.verb_head = nn.Linear(FEATURE_DIM, N_VERB)
        self.noun_head = nn.Linear(FEATURE_DIM, N_NOUN)
        for h in [self.verb_head, self.noun_head]:
            nn.init.trunc_normal_(h.weight, std=0.001)
            nn.init.constant_(h.bias, 0.0)
    def forward(self, x):
        f = self.backbone(x)
        return self.verb_head(f), self.noun_head(f)


# ---- Metrics ----
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


def build_scheduler(opt, warmup, total, mr=1/600):
    def f(e):
        if e < warmup: return (e+1)/warmup
        prog = (e-warmup)/(total-warmup)
        return mr + (1-mr) * 0.5 * (1 + math.cos(math.pi*prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


# ---- Training ----
def train_one_epoch(mode, model, loader, opt, sched, device, epoch, writer):
    model.train()
    n = 0; total_loss = 0.0
    a_correct = 0
    for clip, action, verb, noun in tqdm(loader, desc=f"Train E{epoch:03d}", leave=False):
        clip = clip.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)
        verb = verb.to(device, non_blocking=True)
        noun = noun.to(device, non_blocking=True)
        if mode == "action":
            logits = model(clip)
            loss = F.cross_entropy(logits, action, label_smoothing=0.1)
            act_logits = logits
        else:
            v_lg, n_lg = model(clip)
            loss = (F.cross_entropy(v_lg, verb, label_smoothing=0.1)
                    + F.cross_entropy(n_lg, noun, label_smoothing=0.1))
            with torch.no_grad():
                act_logits = action_logits_from_vn(v_lg, n_lg, device)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        bs = clip.size(0); n += bs; total_loss += loss.item() * bs
        with torch.no_grad():
            ac = topk_acc(act_logits, action, [1])
            a_correct += ac[1]
    sched.step()
    if writer:
        writer.add_scalar("train/loss", total_loss/n, epoch)
        writer.add_scalar("train/action_top1", a_correct/n*100, epoch)
        writer.add_scalar("train/lr", opt.param_groups[0]["lr"], epoch)
    return total_loss/n, a_correct/n


@torch.no_grad()
def validate(mode, model, loader, device, epoch, writer, split):
    model.eval()
    n = 0; total_loss = 0.0
    a_K = [1, 3, 5, 10]; v_K = [1, 3, 5]; n_K = [1, 3, 5]
    a_c = {k: 0 for k in a_K}; v_c = {k: 0 for k in v_K}; n_c = {k: 0 for k in n_K}
    for clip, action, verb, noun in tqdm(loader, desc=f"Val({split}) E{epoch:03d}", leave=False):
        clip = clip.to(device); action = action.to(device)
        verb = verb.to(device); noun = noun.to(device)
        if mode == "action":
            logits = model(clip)
            loss = F.cross_entropy(logits, action)
            act_logits = logits
        else:
            v_lg, n_lg = model(clip)
            loss = F.cross_entropy(v_lg, verb) + F.cross_entropy(n_lg, noun)
            act_logits = action_logits_from_vn(v_lg, n_lg, device)
            cv = topk_acc(v_lg, verb, v_K)
            cn = topk_acc(n_lg, noun, n_K)
            for k in v_K: v_c[k] += cv[k]
            for k in n_K: n_c[k] += cn[k]
        bs = clip.size(0); n += bs; total_loss += loss.item() * bs
        ca = topk_acc(act_logits, action, a_K)
        for k in a_K: a_c[k] += ca[k]

    res = {"loss": total_loss/n, "n": n,
           **{f"action_top{k}": a_c[k]/n*100 for k in a_K}}
    if mode != "action":
        res.update({f"verb_top{k}": v_c[k]/n*100 for k in v_K})
        res.update({f"noun_top{k}": n_c[k]/n*100 for k in n_K})
    if writer:
        writer.add_scalar(f"{split}/loss", total_loss/n, epoch)
        for k in a_K: writer.add_scalar(f"{split}/action_top{k}", a_c[k]/n*100, epoch)
        if mode != "action":
            for k in v_K: writer.add_scalar(f"{split}/verb_top{k}", v_c[k]/n*100, epoch)
            for k in n_K: writer.add_scalar(f"{split}/noun_top{k}", n_c[k]/n*100, epoch)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--view", required=True, choices=VIEWS)
    p.add_argument("--hand", required=True, choices=HANDS)
    p.add_argument("--mode", required=True, choices=["action", "multihead"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--base_lr", type=float, default=0.0096)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--no_pretrain", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    name = f"impact_{args.view}_{args.hand}_{args.mode}"
    ckpt_dir = PROJECT_DIR / "checkpoints" / name
    result_dir = PROJECT_DIR / "results" / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[task] {name}")
    eff_lr = args.base_lr * args.batch_size / 256
    print(f"[lr] eff={eff_lr:.6f}")

    train_ds = IMPACTDataset(args.view, args.hand, "train", is_train=True)
    val_ds   = IMPACTDataset(args.view, args.hand, "val",   is_train=False)
    test_ds  = IMPACTDataset(args.view, args.hand, "test",  is_train=False)

    # NOTE: persistent_workers=False prevents host-memory accumulation from
    # PyAV seek-decode across epochs (which previously caused OOM on
    # left-hand × static-view IMPACT tasks); ~30s/epoch perf cost is acceptable.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=False,
                              prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=False, prefetch_factor=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=False, prefetch_factor=2)

    if args.mode == "action":
        model = IMPACTActionModel(args.dropout, not args.no_pretrain).to(device)
    else:
        model = IMPACTMultiHeadModel(args.dropout, not args.no_pretrain).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] {args.mode}, {n_params:.1f}M params")

    opt = optim.AdamW(model.parameters(), lr=eff_lr, weight_decay=args.weight_decay)
    sched = build_scheduler(opt, args.warmup_epochs, args.epochs)

    start_epoch, best_v = 0, 0.0
    resume = ckpt_dir / "latest.pth"
    if resume.exists():
        c = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["optimizer"])
        sched.load_state_dict(c["scheduler"])
        start_epoch = c["epoch"] + 1
        best_v = c.get("best_val_t1", 0.0)
        print(f"[resume] epoch={c['epoch']} best={best_v:.4f}")

    writer = SummaryWriter(log_dir=str(result_dir / "tensorboard"))

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr_loss, tr_t1 = train_one_epoch(args.mode, model, train_loader, opt, sched, device, epoch, writer)
        va = validate(args.mode, model, val_loader, device, epoch, writer, "val")
        elapsed = time.time() - t0
        extra = ""
        if args.mode != "action":
            extra = f" v_t1={va['verb_top1']:.1f}% n_t1={va['noun_top1']:.1f}%"
        print(f"E{epoch:03d}/{args.epochs}  lr={opt.param_groups[0]['lr']:.2e}  "
              f"tr_loss={tr_loss:.4f} tr_a_t1={tr_t1*100:.2f}%  |  "
              f"va: a_t1={va['action_top1']:.2f}% a_t5={va['action_top5']:.2f}%{extra}  "
              f"({elapsed:.0f}s)")

        c = {"epoch": epoch, "model": model.state_dict(),
             "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
             "best_val_t1": best_v}
        torch.save(c, ckpt_dir / "latest.pth")
        if va["action_top1"]/100 > best_v:
            best_v = va["action_top1"]/100
            c["best_val_t1"] = best_v
            torch.save(c, ckpt_dir / "best.pth")
            print(f"  [*] new best val_action_top1 = {best_v*100:.2f}%")

    writer.close()

    # Final test eval
    print("\n=== Final test (using best.pth) ===")
    c = torch.load(ckpt_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(c["model"])
    va = validate(args.mode, model, val_loader, device, c["epoch"], None, "val_final")
    te = validate(args.mode, model, test_loader, device, c["epoch"], None, "test")
    print(f"VAL  action: t1={va['action_top1']:.2f}% t3={va['action_top3']:.2f}% "
          f"t5={va['action_top5']:.2f}% t10={va['action_top10']:.2f}%")
    print(f"TEST action: t1={te['action_top1']:.2f}% t3={te['action_top3']:.2f}% "
          f"t5={te['action_top5']:.2f}% t10={te['action_top10']:.2f}%")
    if args.mode != "action":
        print(f"TEST verb:   t1={te['verb_top1']:.2f}% t3={te['verb_top3']:.2f}% t5={te['verb_top5']:.2f}%")
        print(f"TEST noun:   t1={te['noun_top1']:.2f}% t3={te['noun_top3']:.2f}% t5={te['noun_top5']:.2f}%")

    with open(result_dir / "test_eval.json", "w") as f:
        json.dump({"view": args.view, "hand": args.hand, "mode": args.mode,
                   "best_epoch": c["epoch"], "best_val_top1": best_v*100,
                   "val": va, "test": te}, f, indent=2)


if __name__ == "__main__":
    main()

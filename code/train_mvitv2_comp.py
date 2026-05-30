"""
MViTv2-S multi-head training on HAViD — Compositional Split.

Uses the compositional split files:
  {subtask}_comp_train.txt       — seen combos only (training)
  {subtask}_comp_val_seen.txt    — seen combos (validation / checkpoint selection)

Checkpoints and results are saved to:
  checkpoints/mvitv2_comp_{subtask}/
  results/mvitv2_comp_{subtask}/

4 classification heads:
  - verb:       11 classes
  - manip_obj:  42 classes
  - target_obj: 42 classes
  - tool:        5 classes
"""

import argparse
import os
import time
import math
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

import av
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEAD_NAMES   = ["verb", "manip_obj", "target_obj", "tool"]
HEAD_CLASSES = [11, 42, 42, 5]

from paths import HAVID_DATA as DATA_BASE
from paths import PROJECT_DIR  # see code/paths.py for layout / env-var overrides

CLIP_LEN      = 16
FRAME_STRIDE  = 4
SPATIAL_SIZE  = 224
RESIZE_SHORT  = 256
FEATURE_DIM   = 768   # MViTv2-S CLS-token output dim


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def code_to_label(code: str) -> int:
    return code.index("1")


class HAViDDataset(Dataset):
    """
    HAViD dataset loader.
    video_roots: list of Path — tried in order until the video file is found.
    """
    def __init__(self, ann_file: Path, video_roots: list, is_train: bool):
        self.video_roots = [Path(r) for r in video_roots]
        self.is_train    = is_train
        self.samples     = self._parse_ann(ann_file)

    def _resolve_video(self, rel_path: str) -> str:
        for root in self.video_roots:
            p = root / rel_path
            if p.exists():
                return str(p)
        raise FileNotFoundError(
            f"Video not found in any root: {rel_path}\n"
            f"  Searched: {[str(r) for r in self.video_roots]}"
        )

    def _parse_ann(self, ann_file: Path):
        samples = []
        with open(ann_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                rel_path = parts[0]
                labels = [
                    code_to_label(parts[6]),   # verb
                    code_to_label(parts[7]),   # manip_obj
                    code_to_label(parts[8]),   # target_obj
                    code_to_label(parts[9]),   # tool
                ]
                samples.append((rel_path, labels))
        return samples

    def _load_frames(self, video_path: str):
        container = av.open(video_path)
        stream = container.streams.video[0]
        raw = []
        for frame in container.decode(stream):
            raw.append(frame.to_ndarray(format="rgb24"))  # (H, W, 3) uint8
        container.close()

        total = len(raw)
        needed = CLIP_LEN * FRAME_STRIDE

        if total >= needed:
            if self.is_train:
                start = np.random.randint(0, total - needed + 1)
            else:
                start = (total - needed) // 2
            indices = [start + i * FRAME_STRIDE for i in range(CLIP_LEN)]
        else:
            indices = [min(i * FRAME_STRIDE, total - 1) for i in range(CLIP_LEN)]

        selected = np.stack([raw[i] for i in indices])     # (T, H, W, 3)
        frames = torch.from_numpy(selected).float() / 255.0
        frames = frames.permute(3, 0, 1, 2)               # (C, T, H, W)
        return frames

    def _train_transform(self, frames):
        C, Tv, H, W = frames.shape
        short = np.random.randint(256, 321)
        scale = short / min(H, W)
        new_H, new_W = int(H * scale), int(W * scale)
        frames = frames.reshape(C * Tv, H, W)
        frames = torch.stack([
            TF.resize(frames[i:i+1], [new_H, new_W],
                      interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
            for i in range(C * Tv)
        ])
        frames = frames.reshape(C, Tv, new_H, new_W)

        i = np.random.randint(0, new_H - SPATIAL_SIZE + 1)
        j = np.random.randint(0, new_W - SPATIAL_SIZE + 1)
        frames = frames[:, :, i:i+SPATIAL_SIZE, j:j+SPATIAL_SIZE]

        if np.random.rand() < 0.5:
            frames = torch.flip(frames, dims=[3])

        frames = frames.reshape(C * Tv, SPATIAL_SIZE, SPATIAL_SIZE)
        jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        frames = torch.stack([
            jitter(frames[i:i+1].reshape(1, SPATIAL_SIZE, SPATIAL_SIZE)).squeeze(0)
            for i in range(C * Tv)
        ])
        frames = frames.reshape(C, Tv, SPATIAL_SIZE, SPATIAL_SIZE)

        mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        std  = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
        return (frames - mean) / std

    def _val_transform(self, frames):
        C, Tv, H, W = frames.shape
        scale = RESIZE_SHORT / min(H, W)
        new_H, new_W = int(H * scale), int(W * scale)
        frames = frames.reshape(C * Tv, H, W)
        frames = torch.stack([
            TF.resize(frames[i:i+1], [new_H, new_W],
                      interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
            for i in range(C * Tv)
        ])
        frames = frames.reshape(C, Tv, new_H, new_W)

        i = (new_H - SPATIAL_SIZE) // 2
        j = (new_W - SPATIAL_SIZE) // 2
        frames = frames[:, :, i:i+SPATIAL_SIZE, j:j+SPATIAL_SIZE]

        mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        std  = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
        return (frames - mean) / std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, labels = self.samples[idx]
        video_path = self._resolve_video(rel_path)
        frames = self._load_frames(video_path)
        frames = self._train_transform(frames) if self.is_train else self._val_transform(frames)
        return frames, torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MViTv2MultiHead(nn.Module):
    def __init__(self, num_classes_per_head, dropout=0.5, pretrained=True):
        super().__init__()
        weights = MViT_V2_S_Weights.DEFAULT if pretrained else None
        base = mvit_v2_s(weights=weights)

        self.conv_proj    = base.conv_proj
        self.pos_encoding = base.pos_encoding
        self.blocks       = base.blocks
        self.norm         = base.norm

        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.heads = nn.ModuleList([
            nn.Linear(FEATURE_DIM, n) for n in num_classes_per_head
        ])
        for head in self.heads:
            nn.init.trunc_normal_(head.weight, std=0.001)
            nn.init.constant_(head.bias, 0.0)

    def forward(self, x):
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_encoding(x)
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        for block in self.blocks:
            x, thw = block(x, thw)
        x = self.norm(x)
        feat = self.dropout(x[:, 0])
        return [head(feat) for head in self.heads]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def topk_correct(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.unsqueeze(0).expand_as(pred))
        return {k: correct[:k].any(dim=0) for k in topk}


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
def build_scheduler(optimizer, warmup_epochs, total_epochs, base_lr, min_lr_ratio=1/600):
    min_lr = base_lr * min_lr_ratio

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Train / val one epoch
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, writer):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    total_loss = 0.0
    head_top1  = [0.0] * 4
    head_top5  = [0.0] * 4
    joint_top1 = 0.0
    n_samples  = 0

    pbar = tqdm(loader, desc=f"Train E{epoch:03d}", leave=False, dynamic_ncols=True)
    for step, (frames, labels) in enumerate(pbar):
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(frames)
        loss = sum(criterion(outputs[i], labels[:, i]) for i in range(4))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = frames.size(0)
        total_loss += loss.item() * bs
        n_samples  += bs

        all_correct_1 = torch.ones(bs, dtype=torch.bool, device=device)
        for i in range(4):
            top_k = (1, min(5, HEAD_CLASSES[i]))
            corr = topk_correct(outputs[i].detach(), labels[:, i], topk=top_k)
            head_top1[i] += corr[1].float().sum().item()
            head_top5[i] += corr[top_k[1]].float().sum().item()
            all_correct_1 &= corr[1]
        joint_top1 += all_correct_1.float().sum().item()

        if step % 50 == 0:
            pbar.set_postfix(loss=f"{loss.item():.3f}")

    scheduler.step()

    avg_loss  = total_loss / n_samples
    avg_top1  = [x / n_samples for x in head_top1]
    avg_top5  = [x / n_samples for x in head_top5]
    avg_joint1 = joint_top1 / n_samples

    if writer is not None:
        writer.add_scalar("train/loss", avg_loss, epoch)
        for i, name in enumerate(HEAD_NAMES):
            writer.add_scalar(f"train/{name}_top1", avg_top1[i] * 100, epoch)
            writer.add_scalar(f"train/{name}_top5", avg_top5[i] * 100, epoch)
        writer.add_scalar("train/joint_top1", avg_joint1 * 100, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

    return avg_loss, avg_top1, avg_top5, avg_joint1


@torch.no_grad()
def validate(model, loader, device, epoch, writer, split="val_seen"):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    head_top1  = [0.0] * 4
    head_top5  = [0.0] * 4
    joint_top1 = 0.0
    n_samples  = 0

    pbar = tqdm(loader, desc=f"Val({split}) E{epoch:03d}", leave=False, dynamic_ncols=True)
    for frames, labels in pbar:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(frames)
        loss = sum(criterion(outputs[i], labels[:, i]) for i in range(4))
        bs = frames.size(0)
        total_loss += loss.item() * bs
        n_samples  += bs

        all_correct_1 = torch.ones(bs, dtype=torch.bool, device=device)
        for i in range(4):
            top_k = (1, min(5, HEAD_CLASSES[i]))
            corr = topk_correct(outputs[i], labels[:, i], topk=top_k)
            head_top1[i] += corr[1].float().sum().item()
            head_top5[i] += corr[top_k[1]].float().sum().item()
            all_correct_1 &= corr[1]
        joint_top1 += all_correct_1.float().sum().item()

    avg_loss   = total_loss / n_samples
    avg_top1   = [x / n_samples for x in head_top1]
    avg_top5   = [x / n_samples for x in head_top5]
    avg_joint1 = joint_top1 / n_samples

    if writer is not None:
        writer.add_scalar(f"{split}/loss", avg_loss, epoch)
        for i, name in enumerate(HEAD_NAMES):
            writer.add_scalar(f"{split}/{name}_top1", avg_top1[i] * 100, epoch)
            writer.add_scalar(f"{split}/{name}_top5", avg_top5[i] * 100, epoch)
        writer.add_scalar(f"{split}/joint_top1", avg_joint1 * 100, epoch)

    return avg_loss, avg_top1, avg_top5, avg_joint1


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(state, ckpt_dir, filename):
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(state, os.path.join(ckpt_dir, filename))


def load_checkpoint(model, optimizer, scheduler, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch   = ckpt["epoch"] + 1
    best_joint_t1 = ckpt.get("best_joint_t1", 0.0)
    print(f"[resume] epoch={ckpt['epoch']}, best_joint_t1={best_joint_t1:.4f}")
    return start_epoch, best_joint_t1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subtask",      type=str,   required=True)
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--num_workers",  type=int,   default=8)
    p.add_argument("--base_lr",      type=float, default=0.0096)
    p.add_argument("--warmup_epochs", type=int,  default=20)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--dropout",      type=float, default=0.5)
    p.add_argument("--ckpt_dir",     type=str,   default=None)
    p.add_argument("--result_dir",   type=str,   default=None)
    p.add_argument("--resume",       type=str,   default=None)
    p.add_argument("--no_pretrain",  action="store_true")
    p.add_argument("--seed",         type=int,   default=42)
    args = p.parse_args()

    if args.ckpt_dir is None:
        args.ckpt_dir = str(PROJECT_DIR / "checkpoints" / f"mvitv2_comp_{args.subtask}")
    if args.result_dir is None:
        args.result_dir = str(PROJECT_DIR / "results" / f"mvitv2_comp_{args.subtask}")

    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subtask = args.subtask
    print(f"[task]   {subtask} (compositional split)")
    print(f"[device] {device}")

    effective_lr = args.base_lr * args.batch_size / 256
    print(f"[lr]     base={args.base_lr}, batch={args.batch_size}, eff={effective_lr:.6f}")

    # ------------------------------------------------------------------
    # Annotation files (compositional split)
    # ------------------------------------------------------------------
    dataset_root = DATA_BASE / subtask
    train_ann    = dataset_root / f"{subtask}_comp_train.txt"
    val_ann      = dataset_root / f"{subtask}_comp_val_seen.txt"

    train_vid = dataset_root / "train" / "videos_train"
    val_vid   = dataset_root / "val"   / "videos_val"

    # ------------------------------------------------------------------
    # Datasets & loaders
    # ------------------------------------------------------------------
    train_ds = HAViDDataset(train_ann, [train_vid], is_train=True)
    val_ds   = HAViDDataset(val_ann,   [val_vid],   is_train=False)
    print(f"[data]   comp_train={len(train_ds)}, comp_val_seen={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = MViTv2MultiHead(HEAD_CLASSES, dropout=args.dropout,
                            pretrained=not args.no_pretrain)
    model = model.to(device)
    print(f"[model]  MViTv2-S + 4 heads, pretrained={not args.no_pretrain}")

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(), lr=effective_lr,
        betas=(0.9, 0.999), weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer, args.warmup_epochs, args.epochs, effective_lr,
    )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch   = 0
    best_joint_t1 = 0.0

    resume_path = args.resume
    if resume_path is None:
        auto = os.path.join(args.ckpt_dir, "latest.pth")
        if os.path.exists(auto):
            resume_path = auto

    if resume_path and os.path.exists(resume_path):
        start_epoch, best_joint_t1 = load_checkpoint(
            model, optimizer, scheduler, resume_path, device
        )

    # ------------------------------------------------------------------
    # TensorBoard
    # ------------------------------------------------------------------
    tb_dir = os.path.join(args.result_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    result_log = []

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        tr_loss, tr_top1, tr_top5, tr_j1 = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, writer
        )
        va_loss, va_top1, va_top5, va_j1 = validate(
            model, val_loader, device, epoch, writer, split="val_seen"
        )

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"tr_loss={tr_loss:.4f}  va_loss={va_loss:.4f}  "
            f"time={elapsed:.0f}s"
        )
        for i, name in enumerate(HEAD_NAMES):
            print(
                f"  {name:12s}  "
                f"tr_top1={tr_top1[i]*100:.2f}% tr_top5={tr_top5[i]*100:.2f}%  |  "
                f"va_top1={va_top1[i]*100:.2f}% va_top5={va_top5[i]*100:.2f}%"
            )
        print(
            f"  {'joint':12s}  "
            f"tr_top1={tr_j1*100:.2f}%  |  va_top1={va_j1*100:.2f}%"
        )

        # Checkpoint
        ckpt_state = {
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "best_joint_t1": best_joint_t1,
        }
        save_checkpoint(ckpt_state, args.ckpt_dir, "latest.pth")

        if va_j1 > best_joint_t1:
            best_joint_t1 = va_j1
            ckpt_state["best_joint_t1"] = best_joint_t1
            save_checkpoint(ckpt_state, args.ckpt_dir, "best.pth")
            print(f"  [*] new best joint_val_seen_top1 = {best_joint_t1*100:.2f}%")

        result_log.append({
            "epoch": epoch,
            "train": {
                "loss": tr_loss,
                **{f"{n}_top1": tr_top1[i] for i, n in enumerate(HEAD_NAMES)},
                **{f"{n}_top5": tr_top5[i] for i, n in enumerate(HEAD_NAMES)},
                "joint_top1": tr_j1,
            },
            "val_seen": {
                "loss": va_loss,
                **{f"{n}_top1": va_top1[i] for i, n in enumerate(HEAD_NAMES)},
                **{f"{n}_top5": va_top5[i] for i, n in enumerate(HEAD_NAMES)},
                "joint_top1": va_j1,
            },
        })

    # ------------------------------------------------------------------
    # Save training log
    # ------------------------------------------------------------------
    os.makedirs(args.result_dir, exist_ok=True)
    result_path = os.path.join(args.result_dir, "train_log.json")
    with open(result_path, "w") as f:
        json.dump(result_log, f, indent=2)
    print(f"\n[done] training log → {result_path}")
    writer.close()

    # Print best epoch summary
    best_ckpt_path = os.path.join(args.ckpt_dir, "best.pth")
    if os.path.exists(best_ckpt_path):
        best_ckpt  = torch.load(best_ckpt_path, map_location="cpu")
        best_epoch = best_ckpt["epoch"]
        best_entry = next((e for e in result_log if e["epoch"] == best_epoch), None)
        if best_entry:
            print(f"\n=== Best model (epoch {best_epoch}) — val_seen ===")
            for name in HEAD_NAMES:
                print(f"  {name:12s}  top1={best_entry['val_seen'][f'{name}_top1']*100:.2f}%")
            print(f"  {'joint':12s}  top1={best_entry['val_seen']['joint_top1']*100:.2f}%")


if __name__ == "__main__":
    main()

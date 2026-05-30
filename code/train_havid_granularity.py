"""
HAViD granularity sweep for RQ4 (and the verb-only / manip-only solo models for RQ1 D1).

Scheme 3: the ACTION LABEL is fixed to the (verb, manip) pair throughout. The
configurations differ only in how many heads predict it / supervise the encoder:

  --config k1          atomic classifier over observed (verb,manip) composites
  --config k2          verb head + manip head            (this is also the D1 'joint')
  --config k3          verb + manip + target(aux) heads
  --config k4          verb + manip + target(aux) + tool(aux) heads
  --config solo_verb   verb head only                    (D1 'solo' for verb)
  --config solo_manip  manip head only                   (D1 'solo' for manip)

All configs are trained on {subtask}_comp_train.txt and evaluated EACH epoch on
{subtask}_comp_val_seen.txt and {subtask}_comp_val_unseen.txt. The metric for
k1..k4 is (verb,manip) joint top-1; for solo_* it is the single-primitive top-1.

Does not touch any existing script/checkpoint. Outputs:
  checkpoints/havid_gran_{config}_{subtask}/{best,latest}.pth
  results/havid_gran_{config}_{subtask}/comp_eval.json

Usage:
  python train_havid_granularity.py --subtask view0_lh_aa --config k2
"""

import argparse, os, time, json, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from train_mvitv2_comp import (
    HAViDDataset, MViTv2MultiHead, build_scheduler,
    DATA_BASE, PROJECT_DIR, HEAD_CLASSES,
)

N_VERB, N_MANIP, N_TARGET, N_TOOL = HEAD_CLASSES   # 11, 42, 42, 5

CONFIGS = ["k1", "k2", "k3", "k4", "solo_verb", "solo_manip"]


def build_composite_map(ann_file):
    """Map every (verb,manip) pair occurring in comp_train to a composite class id."""
    pairs = {}
    with open(ann_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            # one-hot codes: parts[6]=verb, parts[7]=manip
            v = parts[6].index("1") if "1" in parts[6] else -1
            m = parts[7].index("1") if "1" in parts[7] else -1
            if v >= 0 and m >= 0 and (v, m) not in pairs:
                pairs[(v, m)] = len(pairs)
    return pairs


def heads_for_config(config, n_composite):
    """Return (num_classes_per_head list, list of label-index each head supervises).
    label-index: 0=verb 1=manip 2=target 3=tool; -1 = composite."""
    if config == "k1":
        return [n_composite], [-1]
    if config == "k2":
        return [N_VERB, N_MANIP], [0, 1]
    if config == "k3":
        return [N_VERB, N_MANIP, N_TARGET], [0, 1, 2]
    if config == "k4":
        return [N_VERB, N_MANIP, N_TARGET, N_TOOL], [0, 1, 2, 3]
    if config == "solo_verb":
        return [N_VERB], [0]
    if config == "solo_manip":
        return [N_MANIP], [1]
    raise ValueError(config)


def compute_labels(labels4, head_idx, comp_map, device):
    """labels4: (B,4) tensor [v,m,t,tool]. Returns list of target tensors per head."""
    targets = []
    for hi in head_idx:
        if hi == -1:  # composite
            v = labels4[:, 0].tolist(); m = labels4[:, 1].tolist()
            comp = [comp_map.get((vv, mm), -1) for vv, mm in zip(v, m)]
            targets.append(torch.tensor(comp, dtype=torch.long, device=device))
        else:
            targets.append(labels4[:, hi].to(device))
    return targets


@torch.no_grad()
def evaluate(model, loader, device, config, head_idx, comp_map):
    model.eval()
    n = 0
    # joint = (verb,manip) correctness for k1..k4; single-primitive for solo_*
    joint_correct = 0
    head_correct = {hi: 0 for hi in head_idx}
    for frames, labels4 in tqdm(loader, desc="eval", leave=False):
        frames = frames.to(device, non_blocking=True)
        bs = frames.size(0); n += bs
        outs = model(frames)                       # list of head logits
        targets = compute_labels(labels4, head_idx, comp_map, device)
        preds = [o.argmax(1) for o in outs]
        for j, hi in enumerate(head_idx):
            head_correct[hi] += int((preds[j] == targets[j]).sum())
        # joint (verb,manip) top-1
        if config == "k1":
            joint_correct += int((preds[0] == targets[0]).sum())   # composite == (v,m)
        elif config in ("k2", "k3", "k4"):
            v_ok = (preds[0] == targets[0])
            m_ok = (preds[1] == targets[1])
            joint_correct += int((v_ok & m_ok).sum())
        else:  # solo_*
            joint_correct += int((preds[0] == targets[0]).sum())
    res = {"n": n, "joint_top1": joint_correct / n * 100}
    for hi in head_idx:
        nm = {-1: "composite", 0: "verb", 1: "manip", 2: "target", 3: "tool"}[hi]
        res[f"{nm}_top1"] = head_correct[hi] / n * 100
    return res


def train_one_epoch(model, loader, opt, sched, device, head_idx, comp_map, epoch):
    model.train()
    crit = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-1)
    n = 0; total_loss = 0.0
    for frames, labels4 in tqdm(loader, desc=f"Train E{epoch:03d}", leave=False):
        frames = frames.to(device, non_blocking=True)
        outs = model(frames)
        targets = compute_labels(labels4, head_idx, comp_map, device)
        loss = sum(crit(outs[j], targets[j]) for j in range(len(head_idx)))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        bs = frames.size(0); n += bs; total_loss += loss.item() * bs
    sched.step()
    return total_loss / n


def hm(a, b):
    return 2 * a * b / (a + b) if (a + b) > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subtask", required=True)
    p.add_argument("--config", required=True, choices=CONFIGS)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--base_lr", type=float, default=0.0096)
    p.add_argument("--warmup_epochs", type=int, default=20)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name = f"havid_gran_{args.config}_{args.subtask}"
    ckpt_dir = PROJECT_DIR / "checkpoints" / name
    result_dir = PROJECT_DIR / "results" / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    eff_lr = args.base_lr * args.batch_size / 256
    print(f"[task] {name}  eff_lr={eff_lr:.6f}")

    dr = DATA_BASE / args.subtask
    train_ann  = dr / f"{args.subtask}_comp_train.txt"
    seen_ann   = dr / f"{args.subtask}_comp_val_seen.txt"
    unseen_ann = dr / f"{args.subtask}_comp_val_unseen.txt"
    train_vid = dr / "train" / "videos_train"
    val_vid   = dr / "val"   / "videos_val"

    comp_map = build_composite_map(train_ann)
    n_composite = len(comp_map)
    print(f"[comp] {n_composite} observed (verb,manip) composites in comp_train")

    head_classes, head_idx = heads_for_config(args.config, n_composite)
    print(f"[config] {args.config}: heads={head_classes} supervising idx={head_idx}")

    train_ds  = HAViDDataset(train_ann,  [train_vid, val_vid], is_train=True)
    seen_ds   = HAViDDataset(seen_ann,   [train_vid, val_vid], is_train=False)
    unseen_ds = HAViDDataset(unseen_ann, [train_vid, val_vid], is_train=False)
    print(f"[data] comp_train={len(train_ds)} seen={len(seen_ds)} unseen={len(unseen_ds)}")

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.num_workers, pin_memory=True,
                                   drop_last=sh, persistent_workers=True)
    train_loader = dl(train_ds, True)
    seen_loader = dl(seen_ds, False)
    unseen_loader = dl(unseen_ds, False)

    model = MViTv2MultiHead(head_classes, dropout=args.dropout, pretrained=True).to(device)
    opt = optim.AdamW(model.parameters(), lr=eff_lr, betas=(0.9, 0.999),
                      weight_decay=args.weight_decay)
    sched = build_scheduler(opt, args.warmup_epochs, args.epochs, eff_lr)

    # selection metric: HM for k2/3/4; seen joint for k1; primitive top1 for solo_*
    start_epoch, best_sel = 0, 0.0
    resume = ckpt_dir / "latest.pth"
    if resume.exists():
        c = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["optimizer"])
        sched.load_state_dict(c["scheduler"]); start_epoch = c["epoch"] + 1
        best_sel = c.get("best_sel", 0.0)
        print(f"[resume] epoch={c['epoch']} best_sel={best_sel:.4f}")

    writer = SummaryWriter(log_dir=str(result_dir / "tensorboard"))

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, opt, sched, device,
                                  head_idx, comp_map, epoch)
        seen = evaluate(model, seen_loader, device, args.config, head_idx, comp_map)
        unseen = evaluate(model, unseen_loader, device, args.config, head_idx, comp_map)
        cur_hm = hm(seen["joint_top1"], unseen["joint_top1"])
        if args.config == "k1":
            sel = seen["joint_top1"]
        elif args.config in ("k2", "k3", "k4"):
            sel = cur_hm
        else:
            sel = seen["joint_top1"]   # solo: single-primitive seen acc
        elapsed = time.time() - t0
        print(f"E{epoch:03d}/{args.epochs} tr_loss={tr_loss:.4f}  "
              f"seen_j={seen['joint_top1']:.2f}% unseen_j={unseen['joint_top1']:.2f}% "
              f"HM={cur_hm:.2f}  ({elapsed:.0f}s)")
        writer.add_scalar("seen/joint_top1", seen["joint_top1"], epoch)
        writer.add_scalar("unseen/joint_top1", unseen["joint_top1"], epoch)
        writer.add_scalar("unseen/HM", cur_hm, epoch)

        c = {"epoch": epoch, "model": model.state_dict(),
             "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
             "best_sel": best_sel}
        torch.save(c, ckpt_dir / "latest.pth")
        if sel > best_sel:
            best_sel = sel
            c["best_sel"] = best_sel
            torch.save(c, ckpt_dir / "best.pth")
            print(f"  [*] new best selection metric = {best_sel:.2f}")

    writer.close()

    # Final eval with best checkpoint
    print("\n=== Final eval (best.pth) ===")
    c = torch.load(ckpt_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(c["model"])
    seen = evaluate(model, seen_loader, device, args.config, head_idx, comp_map)
    unseen = evaluate(model, unseen_loader, device, args.config, head_idx, comp_map)
    final_hm = hm(seen["joint_top1"], unseen["joint_top1"])
    print(f"SEEN  : {seen}")
    print(f"UNSEEN: {unseen}")
    print(f"HM(joint) = {final_hm:.2f}")

    with open(result_dir / "comp_eval.json", "w") as f:
        json.dump({"subtask": args.subtask, "config": args.config,
                   "n_composite": n_composite, "best_epoch": c["epoch"],
                   "seen": seen, "unseen": unseen, "HM_joint": final_hm}, f, indent=2)
    print(f"[done] -> {result_dir/'comp_eval.json'}")


if __name__ == "__main__":
    main()

"""
IMPACT compositional-split VARIANTS (atomic-comp baseline + D1 solo models).

Complements train_impact_comp.py (which trains the multi-head 'joint' model).
This script trains the non-multihead configurations on the IMPACT comp split:

  --mode action      atomic head over the SEEN (verb,noun) action classes only.
                     Unseen actions have no output slot -> unseen top-1 is
                     structurally 0 (demonstrates the atomic ceiling).
  --mode verb_only   verb head trained alone        (RQ1 D1 'solo' for verb)
  --mode noun_only   noun head trained alone        (RQ1 D1 'solo' for noun)

Reuses IMPACTCompDataset (comp_split_v1) and MViTv2Backbone. Does not touch
train_impact_comp.py or its checkpoints.

Outputs:
  checkpoints/impact_compvar_{mode}_{view}_{hand}/{best,latest}.pth
  results/impact_compvar_{mode}_{view}_{hand}/comp_eval.json
"""

import argparse, time, json, sys
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
from train_impact_mvitv2 import (
    MViTv2Backbone, topk_acc, build_scheduler,
    N_VERB, N_NOUN, FEATURE_DIM, VIEWS, HANDS,
)
from train_impact_comp import IMPACTCompDataset, COMP_DIR, PROJECT_DIR, hm


class SingleHeadModel(nn.Module):
    """MViTv2-S backbone + one linear head of `n_cls` classes."""
    def __init__(self, n_cls, dropout=0.5, pretrained=True):
        super().__init__()
        self.backbone = MViTv2Backbone(dropout, pretrained)
        self.head = nn.Linear(FEATURE_DIM, n_cls)
        nn.init.trunc_normal_(self.head.weight, std=0.001)
        nn.init.constant_(self.head.bias, 0.0)
    def forward(self, x):
        return self.head(self.backbone(x))


def build_seen_action_map(hand):
    """Map each (verb,noun) action appearing in comp_train to a dense class id."""
    amap = {}
    with open(COMP_DIR / hand / "comp_train.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, _, _, v, n, a = line.split(",")
            key = int(a)
            if key not in amap:
                amap[key] = len(amap)
    return amap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--view", required=True, choices=VIEWS)
    p.add_argument("--hand", required=True, choices=HANDS)
    p.add_argument("--mode", required=True, choices=["action", "verb_only", "noun_only"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--base_lr", type=float, default=0.0096)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name = f"impact_compvar_{args.mode}_{args.view}_{args.hand}"
    ckpt_dir = PROJECT_DIR / "checkpoints" / name
    result_dir = PROJECT_DIR / "results" / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    eff_lr = args.base_lr * args.batch_size / 256
    print(f"[task] {name}  eff_lr={eff_lr:.6f}")

    # seen-action map (only needed for atomic mode)
    seen_amap = build_seen_action_map(args.hand) if args.mode == "action" else None
    n_cls = {"action": len(seen_amap) if seen_amap else 0,
             "verb_only": N_VERB, "noun_only": N_NOUN}[args.mode]
    if args.mode == "action":
        print(f"[atomic] {n_cls} seen action classes; unseen actions have no slot")

    train_ds  = IMPACTCompDataset(args.view, args.hand, "comp_train", is_train=True)
    seen_ds   = IMPACTCompDataset(args.view, args.hand, "comp_val_seen", is_train=False)
    unseen_ds = IMPACTCompDataset(args.view, args.hand, "comp_val_unseen", is_train=False)

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.num_workers, pin_memory=True,
                                   drop_last=sh, persistent_workers=False, prefetch_factor=2)
    train_loader = dl(train_ds, True)
    seen_loader = dl(seen_ds, False)
    unseen_loader = dl(unseen_ds, False)

    model = SingleHeadModel(n_cls, args.dropout, pretrained=True).to(device)
    opt = optim.AdamW(model.parameters(), lr=eff_lr, weight_decay=args.weight_decay)
    sched = build_scheduler(opt, args.warmup_epochs, args.epochs)

    def target_of(action, verb, noun):
        """Map a batch to the head's target tensor; -1 if no slot (atomic unseen)."""
        if args.mode == "verb_only":
            return verb
        if args.mode == "noun_only":
            return noun
        # atomic action: dense id of seen action, -1 if unseen
        ids = [seen_amap.get(int(a), -1) for a in action.tolist()]
        return torch.tensor(ids, dtype=torch.long, device=action.device)

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        n = 0; c1 = 0
        for clip, action, verb, noun in tqdm(loader, desc="eval", leave=False):
            clip = clip.to(device); action = action.to(device)
            verb = verb.to(device); noun = noun.to(device)
            logits = model(clip)
            tgt = target_of(action, verb, noun)
            pred = logits.argmax(1)
            c1 += int((pred == tgt).sum())   # tgt=-1 (atomic unseen) never matches
            n += clip.size(0)
        return c1 / n * 100

    # Robust resume: latest.pth then best.pth fallback (corruption insurance).
    start_epoch, best_sel = 0, 0.0
    for cand in ["latest.pth", "best.pth"]:
        cpath = ckpt_dir / cand
        if not cpath.exists():
            continue
        try:
            c = torch.load(cpath, map_location=device, weights_only=False)
            model.load_state_dict(c["model"]); opt.load_state_dict(c["optimizer"])
            sched.load_state_dict(c["scheduler"]); start_epoch = c["epoch"] + 1
            best_sel = c.get("best_sel", 0.0)
            print(f"[resume] from {cand}: epoch={c['epoch']} best_sel={best_sel:.4f}")
            break
        except Exception as e:
            print(f"[resume] {cand} unreadable ({e}); trying fallback")

    writer = SummaryWriter(log_dir=str(result_dir / "tensorboard"))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-1)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        n = 0; tot = 0.0
        for clip, action, verb, noun in tqdm(train_loader, desc=f"E{epoch:03d}", leave=False):
            clip = clip.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            verb = verb.to(device, non_blocking=True)
            noun = noun.to(device, non_blocking=True)
            logits = model(clip)
            loss = crit(logits, target_of(action, verb, noun))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            bs = clip.size(0); n += bs; tot += loss.item() * bs
        sched.step()
        seen_acc = evaluate(seen_loader)
        unseen_acc = evaluate(unseen_loader)
        sel = hm(seen_acc, unseen_acc) if args.mode != "action" else seen_acc
        elapsed = time.time() - t0
        print(f"E{epoch:03d}/{args.epochs} loss={tot/n:.4f}  "
              f"seen={seen_acc:.2f}% unseen={unseen_acc:.2f}%  ({elapsed:.0f}s)")
        writer.add_scalar("seen/top1", seen_acc, epoch)
        writer.add_scalar("unseen/top1", unseen_acc, epoch)

        c = {"epoch": epoch, "model": model.state_dict(),
             "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
             "best_sel": best_sel}
        # Atomic save: tmp + rename.
        tmp = ckpt_dir / "latest.pth.tmp"
        torch.save(c, tmp); tmp.replace(ckpt_dir / "latest.pth")
        if sel > best_sel:
            best_sel = sel
            c["best_sel"] = best_sel
            tmp = ckpt_dir / "best.pth.tmp"
            torch.save(c, tmp); tmp.replace(ckpt_dir / "best.pth")
            print(f"  [*] new best = {best_sel:.2f}")

    writer.close()
    c = torch.load(ckpt_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(c["model"])
    seen_acc = evaluate(seen_loader)
    unseen_acc = evaluate(unseen_loader)
    print(f"\n=== Final (best.pth) === seen={seen_acc:.2f}% unseen={unseen_acc:.2f}% "
          f"HM={hm(seen_acc, unseen_acc):.2f}")
    with open(result_dir / "comp_eval.json", "w") as f:
        json.dump({"view": args.view, "hand": args.hand, "mode": args.mode,
                   "n_cls": n_cls, "best_epoch": c["epoch"],
                   "seen_top1": seen_acc, "unseen_top1": unseen_acc,
                   "HM": hm(seen_acc, unseen_acc)}, f, indent=2)
    print(f"[done] -> {result_dir/'comp_eval.json'}")


if __name__ == "__main__":
    main()

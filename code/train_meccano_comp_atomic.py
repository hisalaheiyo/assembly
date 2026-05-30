"""
MECCANO atomic-comp baseline (RQ4 k=1 cross-check + anchor table).

Trains a single atomic head over the SEEN action classes of the MECCANO
compositional split. Unseen actions have no output slot, so unseen top-1 is
structurally 0 -- the concrete demonstration of the atomic ceiling on MECCANO.

Reuses MeccanoCompDataset / MViTv2Backbone / build_scheduler from
train_meccano_comp.py. Does not touch existing scripts or checkpoints.

Output: checkpoints/meccano_comp_atomic/{best,latest}.pth
        results/meccano_comp_atomic/comp_eval.json
"""

import argparse, csv, time, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from train_meccano_comp import (
    MeccanoCompDataset, MViTv2Backbone, build_scheduler,
    COMP_DIR, PROJECT_DIR, FEATURE_DIM,
)


class AtomicHeadModel(nn.Module):
    def __init__(self, n_cls, dropout=0.5, pretrained=True):
        super().__init__()
        self.backbone = MViTv2Backbone(dropout, pretrained)
        self.head = nn.Linear(FEATURE_DIM, n_cls)
        nn.init.trunc_normal_(self.head.weight, std=0.001)
        nn.init.constant_(self.head.bias, 0.0)
    def forward(self, x):
        return self.head(self.backbone(x))


def build_seen_action_map():
    """action_id -> dense class id, over actions present in comp_train.csv."""
    amap = {}
    with open(COMP_DIR / "comp_train.csv") as f:
        for row in csv.DictReader(f):
            a = int(row["action_id"])
            if a not in amap:
                amap[a] = len(amap)
    return amap


def hm(a, b):
    return 2 * a * b / (a + b) if (a + b) > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--base_lr", type=float, default=0.0096)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = PROJECT_DIR / "checkpoints" / "meccano_comp_atomic"
    result_dir = PROJECT_DIR / "results" / "meccano_comp_atomic"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    eff_lr = args.base_lr * args.batch_size / 256

    amap = build_seen_action_map()
    n_cls = len(amap)
    print(f"[atomic] {n_cls} seen action classes in MECCANO comp_train")

    train_ds = MeccanoCompDataset("comp_train",      is_train=True)
    seen_ds  = MeccanoCompDataset("comp_val_seen",   is_train=False)
    unsn_ds  = MeccanoCompDataset("comp_val_unseen", is_train=False)
    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.num_workers, pin_memory=True,
                                   drop_last=sh, persistent_workers=True)
    train_loader = dl(train_ds, True)
    seen_loader = dl(seen_ds, False)
    unsn_loader = dl(unsn_ds, False)

    model = AtomicHeadModel(n_cls, args.dropout, pretrained=True).to(device)
    opt = optim.AdamW(model.parameters(), lr=eff_lr, weight_decay=args.weight_decay)
    sched = build_scheduler(opt, args.warmup_epochs, args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-1)

    def remap(action):
        ids = [amap.get(int(a), -1) for a in action.tolist()]
        return torch.tensor(ids, dtype=torch.long, device=action.device)

    @torch.no_grad()
    def evaluate(loader):
        model.eval(); n = 0; c1 = 0
        for clip, action, verb, noun_vec in tqdm(loader, desc="eval", leave=False):
            clip = clip.to(device); action = action.to(device)
            pred = model(clip).argmax(1)
            tgt = remap(action)
            c1 += int((pred == tgt).sum()); n += clip.size(0)
        return c1 / n * 100

    start_epoch, best_seen = 0, 0.0
    resume = ckpt_dir / "latest.pth"
    if resume.exists():
        c = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["optimizer"])
        sched.load_state_dict(c["scheduler"]); start_epoch = c["epoch"] + 1
        best_seen = c.get("best_seen", 0.0)
        print(f"[resume] epoch={c['epoch']} best_seen={best_seen:.4f}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train(); n = 0; tot = 0.0
        for clip, action, verb, noun_vec in tqdm(train_loader, desc=f"E{epoch:03d}", leave=False):
            clip = clip.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            loss = crit(model(clip), remap(action))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            bs = clip.size(0); n += bs; tot += loss.item() * bs
        sched.step()
        seen_acc = evaluate(seen_loader)
        unseen_acc = evaluate(unsn_loader)
        print(f"E{epoch:03d}/{args.epochs} loss={tot/n:.4f}  "
              f"seen={seen_acc:.2f}% unseen={unseen_acc:.2f}%  ({time.time()-t0:.0f}s)")
        c = {"epoch": epoch, "model": model.state_dict(),
             "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
             "best_seen": best_seen}
        torch.save(c, ckpt_dir / "latest.pth")
        if seen_acc > best_seen:
            best_seen = seen_acc
            c["best_seen"] = best_seen
            torch.save(c, ckpt_dir / "best.pth")
            print(f"  [*] new best seen = {best_seen:.2f}")

    c = torch.load(ckpt_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(c["model"])
    seen_acc = evaluate(seen_loader)
    unseen_acc = evaluate(unsn_loader)
    print(f"\n=== Final === seen={seen_acc:.2f}% unseen={unseen_acc:.2f}% HM={hm(seen_acc,unseen_acc):.2f}")
    with open(result_dir / "comp_eval.json", "w") as f:
        json.dump({"mode": "atomic", "n_cls": n_cls, "best_epoch": c["epoch"],
                   "seen_top1": seen_acc, "unseen_top1": unseen_acc,
                   "HM": hm(seen_acc, unseen_acc)}, f, indent=2)
    print(f"[done] -> {result_dir/'comp_eval.json'}")


if __name__ == "__main__":
    main()

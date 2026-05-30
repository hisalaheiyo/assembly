"""
Train MViTv2-S on the IMPACT COMPOSITIONAL split for one (view, hand).

Uses comp_split_v1 (segment-level, ~15% of (verb,noun) pairs held out).
Multi-head verb+noun model; action top-K reconstructed from (v,n) via the
bijective lookup. Reports seen / unseen / HM each epoch.

Does NOT touch the standard-split scripts or their checkpoints/results.
Checkpoints -> checkpoints/impact_comp_{view}_{hand}/
Results     -> results/impact_comp_{view}_{hand}/

Usage:
  python train_impact_comp.py --view front --hand left
"""

import argparse, os, time, math, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import av     # used by CLIP-MODE _decode_segment override below

sys.path.insert(0, str(Path(__file__).parent))
from train_impact_mvitv2 import (
    IMPACTDataset, IMPACTMultiHeadModel,
    action_logits_from_vn, build_action_lookup, topk_acc, build_scheduler,
    N_VERB, N_NOUN, N_ACTION, DATA_BASE, VIEWS, HANDS,
    CLIP_LEN, SPATIAL_SIZE,
)


def _compute_sample_indices(N, is_train):
    """Replicates IMPACTDataset._sample_frames index-selection logic EXACTLY.
    Returns CLIP_LEN integer indices into a length-N frame array.
    Consumes the same RNG draws (and same order) as _sample_frames would.
    Equivalence is verified in scripts/test_clip_mode_equivalence.py."""
    if N <= 0:
        raise RuntimeError("N must be > 0")
    if N >= CLIP_LEN:
        if is_train:
            step = N / CLIP_LEN
            offsets = np.random.uniform(0, step, CLIP_LEN)   # <-- 16 RNG draws
            return [min(N-1, int(i*step + offsets[i])) for i in range(CLIP_LEN)]
        return np.linspace(0, N-1, CLIP_LEN, dtype=int).tolist()
    # N < CLIP_LEN
    return np.linspace(0, N-1, CLIP_LEN, dtype=int).tolist()

COMP_DIR = DATA_BASE / "annotations/TAS-B/comp_split_v1"
from paths import PROJECT_DIR  # see code/paths.py for layout / env-var overrides

# comp_split_v1 was built from front (30 fps) annotations; its frame indices are
# in the 30-fps front timeline. For views whose own fps != 30 (only ego @ 24.92
# in this dataset), comp_split frames must be converted to view-native frames
# before SEEKing the mp4 — otherwise the decoded time window is wrong.
COMP_SPLIT_NATIVE_FPS = 30.0

def _comp_to_view_frame(comp_f, view_fps):
    return int(round(comp_f * view_fps / COMP_SPLIT_NATIVE_FPS))


class IMPACTCompDataset(IMPACTDataset):
    """Compositional-split dataset. Reads segment-level comp_split_v1 files.

    Inherits _decode_segment / _sample_frames / _transform / __getitem__ from
    IMPACTDataset; only __init__ differs (segment lines instead of annotations).
    """

    def __init__(self, view, hand, comp_split, is_train):
        # comp_split in {comp_train, comp_val_seen, comp_val_unseen}
        self.is_train = is_train
        self.view = view
        self.samples = []
        # Optional CLIP-MODE: if env var IMPACT_CLIP_DIR=/path is set AND this
        # dataset's view has pre-extracted clips at that path, redirect to them.
        # Clips are exact per-segment mp4s, so decoding skips the seek logic.
        clip_root = os.environ.get("IMPACT_CLIP_DIR", "")
        self.clip_mode = bool(clip_root) and (Path(clip_root) / view).exists()
        clip_dir_view = Path(clip_root) / view if self.clip_mode else None

        split_file = COMP_DIR / hand / f"{comp_split}.txt"
        video_dir = DATA_BASE / f"videos/{view}"
        ann_dir = DATA_BASE / f"annotations/TAS-B/{view}"

        # fps per recording (load once from annotations)
        fps_cache = {}

        def get_fps(rid):
            if rid not in fps_cache:
                af = ann_dir / f"{rid}_{view}.json"
                fps_cache[rid] = json.load(open(af))["meta_data"]["fps"] if af.exists() else 30.0
            return fps_cache[rid]

        n_missing = 0; n_missing_clip = 0
        any_fps = None
        for line in open(split_file):
            line = line.strip()
            if not line:
                continue
            rid, start, end, verb, noun, action = line.split(",")
            comp_s, comp_e = int(start), int(end)         # comp_split = 30-fps front timeline
            view_fps = get_fps(rid)
            any_fps = view_fps
            # CRITICAL: convert comp_split frame to view-native frame so the SEEK
            # operation lands on the right time. Identity for 30-fps views;
            # only ego (24.92 fps) actually shifts.
            view_s = _comp_to_view_frame(comp_s, view_fps)
            view_e = _comp_to_view_frame(comp_e, view_fps)
            if self.clip_mode:
                # Clip filename uses comp_split (30-fps) indices for naming
                # consistency with extract_ego_clips.py (which also keys by
                # comp_split indices). The clip CONTENT inside is the view-
                # native segment after fps-conversion at extract time.
                clip_file = clip_dir_view / f"{rid}__{comp_s}_{comp_e}.mp4"
                if not clip_file.exists():
                    n_missing_clip += 1
                    continue
                video_path = str(clip_file)
            else:
                video_file = video_dir / f"{rid}_{view}.mp4"
                if not video_file.exists():
                    n_missing += 1
                    continue
                video_path = str(video_file)
            self.samples.append({
                "video": video_path,
                # Store VIEW-NATIVE frame indices; _decode_segment seeks with these.
                "start": view_s, "end": view_e,
                "action": int(action), "verb": int(verb), "noun": int(noun),
                "fps": view_fps,
            })
        mode_str = "CLIP-MODE" if self.clip_mode else "source-video"
        fps_note = "" if any_fps is None or abs(any_fps - COMP_SPLIT_NATIVE_FPS) < 0.01 \
                       else f" [comp_split 30fps → {any_fps:.2f}fps converted]"
        print(f"  [{view}/{hand}/{comp_split} {mode_str}{fps_note}] {len(self.samples)} segments "
              f"(dropped {n_missing} missing src, {n_missing_clip} missing clip)")

    def _decode_segment(self, path, start_f, end_f, fps):
        """Override: in CLIP-MODE, the file IS the segment (no seek needed).

        Memory-safe streaming variant: rather than decode all frames of the clip
        into a list (which OOMs for high-res ego, ~6MB/frame * 500 frames * N workers),
        we (a) read N from container metadata, (b) compute the SAME T_CLIP indices
        and consume the SAME RNG draws as base _sample_frames would, (c) stream
        through decode keeping only the T_CLIP target frames at full resolution.

        Provably equivalent to: super()._decode_segment(...) + super()._sample_frames(...)
        provided stream.frames is accurate (verified in test_clip_mode_equivalence.py).
        __getitem__ override below skips _sample_frames to avoid double-sampling."""
        if not self.clip_mode:
            return super()._decode_segment(path, start_f, end_f, fps)
        c = av.open(path)
        s = c.streams.video[0]
        N = s.frames
        if N <= 0:
            # Defensive fallback: container didn't advertise frame count.
            frames_all = [fr.to_ndarray(format="rgb24") for fr in c.decode(s)]
            c.close()
            return frames_all
        # Sample indices (consumes 16 uniform draws in train mode -> identical RNG
        # consumption to _sample_frames so downstream _transform draws line up).
        target_idxs = _compute_sample_indices(N, self.is_train)
        unique_sorted = sorted(set(target_idxs))
        max_needed = unique_sorted[-1]
        picked = {}
        for i, fr in enumerate(c.decode(s)):
            if i in picked:  # shouldn't recur, but cheap guard
                continue
            if i in unique_sorted:
                picked[i] = fr.to_ndarray(format="rgb24")
                if len(picked) == len(unique_sorted):
                    break
            if i > max_needed:
                break
        c.close()
        if not picked:
            # Should not happen for non-empty clip; let __getitem__ catch.
            raise RuntimeError(f"streaming decode produced no frames for {path}")
        # Assemble in target order (duplicates from N<CLIP_LEN preserved)
        out = []
        for ti in target_idxs:
            if ti in picked:
                out.append(picked[ti])
            else:
                nearest = min(picked.keys(), key=lambda k: abs(k - ti))
                out.append(picked[nearest])
        return out

    def __getitem__(self, idx):
        """CLIP-MODE: _decode_segment already returns CLIP_LEN sampled frames,
        so skip _sample_frames (which would consume 16 additional RNG draws and
        desync the random crop/flip in _transform from the OLD code path).
        Non-CLIP-MODE: defer to base __getitem__ unchanged."""
        if not self.clip_mode:
            return super().__getitem__(idx)
        s = self.samples[idx]
        try:
            frames = self._decode_segment(s["video"], s["start"], s["end"], s["fps"])
            clip = self._transform(frames)
        except Exception as e:
            print(f"[warn] CLIP-MODE decode error {s['video']}: {e}")
            clip = torch.zeros(3, CLIP_LEN, SPATIAL_SIZE, SPATIAL_SIZE)
        return (clip,
                torch.tensor(s["action"], dtype=torch.long),
                torch.tensor(s["verb"], dtype=torch.long),
                torch.tensor(s["noun"], dtype=torch.long))


@torch.no_grad()
def evaluate(model, loader, device, split_label):
    model.eval()
    n = 0
    a_K = [1, 3, 5, 10]; v_K = [1, 3, 5]; nn_K = [1, 3, 5]
    a_c = {k: 0 for k in a_K}; v_c = {k: 0 for k in v_K}; n_c = {k: 0 for k in nn_K}
    # per-component error decomposition (top-1)
    cell = {"both": 0, "verb_only": 0, "noun_only": 0, "neither": 0}
    for clip, action, verb, noun in tqdm(loader, desc=f"Eval({split_label})", leave=False):
        clip = clip.to(device); action = action.to(device)
        verb = verb.to(device); noun = noun.to(device)
        v_lg, n_lg = model(clip)
        a_lg = action_logits_from_vn(v_lg, n_lg, device)
        bs = clip.size(0); n += bs
        ca = topk_acc(a_lg, action, a_K)
        cv = topk_acc(v_lg, verb, v_K)
        cn = topk_acc(n_lg, noun, nn_K)
        for k in a_K: a_c[k] += ca[k]
        for k in v_K: v_c[k] += cv[k]
        for k in nn_K: n_c[k] += cn[k]
        # error decomposition
        pv = v_lg.argmax(1); pn = n_lg.argmax(1)
        v_ok = (pv == verb); n_ok = (pn == noun)
        cell["both"]      += int((v_ok & n_ok).sum())
        cell["verb_only"] += int((v_ok & ~n_ok).sum())
        cell["noun_only"] += int((~v_ok & n_ok).sum())
        cell["neither"]   += int((~v_ok & ~n_ok).sum())
    res = {"n": n,
           **{f"action_top{k}": a_c[k]/n*100 for k in a_K},
           **{f"verb_top{k}": v_c[k]/n*100 for k in v_K},
           **{f"noun_top{k}": n_c[k]/n*100 for k in nn_K},
           "err_decomp": {k: cell[k]/n*100 for k in cell}}
    return res


def train_one_epoch(model, loader, opt, sched, device, epoch, writer):
    model.train()
    n = 0; total_loss = 0.0; a_correct = 0
    for clip, action, verb, noun in tqdm(loader, desc=f"Train E{epoch:03d}", leave=False):
        clip = clip.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)
        verb = verb.to(device, non_blocking=True)
        noun = noun.to(device, non_blocking=True)
        v_lg, n_lg = model(clip)
        loss = (F.cross_entropy(v_lg, verb, label_smoothing=0.1)
                + F.cross_entropy(n_lg, noun, label_smoothing=0.1))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        bs = clip.size(0); n += bs; total_loss += loss.item() * bs
        with torch.no_grad():
            a_lg = action_logits_from_vn(v_lg, n_lg, device)
            a_correct += topk_acc(a_lg, action, [1])[1]
    sched.step()
    if writer:
        writer.add_scalar("train/loss", total_loss/n, epoch)
        writer.add_scalar("train/action_top1", a_correct/n*100, epoch)
    return total_loss/n, a_correct/n


def hm(a, b):
    return 2*a*b/(a+b) if (a+b) > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--view", required=True, choices=VIEWS)
    p.add_argument("--hand", required=True, choices=HANDS)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--base_lr", type=float, default=0.0096)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    name = f"impact_comp_{args.view}_{args.hand}"
    ckpt_dir = PROJECT_DIR / "checkpoints" / name
    result_dir = PROJECT_DIR / "results" / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eff_lr = args.base_lr * args.batch_size / 256
    print(f"[task] {name}  eff_lr={eff_lr:.6f}")

    train_ds = IMPACTCompDataset(args.view, args.hand, "comp_train", is_train=True)
    seen_ds  = IMPACTCompDataset(args.view, args.hand, "comp_val_seen", is_train=False)
    unseen_ds = IMPACTCompDataset(args.view, args.hand, "comp_val_unseen", is_train=False)

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.num_workers, pin_memory=True,
                                   drop_last=sh, persistent_workers=False, prefetch_factor=2)
    train_loader = dl(train_ds, True)
    seen_loader = dl(seen_ds, False)
    unseen_loader = dl(unseen_ds, False)

    model = IMPACTMultiHeadModel(args.dropout, pretrained=True).to(device)
    build_action_lookup(device)
    opt = optim.AdamW(model.parameters(), lr=eff_lr, weight_decay=args.weight_decay)
    sched = build_scheduler(opt, args.warmup_epochs, args.epochs)

    # Robust resume: latest.pth then best.pth fallback (in case latest.pth was
    # interrupted mid-write). Both checkpoints are full state.
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
        t0 = time.time()
        tr_loss, tr_t1 = train_one_epoch(model, train_loader, opt, sched, device, epoch, writer)
        seen = evaluate(model, seen_loader, device, "seen")
        unseen = evaluate(model, unseen_loader, device, "unseen")
        cur_hm = hm(seen["action_top1"], unseen["action_top1"])
        elapsed = time.time() - t0
        print(f"E{epoch:03d}/{args.epochs}  tr_loss={tr_loss:.4f}  "
              f"seen a_t1={seen['action_top1']:.2f}%  "
              f"unseen a_t1={unseen['action_top1']:.2f}%  HM={cur_hm:.2f}  ({elapsed:.0f}s)")
        if writer:
            writer.add_scalar("seen/action_top1", seen["action_top1"], epoch)
            writer.add_scalar("unseen/action_top1", unseen["action_top1"], epoch)
            writer.add_scalar("unseen/HM", cur_hm, epoch)

        c = {"epoch": epoch, "model": model.state_dict(),
             "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
             "best_hm": best_hm}
        # Atomic save: tmp + rename so a interruption cannot corrupt latest.pth.
        tmp = ckpt_dir / "latest.pth.tmp"
        torch.save(c, tmp); tmp.replace(ckpt_dir / "latest.pth")
        if cur_hm > best_hm:
            best_hm = cur_hm
            c["best_hm"] = best_hm
            tmp = ckpt_dir / "best.pth.tmp"
            torch.save(c, tmp); tmp.replace(ckpt_dir / "best.pth")
            print(f"  [*] new best HM = {best_hm:.2f}")

    writer.close()

    # Final eval with best checkpoint
    print("\n=== Final eval (best.pth) ===")
    c = torch.load(ckpt_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(c["model"])
    seen = evaluate(model, seen_loader, device, "seen")
    unseen = evaluate(model, unseen_loader, device, "unseen")
    final_hm = hm(seen["action_top1"], unseen["action_top1"])
    print(f"SEEN   action: t1={seen['action_top1']:.2f}% t5={seen['action_top5']:.2f}%  "
          f"verb_t1={seen['verb_top1']:.2f}% noun_t1={seen['noun_top1']:.2f}%")
    print(f"UNSEEN action: t1={unseen['action_top1']:.2f}% t5={unseen['action_top5']:.2f}%  "
          f"verb_t1={unseen['verb_top1']:.2f}% noun_t1={unseen['noun_top1']:.2f}%")
    print(f"HM(action_top1) = {final_hm:.2f}")
    print(f"UNSEEN err-decomp: {unseen['err_decomp']}")

    with open(result_dir / "comp_eval.json", "w") as f:
        json.dump({"view": args.view, "hand": args.hand, "best_epoch": c["epoch"],
                   "seen": seen, "unseen": unseen, "HM_action_top1": final_hm}, f, indent=2)
    print(f"[done] -> {result_dir/'comp_eval.json'}")


if __name__ == "__main__":
    main()

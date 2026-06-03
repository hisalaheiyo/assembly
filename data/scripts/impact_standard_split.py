"""
Build recording-level train/val/test split for IMPACT TAS-B.
Per hand (left/right), independently. View is irrelevant: same recording → same split.

Uses iterative stratification (multi-label) to balance action class distribution
across the 3 splits. Hard constraint: every action class with >=3 videos must
appear in all 3 splits with >=1 video.

Output: splits_v1/{left,right}/{train,val,test}.txt + split_stats.json
"""

import json, random, math
from pathlib import Path
from collections import Counter, defaultdict

from paths import IMPACT_DATA as BASE  # see code/paths.py for layout / env-var overrides
ANN_DIR = BASE / "annotations/TAS-B"
OUT_DIR = BASE / "annotations/TAS-B/splits_v1"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (implicit)
SEED = 42


def load_recording_actions(hand):
    """Returns: dict {recording_id: Counter(action_label)} for given hand.
    Reads from front view annotations (segments are identical across views)."""
    rec_to_actions = {}
    for f in sorted((ANN_DIR/"front").glob("*.json")):
        with open(f) as fp: ann = json.load(fp)
        # recording_id strips '_front' suffix
        rec_id = ann["video_id"].rsplit("_", 1)[0]
        c = Counter()
        for s in ann["segments"]:
            if s["entity"] != hand: continue
            a = s["action_label"]
            if a > 0:  # exclude null
                c[a] += 1
        rec_to_actions[rec_id] = c
    return rec_to_actions


def iterative_stratification(rec_to_actions, ratios={"train": 0.70, "val": 0.15, "test": 0.15}, seed=42):
    """Iterative multi-label stratification (Sechidis et al., 2011 — proper version).

    Algorithm:
      1. Compute desired #videos per split (size targets) and per (split, class) targets.
      2. Repeat until all videos assigned:
         a. Find class with smallest remaining unassigned count (rarest class first).
         b. For each unassigned video containing that class:
              - Assign to split with largest (target - current) for that class.
              - Tie-break: split with most overall capacity remaining.
              - Random as final tiebreaker.
         c. Decrement remaining counts for ALL classes in the assigned video.
      3. After main loop, place any leftover videos (no remaining classes) by capacity.
    """
    rng = random.Random(seed)
    recordings = list(rec_to_actions.keys())
    rng.shuffle(recordings)
    rec_to_classes = {r: set(rec_to_actions[r].keys()) for r in recordings}

    # Per class: how many videos contain it
    class_video_count = defaultdict(int)
    for rec in recordings:
        for cls in rec_to_classes[rec]:
            class_video_count[cls] += 1

    # Target #videos per split per class
    target_class = {sp: {c: class_video_count[c] * r for c in class_video_count}
                    for sp, r in ratios.items()}
    # Target overall #videos per split
    target_size = {sp: int(round(len(recordings) * r)) for sp, r in ratios.items()}
    diff = len(recordings) - sum(target_size.values())
    target_size["train"] += diff

    # State
    assigned = {sp: [] for sp in ratios}
    cur_class = {sp: defaultdict(int) for sp in ratios}
    cur_size = {sp: 0 for sp in ratios}
    unassigned = set(recordings)
    # Per-class remaining count (in unassigned pool)
    class_remaining = dict(class_video_count)

    while unassigned:
        # Find rarest class with at least 1 remaining
        active_classes = {c: class_remaining[c] for c in class_remaining if class_remaining[c] > 0}
        if not active_classes:
            # No more videos with any active class → assign rest by capacity
            for rec in list(unassigned):
                # Pick split with largest remaining capacity (target - current)
                best_sp = max(ratios.keys(), key=lambda sp: target_size[sp] - cur_size[sp])
                assigned[best_sp].append(rec)
                cur_size[best_sp] += 1
                for c in rec_to_classes[rec]:
                    cur_class[best_sp][c] += 1
                unassigned.discard(rec)
            break

        rarest = min(active_classes.items(), key=lambda x: x[1])[0]
        # Assign all unassigned videos containing rarest, one by one
        candidates = [r for r in unassigned if rarest in rec_to_classes[r]]
        for rec in candidates:
            if rec not in unassigned: continue
            # Pick split with largest deficit for this class
            def score(sp):
                deficit_class = target_class[sp][rarest] - cur_class[sp][rarest]
                deficit_size  = target_size[sp]  - cur_size[sp]
                # Primary: class deficit; secondary: size deficit; final: random
                return (deficit_class, deficit_size, rng.random())
            best_sp = max(ratios.keys(), key=score)
            assigned[best_sp].append(rec)
            cur_size[best_sp] += 1
            for c in rec_to_classes[rec]:
                cur_class[best_sp][c] += 1
                class_remaining[c] -= 1
            unassigned.discard(rec)

    return assigned, cur_class, class_video_count


def repair_singleton_classes(assigned, cur_per_class, class_video_count, rec_to_classes):
    """Ensure every class with >=3 videos appears in all 3 splits.
    If a splittable class is missing from a split, swap a video that has it from
    another split into the missing one (best-effort)."""
    splittable_classes = [c for c, cnt in class_video_count.items() if cnt >= 3]
    moved = 0
    for cls in splittable_classes:
        for sp in ["train", "val", "test"]:
            if cur_per_class[sp][cls] >= 1:
                continue
            # Find another split that has this class with >=2 videos
            donors = [sp2 for sp2 in ["train", "val", "test"]
                      if sp2 != sp and cur_per_class[sp2][cls] >= 2]
            if not donors: continue
            donor = max(donors, key=lambda d: cur_per_class[d][cls])
            # Find a recording in donor that has cls and is OK to move
            # (prefer one with fewest total classes, to minimize disruption)
            recs_with_cls = [r for r in assigned[donor] if cls in rec_to_classes[r]]
            if not recs_with_cls: continue
            rec_to_move = min(recs_with_cls, key=lambda r: len(rec_to_classes[r]))
            assigned[donor].remove(rec_to_move)
            assigned[sp].append(rec_to_move)
            moved += 1
            for c in rec_to_classes[rec_to_move]:
                cur_per_class[donor][c] -= 1
                cur_per_class[sp][c] += 1
    return moved


def report(hand, assigned, cur_per_class, class_video_count, rec_to_actions):
    print(f"\n=== {hand.upper()} hand split summary ===")
    total = sum(len(v) for v in assigned.values())
    for sp, recs in assigned.items():
        print(f"  {sp}: {len(recs)} recordings ({len(recs)/total*100:.1f}%)")

    # Class coverage
    n_classes = len(class_video_count)
    print(f"\n  Action classes total: {n_classes}")
    in_train = sum(1 for c in class_video_count if cur_per_class["train"][c] > 0)
    in_val   = sum(1 for c in class_video_count if cur_per_class["val"][c] > 0)
    in_test  = sum(1 for c in class_video_count if cur_per_class["test"][c] > 0)
    in_all3  = sum(1 for c in class_video_count
                   if cur_per_class["train"][c] > 0 and cur_per_class["val"][c] > 0
                   and cur_per_class["test"][c] > 0)
    print(f"  Classes in train: {in_train}, val: {in_val}, test: {in_test}")
    print(f"  Classes in ALL 3 splits: {in_all3} / {n_classes}")
    splittable = sum(1 for c, cnt in class_video_count.items() if cnt >= 3)
    in_all3_splittable = sum(1 for c, cnt in class_video_count.items()
                             if cnt >= 3 and cur_per_class["train"][c] > 0
                             and cur_per_class["val"][c] > 0
                             and cur_per_class["test"][c] > 0)
    print(f"  Splittable classes (>=3 videos): {splittable}, of which in ALL 3 splits: {in_all3_splittable}")

    # Segment counts per split (sum of action counts in each recording)
    print(f"\n  Total non-null segments per split:")
    for sp, recs in assigned.items():
        n_seg = sum(sum(rec_to_actions[r].values()) for r in recs)
        print(f"    {sp}: {n_seg}")

    # Classes missing from train (model can't learn these)
    missing_from_train = [c for c in class_video_count if cur_per_class["train"][c] == 0]
    print(f"\n  Classes MISSING from train: {len(missing_from_train)} (e.g., {missing_from_train[:5]})")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--repair", action="store_true", default=False,
                   help="Aggressively rebalance to ensure splittable classes are in all 3 splits "
                        "(may distort 70/15/15 ratio). Default: off.")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for hand in ["left", "right"]:
        rec_to_actions = load_recording_actions(hand)
        rec_to_classes = {r: set(c.keys()) for r, c in rec_to_actions.items()}
        assigned, cur, cls_count = iterative_stratification(rec_to_actions, seed=SEED)
        if args.repair:
            n_moved = repair_singleton_classes(assigned, cur, cls_count, rec_to_classes)
            if n_moved > 0:
                print(f"[{hand}] repair: moved {n_moved} recordings to fix split coverage")

        report(hand, assigned, cur, cls_count, rec_to_actions)

        # Save split files
        hand_dir = OUT_DIR / hand
        hand_dir.mkdir(parents=True, exist_ok=True)
        for sp in ["train", "val", "test"]:
            recs_sorted = sorted(assigned[sp])
            with open(hand_dir / f"{sp}.txt", "w") as f:
                for r in recs_sorted:
                    f.write(f"{r}\n")

        # Per-split stats
        summary[hand] = {
            sp: {
                "n_recordings": len(assigned[sp]),
                "n_segments_nonnull": sum(sum(rec_to_actions[r].values()) for r in assigned[sp]),
                "n_classes_present": sum(1 for c in cls_count if cur[sp][c] > 0),
            }
            for sp in ["train", "val", "test"]
        }
        summary[hand]["n_total_classes"] = len(cls_count)

    with open(OUT_DIR / "split_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

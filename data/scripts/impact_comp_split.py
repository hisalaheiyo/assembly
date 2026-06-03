"""
Build a COMPOSITIONAL split for IMPACT TAS-B, per hand, mirroring HAViD/MECCANO.

Holds out ~15% of observed (verb, noun) pairs as A_unseen, subject to:
  (i)  every verb in A_unseen still appears in A_seen (paired with another noun);
  (ii) every noun in A_unseen still appears in A_seen (paired with another verb);
  (iii) every held-out pair has >= MIN_SEG segments (enough for meaningful test).

A_seen segments reuse the recording-level partition of splits_v1 (train/val/test).
A_unseen segments (the held-out pairs) all go to comp_val_unseen, regardless of
recording -- the model never trains on these pairs anyway.

Segment-level output (a recording contributes seen-pair segments to comp_train and
unseen-pair segments to comp_val_unseen simultaneously, exactly as in HAViD).

Output: comp_split_v1/{left,right}/{comp_train,comp_val_seen,comp_val_unseen}.txt
        comp_split_v1/comp_split_stats.json
Each line: recording_id,start_frame,end_frame,verb,noun,action_label
"""

import json, random
from pathlib import Path
from collections import defaultdict, Counter

from paths import IMPACT_DATA as BASE  # see code/paths.py for layout / env-var overrides
ANN_DIR = BASE / "annotations/TAS-B/front"          # segments identical across views
SPLITS_V1 = BASE / "annotations/TAS-B/splits_v1"     # existing recording-level split
OUT_DIR = BASE / "annotations/TAS-B/comp_split_v1"

HOLDOUT_FRAC = 0.15      # fraction of pairs to hold out
MIN_SEG = 10             # min segments for a pair to be holdout-eligible
SEED = 42
N_TIERS = 3              # frequency tiers for stratified holdout


def load_hand_segments(hand):
    """Return list of segment dicts and per-recording grouping for one hand."""
    segs = []
    for f in sorted(ANN_DIR.glob("*.json")):
        ann = json.load(open(f))
        rid = ann["video_id"].rsplit("_", 1)[0]
        for s in ann["segments"]:
            if s["entity"] != hand:
                continue
            v, n, a = s["verb"], s["noun"], s["action_label"]
            if v < 0 or n < 0 or a == 0:        # skip null
                continue
            segs.append({"rec": rid, "start": s["start_frame"],
                         "end": s["end_frame"], "verb": v, "noun": n, "action": a})
    return segs


def select_unseen_pairs(segs, seed=SEED):
    """Greedy stratified holdout of ~HOLDOUT_FRAC of (v,n) pairs."""
    rng = random.Random(seed)
    pair_segs = Counter((s["verb"], s["noun"]) for s in segs)
    verb_partners = defaultdict(set)
    noun_partners = defaultdict(set)
    for (v, n) in pair_segs:
        verb_partners[v].add(n)
        noun_partners[n].add(v)

    n_pairs = len(pair_segs)
    target = round(n_pairs * HOLDOUT_FRAC)

    # Eligible: removing the pair leaves verb and noun still covered, and >= MIN_SEG segs
    eligible = [p for p in pair_segs
                if len(verb_partners[p[0]]) >= 2
                and len(noun_partners[p[1]]) >= 2
                and pair_segs[p] >= MIN_SEG]

    # Stratify eligible into N_TIERS frequency tiers
    eligible.sort(key=lambda p: pair_segs[p])
    tiers = [eligible[i::N_TIERS] for i in range(N_TIERS)]
    for t in tiers:
        rng.shuffle(t)

    # Track remaining seen-degree as we hold pairs out
    seen_verb_deg = {v: len(ns) for v, ns in verb_partners.items()}
    seen_noun_deg = {n: len(vs) for n, vs in noun_partners.items()}
    unseen = []
    per_tier = [target // N_TIERS + (1 if i < target % N_TIERS else 0)
                for i in range(N_TIERS)]

    for tier, want in zip(tiers, per_tier):
        got = 0
        for p in tier:
            if got >= want:
                break
            v, n = p
            # holding p out must keep verb v and noun n covered in seen
            if seen_verb_deg[v] >= 2 and seen_noun_deg[n] >= 2:
                unseen.append(p)
                seen_verb_deg[v] -= 1
                seen_noun_deg[n] -= 1
                got += 1

    return set(unseen), pair_segs, verb_partners, noun_partners


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for hand in ["left", "right"]:
        segs = load_hand_segments(hand)
        unseen_pairs, pair_segs, verb_partners, noun_partners = select_unseen_pairs(segs)

        # Recording-level partition from splits_v1
        rec_train = set(l.strip() for l in open(SPLITS_V1 / hand / "train.txt") if l.strip())
        rec_val   = set(l.strip() for l in open(SPLITS_V1 / hand / "val.txt") if l.strip())
        rec_test  = set(l.strip() for l in open(SPLITS_V1 / hand / "test.txt") if l.strip())
        rec_seen_eval = rec_val | rec_test    # A_seen eval pool

        comp_train, comp_val_seen, comp_val_unseen = [], [], []
        for s in segs:
            pair = (s["verb"], s["noun"])
            line = f"{s['rec']},{s['start']},{s['end']},{s['verb']},{s['noun']},{s['action']}"
            if pair in unseen_pairs:
                comp_val_unseen.append(line)            # held-out pair -> always unseen eval
            elif s["rec"] in rec_train:
                comp_train.append(line)
            elif s["rec"] in rec_seen_eval:
                comp_val_seen.append(line)
            # (recordings not in any splits_v1 list: none expected)

        # Write files
        hand_dir = OUT_DIR / hand
        hand_dir.mkdir(parents=True, exist_ok=True)
        for fname, rows in [("comp_train.txt", comp_train),
                            ("comp_val_seen.txt", comp_val_seen),
                            ("comp_val_unseen.txt", comp_val_unseen)]:
            with open(hand_dir / fname, "w") as f:
                f.write("\n".join(rows) + ("\n" if rows else ""))

        # Validation checks
        seen_pairs = set(pair_segs) - unseen_pairs
        seen_verbs = set(v for (v, n) in seen_pairs)
        seen_nouns = set(n for (v, n) in seen_pairs)
        unseen_verbs = set(v for (v, n) in unseen_pairs)
        unseen_nouns = set(n for (v, n) in unseen_pairs)
        verbs_ok = unseen_verbs <= seen_verbs
        nouns_ok = unseen_nouns <= seen_nouns

        print(f"=== {hand.upper()} hand ===")
        print(f"  total pairs={len(pair_segs)}  held-out={len(unseen_pairs)} "
              f"({len(unseen_pairs)/len(pair_segs)*100:.1f}%)")
        print(f"  comp_train={len(comp_train)}  comp_val_seen={len(comp_val_seen)}  "
              f"comp_val_unseen={len(comp_val_unseen)} segments")
        print(f"  constraint -- every held-out verb in seen: {verbs_ok}")
        print(f"  constraint -- every held-out noun in seen: {nouns_ok}")
        us_counts = sorted(pair_segs[p] for p in unseen_pairs)
        print(f"  held-out pair seg-counts: min={us_counts[0]} "
              f"median={us_counts[len(us_counts)//2]} max={us_counts[-1]}")

        summary[hand] = {
            "n_pairs_total": len(pair_segs),
            "n_pairs_unseen": len(unseen_pairs),
            "unseen_pairs": sorted([list(p) for p in unseen_pairs]),
            "n_seg_comp_train": len(comp_train),
            "n_seg_comp_val_seen": len(comp_val_seen),
            "n_seg_comp_val_unseen": len(comp_val_unseen),
            "constraint_verbs_ok": verbs_ok,
            "constraint_nouns_ok": nouns_ok,
        }
        print()

    with open(OUT_DIR / "comp_split_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

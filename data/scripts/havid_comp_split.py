"""
Compositional split for the HAViD dataset (hold-out composition protocol).

For every (view, hand) subtask we write three new files alongside the
upstream train/val lists (originals are not modified):
  *_comp_train.txt       train segments minus held-out (verb, manip-object) pairs
  *_comp_val_seen.txt    val segments minus held-out pairs
  *_comp_val_unseen.txt  every segment whose pair is held out (from train + val)

The held-out (verb, manip-object) pairs are chosen manually under fixed
quantitative constraints:
  1. The pair must occur at least 15 times in the dataset.
  2. After removal, the verb still pairs with at least 5 other objects.
  3. After removal, the object still pairs with at least 3 other verbs.
  4. The held-out set spans multiple verb categories.
"""

import os
import collections
from collections import defaultdict

# ----------------------------------------------------------------
# 1. Dataset root and the list of (view, hand) subtasks
# ----------------------------------------------------------------
from paths import HAVID_DATA as DATASET_ROOT

# (subdir, train_file, val_file)
SPLITS = [
    ("view0_lh_aa", "view0_lh_aa_train_list_videos_parsed.txt",            "view0_lh_aa_val_list_videos_parsed.txt"),
    ("view0_lh_pt", "view0_lh_pt_train_list_videos_parsed_filtered.txt",    "view0_lh_pt_val_list_videos_parsed_filtered.txt"),
    ("view0_rh_aa", "view0_rh_aa_train_list_videos_parsed.txt",             "view0_rh_aa_val_list_videos_parsed.txt"),
    ("view0_rh_pt", "view0_rh_pt_train_list_videos_parsed_filtered.txt",    "view0_rh_pt_val_list_videos_parsed_filtered.txt"),
    ("view1_lh_aa", "view1_lh_aa_train_list_videos_parsed.txt",            "view1_lh_aa_val_list_videos_parsed.txt"),
    ("view1_lh_pt", "view1_lh_pt_train_list_videos_parsed_filtered.txt",    "view1_lh_pt_val_list_videos_parsed_filtered.txt"),
    ("view1_rh_aa", "view1_rh_aa_train_list_videos_parsed.txt",             "view1_rh_aa_val_list_videos_parsed.txt"),
    ("view1_rh_pt", "view1_rh_pt_train_list_videos_parsed_filtered.txt",    "view1_rh_pt_val_list_videos_parsed_filtered.txt"),
    ("view2_lh_aa", "view2_lh_aa_train_list_videos_parsed.txt",            "view2_lh_aa_val_list_videos_parsed.txt"),
    ("view2_lh_pt", "view2_lh_pt_train_list_videos_parsed_filtered.txt",    "view2_lh_pt_val_list_videos_parsed_filtered.txt"),
    ("view2_rh_aa", "view2_rh_aa_train_list_videos_parsed.txt",             "view2_rh_aa_val_list_videos_parsed.txt"),
    ("view2_rh_pt", "view2_rh_pt_train_list_videos_parsed_filtered.txt",    "view2_rh_pt_val_list_videos_parsed_filtered.txt"),
]


# ----------------------------------------------------------------
# 2. Parse a list-file line -> (raw_line, verb, manip_object)
# ----------------------------------------------------------------
def parse_line(line):
    line = line.rstrip("\n")
    parts = line.split(",")
    verb   = parts[2] if len(parts) > 2 else "null"
    manip  = parts[3] if len(parts) > 3 else "null"
    return line, verb, manip


def load_file(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(parse_line(line))
    return rows


# ----------------------------------------------------------------
# 3. Use view2_lh_aa as the global statistics source for picking
#    the held-out pairs. All views share the same clip IDs and
#    labels, so any single subtask gives an equivalent count.
# ----------------------------------------------------------------
ref_train_path = os.path.join(DATASET_ROOT, "view2_lh_aa",
                              "view2_lh_aa_train_list_videos_parsed.txt")
ref_val_path   = os.path.join(DATASET_ROOT, "view2_lh_aa",
                              "view2_lh_aa_val_list_videos_parsed.txt")

ref_all = load_file(ref_train_path) + load_file(ref_val_path)
non_null = [(v, mo) for _, v, mo in ref_all if v != "null"]

combo_counts   = collections.Counter(non_null)
verb_to_objs   = defaultdict(set)
obj_to_verbs   = defaultdict(set)
for v, mo in non_null:
    verb_to_objs[v].add(mo)
    obj_to_verbs[mo].add(v)


# ----------------------------------------------------------------
# 4. Manually selected held-out (verb, manip-object) pairs.
#
#    Selection guidelines:
#      - At most 1-2 objects per verb (keeps the training set covered).
#      - Prefer objects that already pair with >= 6 other verbs
#        (so removing the pair still leaves the object well-covered).
#      - 7 verbs each contribute 1-2 pairs; total ~8-9% of non-null
#        segments are held out.
#      - Avoid rare-vocabulary verbs (rotate / slide / disassemble);
#        their constraints are too tight.
# ----------------------------------------------------------------
HELD_OUT = {
    # verb           manip_obj
    ("approach",    "small_gear"),       # count=55,  sg: 6 verbs remain after removal
    ("approach",    "cylinder_base"),    # count=41,  cb: 7 verbs -> 6
    ("grasp",       "bolt"),             # count=75,  bolt: 8 verbs -> 7
    ("grasp",       "large_gear"),       # count=70,  lg: 6 verbs -> 5
    ("hold",        "cylinder_bracket"), # count=62,  cbracket: 7 verbs -> 6
    ("hold",        "worm_gear"),        # count=49,  wg: 5 verbs -> 4  (place+wg also removed below)
    ("insert",      "large_spacer"),     # count=79,  ls: 6 verbs -> 5
    ("insert",      "small_spacer"),     # count=87,  ss: 5 verbs -> 4
    ("move",        "cylinder_cap"),     # count=44,  cap: 7 verbs -> 6
    ("move",        "usb_male"),         # count=58,  usb: 5 verbs -> 4
    ("place",       "cylinder_base"),    # count=22,  cb: 7 verbs -> 6 (after approach+cb removed)
    ("place",       "worm_gear"),        # count=55,  wg: 5 verbs -> 3  (hold+wg also removed)
    ("screw",       "gear_shaft"),       # count=123, gs: 7 verbs -> 6
}

# Note: in the HAViD list files "small gear" etc. carry a literal space,
# while the held-out set above uses underscores; convert before matching.
HELD_OUT_RAW = set()
for v, mo in HELD_OUT:
    HELD_OUT_RAW.add((v, mo.replace("_", " ")))


# ----------------------------------------------------------------
# 5. Verify the held-out set satisfies the stated constraints on
#    the full dataset before generating any files.
# ----------------------------------------------------------------
def verify_constraints():
    print("=" * 60)
    print("Verifying held-out constraints on view2_lh_aa (train+val)")
    print("=" * 60)

    total_heldout = sum(combo_counts.get(pair, 0) for pair in HELD_OUT_RAW)
    total_nonnull = len(non_null)
    print(f"\nHeld-out combos: {len(HELD_OUT_RAW)}")
    print(f"Held-out samples: {total_heldout} / {total_nonnull} non-null"
          f"  ({100*total_heldout/total_nonnull:.1f}%)")

    # Recompute the remaining (verb, obj) coverage map after removing the held-out pairs.
    remaining_verb_objs = defaultdict(set)
    remaining_obj_verbs = defaultdict(set)
    for v, mo in non_null:
        if (v, mo) not in HELD_OUT_RAW:
            remaining_verb_objs[v].add(mo)
            remaining_obj_verbs[mo].add(v)

    print("\n[Combo details]")
    ok = True
    for (v, mo) in sorted(HELD_OUT_RAW):
        cnt          = combo_counts.get((v, mo), 0)
        v_remain     = len(remaining_verb_objs.get(v, set()))
        mo_remain    = len(remaining_obj_verbs.get(mo, set()))
        c1 = cnt >= 15
        c2 = v_remain >= 5
        c3 = mo_remain >= 3
        status = "OK" if (c1 and c2 and c3) else "FAIL"
        if not (c1 and c2 and c3):
            ok = False
        print(f"  [{status}] ({v}, {mo}): count={cnt}, "
              f"verb_remain={v_remain}(>=5?{c2}), "
              f"obj_remain={mo_remain}(>=3?{c3})")

    print(f"\nAll constraints satisfied: {ok}")
    print()
    return ok


# ----------------------------------------------------------------
# 6. For each (view, hand) subtask, write the three split files.
# ----------------------------------------------------------------
def split_one(subdir, train_file, val_file):
    base = os.path.join(DATASET_ROOT, subdir)
    train_rows = load_file(os.path.join(base, train_file))
    val_rows   = load_file(os.path.join(base, val_file))

    comp_train       = []  # seen pairs from upstream train
    comp_val_seen    = []  # seen pairs from upstream val
    comp_val_unseen  = []  # held-out pairs (from train + val combined)

    for raw, verb, mo in train_rows:
        if (verb, mo) in HELD_OUT_RAW:
            comp_val_unseen.append(raw)
        else:
            comp_train.append(raw)

    for raw, verb, mo in val_rows:
        if (verb, mo) in HELD_OUT_RAW:
            comp_val_unseen.append(raw)
        else:
            comp_val_seen.append(raw)

    prefix = subdir  # e.g. "view2_lh_aa"
    out_train   = os.path.join(base, f"{prefix}_comp_train.txt")
    out_seen    = os.path.join(base, f"{prefix}_comp_val_seen.txt")
    out_unseen  = os.path.join(base, f"{prefix}_comp_val_unseen.txt")

    def write(path, rows):
        with open(path, "w") as f:
            for row in rows:
                f.write(row + "\n")

    write(out_train,  comp_train)
    write(out_seen,   comp_val_seen)
    write(out_unseen, comp_val_unseen)

    return len(comp_train), len(comp_val_seen), len(comp_val_unseen)


# ----------------------------------------------------------------
# 7. Entry point: verify, then generate every subtask's files.
# ----------------------------------------------------------------
if __name__ == "__main__":
    ok = verify_constraints()
    if not ok:
        print("Constraint check failed. Aborting.")
        exit(1)

    print("=" * 60)
    print("Generating compositional split files ...")
    print("=" * 60)
    print(f"\n{'Split':<18} {'comp_train':>12} {'val_seen':>10} {'val_unseen':>12}")
    print("-" * 56)

    for subdir, train_file, val_file in SPLITS:
        n_train, n_seen, n_unseen = split_one(subdir, train_file, val_file)
        print(f"{subdir:<18} {n_train:>12,} {n_seen:>10,} {n_unseen:>12,}")

    print()
    print("Done. New files created (originals untouched):")
    print("  *_comp_train.txt       seen pairs, for training")
    print("  *_comp_val_seen.txt    seen pairs, for validation")
    print("  *_comp_val_unseen.txt  held-out pairs, for compositional test")

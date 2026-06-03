"""
Build compositional split for MECCANO (Plan A: 6 unseen pairs).

Strategy (mirrors HAViD's compositional split):
  - 6 (verb, primary_noun) pairs are held out as UNSEEN
  - Each unseen verb / noun separately appears in training (with other partners)
  - comp_val_unseen: ALL samples (across original train/val/test) whose pair is in unseen
  - comp_train:     original train samples whose pair is in seen pairs
  - comp_val_seen:  original (val + test) samples whose pair is in seen pairs

Output: 3 CSV files in $MECCANO_DATA/comp_split/  (see code/paths.py for path config)
  Same columns as the *_index_expanded.csv files (clip_path, action_id, action_name,
  verb_id, noun_*).
"""

import csv
import json
from pathlib import Path

from paths import MECCANO_DATA as BASE  # see code/paths.py for layout / env-var overrides
OUT_DIR = BASE / "comp_split"
OUT_DIR.mkdir(exist_ok=True)

# ---- Plan A: 6 unseen (verb, primary_noun) pairs ----
UNSEEN_PAIRS = [
    ("take", "booklet"),
    ("put", "screw"),
    ("put", "booklet"),
    ("put", "bolt"),
    ("take", "rod"),
    ("take", "handlebar"),
]


def read_split(split):
    """Read MECCANO {split}_index_expanded.csv."""
    rows = []
    with open(BASE / f"{split}_index_expanded.csv") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        for row in reader:
            row["_split_origin"] = split
            rows.append(row)
    return rows, fields


def get_primary_noun(row, noun_cols):
    """Return the alphabetically first noun marked as 1, or None."""
    nouns = sorted([c.replace("noun_", "") for c in noun_cols if row[c] == "1"])
    return nouns[0] if nouns else None


def get_verb_name(row):
    """Recover verb name from action_name (first token)."""
    return row["action_name"].split("_")[0]


# Read all 3 splits
all_rows = []
fieldnames = None
for split in ["train", "val", "test"]:
    rows, fields = read_split(split)
    if fieldnames is None:
        fieldnames = fields
    all_rows.extend(rows)
print(f"Total samples: {len(all_rows)}")

noun_cols = [c for c in fieldnames if c.startswith("noun_")]

# Build unseen pair lookup: (verb_name, primary_noun) → True
unseen_pair_set = set(UNSEEN_PAIRS)

# Categorize samples
comp_train, comp_val_seen, comp_val_unseen = [], [], []
n_dropped = 0  # samples with no noun

for row in all_rows:
    verb_name = get_verb_name(row)
    primary_noun = get_primary_noun(row, noun_cols)
    if primary_noun is None:
        n_dropped += 1
        continue

    pair = (verb_name, primary_noun)
    is_unseen = pair in unseen_pair_set

    if is_unseen:
        comp_val_unseen.append(row)
    else:
        if row["_split_origin"] == "train":
            comp_train.append(row)
        else:  # val or test
            comp_val_seen.append(row)

print(f"\nNo-noun samples dropped: {n_dropped}")
print(f"comp_train:      {len(comp_train)}")
print(f"comp_val_seen:   {len(comp_val_seen)}")
print(f"comp_val_unseen: {len(comp_val_unseen)}")
print(f"Total kept:      {len(comp_train)+len(comp_val_seen)+len(comp_val_unseen)}")


def write_csv(rows, out_file, fieldnames):
    """Write rows to CSV, dropping internal fields like _split_origin."""
    cols = [c for c in fieldnames if not c.startswith("_")]
    with open(out_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            out = {c: r[c] for c in cols}
            w.writerow(out)
    print(f"  Saved {len(rows)} → {out_file}")


print("\nWriting CSV files...")
write_csv(comp_train,      OUT_DIR / "comp_train.csv",      fieldnames)
write_csv(comp_val_seen,   OUT_DIR / "comp_val_seen.csv",   fieldnames)
write_csv(comp_val_unseen, OUT_DIR / "comp_val_unseen.csv", fieldnames)

# ---- Sanity checks ----
print("\n" + "="*60)
print("SANITY CHECKS")
print("="*60)


def get_pairs_in(rows):
    pairs = set()
    for r in rows:
        v = get_verb_name(r)
        n = get_primary_noun(r, noun_cols)
        if n: pairs.add((v, n))
    return pairs


train_pairs   = get_pairs_in(comp_train)
seen_pairs    = get_pairs_in(comp_val_seen)
unseen_pairs  = get_pairs_in(comp_val_unseen)

print(f"\nUnique (verb, noun) pairs:")
print(f"  comp_train:      {len(train_pairs)}")
print(f"  comp_val_seen:   {len(seen_pairs)}")
print(f"  comp_val_unseen: {len(unseen_pairs)}")

print(f"\nUnseen pairs verification:")
for p in UNSEEN_PAIRS:
    in_train = p in train_pairs
    in_unseen = p in unseen_pairs
    status = "OK" if (not in_train and in_unseen) else "FAIL"
    print(f"  {p}: in_train={in_train}, in_unseen={in_unseen} [{status}]")

print(f"\nLeakage check (train ∩ unseen): {len(train_pairs & unseen_pairs)}  (should be 0)")
print(f"Leakage check (val_seen ∩ unseen): {len(seen_pairs & unseen_pairs)}  (should be 0)")

print(f"\nVerb coverage in train (unseen verbs need to appear with other nouns):")
for verb, noun in UNSEEN_PAIRS:
    other_nouns_in_train = [n for (v, n) in train_pairs if v == verb]
    print(f"  '{verb}': appears with {len(other_nouns_in_train)} other nouns in train")

print(f"\nNoun coverage in train (unseen nouns need to appear with other verbs):")
for verb, noun in UNSEEN_PAIRS:
    other_verbs_in_train = [v for (v, n) in train_pairs if n == noun]
    print(f"  '{noun}': appears with {len(other_verbs_in_train)} other verbs in train")

# Save metadata
meta = {
    "plan": "A (6 unseen pairs)",
    "unseen_pairs": [list(p) for p in UNSEEN_PAIRS],
    "stats": {
        "comp_train":      len(comp_train),
        "comp_val_seen":   len(comp_val_seen),
        "comp_val_unseen": len(comp_val_unseen),
        "n_train_pairs":   len(train_pairs),
        "n_unseen_pairs":  len(unseen_pairs),
    }
}
with open(OUT_DIR / "split_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"\nMetadata saved → {OUT_DIR/'split_metadata.json'}")

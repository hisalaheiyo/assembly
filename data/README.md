# Dataset preparation

This repository does **not** redistribute raw videos for any of the three
datasets used in the paper. Each dataset has its own release agreement and
must be obtained from its original authors.

After downloading, point the corresponding environment variable at the local
root and the training / evaluation code will read from it transparently:

```bash
export MECCANO_DATA=/path/to/MECCANO
export HAVID_DATA=/path/to/HAViD
export IMPACT_DATA=/path/to/IMPACT
```

The defaults in `code/paths.py` look in `./data/{meccano,havid,impact}/`.

---

## MECCANO

- **Origin**: Ragusa et al., *MECCANO: A Multimodal Egocentric Dataset for
  Humans Behavior Understanding in the Industrial-Like Domain*, WACV 2021.
- **Download**: follow the request form at the project page.
- **Expected layout under `$MECCANO_DATA`**:
  ```
  $MECCANO_DATA/
  ├── MECCANO_RGB_Videos/        full untrimmed RGB recordings
  ├── train/                     pre-segmented clips referenced by train_index.csv
  ├── val/                       pre-segmented clips referenced by val_index.csv
  ├── test/                      pre-segmented clips referenced by test_index.csv
  ├── train_index.csv            (clip_path, action_id, action_name)
  ├── val_index.csv
  └── test_index.csv
  ```
- **Notes**: the CSV indices ship with absolute clip paths from another
  machine. Either pre-rewrite the `clip_path` column to your local paths, or
  set `MECCANO_CSV_OLD_PREFIX` and `MECCANO_CSV_NEW_PREFIX` in your
  environment and the training scripts will remap on read.

### Compositional split

The 6 held-out (verb, object) pairs are hard-coded in
`code/train_meccano_comp_atomic.py` / `train_meccano_comp.py` (look for the
`HELD_OUT` constant). No external script is needed — the split list is the
same for every run.

---

## HAViD

- **Origin**: HAViD: a *Hand-Action Video Dataset*. See the dataset paper
  cited in our submission.
- **Download**: per the dataset authors.
- **Expected layout under `$HAVID_DATA`** (six (view, hand) subtasks plus
  the corresponding `_pt` variants, mirroring the upstream release):
  ```
  $HAVID_DATA/
  ├── view0_lh_aa/
  │   ├── view0_lh_aa_train_list_videos_parsed.txt
  │   ├── view0_lh_aa_val_list_videos_parsed.txt
  │   └── ... (video files referenced by the list)
  ├── view0_rh_aa/
  ├── view1_lh_aa/
  ├── view1_rh_aa/
  ├── view2_lh_aa/
  ├── view2_rh_aa/
  └── ... (corresponding _pt subtasks)
  ```

### Compositional split

Run the construction script once to generate the three new
`*_comp_train.txt`, `*_comp_val_seen.txt`, `*_comp_val_unseen.txt` files for
every subtask:

```bash
python data/scripts/havid_comp_split.py
```

This script

1. picks 13 held-out (verb, manip-object) pairs under fixed quantitative
   constraints (≥ 15 occurrences each, the held-out verb still pairs with
   ≥ 5 other objects, the held-out object still pairs with ≥ 3 other verbs);
2. verifies the constraints on the dataset before writing anything;
3. writes the new split files alongside the upstream list files (the
   originals are not modified).

The selected pairs are reproducible verbatim — there is no randomness; the
same 13 pairs are used in every run reported in the paper.

---

## IMPACT

- **Origin**: see the dataset paper cited in our submission.
- **Download**: per the dataset authors.
- **Expected layout under `$IMPACT_DATA`**:
  ```
  $IMPACT_DATA/
  ├── videos/
  │   ├── front/
  │   ├── left/
  │   ├── right/
  │   ├── top/
  │   └── ego/
  └── annotations/
      └── TAS-B/
          ├── splits_v1/         standard split (left/, right/) + split_stats.json
          ├── front/             per-view per-segment annotations
          ├── left/
          ├── right/
          ├── top/
          └── ego/
  ```

### Standard split (constructed here)

The recording-level, hand-stratified standard partition (~70/15/15%) is built
via iterative multilabel stratification by the training code on first run;
the resulting `splits_v1/<hand>/{train,val,test}.txt` files are persisted
and re-used.

### Compositional split (constructed here)

20 of 134 (verb, noun) pairs per hand are held out (≈ 14.9 %; each held-out
pair has at least 10 segments). The held-out pair list is the same for every
run and is recreated by `code/train_impact_comp.py` on first invocation
(it writes `comp_train.txt`, `comp_val_seen.txt`, `comp_val_unseen.txt`
under each per-view annotation directory).

---

## What lives under `data/splits/`

The `data/splits/{meccano,havid,impact}/` subdirectories are intentionally
empty in the repository. They are populated by the construction scripts above
the first time you run training, and contain plain-text segment lists only
(no media). Generated splits are deterministic across runs.

---

## Sanity check

After preparing all three datasets, the following should succeed without
error and print the configured roots:

```bash
python -c "from paths import MECCANO_DATA, HAVID_DATA, IMPACT_DATA, PROJECT_DIR; \
print('MECCANO_DATA =', MECCANO_DATA); print('HAVID_DATA   =', HAVID_DATA); \
print('IMPACT_DATA  =', IMPACT_DATA); print('PROJECT_DIR  =', PROJECT_DIR)"
```

(Run from inside the `code/` directory, or add `code/` to `PYTHONPATH`.)

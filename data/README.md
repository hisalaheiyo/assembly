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
- **Download**: <https://iplab.dmi.unict.it/MECCANO/> — use the request
  form on the project page; we do not redistribute the data.
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

Run the construction script once to build the three split CSVs (6 held-out
(verb, primary-noun) pairs):

```bash
python data/scripts/meccano_comp_split.py
```

This writes:

```
$MECCANO_DATA/comp_split/
├── comp_train.csv
├── comp_val_seen.csv
├── comp_val_unseen.csv
└── split_metadata.json
```

The held-out pair list is hard-coded inside the script and is the same for
every run reported in the paper.

---

## HAViD

- **Origin**: Zheng, Lee, and Lu, *HA-ViD: A Human Assembly Video Dataset
  for Comprehensive Assembly Knowledge Understanding*, NeurIPS 2023.
- **Download**: <https://iai-hrc.github.io/ha-vid> — submit an access
  request on the project page (the data is hosted on Dropbox); the
  benchmarking code is at <https://github.com/iai-hrc/ha-vid>.
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
`*_comp_train.txt`, `*_comp_val_seen.txt`, `*_comp_val_unseen.txt` files
alongside the upstream list files (originals are not modified):

```bash
python data/scripts/havid_comp_split.py
```

The script (1) picks 13 held-out (verb, manip-object) pairs under fixed
quantitative constraints (≥ 15 occurrences each, the held-out verb still
pairs with ≥ 5 other objects, the held-out object still pairs with ≥ 3 other
verbs), (2) verifies the constraints on the full dataset before writing
anything, (3) writes the new split files. The selected pairs are reproducible
verbatim — no randomness.

---

## IMPACT

- **Origin**: Wen, Zhong, Schneider, Peng et al., *IMPACT: A Dataset for
  Multi-Granularity Human Procedural Action Understanding in Industrial
  Assembly*.
- **Download**: <https://github.com/Kratos-Wen/IMPACT> — the project page
  links to the full release bundles (annotations, multi-view RGB and depth
  archives, ego audio, ego eye-tracking traces, feature packs, quick-start
  sample subset). A public Hugging Face mirror of the release bundle is
  available at `KratosWen/IMPACT` (use `huggingface_hub` to fetch).
- **Expected layout under `$IMPACT_DATA`**:
  ```
  $IMPACT_DATA/
  ├── videos/
  │   ├── front/  left/  right/  top/  ego/
  └── annotations/
      └── TAS-B/
          ├── splits_v1/        standard split (left/, right/) + split_stats.json
          ├── comp_split_v1/    compositional split files (generated)
          ├── front/  left/  right/  top/  ego/    per-view per-segment annotations
  ```

### Standard split

The recording-level hand-stratified standard partition (~70/15/15%) is built
via iterative multilabel stratification. Run once:

```bash
python data/scripts/impact_standard_split.py
```

This populates `$IMPACT_DATA/annotations/TAS-B/splits_v1/{left,right}/{train,val,test}.txt`
and a `split_stats.json` summarising per-split counts. The stratifier seed is
fixed inside the script, so successive runs produce byte-identical splits.

### Compositional split

20 of 134 (verb, noun) pairs per hand are held out (≈ 14.9%; each held-out
pair has ≥ 10 segments). Run once:

```bash
python data/scripts/impact_comp_split.py
```

This writes `comp_train.txt`, `comp_val_seen.txt`, `comp_val_unseen.txt`
under the per-view annotation directories. The held-out pair list is fixed in
the script (no randomness).

---

## What lives under `data/splits/`

The `data/splits/{meccano,havid,impact}/` subdirectories are intentionally
empty in the repository. They are populated by the construction scripts above
the first time you run them, and contain plain-text segment lists only
(no media). Every generated split is deterministic across runs.

---

## Sanity check

After preparing all three datasets, the following should succeed without
error and print the configured roots:

```bash
cd code   # so that paths.py is importable
python -c "from paths import MECCANO_DATA, HAVID_DATA, IMPACT_DATA, PROJECT_DIR; \
print('MECCANO_DATA =', MECCANO_DATA); print('HAVID_DATA   =', HAVID_DATA); \
print('IMPACT_DATA  =', IMPACT_DATA); print('PROJECT_DIR  =', PROJECT_DIR)"
```

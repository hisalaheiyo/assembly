# Diagnosing Verb–Noun Decomposition in Assembly Action Recognition

**A Cross-Dataset Study of Compositional Generalization**

> Anonymous code release for double-blind review.
> Author identities will be added in the camera-ready version.

This repository contains the code, generated splits, and analysis pipelines
that reproduce every number, table, and figure of the paper.

---

## Abstract

Recognizing an assembly action means recognizing a *combination*: a manipulation
applied to a part or tool. Atomic action classifiers cannot generalize to
combinations they have not seen — they assign zero probability to every unseen
(verb, noun) pair by construction. The field's escape is verb–noun
decomposition: predict each component with its own head and recombine the
scores, so the output spans the full composition grid. This design is adopted
by convention and seldom tested as a hypothesis. We make its four tacit
assumptions explicit — visual separability, statistical factorizability,
symmetric per-component difficulty, and an appropriate factoring granularity —
and pair each with a controlled diagnostic. On three assembly datasets
(MECCANO, HAViD, IMPACT) that span egocentric and static views, single- and
dual-handed manipulation, and two- to four-way factoring, decomposition
escapes the atomic ceiling but realizes only part of the space it unlocks:
unseen accuracy tracks training co-occurrence and is largely grid
interpolation; failure concentrates on the larger-vocabulary component, which
IMPACT's verb-heavy vocabulary flips from the usual object bottleneck to a
verb one; and the common shared encoder forfeits much of the achievable gain
by entangling the components, trailing independent recombination by up to
$6.0\times$ in harmonic mean. Each diagnosis becomes a concrete design lever
that is predictable from dataset statistics before training, and we release
the diagnostics as a portable protocol.

---

## TL;DR

**Atomic action classifiers cannot generalize to unseen (verb, noun)
combinations**: they assign probability $0$ to every held-out pair by
construction. Some form of decomposition is therefore *necessary*, not
optional, and the right question is not whether decomposition beats atomic
prediction on average but **what limits compositional generalization once that
ceiling has been removed**.

We audit verb–noun decomposition through four assumptions $\rightarrow$ four
diagnostics, on three datasets (MECCANO, HAViD, IMPACT) and three reference
architectures (atomic, shared encoder, independent recombination — the
latter taken as the strongest realization and the primary diagnosed object):

- **(A1) Visual separability** — Verbs / nouns can be recognized
  independently. Tested by **D1**, the shortcut diagnostic.
- **(A2) Statistical factorizability** — Per-component predictions
  recombine without losing structure. Tested by **D2**, the
  grid-density correlation.
- **(A3) Symmetric per-component difficulty** — Neither component
  dominates the error budget. Tested by **D3**, the per-component
  error decomposition.
- **(A4) Right granularity** — Two-way factoring is the appropriate
  abstraction level. Tested by **D4**, the granularity Pareto sweep.

We find:

1. **Unseen success is mostly coverage interpolation, not free
   recombination.** Per-pair accuracy correlates strongly with each pair's
   training co-occurrence ($\rho{=}0.49{-}0.69$).
2. **Failure concentrates on the larger-vocabulary component**, identifiable
   from vocabulary sizes before any training. IMPACT's verb-heavy vocabulary
   flips the bottleneck from object to verb, confirming the regularity by
   inversion.
3. **The shared encoder forfeits a $2.2{-}6.0\times$ harmonic-mean gap** to
   independent recombination, driven not by capacity (it carries fewer
   parameters yet posts higher seen accuracy) but by **component entanglement**:
   a co-occurrence shortcut that shared training writes into the encoder and
   does not transfer to unseen pairs.

---

## Research questions

The paper follows a phenomenon $\rightarrow$ localization $\rightarrow$
mechanism $\rightarrow$ intervention arc:

| RQ  | Question | Diagnostic | Tests |
|----:|----------|:----------:|:-----:|
| **RQ1** | Is decomposition's unseen accuracy genuine recombination, or interpolation within densely observed regions of the compositional space? | D2 | A2 |
| **RQ2** | Where do failures occur, and is the dominant source of error predictable before training? | D3 | A3 |
| **RQ3** | Why does the field-standard shared encoder generalize worse than independent recombination? | D1 | A1 |
| **RQ4** | Can the limitations be alleviated through alternative factorization granularities, architectures, or compositional objectives? | D4 | A4 |

---

## Contributions

1. **First systematic analysis of verb–noun decomposition under
   compositional shift**, with a portable diagnostic toolkit that connects
   observed behaviour to the assumptions underlying factorized action
   recognition.
2. **A predictable bottleneck identified by vocabulary structure**: the
   larger-vocabulary component dominates unseen failure on every dataset,
   independent of whether that component is the verb or the noun.
3. **Component entanglement explains the shared/independent gap**, gives
   architecture (de-entanglement) as the dominant lever, and yields practical
   guidance for the design of compositional recognition systems.

---

## Framework

The three architectural variants compared throughout the paper:

![Framework](assets/F1_framework.png)

The same input segment feeds all three pipelines. Atomic's output space is a
strict subset of the verb×noun grid, so $P(\text{unseen}){=}0$ for every
unseen pair by construction; shared and independent both span the full grid,
and the paper measures how much of that reachable space each realizes.

---

## Headline results

**Compositional split, harmonic mean of seen and unseen action top-1** (the
strongest comparator per dataset is highlighted):

|              | Atomic | Shared | Independent | C2C-shared | C2C-indep |
|--------------|:------:|:------:|:-----------:|:----------:|:---------:|
| **MECCANO**  |  0.0   |  3.3   |   **19.8**  |   17.9     |   24.3    |
| **HAViD**    |  0.0   | 13.7   |   **30.2**  |   28.9     |   30.7    |
| **IMPACT**   |  0.0   | 10.6   |   **31.2**  |   22.1     |   30.9    |

All numbers use a unified argmax-over-valid-pairs scoring rule
($\arg\max_{(v,n)} \log P(v){+}\log P(n)$).

The paper additionally reports per-component breakdowns, ranking-based top-$K$
recovery, the per-component shortcut diagnostic D1, the grid-density
correlation D2, the per-component error decomposition D3, the granularity
sweep D4, calibration with bootstrap CIs, a per-loss C2C objective ablation,
and the standard-split breakdown — all reproducible from this repository (see
*Reproducing paper exhibits* below).

---

## Repository layout

```
.
├── README.md                  this file
├── LICENSE                    MIT
├── requirements.txt
├── .gitignore
├── assets/                    figures used in this README
│
├── code/                      all training / evaluation / analysis scripts
│   ├── paths.py               centralized path config (override via env vars)
│   │
│   ├── train_*.py             training entry-points
│   ├── eval_*.py /
│   │   comprehensive_eval.py  evaluation passes that emit the JSON files used downstream
│   ├── compute_d1b.py         diagnostic D1 (per-component shortcut)
│   ├── compute_d2d3.py        diagnostics D2 (coverage) and D3 (error decomp)
│   ├── make_tables.py         regenerate paper tables T1–T5
│   ├── make_supp_tables.py    regenerate supplementary tables S1–S14
│   ├── fig_F*.py              regenerate paper figures F2–F6
│   ├── fig_supp.py            regenerate supplementary figures SF1–SF3
│   └── plotstyle.py           shared matplotlib style
│
├── data/
│   ├── README.md              dataset download + preparation instructions
│   ├── splits/                generated split files (populated by the scripts below)
│   │   ├── meccano/  havid/  impact/
│   └── scripts/
│       ├── meccano_comp_split.py     MECCANO compositional split construction
│       ├── havid_comp_split.py       HAViD compositional split construction
│       ├── impact_comp_split.py      IMPACT compositional split construction
│       └── impact_standard_split.py  IMPACT standard split (iterative multilabel)
│
├── configs/                   YAML configs for the named experiments (optional)
└── checkpoints/               drop downloaded pre-trained weights here (see below)
```

---

## Setup

The code was developed with PyTorch 2.x on a single GPU (≥ 40 GB VRAM is
comfortable for MViTv2-S at the published batch / clip sizes).

```bash
conda create -n vn-decomp python=3.10 -y
conda activate vn-decomp

# PyTorch + torchvision for your CUDA build (https://pytorch.org)
pip install torch torchvision   # adapt to your CUDA
pip install -r requirements.txt
```

### Paths

All filesystem paths are read from environment variables (see `code/paths.py`)
with safe relative defaults. Override any of the following before running:

```bash
export PROJECT_DIR=$PWD/outputs       # where checkpoints / results land
export MECCANO_DATA=/path/to/MECCANO  # dataset roots (download separately)
export HAVID_DATA=/path/to/HAViD
export IMPACT_DATA=/path/to/IMPACT
```

Without overrides, outputs land in `./outputs/{checkpoints,results,tables,figures}`
and dataset roots default to `./data/{meccano,havid,impact}`.

---

## Datasets

Three datasets are used. Each has its own release agreement; **only download
links and preparation instructions are provided here, no raw videos are
redistributed.** See [`data/README.md`](data/README.md) for full step-by-step
instructions.

| Dataset | View | Hands | Vocab (V / N) | Compositional split | Download |
|---|---|---|---|---|---|
| **MECCANO** | ego | single | 12 / 23 | 6 held-out (verb, object) pairs | [Project page (TBA)](#dataset-download-links) |
| **HAViD**   | 3 static + ego | bimanual | 11 / 29 (+ target 25, tool 5) | 13 held-out (verb, manip-object) pairs | [Project page (TBA)](#dataset-download-links) |
| **IMPACT**  | 5 (4 static + ego) | bimanual | 22 / 18 | 20 held-out pairs per hand | [Project page (TBA)](#dataset-download-links) |

### Two evaluation regimes

- **Standard split** — official train / val / test partition (or, for
  IMPACT, a recording-level hand-stratified partition we construct via
  iterative multilabel stratification). Every action class seen at test
  time has appeared in training; this is the routine setting on which
  decomposition's compositional behaviour is hidden.
- **Compositional split** — a set of (verb, object) pairs is held out
  from training, while every held-out verb and held-out object continues
  to appear paired with other partners. This is the regime the paper
  studies.

Compositional splits are reproduced exactly by the construction scripts in
`data/scripts/` (the held-out pair lists are hard-coded under fixed
quantitative constraints; no randomness):

```bash
python data/scripts/meccano_comp_split.py
python data/scripts/havid_comp_split.py
python data/scripts/impact_comp_split.py
python data/scripts/impact_standard_split.py   # IMPACT only (standard split)
```

### Dataset download links

> 🚧 **To be added before the camera-ready release.** Each dataset has
> independent licensing terms; the camera-ready version of this README
> will link directly to each dataset's official project page.

---

## Reproducing the paper exhibits

The pipeline is: **train → evaluate → JSON results → tables / figures**.

### 1. Train

```bash
# MECCANO compositional split — shared encoder (2-head verb + object)
python code/train_meccano_comp.py --arch shared --epochs 50 --seed 42

# MECCANO atomic on the compositional split
python code/train_meccano_comp_atomic.py --epochs 50 --seed 42

# IMPACT independent recombination (per-component heads, trained separately)
python code/train_impact_comp_variants.py --view front --hand left --mode verb_only
python code/train_impact_comp_variants.py --view front --hand left --mode noun_only

# HAViD k=2 multi-head shared backbone (verb + manipulated object)
python code/train_havid_granularity.py --config k2 --subtask view0_lh_aa

# C2C objective on the shared encoder (--dataset {meccano,impact,havid})
python code/train_c2c_shared.py --dataset meccano --epochs 100

# C2C-indep: two-stage on already-trained Independent backbones
python code/train_c2c_indep.py --dataset meccano --epochs 200
```

Every script accepts `--help` and lists its required arguments.

### 2. Evaluate

```bash
# Comprehensive evaluation pass (aggregates -> JSON used by D1/D2/D3 + S6)
python code/comprehensive_eval.py --dataset meccano --tag meccano_indep
python code/comprehensive_eval.py --dataset impact  --view front --hand left
python code/comprehensive_eval.py --dataset havid   --subtask view0_lh_aa

# Action-level argmax-pair scoring used by the paper anchor (T2)
python code/eval_impact_indep_action.py --view front --hand left
python code/eval_meccano_indep_action.py

# HAViD top-K with the same argmax-pair scoring
python code/eval_havid_topk_F6.py
```

Each evaluation writes a small JSON under `${RESULTS_DIR}/<tag>/`.

### 3. Tables and figures

```bash
python code/make_tables.py            # T1..T5 -> ${TABLES_DIR}/T*.tex
python code/make_supp_tables.py       # S1..S14 -> ${TABLES_DIR}/S*.tex

python code/fig_F2_density.py         # F2 coverage scatter
python code/fig_F3_bottleneck.py      # F3 component bottleneck
python code/fig_F4_F5.py              # F4 granularity + F5 interventions
python code/fig_F6_topk.py            # F6 top-K
python code/fig_supp.py               # SF1..SF3
```

PDF figures land in `${FIGURES_DIR}/`.

---

## Pre-trained checkpoints

We release **142 pre-trained checkpoints** (one `best.pth` per dataset ×
configuration × view × hand) covering every experiment reported in the paper.
Together they consume ~49 GB; the full inventory and per-checkpoint role
mapping lives in `MANIFEST.json` inside the bundle.

> 📦 **Checkpoint bundle (Hugging Face Hub)**: 🚧 *Link to be added before
> camera-ready.* The bundle is hosted as a Hugging Face model repository
> (anonymous account during the review period).

### Download

```bash
# Option 1: huggingface_hub (recommended)
pip install huggingface_hub
huggingface-cli download <ANONYMOUS_REPO_ID> \
    --repo-type model --local-dir checkpoints/

# Option 2: git-lfs
git lfs install
git clone https://huggingface.co/<ANONYMOUS_REPO_ID> checkpoints/
```

After download, `checkpoints/` will contain one directory per configuration
tag, mirroring the layout the scripts expect — no path rewriting is needed.

### Expected layout

```
checkpoints/
├── MANIFEST.json                            inventory + per-ckpt role
├── meccano_comp_atomic/best.pth             MECCANO atomic, compositional split
├── meccano_comp_shared/best.pth             MECCANO shared 2-head
├── meccano_comp_indep_{verb,noun}/best.pth  MECCANO Independent component heads
├── meccano_mvitv2_{action,multihead}/       MECCANO standard split
├── c2c_{shared,indep}_meccano/              MECCANO C2C variants
├── c2c_shared_meccano_abl_*/                S8 C2C objective ablation
├── meccano_comp_shared_s{1,7}/              S8(b) multi-seed
├── impact_{view}_{hand}_{action,multihead}/ IMPACT standard split (10 configs)
├── impact_comp_{view}_{hand}/               IMPACT shared comp
├── impact_compvar_{action,verb_only,noun_only}_*/  IMPACT atomic + per-component
├── c2c_{shared,indep}_impact_{view}_{hand}/        IMPACT C2C variants
├── havid_gran_k{1,2,3}_<subtask>/           HAViD granularity (k=4 results JSON-only)
├── havid_gran_solo_{verb,manip}_<subtask>/  HAViD solo heads (D1 diagnostic)
├── c2c_{shared,indep}_havid_<subtask>/      HAViD C2C variants
└── mvitv2_comp_<subtask>/                   HAViD shared multi-head backbone
```

Configuration tags exactly match the experiment names used by the training
and evaluation scripts. `<view> ∈ {front, left, right, top, ego}`,
`<hand> ∈ {left, right}`, `<subtask> ∈ {view0_lh_aa, view0_rh_aa,
view1_lh_aa, view1_rh_aa, view2_lh_aa, view2_rh_aa}`.

### What's *not* in the bundle

- HAViD `k=4` granularity checkpoints (`havid_gran_k4_*`): the underlying
  ckpts were cleaned after the per-configuration evaluation JSONs were
  produced. The `havid_gran_k4_*/comp_eval.json` numbers used in Table T5
  remain reproducible from the per-config JSON, but the k=4 model cannot be
  re-instantiated for fresh inference; rerun
  `python code/train_havid_granularity.py --config k4 ...` to regenerate.

---

## Method ingredients (one-paragraph reference)

We use a single backbone family (MViTv2-S, Kinetics-400 pretrained) and a
linear classifier head per component, so every reported difference reflects
the paradigm rather than the encoder. Input clips are decoded into 16 RGB
frames at 224² resolution with standard augmentation. Optimization uses AdamW
with cosine schedule after linear warm-up. Atomic / shared / k-way /
independent backbones train for 50 epochs on MECCANO and IMPACT and 100 on
HAViD; the C2C variants follow the protocol of the original C2C method
(C2C-shared 100 epochs end-to-end; C2C-indep two-stage, training a
lightweight head on top of frozen Independent backbones for 200 head-only
epochs). No unseen pair ever enters training (the compositional split removes
them from `comp_train` by construction); only checkpoint selection varies
between configurations — atomic and per-component Indep heads use seen-only
validation, while joint shared multi-head models and both C2C variants use
the harmonic mean of seen and unseen action top-1, the standard CZSL
selection protocol on splits without a separate held-out test partition.

---

## Citing

Citation block will be added in the camera-ready release. For the review
period, please refer to the submission with the anonymous title:

> *Diagnosing Verb–Noun Decomposition in Assembly Action Recognition: A
> Cross-Dataset Study of Compositional Generalization.*

---

## License

The code in this repository is released under the MIT license; see
[LICENSE](LICENSE). The three datasets remain under their original licenses
and must be obtained from their respective authors.

# Reachability Is Not Generalization
## Understanding Verb–Noun Decomposition in Assembly Action Recognition

> Anonymous code release for double-blind review.
> Author identities will be added in the camera-ready version.

This repository contains the code, generated splits, and analysis pipelines
that reproduce every number, table, and figure of the paper.

---

## Abstract

Assembly actions are compositional: they combine a manipulation with a part
or tool. In deployment, systems routinely encounter novel combinations of
familiar components, yet an atomic action classifier assigns every unseen
combination exactly zero probability by construction. The prevailing
solution is verb–noun decomposition, which predicts components separately
and recombines them to reach unseen actions. While widely adopted, how
decomposition generalizes under compositional shift remains poorly
understood. We present a systematic analysis of verb–noun decomposition
across three assembly datasets (MECCANO, HAViD, and IMPACT). Although
decomposition escapes the *atomic ceiling*, its generalization extends only
partially beyond it. Unseen-composition performance remains strongly tied to
the co-occurrence structure of the training data, indicating that much of
the observed gain arises from interpolation within densely supported regions
of the compositional space rather than from unconstrained recombination.
Across datasets, failures consistently concentrate on the larger-vocabulary
component, and IMPACT's verb-heavy vocabulary reverses the bottleneck from
nouns to verbs. We further show that shared-encoder training introduces
component entanglement, encouraging reliance on co-occurrence patterns that
transfer poorly to unseen compositions and trailing independent
recombination by up to $6.0\times$ in harmonic mean. Taken together, these
findings explain why decomposition achieves only partial compositional
generalization in practice. By identifying **primitive support**,
**vocabulary asymmetry**, and **component entanglement** as connected
sources of error, we provide a portable diagnostic framework for studying
compositional recognition beyond aggregate accuracy.

---

## Highlights

**The atomic ceiling makes decomposition structurally necessary.** Atomic
action classifiers assign probability zero to every unseen (verb, noun)
combination by construction. Some form of decomposition is therefore not
optional but required. The right question is not whether decomposition beats
atomic prediction on average, but **what limits compositional generalization
once that ceiling has been removed**.

We organize the analysis around four research questions that proceed from
phenomenon to localization, mechanism, and intervention:

- **RQ1 — Recombination or interpolation?**
  Does the unseen-composition accuracy of decomposition reflect genuine
  recombination, or interpolation within densely observed regions of the
  composition space?
- **RQ2 — Where do failures occur, and can the bottleneck be predicted?**
  When decomposition errs on a novel composition, which component fails,
  and is the dominant source of error predictable before training?
- **RQ3 — Why does the shared encoder generalize worse?**
  The field-standard shared encoder consistently trails independent
  recombination. What mechanism drives this gap, and is it a capacity issue
  or something else?
- **RQ4 — Can the limitation be engineered away?**
  Can changing the factorization granularity, the architecture, or the
  training objective alleviate these limitations, or is the asymmetry
  structural?

Our analysis surfaces three connected sources of error:

1. **Primitive support drives unseen performance.** Per-pair unseen accuracy
   correlates strongly with the training co-occurrence of its constituent
   primitives ($\rho = 0.49$ to $0.69$, rising with grid density). Much of
   decomposition's apparent compositional ability is interpolation from
   dense coverage rather than free recombination.
2. **Vocabulary asymmetry produces a predictable bottleneck.** Unseen
   failures concentrate on the larger-vocabulary component on every dataset,
   identifiable from vocabulary sizes alone before training. IMPACT's
   verb-heavy vocabulary reverses the bottleneck from nouns to verbs,
   overturning the common belief that the object is intrinsically the hard
   part.
3. **Component entanglement explains the shared-vs-independent gap.**
   Shared training adds a non-transferable co-occurrence shortcut that
   improves neither seen accuracy nor unseen transfer, leaving the harmonic
   mean trailing independent recombination by up to $6.0\times$. The gap is
   mechanistic, not a capacity deficit, and identifies architecture
   (de-entanglement) as the dominant lever.

These three findings turn into concrete, dataset-predictable design levers:
de-entangle the encoders, allocate capacity to the larger-vocabulary head,
keep the two-way factoring, and calibrate on unseen compositions.

---

## Framework

The three recognition formulations studied throughout the paper:

![Framework](assets/assembly-framework.png)

Top: *atomic prediction*; middle: *shared decomposition*; bottom:
*independent decomposition*. Given an input clip ($T = 16$ frames), atomic
prediction treats each action as a single class and predicts directly over
the training action vocabulary $\mathcal{A}_{\text{train}}$. Shared
decomposition predicts verbs and nouns using separate heads on a shared
backbone, whereas independent decomposition trains one backbone per
component and combines predictions through $\log P(v)+\log P(n)$. The
coloured output boxes indicate the reachable output space of each
formulation. Atomic prediction is restricted to actions observed during
training, whereas both decomposition variants span the full
verb $\times$ noun composition grid and can therefore assign probability to
unseen compositions.

---

## Headline results

**Compositional split, harmonic mean of seen and unseen action top-1** (the
strongest comparator per dataset is highlighted):

|              | Atomic | Shared | Independent | C2C-shared | C2C-indep |
|--------------|:------:|:------:|:-----------:|:----------:|:---------:|
| **MECCANO**  |  0.0   |  3.3   |   **19.8**  |   17.9     |   24.3    |
| **HAViD**    |  0.0   | 13.7   |   **30.2**  |   28.9     |   30.7    |
| **IMPACT**   |  0.0   | 10.6   |   **31.2**  |   22.1     |   30.9    |

All numbers use a unified argmax-over-valid-pairs scoring rule:

$$\arg\max_{(v,n)\,\in\,\mathcal{V}\times\mathcal{N}} \bigl[\log P(v) + \log P(n)\bigr]$$

The paper additionally reports per-component breakdowns, ranking-based
top-$K$ recovery, the per-component shortcut analysis (component
entanglement), the grid-density correlation (primitive support), the
per-component error decomposition (vocabulary asymmetry), the granularity
sweep, calibration with bootstrap CIs, a per-loss compositional-objective
ablation, and the standard-split breakdown — all reproducible from this
repository (see *Reproducing the paper exhibits* below).

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
│   ├── compute_d1b.py         per-component shortcut / entanglement analysis
│   ├── compute_d2d3.py        primitive-support correlation + per-component error decomposition
│   ├── make_tables.py         regenerate paper tables
│   ├── make_supp_tables.py    regenerate supplementary tables
│   ├── fig_F*.py              regenerate paper figures
│   ├── fig_supp.py            regenerate supplementary figures
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

The paper uses three assembly datasets. Each has its own release agreement;
**this repository does not redistribute raw videos**. Download each dataset
from its original release page, then run the construction scripts in
`data/scripts/` to reproduce the standard / compositional splits used in the
paper (see [`data/README.md`](data/README.md) for the full step-by-step).

| Dataset | View | Hands | Vocab (V / N) | Original release |
|---|---|---|---|---|
| **MECCANO** | ego | single | 12 / 23 | <https://iplab.dmi.unict.it/MECCANO/> |
| **HA-ViD** | 3 static + ego | bimanual | 11 / 29 | <https://iai-hrc.github.io/ha-vid> · code: <https://github.com/iai-hrc/ha-vid> |
| **IMPACT** | 5 (4 static + ego) | bimanual | 22 / 18 | <https://github.com/Kratos-Wen/IMPACT> · HF mirror: `KratosWen/IMPACT` |

> ℹ️ The dataset URLs above point to the original authors' releases.
> Please obtain the data under their licence terms — we do not host or
> mirror any video files. The pair-level split files that the paper
> reports are reproduced by running the four scripts in
> `data/scripts/` after the raw data is in place.

### Two evaluation regimes

- **Standard split** — the official train / val / test partition, or, for
  IMPACT, a recording-level hand-stratified partition built via iterative
  multilabel stratification. Every action class seen at test time has
  appeared in training; this is the routine setting that hides
  decomposition's compositional behaviour.
- **Compositional split** — a fixed set of (verb, object) pairs is held out
  from training. Every held-out verb and held-out object continues to appear
  paired with other partners, so the task is recombination of familiar
  primitives rather than discovery of new ones. This is the regime the
  paper studies.

Generate every split deterministically (no randomness; held-out pair lists
are hard-coded in the scripts):

```bash
python data/scripts/meccano_comp_split.py
python data/scripts/havid_comp_split.py
python data/scripts/impact_standard_split.py   # IMPACT only
python data/scripts/impact_comp_split.py
```

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
# Comprehensive evaluation pass (aggregates -> JSON used by analysis below)
python code/comprehensive_eval.py --dataset meccano --tag meccano_indep
python code/comprehensive_eval.py --dataset impact  --view front --hand left
python code/comprehensive_eval.py --dataset havid   --subtask view0_lh_aa

# Action-level argmax-pair scoring used by the paper anchor table
python code/eval_impact_indep_action.py --view front --hand left
python code/eval_meccano_indep_action.py

# HAViD top-K with the same argmax-pair scoring
python code/eval_havid_topk_F6.py
```

Each evaluation writes a small JSON under `${RESULTS_DIR}/<tag>/`.

### 3. Tables and figures

```bash
python code/make_tables.py            # paper tables -> ${TABLES_DIR}/T*.tex
python code/make_supp_tables.py       # supplementary tables -> ${TABLES_DIR}/S*.tex

python code/fig_F2_density.py         # primitive-support scatter
python code/fig_F3_bottleneck.py      # component-bottleneck breakdown
python code/fig_F4_F5.py              # granularity sweep + interventions
python code/fig_F6_topk.py            # top-K recovery
python code/fig_supp.py               # supplementary figures
```

PDF figures land in `${FIGURES_DIR}/`.

---

## Pre-trained checkpoints

We release **142 pre-trained checkpoints** (one `best.pth` per dataset ×
configuration × view × hand) covering every experiment reported in the paper.
Together they consume ~49 GB, distributed as six ~9 GB tar shards
(`checkpoints.tar.part00` … `checkpoints.tar.part05`); the full inventory and
per-checkpoint role mapping lives in `MANIFEST.json` inside the bundle.

### Download

🤗 **Hugging Face Hub**: <https://huggingface.co/<ANONYMOUS_REPO_ID>>

Fetch the shards from the Hub, then reassemble and extract in one stream:

```bash
pip install huggingface_hub
huggingface-cli download <ANONYMOUS_REPO_ID> \
    --repo-type model --local-dir checkpoints/

# Reassemble + extract in one stream (no 49 GB intermediate file)
cd checkpoints/
cat checkpoints.tar.part* | tar -xf -

# Optional: remove the shards once extraction succeeds
rm checkpoints.tar.part*
```

After extraction, `checkpoints/` contains one directory per configuration
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
├── c2c_shared_meccano_abl_*/                MECCANO C2C objective ablation
├── meccano_comp_shared_s{1,7}/              MECCANO multi-seed
├── impact_{view}_{hand}_{action,multihead}/ IMPACT standard split (10 configs)
├── impact_comp_{view}_{hand}/               IMPACT shared comp
├── impact_compvar_{action,verb_only,noun_only}_*/  IMPACT atomic + per-component
├── c2c_{shared,indep}_impact_{view}_{hand}/        IMPACT C2C variants
├── havid_gran_k{1,2,3}_<subtask>/           HAViD granularity
├── havid_gran_solo_{verb,manip}_<subtask>/  HAViD solo heads (entanglement analysis)
├── c2c_{shared,indep}_havid_<subtask>/      HAViD C2C variants
└── mvitv2_comp_<subtask>/                   HAViD shared multi-head backbone
```

Configuration tags exactly match the experiment names used by the training
and evaluation scripts. `<view> ∈ {front, left, right, top, ego}`,
`<hand> ∈ {left, right}`, `<subtask> ∈ {view0_lh_aa, view0_rh_aa,
view1_lh_aa, view1_rh_aa, view2_lh_aa, view2_rh_aa}`.

---

## Citing

Citation block will be added in the camera-ready release. For the review
period, please refer to the submission with the anonymous title:

> *Reachability Is Not Generalization: Understanding Verb–Noun Decomposition
> in Assembly Action Recognition.*

---

## License

The code in this repository is released under the MIT license; see
[LICENSE](LICENSE). The three datasets remain under their original licenses
and must be obtained from their respective authors.

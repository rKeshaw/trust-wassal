# WASSAL: Wasserstein-Distance-Based Targeted Active Learning

This repository contains the code and experiment pipeline for **WASSAL**, a
targeted active learning strategy that selects samples from an unlabeled pool
by minimizing the Sinkhorn (entropy-regularized Wasserstein) distance between
class-conditional query distributions and a learned simplex weighting over the
unlabeled pool. WASSAL is evaluated on class-imbalanced multiclass image
classification against standard active learning baselines (BADGE, GLISTER,
GradMatch-TSS, CoreSet, uncertainty/margin/least-confidence sampling, random).

The codebase is a research fork of [TRUST](https://github.com/decile-team/trust)
(targeted subset selection toolkit) and uses
[DISTIL](https://github.com/decile-team/distil) for the baseline AL strategies.

## Installation

Python 3.9+ with a CUDA-capable GPU is required.

```bash
git clone <this-repository>
cd trust-wassal
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/requirements.txt
```

The requirements file pins all dependencies, including `decile_distil` and
`submodlib` installed directly from their GitHub repositories.

## Datasets

All datasets are placed under `data/` in the repository root (gitignored).

| Dataset | Setup |
|---------|-------|
| CIFAR-10 | Downloaded automatically via torchvision on first run |
| SVHN | Downloaded automatically via torchvision on first run |
| PneumoniaMNIST | Downloaded automatically via the bundled MedMNIST loader |
| STL-10 | Run `python prepare_stl10_channel_first.py` to download STL-10 and produce `data/stl10_raw_channel_first.npz` |
| Caltech-101 | Run `python prepare_caltech.py` to download via TensorFlow Datasets and produce a stratified 90/10 split as 32x32 images in `data/caltech101_raw_channel_first.npz` (requires `tensorflow-datasets`) |

## Repository layout

| Path | Contents |
|------|----------|
| `trust/strategies/` | Selection strategies. `wassal_multiclass.py` is the primary WASSAL implementation (used by CIFAR-10, SVHN, STL-10, Caltech-101). `wassal_multiclass_v2.py` is the variant used by PneumoniaMNIST. |
| `trust/utils/` | Dataset loaders (`custom_dataset.py`, `custom_dataset_medmnist.py`), models (`models/resnet*.py`), statistics (`CalcStatistics_*.py`), plotting, and `paths.py` (repo-relative path constants) |
| `tutorials/All_Wassal/` | One experiment driver per dataset (see below) |
| `scripts/` | Result auditing and analysis tools |
| `*.sh` (repo root) | Experiment launchers that parallelize strategy/budget subsets across GPUs |

### Canonical experiment drivers

| Dataset | Driver script | WASSAL implementation |
|---------|--------------|----------------------|
| CIFAR-10 | `tutorials/All_Wassal/wassal_cifar10_multiclass_vanilla2.py` | `wassal_multiclass.py` |
| SVHN | `tutorials/All_Wassal/wassal_svhn_multiclass_vanilla.py` | `wassal_multiclass.py` |
| PneumoniaMNIST | `tutorials/All_Wassal/wassal_pneumonia_multiclass_vanilla.py` | `wassal_multiclass_v2.py` |
| STL-10 | `tutorials/All_Wassal/wassal_stl10.py` | `wassal_multiclass.py` |
| Caltech-101 | `tutorials/All_Wassal/wassal_caltech_multiclass_vanilla2.py` | `wassal_multiclass.py` |

## Running experiments

Every driver takes six positional arguments:

```
python3 -u <driver.py> "<SKIP_STRATEGIES>" "<SKIP_METHODS>" "<SKIP_BUDGETS>" <DEVICE_ID> <EXPERIMENT_NAME> <SOFT_LOSS_HYPERPARAM>
```

- `SKIP_STRATEGIES` / `SKIP_METHODS` / `SKIP_BUDGETS` — space-separated lists
  of strategy groups, methods, and budgets to *skip*, allowing complementary
  subsets to run in parallel on different GPUs.
- `DEVICE_ID` — CUDA device index.
- `EXPERIMENT_NAME` — top-level results folder (`onlywassal`, `onlyal`,
  `inpaper`, ...).
- `SOFT_LOSS_HYPERPARAM` — weight of the soft-simplex loss term for the
  `*_WITHSOFT` variants.

Example — run only WASSAL on CIFAR-10 at all budgets on GPU 0:

```bash
python3 -u tutorials/All_Wassal/wassal_cifar10_multiclass_vanilla2.py \
  "random AL AL_WITHSOFT WASSAL_WITHSOFT" "WASSAL" "" 0 onlywassal 0.3
```

The shell launchers in the repository root (`cifar10WASSALonly1.sh`,
`caltechwassal.sh`, `wassalonlystl1.sh`, ...) encode the strategy/budget/GPU
splits used for the paper. See `REPRODUCE.md` for the full reproduction
recipe.

## Results layout

Results are written under `tutorials/results/` (gitignored due to size) with
the convention:

```
tutorials/results/{experiment_name}/{dataset}/classimb/rounds{N}/{strategy}/{budget}/{expK}/
    results_{method}_{budget}.json   # per-round test accuracy, per-class selections
    results_{method}_{budget}.csv    # per-class accuracy per round
```

`exp1`..`exp4` are independent runs with fixed seeds (`exp2/exp3/exp4` use
seeds 48/86/28). The curated paper results live under
`tutorials/results/inpaper/`.

## Reproducing paper tables and figures

```bash
# Completeness audit of the results tree
python scripts/audit_results.py

# Per-dataset statistics (mean/std gain tables, plots, LaTeX)
python trust/utils/CalcStatistics_cifar10.py
python trust/utils/CalcStatistics_svhn.py
python trust/utils/CalcStatistics_pneumonia.py
python trust/utils/CalcStatistics_stl10.py
python trust/utils/CalcStatistics_caltech.py
```

## Sanity check

A lightweight smoke test runs the WASSAL selection loop on dummy data:

```bash
python test_wassal_smoke.py
```

## Hardware

Experiments were run on NVIDIA GPUs (one experiment process per GPU). A full
strategy x budget sweep for one dataset and one seed takes on the order of a
GPU-day; the shell launchers split the sweep across devices.

## Acknowledgments

This library builds on [TRUST](https://github.com/decile-team/trust),
[DISTIL](https://github.com/decile-team/distil), and
[Submodlib](https://github.com/decile-team/submodlib) by the DECILE team.

## References

[1] Kothawade S, Kaushal V, Ramakrishnan G, Bilmes J, Iyer R. PRISM: A Rich
Class of Parameterized Submodular Information Measures for Guided Subset
Selection. AAAI 2022.

[2] Iyer R, Khargoankar N, Bilmes J, Asanani H. Submodular combinatorial
information measures with applications in machine learning. ALT 2021.

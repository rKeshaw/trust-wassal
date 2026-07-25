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
Path helpers live in `trust/utils/paths.py` (`REPO_ROOT`, `RESULTS_DIR`,
`DATA_DIR`).

| Dataset | Setup |
|---------|-------|
| CIFAR-10 | Downloaded automatically via torchvision on first run |
| SVHN | Downloaded automatically via torchvision on first run |
| PneumoniaMNIST | Downloaded automatically via the bundled MedMNIST loader |
| STL-10 | `python prepare_stl10_channel_first.py` → `data/stl10_raw_channel_first.npz` |
| Caltech-101 | `python prepare_caltech.py` → `data/caltech101_train_90_32.npz` and `data/caltech101_test_10_32.npz` (requires `tensorflow-datasets`) |

## Repository layout

| Path | Contents |
|------|----------|
| `trust/strategies/` | Selection strategies. `wassal_multiclass.py` is the primary WASSAL implementation (CIFAR-10, SVHN, STL-10, Caltech-101). `wassal_multiclass_v2.py` is used by PneumoniaMNIST. |
| `trust/utils/` | Dataset loaders, models (`models/resnet*.py`), statistics (`CalcStatistics_*.py`), plotting, and `paths.py` |
| `tutorials/All_Wassal/` | One experiment driver per dataset (see below) |
| `scripts/` | Result merging, auditing, and long-running launch helpers |
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
  of strategy groups, methods, and budgets to *skip*, so complementary
  subsets can run in parallel on different GPUs.
- `DEVICE_ID` — CUDA device index.
- `EXPERIMENT_NAME` — top-level results folder (`onlywassal`, `onlyal`,
  `inpaper`, ...).
- `SOFT_LOSS_HYPERPARAM` — weight of the soft-simplex loss term for the
  `*_WITHSOFT` variants (paper default: `0.3`).

Example — run only WASSAL on CIFAR-10 at all budgets on GPU 0:

```bash
python3 -u tutorials/All_Wassal/wassal_cifar10_multiclass_vanilla2.py \
  "random AL AL_WITHSOFT WASSAL_WITHSOFT" "WASSAL" "" 0 onlywassal 0.3
```

Long GPU jobs should be launched inside **tmux** (or equivalent) so they
survive closing the IDE/SSH session. For the Caltech-101 paper rerun:

```bash
bash scripts/run_caltech_tmux.sh
tmux attach -t caltech-rerun   # Ctrl-b d to detach
```

See `REPRODUCE.md` for the full per-dataset launcher list and the experiment
matrix.

## Results layout

Results are written under `tutorials/results/` (gitignored due to size):

```
tutorials/results/{experiment_name}/{dataset}/classimb/rounds{N}/{strategy}/{budget}/{expK}/
    *_{method}_budget:{budget}_rounds:{N}_runs_{expK}.json
    *_{method}_budget:{budget}_rounds:{N}_runs_{expK}.csv
```

`exp2`/`exp3`/`exp4` are the paper seeds (48 / 86 / 28). Curated paper
results live under `tutorials/results/inpaper/`. Assemble / refresh that tree
with:

```bash
python scripts/merge_inpaper.py
python scripts/audit_results.py
```

### Paper-results status

| Dataset | Status |
|---------|--------|
| CIFAR-10 | Complete (11 strategies × 7 budgets × exp2–4) |
| SVHN | Complete (`WASSAL_WITHSOFT` + AL baselines × 5 budgets × exp2–4) |
| PneumoniaMNIST | Complete (`WASSAL_WITHSOFT` + AL baselines × 9 budgets × exp2–4) |
| STL-10 | Merged into `inpaper/` from `onlywassal` + `onlyal` |
| Caltech-101 | Multi-seed rerun (exp2–4, budgets including 175) in progress |

## Reproducing paper tables and figures

```bash
python scripts/audit_results.py

python trust/utils/CalcStatistics_cifar10.py
python trust/utils/CalcStatistics_svhn.py
python trust/utils/CalcStatistics_pneumonia.py
python trust/utils/CalcStatistics_stl10.py
python trust/utils/CalcStatistics_caltech.py   # after Caltech rerun finishes
```

Each script writes `_allclasses.csv`, `.png`, and `.tex` next to the results
it reads.

## Sanity check

```bash
python test_wassal_smoke.py
```

Runs `WASSAL_Multiclass.select_only_for_query` on small class-structured
dummy data and validates the selection and per-class simplexes.

## Hardware

Experiments were run on NVIDIA GPUs (one process per GPU). A full
strategy × budget sweep for one dataset and one seed is on the order of a
GPU-day; the shell launchers and `scripts/run_caltech_tmux.sh` split the
sweep across devices.

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

# Reproducing the WASSAL Experiments

This document lists the exact commands used to produce the results reported
in the paper. All commands are run from the repository root with the
virtual environment activated (`source .venv/bin/activate`).

## 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/requirements.txt
```

## 2. Data preparation

```bash
# CIFAR-10, SVHN, PneumoniaMNIST download automatically on first run.

# STL-10 -> data/stl10_raw_channel_first.npz
python prepare_stl10_channel_first.py

# Caltech-101 -> data/caltech101_raw_channel_first.npz (needs tensorflow-datasets)
python prepare_caltech.py
```

## 3. Experiment matrix

Strategies (11): `WASSAL`, `WASSAL_WITHSOFT`, and the AL baselines
`badge`, `us`, `glister`, `gradmatch-tss`, `coreset`, `leastconf`, `margin`,
`random` (plus their `_withsoft` variants where applicable).

| Dataset | Budgets | Rounds | Seeds (runs) |
|---------|---------|--------|--------------|
| CIFAR-10 | 25 50 100 125 150 175 200 | 10 | exp2-4 (48/86/28) |
| SVHN | 25 50 100 175 200 | 10 | exp2-4 |
| PneumoniaMNIST | 20-100 step 10 | 10 | exp2-4 |
| STL-10 | 25 50 100 125 150 175 200 | 10 | exp1-4 |
| Caltech-101 | 25 50 100 175 200 | 8 | exp2-4 |

## 4. Running the sweeps

Each launcher runs the complement of its skip lists, so complementary
launchers can run concurrently on different GPUs. Edit `DEVICE_ID` inside
each script to match your machine.

```bash
# CIFAR-10
bash cifar10WASSALonly1.sh      # WASSAL
bash cifar10WASSALSOFTonly1.sh  # WASSAL_WITHSOFT
bash cifar10ALonly.sh           # AL baselines

# SVHN
bash svhnWASSALonly1.sh
bash svhnWASSALSOFTonly1.sh
bash svhn10ALonly.sh

# PneumoniaMNIST
bash pneumoWASSALOnly1.sh
bash pneumoWASSALSOFTOnly1.sh
bash pneumoAL.sh

# STL-10
bash wassalonlystl1.sh          # WASSAL
bash wassalsoftstl.sh           # WASSAL_WITHSOFT
bash wassalonlystl2.sh          # AL baselines

# Caltech-101 (multi-seed rerun, all strategies)
bash caltech_rerun_all.sh       # see section 5
```

Results accumulate under
`tutorials/results/{experiment_name}/{dataset}/classimb/rounds{N}/...`.

## 5. Caltech-101 multi-seed rerun

`caltech_rerun_all.sh` runs all 11 strategies for seeds exp2-4 at budgets
25/50/100/175/200 (8 AL rounds). It accepts an optional GPU id:

```bash
bash caltech_rerun_all.sh 0
```

## 6. Assembling the paper results tree

STL-10 results were produced in the `onlywassal` and `onlyal` experiment
folders; merge them into the unified `inpaper` tree:

```bash
python scripts/merge_stl10_inpaper.py
```

Verify completeness of all five datasets:

```bash
python scripts/audit_results.py
```

## 7. Tables and figures

```bash
python trust/utils/CalcStatistics_cifar10.py
python trust/utils/CalcStatistics_svhn.py
python trust/utils/CalcStatistics_pneumonia.py
python trust/utils/CalcStatistics_stl10.py
python trust/utils/CalcStatistics_caltech.py
```

Each script writes `output_statistics_*_allclasses.csv`, a `.png` plot, and a
`.tex` LaTeX table next to the results it reads.

## 8. Smoke test

```bash
python test_wassal_smoke.py
```

Runs the WASSAL selection loop end-to-end on dummy data (CPU) and reports
Wasserstein distances before/after selection.

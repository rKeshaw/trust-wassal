# Reproducing the WASSAL Experiments

Exact commands used to produce the results reported in the paper. Run
everything from the repository root with the virtual environment activated
(`source .venv/bin/activate`).

**Long GPU jobs:** prefer `tmux` (or `screen`) so processes survive closing
SSH / the IDE. Do not rely on Cursor/IDE background terminals for multi-day
sweeps.

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

# Caltech-101 -> data/caltech101_{train_90,test_10}_32.npz
# (needs tensorflow-datasets)
python prepare_caltech.py
```

## 3. Experiment matrix

Strategies (11): `WASSAL`, `WASSAL_WITHSOFT`, and the AL baselines
`badge`, `us`, `glister`, `gradmatch-tss`, `coreset`, `leastconf`, `margin`,
`random` (plus their `_withsoft` variants where applicable). Soft-loss
hyperparameter used for paper runs: `0.3`.

| Dataset | Budgets | Rounds | Seeds (runs) | Paper status |
|---------|---------|--------|--------------|--------------|
| CIFAR-10 | 25 50 100 125 150 175 200 | 10 | exp2–4 (48/86/28) | Complete in `inpaper/` |
| SVHN | 25 50 100 175 200 | 10 | exp2–4 | Complete (`WASSAL_WITHSOFT` + AL) |
| PneumoniaMNIST | 20–100 step 10 | 10 | exp2–4 | Complete (`WASSAL_WITHSOFT` + AL) |
| STL-10 | 25 50 100 125 150 175 200 | 10 | exp2–4 | Complete in `inpaper/` |
| Caltech-101 | 25 50 100 175 200 | 8 | exp2–4 | Complete in `inpaper/` |

STL-10 also has an `exp1` seed on disk for most cells. It is excluded from the
paper because it was trained from a different initial model, so all five
datasets are reported over the same three seeds.

## 4. Running the sweeps

Each launcher runs the complement of its skip lists, so complementary
launchers can run concurrently on different GPUs. Edit `DEVICE_ID` inside
each script to match your machine. For multi-day jobs, wrap with tmux:

```bash
tmux new -s cifar-wassal
bash cifar10WASSALonly1.sh
# Ctrl-b d to detach
```

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
```

Results accumulate under
`tutorials/results/{experiment_name}/{dataset}/classimb/rounds{N}/...`.

## 5. Caltech-101 multi-seed rerun

Recommended: two-GPU split inside a detached tmux session.

```bash
bash scripts/run_caltech_tmux.sh
# session name: caltech-rerun
#   window gpu0 -> budgets 25 / 100 / 200
#   window gpu1 -> budgets 50 / 175

tmux attach -t caltech-rerun          # watch live (Ctrl-b 0/1, Ctrl-b d)
tail -f tutorials/results/inpaper/caltech101/caltech_rerun_gpu0.log
```

Single-GPU alternative (slower):

```bash
bash caltech_rerun_all.sh 0
```

The driver loops `experiments=["exp2","exp3","exp4"]` with seeds
`[48, 86, 28]` and budgets `[25, 50, 100, 175, 200]` (8 AL rounds). Results
are written under `tutorials/results/inpaper/caltech101/`.

When the rerun finishes:

```bash
python scripts/audit_results.py
python trust/utils/CalcStatistics_caltech.py
```

### Re-running a single cell

`wassal_stl10.py` takes three optional arguments after the soft-loss
hyperparameter — experiments, seeds, and budgets — which narrow the sweep to a
single cell. Point `experiment_name` at a scratch folder so the run cannot
overwrite anything in `inpaper/`, then copy the cell across once it looks
right. `scripts/rerun_stl10_glister_cell.sh` is a worked example that
regenerates `glister` at budget 25 for `exp2`:

```bash
bash scripts/rerun_stl10_glister_cell.sh
tmux attach -t stl10-fix
```

Be aware that round 0 evaluates a *cached* initial model shared by every seed
within an `experiment_name`, and that the driver re-saves that checkpoint at
the end of round 0. A cell re-run long after the original sweep therefore
starts from a different initial model than its siblings, which leaves its
final accuracy comparable but its accuracy *gain* not.

### The one repaired cell

STL-10 `glister` / budget 25 / `exp2` wrote an all-100.0 placeholder as its
round-0 row and omitted the corresponding `test_acc` entry; the rest of that
run is intact. Because round 0 records the shared initial model, the correct
row was recorded verbatim by the 10 other strategies in the same
budget/seed group, so it was restored rather than re-run (a re-run cannot
recover the original checkpoint, per the note above):

```bash
python scripts/repair_stl10_glister_cell.py --dry-run   # inspect
python scripts/repair_stl10_glister_cell.py             # apply
```

The script refuses to run if the sibling strategies disagree on the round-0
row, is a no-op once applied, and leaves `.orig` copies of both files
alongside the repaired ones.

## 6. Assembling the paper results tree

Results produced under `onlywassal/` / `onlyal/` are merged into the unified
`inpaper/` tree (existing files are never overwritten):

```bash
python scripts/merge_inpaper.py
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

Runs the WASSAL selection loop end-to-end on dummy data and validates the
returned indices and per-class simplexes.

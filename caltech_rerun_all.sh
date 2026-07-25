#!/bin/bash
# Caltech-101 multi-seed rerun for the paper: all strategies, seeds exp2-4,
# budgets 25/50/100/175/200 (8 AL rounds). Results go to
# tutorials/results/inpaper/caltech101/.
#
# For multi-day / multi-GPU runs that must survive closing the terminal,
# prefer:  bash scripts/run_caltech_tmux.sh
#
# Usage: bash caltech_rerun_all.sh [GPU_ID]
DEVICE_ID="${1:-0}"

SKIP_STRATEGIES="AL_WITHSOFT"   # paper uses the plain AL baselines
SKIP_METHODS=""
SKIP_BUDGETS=""
EXPERIMENT_NAME="inpaper"
SOFT_LOSS_HYPERPARAM="0.3"

mkdir -p tutorials/results/inpaper/caltech101
python3 -u tutorials/All_Wassal/wassal_caltech_multiclass_vanilla2.py "$SKIP_STRATEGIES" "$SKIP_METHODS" "$SKIP_BUDGETS" "$DEVICE_ID" "$EXPERIMENT_NAME" "$SOFT_LOSS_HYPERPARAM" 2>&1 | tee tutorials/results/inpaper/caltech101/caltech_rerun_gpu${DEVICE_ID}.log

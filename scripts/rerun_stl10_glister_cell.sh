#!/bin/bash
# Re-run the one corrupted STL-10 cell: AL/glister, budget 25, exp2.
#
# The shipped run for that cell logged an all-100.0 first row (initial-model
# accuracy), which makes its accuracy-gain meaningless. Everything else in the
# STL-10 grid is fine, so this writes to a scratch experiment folder
# (tutorials/results/stl10fix/) and the good cell is copied into inpaper/
# afterwards -- nothing existing is overwritten by the run itself.
#
# Usage:
#   bash scripts/rerun_stl10_glister_cell.sh
#   tmux attach -t stl10-fix          # watch live (Ctrl-b d to detach)
#   tmux kill-session -t stl10-fix    # stop
set -euo pipefail
cd "$(dirname "$0")/.."

SESSION="stl10-fix"
DRIVER="tutorials/All_Wassal/wassal_stl10.py"
LOGDIR="tutorials/results/stl10fix/stl10"

# Leave only the plain "AL" strategy group active...
SKIP_STRATEGIES="WASSAL WASSAL_WITHSOFT AL_WITHSOFT random"
# ...and within it skip every method except glister at the budget we run.
SKIP_METHODS="gradmatch-tss coreset leastconf margin badge badge_withsoft us"
SKIP_BUDGETS="25"
DEVICE_ID="0"
EXPERIMENT_NAME="stl10fix"
SOFT_LOSS_HYPERPARAM="0.3"
EXPERIMENTS="exp2"
SEEDS="48"
BUDGETS="25"

# The driver caches one initial model per experiment_name and reuses it for
# every seed. Seed the scratch folder with the checkpoint the original STL-10
# AL runs used, otherwise the re-run trains a fresh initial model and its
# round-0 accuracy no longer lines up with the other seeds in the cell.
INIT_MODEL="stl10_ResNet18_features_0.0001"
REF_INIT_MODEL="tutorials/results/onlyal/$INIT_MODEL"

mkdir -p "$LOGDIR" "tutorials/results/$EXPERIMENT_NAME"
if [ ! -f "tutorials/results/$EXPERIMENT_NAME/$INIT_MODEL" ]; then
  if [ ! -f "$REF_INIT_MODEL" ]; then
    echo "Missing reference initial model: $REF_INIT_MODEL" >&2
    exit 1
  fi
  cp "$REF_INIT_MODEL" "tutorials/results/$EXPERIMENT_NAME/$INIT_MODEL"
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach with:"
  echo "  tmux attach -t $SESSION"
  exit 1
fi

CMD="source .venv/bin/activate && python3 -u $DRIVER \"$SKIP_STRATEGIES\" \"$SKIP_METHODS\" \"$SKIP_BUDGETS\" $DEVICE_ID $EXPERIMENT_NAME $SOFT_LOSS_HYPERPARAM \"$EXPERIMENTS\" \"$SEEDS\" \"$BUDGETS\" 2>&1 | tee -a $LOGDIR/stl10_glister_b25_exp2.log; echo EXIT:\$?; exec bash"

tmux new-session -d -s "$SESSION" -n gpu0 -c "$PWD"
tmux send-keys -t "$SESSION:gpu0" "$CMD" C-m

echo "Started tmux session '$SESSION'."
echo "  tmux attach -t $SESSION"
echo "  tail -f $LOGDIR/stl10_glister_b25_exp2.log"

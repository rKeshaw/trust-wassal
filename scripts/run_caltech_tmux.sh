#!/bin/bash
# Launch the Caltech-101 paper rerun inside a detached tmux session so the
# jobs survive closing the terminal / IDE. Splits budgets across two GPUs.
#
# Session name: caltech-rerun
#   window gpu0 -> budgets 25 / 100 / 200
#   window gpu1 -> budgets 50 / 175
#
# Usage:
#   bash scripts/run_caltech_tmux.sh
#   tmux attach -t caltech-rerun   # watch live (Ctrl-b d to detach)
#   tmux kill-session -t caltech-rerun   # stop both jobs
set -euo pipefail
cd "$(dirname "$0")/.."

SESSION="caltech-rerun"
SKIP_STRATEGIES="AL_WITHSOFT"
ALL_METHODS="WASSAL WASSAL_WITHSOFT glister gradmatch-tss coreset leastconf margin random badge badge_withsoft us"
EXPERIMENT_NAME="inpaper"
SOFT_LOSS_HYPERPARAM="0.3"
DRIVER="tutorials/All_Wassal/wassal_caltech_multiclass_vanilla2.py"
LOGDIR="tutorials/results/inpaper/caltech101"

mkdir -p "$LOGDIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach with:"
  echo "  tmux attach -t $SESSION"
  exit 1
fi

run_cmd() {
  local gpu="$1"
  local skip_budgets="$2"
  local logfile="$3"
  # shellcheck disable=SC2086
  echo "source .venv/bin/activate && python3 -u $DRIVER \"$SKIP_STRATEGIES\" \"$ALL_METHODS\" \"$skip_budgets\" $gpu $EXPERIMENT_NAME $SOFT_LOSS_HYPERPARAM 2>&1 | tee -a $logfile; echo EXIT:\$?; exec bash"
}

tmux new-session -d -s "$SESSION" -n gpu0 -c "$PWD"
tmux send-keys -t "$SESSION:gpu0" "$(run_cmd 0 "50 175" "$LOGDIR/caltech_rerun_gpu0.log")" C-m

tmux new-window -t "$SESSION" -n gpu1 -c "$PWD"
tmux send-keys -t "$SESSION:gpu1" "$(run_cmd 1 "25 100 200" "$LOGDIR/caltech_rerun_gpu1.log")" C-m

echo "Started tmux session '$SESSION' (2 windows: gpu0, gpu1)."
echo "  tmux attach -t $SESSION"
echo "  tail -f $LOGDIR/caltech_rerun_gpu0.log"
echo "  python scripts/audit_results.py   # check completeness later"

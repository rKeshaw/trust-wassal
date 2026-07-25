import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trust.utils.paths import RESULTS_DIR
from trust.utils.calc_statistics_common import run_statistics

rounds = 10
base_dir = str(RESULTS_DIR / "inpaper" / "svhn" / "classimb"
               / f"rounds{rounds}")

run_statistics(
    base_dir=base_dir,
    budgets=[25, 50, 100, 175, 200],
    rounds=rounds,
    avg_col=10,  # 10 classes + average column
    strategies=['WASSAL_WITHSOFT', 'badge', 'glister', 'gradmatch-tss', 'us',
                'coreset', 'leastconf', 'margin', 'random'],
    experiments=['exp2', 'exp3', 'exp4'],
    filename="output_statistics_svhn_vanilla",
    strategy_group="AL_WITHSOFT",
    dataset_label="SVHN",
)

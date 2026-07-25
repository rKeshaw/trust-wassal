import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trust.utils.paths import RESULTS_DIR
from trust.utils.calc_statistics_common import run_statistics

rounds = 10
base_dir = str(RESULTS_DIR / "inpaper" / "pneumoniamnist" / "classimb"
               / f"rounds{rounds}")

run_statistics(
    base_dir=base_dir,
    budgets=[20, 30, 40, 50, 60, 70, 80, 90, 100],
    rounds=rounds,
    avg_col=2,  # 2 classes + average column
    strategies=['WASSAL_WITHSOFT', 'badge', 'glister', 'gradmatch-tss', 'us',
                'coreset', 'leastconf', 'margin', 'random'],
    experiments=['exp2', 'exp3', 'exp4'],
    filename="output_statistics_pneumo_vanilla",
    strategy_group="AL_WITHSOFT",
    dataset_label="PneumoniaMNIST",
)

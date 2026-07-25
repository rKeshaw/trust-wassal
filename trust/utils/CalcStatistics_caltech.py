import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trust.utils.paths import RESULTS_DIR
from trust.utils.calc_statistics_common import run_statistics

rounds = 8
base_dir = str(RESULTS_DIR / "inpaper" / "caltech101" / "classimb"
               / f"rounds{rounds}")

run_statistics(
    base_dir=base_dir,
    budgets=[25, 50, 100, 175, 200],
    rounds=rounds,
    avg_col=102,  # 102 classes + average column
    strategies=['WASSAL', 'WASSAL_WITHSOFT', 'badge', 'badge_withsoft', 'us',
                'glister', 'gradmatch-tss', 'coreset', 'leastconf', 'margin',
                'random'],
    experiments=['exp2', 'exp3', 'exp4'],
    filename="output_statistics_caltech101_vanilla",
    strategy_group="AL_WITHSOFT",
    dataset_label="Caltech101",
)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trust.utils.paths import RESULTS_DIR
from trust.utils.calc_statistics_common import run_statistics

rounds = 10
base_dir = str(RESULTS_DIR / "inpaper" / "stl10" / "classimb"
               / f"rounds{rounds}")

run_statistics(
    base_dir=base_dir,
    budgets=[25, 50, 100, 125, 150, 175, 200],
    rounds=rounds,
    avg_col=10,  # 10 classes + average column
    strategies=['WASSAL', 'WASSAL_WITHSOFT', 'badge', 'badge_withsoft', 'us',
                'glister', 'gradmatch-tss', 'coreset', 'leastconf', 'margin',
                'random'],
    experiments=['exp2', 'exp3', 'exp4'],
    filename="output_statistics_stl10_vanilla",
    strategy_group="AL_WITHSOFT",
    dataset_label="STL10",
)

"""Audit completeness of the `inpaper` results tree for all five datasets.

For every expected (method, budget, experiment) cell, checks that a result
JSON exists and contains the expected number of AL rounds in `test_acc`.
Result files follow two naming schemes:

  {dataset}_{ncls}_{method}_budget:{budget}_rounds:{R}_runs_{exp}.json
  results_{method}_{budget}.json          (Caltech-101, under .../{exp}/)

Directory names are not trusted, since older runs stored WASSAL_WITHSOFT
results inside a `WASSAL/` directory. Prints a per-dataset summary and lists
missing or incomplete cells; exits nonzero if anything is missing.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trust.utils.paths import RESULTS_DIR

AL_BASELINES = ["badge", "coreset", "glister", "gradmatch-tss",
                "leastconf", "margin", "random", "us"]

DATASETS = {
    "cifar10": {
        "rounds": 10,
        "budgets": [25, 50, 100, 125, 150, 175, 200],
        "experiments": ["exp2", "exp3", "exp4"],
        "methods": ["WASSAL", "WASSAL_WITHSOFT"] + AL_BASELINES,
    },
    "svhn": {
        "rounds": 10,
        "budgets": [25, 50, 100, 175, 200],
        "experiments": ["exp2", "exp3", "exp4"],
        "methods": ["WASSAL_WITHSOFT"] + AL_BASELINES,
    },
    "pneumoniamnist": {
        "rounds": 10,
        "budgets": [20, 30, 40, 50, 60, 70, 80, 90, 100],
        "experiments": ["exp2", "exp3", "exp4"],
        "methods": ["WASSAL_WITHSOFT"] + AL_BASELINES,
    },
    "stl10": {
        "rounds": 10,
        "budgets": [25, 50, 100, 125, 150, 175, 200],
        # An extra exp1 seed exists on disk for most STL-10 cells but is not
        # part of the paper matrix (it was trained from a different initial
        # model), so all datasets are audited over the same three seeds.
        "experiments": ["exp2", "exp3", "exp4"],
        "methods": ["WASSAL", "WASSAL_WITHSOFT"] + AL_BASELINES,
    },
    "caltech101": {
        "rounds": 8,
        "budgets": [25, 50, 100, 175, 200],
        "experiments": ["exp2", "exp3", "exp4"],
        "methods": ["WASSAL", "WASSAL_WITHSOFT"] + AL_BASELINES,
    },
}


def find_cell_json(index, method, budget, rounds, exp):
    """Return the result JSON path for a cell, or None."""
    long_marker = f"_{method}_budget:{budget}_rounds:{rounds}_runs"
    short_name = f"results_{method}_{budget}.json"
    for path in index:
        name = path.name
        if name == short_name and path.parent.name == exp:
            return path
        if long_marker in name and name.endswith(f"{exp}.json"):
            # Guard against method being a prefix of another method
            # (e.g. WASSAL vs WASSAL_WITHSOFT): marker match is exact
            # because the full "_budget:" suffix follows the method name.
            return path
    return None


def check_rounds(path: Path, rounds: int) -> bool:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return len(data.get("test_acc", [])) >= rounds


def main():
    overall_missing = 0
    for dataset, cfg in DATASETS.items():
        base = (RESULTS_DIR / "inpaper" / dataset / "classimb"
                / f"rounds{cfg['rounds']}")
        index = list(base.rglob("*.json")) if base.exists() else []
        ok = 0
        problems = []
        for method in cfg["methods"]:
            for budget in cfg["budgets"]:
                for exp in cfg["experiments"]:
                    path = find_cell_json(index, method, budget,
                                          cfg["rounds"], exp)
                    if path is None:
                        problems.append((method, budget, exp, "missing"))
                    elif not check_rounds(path, cfg["rounds"]):
                        problems.append((method, budget, exp, "incomplete"))
                    else:
                        ok += 1
        total = (len(cfg["methods"]) * len(cfg["budgets"])
                 * len(cfg["experiments"]))
        print(f"\n=== {dataset} ===")
        print(f"  complete: {ok}/{total} cells")
        if problems:
            overall_missing += len(problems)
            for method, budget, exp, status in problems:
                print(f"  {status:>10}: {method} / budget {budget} / {exp}")
    print(f"\nTotal missing or incomplete cells: {overall_missing}")
    return 1 if overall_missing else 0


if __name__ == "__main__":
    sys.exit(main())

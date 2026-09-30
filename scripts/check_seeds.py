"""Report per-dataset method/budget cells that have fewer than 3 seeds.

A "seed" = an exp subdir (exp2/exp3/exp4) containing at least one result json.
Prints a compact list of cells with 1 or 2 seeds so the missing-seed jobs can
be targeted at exactly those cells.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trust.utils.paths import RESULTS_DIR

DATASETS = {
    "cifar10": ("cifar10_full/cifar10", 10, [25, 50, 100, 125, 150, 175, 200]),
    "svhn": ("svhn_full/svhn", 10, [25, 50, 100, 175, 200]),
    "pneumoniamnist": ("pneumo_full/pneumoniamnist", 10, [20, 30, 40, 50, 60, 70, 80, 90, 100]),
    "stl10": ("stl10_full/stl10", 10, [25, 50, 100, 125, 150, 175, 200]),
    "caltech101": ("caltech_full/caltech101", 8, [25, 50, 100, 175, 200]),
}


def seeds_with_jsons(bdir: Path):
    out = []
    for expdir in sorted(bdir.iterdir()) if bdir.exists() else []:
        if expdir.is_dir() and any(expdir.glob("*.json")):
            out.append(expdir.name)
    return out


def main():
    total_thin = 0
    for ds, (rel, rounds, budgets) in DATASETS.items():
        base = RESULTS_DIR / rel / "classimb" / f"rounds{rounds}"
        if not base.exists():
            print(f"[skip] {ds}: {base} missing")
            continue
        print(f"\n=== {ds} ===")
        for mdir in sorted(p for p in base.iterdir() if p.is_dir()):
            thin = []
            for b in budgets:
                s = seeds_with_jsons(mdir / str(b))
                if len(s) == 1:
                    thin.append(f"b{b}({s[0]})")
                elif len(s) == 2:
                    thin.append(f"b{b}({'+'.join(s)})")
            if thin:
                total_thin += len(thin)
                print(f"  {mdir.name:<22} {' '.join(thin)}")
    print(f"\ncells with <3 seeds: {total_thin}")


if __name__ == "__main__":
    main()

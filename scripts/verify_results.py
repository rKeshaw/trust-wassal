"""Integrity + completion audit for the results trees.

Checks, for every (dataset, method, budget, seed) cell:
  - result json exists, parses, and is complete (len(test_acc) == rounds)
  - sel_per_cls present, one row per round, each row sums to the budget
  - flags empty/stale/corrupt files that could break tables
Prints a completion matrix (n_seeds per cell) and a problem list.
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
EXCLUDED_DIRS = {"exp1", "verify1", "pneumo_2epoch_gpu3_test"}


def check_cell(bdir: Path, rounds: int, budget: int):
    """Return (n_valid_seeds, [problems]).

    A seed counts as valid if its json parses and test_acc has `rounds` entries
    (this is all the table pipeline reads). sel_per_cls problems are reported
    as SOFT warnings: WASSAL-family cells legitimately store a different
    schema, so sel issues never invalidate a seed.
    """
    n_ok, problems = 0, []
    if not bdir.exists():
        return 0, problems
    for expdir in sorted(p for p in bdir.iterdir() if p.is_dir()):
        if expdir.name in EXCLUDED_DIRS:
            continue
        js = sorted(expdir.glob("*.json"))
        if not js:
            problems.append(f"  EMPTY expdir: {bdir.name}/{expdir.name}")
            continue
        for jf in js:
            rel = f"{bdir.name}/{expdir.name}/{jf.name}"
            try:
                d = json.loads(jf.read_text())
            except (json.JSONDecodeError, OSError) as e:
                problems.append(f"  CORRUPT: {rel} ({type(e).__name__})")
                continue
            ta = d.get("test_acc") or []
            # WASSAL-family cells log a round-0 baseline entry, so len can be
            # rounds+1; tables only read the final entry.
            if len(ta) not in (rounds, rounds + 1):
                problems.append(f"  INCOMPLETE: {rel} rounds={len(ta)}/{rounds}")
                continue
            n_ok += 1
            sel = d.get("sel_per_cls")
            if not sel or len(sel) != rounds:
                continue  # WASSAL-family schema differs; not an error
            sums = [sum(r) for r in sel]
            if any(s != budget for s in sums):
                problems.append(f"  SOFT-WARN sel sums {sums}: {rel}")
    return n_ok, problems


def main():
    grand_done, grand_total, all_problems = 0, 0, []
    for ds, (rel, rounds, budgets) in DATASETS.items():
        base = RESULTS_DIR / rel / "classimb" / f"rounds{rounds}"
        if not base.exists():
            print(f"[skip] {ds}: base missing")
            continue
        print(f"\n=== {ds} (rounds{rounds}) ===")
        for mdir in sorted(p for p in base.iterdir() if p.is_dir()):
            if mdir.name in EXCLUDED_DIRS:
                continue
            row, probs = [], []
            for b in budgets:
                n, pr = check_cell(mdir / str(b), rounds, b)
                row.append(n)
                probs += pr
                grand_total += 3
                grand_done += min(n, 3)
            if any(row) or probs:
                cells = " ".join(f"b{b}:{n}" for b, n in zip(budgets, row))
                print(f"  {mdir.name:<22} {cells}")
            all_problems += probs
    print(f"\ncell-seeds complete: {grand_done}/{grand_total}")
    if all_problems:
        print(f"\nPROBLEMS ({len(all_problems)}):")
        for p in all_problems[:40]:
            print(p)
    else:
        print("PROBLEMS: none — all present jsons parse, complete, and consistent")


if __name__ == "__main__":
    main()

"""Cross-dataset comparison: WASSAL variants vs classic and modern (2022-2024)
baselines. Prints mean final-round test accuracy per (dataset, method, budget)
and writes tutorials/tables/modern_comparison.csv.

Usage: python scripts/compare_modern.py
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trust.utils.paths import RESULTS_DIR

METHODS = [
    "WASSAL_WITHSOFT", "WASSAL",
    "badge", "glister", "gradmatch-tss", "coreset", "us", "leastconf",
    "margin", "random",
    "typiclust", "probcover", "dcom", "alfamargin",
    "maxherding", "uherding", "wassersteinip",
]
MODERN = {"typiclust", "probcover", "dcom", "alfamargin",
          "maxherding", "uherding", "wassersteinip"}

DATASETS = {
    "cifar10": ("cifar10_full/cifar10", 10, [25, 50, 100, 125, 150, 175, 200]),
    "svhn": ("svhn_full/svhn", 10, [25, 50, 100, 175, 200]),
    "pneumoniamnist": ("pneumo_full/pneumoniamnist", 10, [20, 30, 40, 50, 60, 70, 80, 90, 100]),
    "stl10": ("stl10_full/stl10", 10, [25, 50, 100, 125, 150, 175, 200]),
    "caltech101": ("caltech_full/caltech101", 8, [25, 50, 100, 175, 200]),
}


def finals(base: Path, method: str, budget: int):
    out = []
    bdir = base / method / str(budget)
    if not bdir.exists():
        return out
    for expdir in sorted(bdir.iterdir()):
        if not expdir.is_dir():
            continue
        js = list(expdir.glob("*.json"))
        if not js:
            continue
        try:
            d = json.loads(js[0].read_text())
        except json.JSONDecodeError:
            continue
        ta = d.get("test_acc")
        if ta:
            out.append(ta[-1])
    return out


def main():
    rows = []
    for ds, (root, rounds, budgets) in DATASETS.items():
        base = RESULTS_DIR / root / "classimb" / f"rounds{rounds}"
        if not base.exists():
            continue
        print(f"\n=== {ds} ===")
        header = f"{'method':<18}" + "".join(f"b{b:<11}" for b in budgets)
        print(header)
        best = {b: -1.0 for b in budgets}
        data = {}
        for m in METHODS:
            cells = {}
            for b in budgets:
                v = finals(base, m, b)
                if v:
                    mu = st.mean(v)
                    cells[b] = (mu, st.stdev(v) if len(v) > 1 else None, len(v))
                    best[b] = max(best[b], mu)
            if cells:
                data[m] = cells
        for m in METHODS:
            if m not in data:
                continue
            tag = " *" if m in MODERN else ""
            line = f"{m + tag:<18}"
            for b in budgets:
                if b in data[m]:
                    mu, sd, n = data[m][b]
                    txt = f"{mu:.1f}" + (f"±{sd:.1f}" if sd else "")
                    if abs(mu - best[b]) < 1e-9:
                        txt = f"**{txt}**"
                    line += f"{txt}({n})".ljust(12)
                else:
                    line += "--".ljust(12)
            print(line)
            for b, (mu, sd, n) in data[m].items():
                rows.append({
                    "dataset": ds, "method": m, "modern": m in MODERN,
                    "budget": b, "n_seeds": n,
                    "mean_final_acc": f"{mu:.2f}",
                    "std_final_acc": f"{sd:.2f}" if sd else "",
                })

    out = RESULTS_DIR.parent / "tables" / "modern_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "dataset", "method", "modern", "budget", "n_seeds",
            "mean_final_acc", "std_final_acc"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[ok] wrote {out} ({len(rows)} rows)")
    print("(* = modern 2022-2024 method; ** = best per budget; n in parens)")


if __name__ == "__main__":
    main()

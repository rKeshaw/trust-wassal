"""Audit the paper's LaTeX tables against tutorials/tables/all_datasets_cells.csv.

Parses the per-budget tables in iclr2027_paper/sec/appendix.tex and the summary
table in sec/experiments.tex, then checks (a) every mean matches the CSV average
of the seeds on disk, (b) every bold entry is the column maximum, and (c) the
summary table equals the budget-average of the per-budget table.

Run:  python scripts/audit_paper_tables.py
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "iclr2027_paper" / "sec"
CSV = ROOT / "tutorials" / "tables" / "all_datasets_cells.csv"

NAME2KEY = {
    r"\modelwithsoft{}": "WASSAL_WITHSOFT",
    r"\model{}": "WASSAL",
    "BADGE": "badge", "GLISTER": "glister", "GradMatch-TSS": "gradmatch-tss",
    "CoreSet": "coreset", "LeastConf": "leastconf", "Margin": "margin",
    "Random": "random", "DCoM": "dcom", "MaxHerding": "maxherding",
    "UHerding": "uherding", "Wasserstein-IP": "wassersteinip",
}

PERBUDGET = {
    "tab:cifar10": "cifar10",
    "tab:svhn": "svhn",
    "tab:pneumo": "pneumoniamnist",
    "tab:stl10": "stl10",
    "tab:caltech": "caltech101",
}


def load_csv():
    data = {}
    for r in csv.DictReader(open(CSV)):
        data[(r["dataset"], r["strategy"], int(r["budget"]))] = \
            (float(r["mean_final_acc"]), int(r["n_seeds"]))
    return data


def parse_tables(text):
    """Yield (label, [ (name, [cells]) ]) for each table environment."""
    for m in re.finditer(
            r"\\label\{(tab:[^}]+)\}(.*?)\\end\{tabular\}", text, re.S):
        label, body = m.group(1), m.group(2)
        rows = []
        for line in body.splitlines():
            line = line.strip()
            mm = re.match(r"^(.*?)\s*&\s*(.*?)\s*\\\\$", line)
            if not mm:
                continue
            name = mm.group(1).strip()
            if name in ("Strategy", r"\toprule", r"\midrule") or "\\rule" in name:
                continue
            if name not in NAME2KEY:
                continue
            cells = [c.strip() for c in mm.group(2).split("&")]
            rows.append((name, cells))
        if rows:
            yield label, rows


def main():
    csvd = load_csv()
    text = (PAPER / "appendix.tex").read_text() + \
        (PAPER / "experiments.tex").read_text()
    problems = 0
    for label, rows in parse_tables(text):
        if label not in PERBUDGET:
            continue
        ds = PERBUDGET[label]
        # budgets = sorted csv budgets for this dataset
        budgets = sorted({b for (d, s, b) in csvd if d == ds})
        best = {}
        for b in budgets:
            best[b] = max(csvd[(ds, s, b)][0] for s in set(NAME2KEY.values())
                          if (ds, s, b) in csvd)
        for name, cells in rows:
            key = NAME2KEY[name]
            for b, cell in zip(budgets, cells):
                bold = cell.startswith(r"\textbf{")
                val = float(re.sub(r"\\textbf\{|\}|", "", cell))
                true = csvd[(ds, key, b)][0]
                if abs(val - round(true, 2)) > 1e-9:
                    print(f"[MISMATCH] {ds} {key} b{b}: tex={val} csv={true:.2f}")
                    problems += 1
                is_best = abs(true - best[b]) < 1e-9
                if bold != is_best:
                    print(f"[BOLD] {ds} {key} b{b}: bold={bold} best={is_best} "
                          f"(true={true:.2f})")
                    problems += 1
        print(f"[ok] {label}: {len(rows)} methods x {len(budgets)} budgets")

    # summary table check
    summary = None
    for label, rows in parse_tables(text):
        if label == "tab:summary":
            summary = rows
    if summary:
        ds_order = ["cifar10", "svhn", "pneumoniamnist", "stl10", "caltech101"]
        for name, cells in summary:
            key = NAME2KEY[name]
            for ds, cell in zip(ds_order, cells):
                budgets = [b for (d, s, b) in csvd if d == ds and s == key]
                mean = sum(csvd[(ds, key, b)][0] for b in budgets) / len(budgets)
                val = float(re.sub(r"\\textbf\{|\}|", "", cell))
                if abs(val - round(mean, 2)) > 1e-9:
                    print(f"[SUMMARY MISMATCH] {ds} {key}: tex={val} "
                          f"csv={mean:.2f}")
                    problems += 1
        print(f"[ok] tab:summary: {len(summary)} methods x 5 datasets")

    print(f"\n{'PASS: no problems' if problems == 0 else str(problems)+' problems'}")


if __name__ == "__main__":
    main()

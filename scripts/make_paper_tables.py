"""Emit paper-ready LaTeX result tables from tutorials/tables/*_table.md.

The markdown tables in `tutorials/tables/` are the authoritative source (produced
by `make_tables.py` from the on-disk result JSONs). This script turns them into
ICLR-ready `tabular` bodies: means only, bold column maximum, methods grouped into
"our methods + classical baselines" and "recent low-budget / optimal-transport
methods". It also builds one compact table summarising the soft-loss transfer
ablation (applying the weighted soft loss on top of each classical query rule),
reported as the budget-averaged accuracy change per dataset.

Usage:  python scripts/make_paper_tables.py > /tmp/paper_tables.tex
"""
from pathlib import Path

TABLES = Path(__file__).resolve().parents[1] / "tutorials" / "tables"

DATASET_TITLE = {"cifar10": "CIFAR-10", "svhn": "SVHN",
                 "pneumoniamnist": "PneumoniaMNIST", "stl10": "STL-10",
                 "caltech101": "Caltech-101"}

# Order and display names. The two blocks are rendered separately.
CORE = [
    ("WASSAL_WITHSOFT", r"\modelwithsoft{}"),
    ("WASSAL", r"\model{}"),
    ("badge", "BADGE"),
    ("glister", "GLISTER"),
    ("gradmatch-tss", "GradMatch-TSS"),
    ("coreset", "CoreSet"),
    ("leastconf", "LeastConf"),
    ("margin", "Margin"),
    ("random", "Random"),
]
MODERN = [
    ("dcom", "DCoM"),
    ("maxherding", "MaxHerding"),
    ("uherding", "UHerding"),
    ("wassersteinip", "Wasserstein-IP"),
]
TRANSFER = ["badge", "glister", "gradmatch-tss", "coreset",
            "leastconf", "margin"]
TRANSFER_NAME = {
    "badge": "BADGE", "glister": "GLISTER", "gradmatch-tss": "GradMatch-TSS",
    "coreset": "CoreSet", "leastconf": "LeastConf", "margin": "Margin",
}

DATASETS = ["cifar10", "svhn", "pneumoniamnist", "stl10", "caltech101"]


def parse_md(path):
    """Return {strategy: {budget(int): (mean, std_or_None)}} for one dataset."""
    rows = {}
    budgets = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] == "Strategy":
            budgets = [int(c) for c in cells[1:]]
            continue
        if set(cells[0]) <= set("-") or not budgets:
            continue
        strat = cells[0]
        vals = {}
        for b, c in zip(budgets, cells[1:]):
            c = c.replace("*", "")
            if c in ("--", ""):
                continue
            mean = c.split("±")[0].strip()
            vals[b] = float(mean)
        rows[strat] = vals
    return budgets, rows


def fmt(v):
    return f"{v:.2f}"


def emit_dataset(budgets, rows):
    """Return the tabular body with the two method blocks separated."""
    best = {b: max(rows[s][b] for s, _ in CORE + MODERN) for b in budgets}
    return (render_block(CORE, rows, budgets, best) + "\n\\midrule\n"
            + render_block(MODERN, rows, budgets, best))


def render_block(block, rows, budgets, best):
    lines = []
    for key, name in block:
        if key not in rows:
            continue
        cells = []
        for b in budgets:
            v = rows[key][b]
            s = fmt(v)
            if abs(v - best[b]) < 1e-9:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def main():
    parsed = {}
    for ds in DATASETS:
        budgets, rows = parse_md(TABLES / f"{ds}_table.md")
        parsed[ds] = (budgets, rows)

    # ---- per-dataset result tables -------------------------------------
    for ds in DATASETS:
        budgets, rows = parsed[ds]
        label = ds if ds != "pneumoniamnist" else "pneumo"
        cols = "l" + "c" * len(budgets)
        tv = parsed[ds][0]
        print(f"% ---- {ds} ----")
        print(f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}")
        print(r"\toprule")
        print("Strategy & " + " & ".join(str(b) for b in budgets) + r" \\")
        print(r"\midrule")
        print(emit_dataset(budgets, rows))
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print()

    # ---- soft-loss transfer ablation -----------------------------------
    print("% ---- soft-loss transfer ablation (budget-averaged delta) ----")
    pretty = {"cifar10": "CIFAR-10", "svhn": "SVHN",
              "pneumoniamnist": "Pneumo", "stl10": "STL-10",
              "caltech101": "Caltech-101"}
    header = ["Rule"]
    for ds in DATASETS:
        header.append(pretty[ds])
    print(r"\begin{tabular}{@{}l" + "c" * len(DATASETS) + r"@{}}")
    print(r"\toprule")
    print(" & ".join(header) + r" \\")
    print(r"\midrule")
    grids = {}
    for ds in DATASETS:
        budgets, rows = parsed[ds]
        col = {}
        for base in TRANSFER:
            ws = base + "_withsoft"
            if ws in rows and base in rows:
                bs = [b for b in budgets if b in rows[ws] and b in rows[base]]
                if bs:
                    col[base] = sum(rows[ws][b] - rows[base][b]
                                    for b in bs) / len(bs)
        grids[ds] = col
    best = {ds: max(grids[ds].values()) for ds in DATASETS if grids[ds]}
    for base in TRANSFER:
        cells = []
        for ds in DATASETS:
            if base in grids[ds]:
                v = grids[ds][base]
                s = f"{v:+.2f}"
                if abs(v - best[ds]) < 1e-9:
                    s = r"\textbf{" + s + "}"
                cells.append(s)
            else:
                cells.append("--")
        print(TRANSFER_NAME[base] + " & " + " & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()

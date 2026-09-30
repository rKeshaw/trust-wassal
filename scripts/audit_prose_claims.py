#!/usr/bin/env python
"""Audit: does the Results prose match the LaTeX tables?

Checks, for each per-budget table:
  * the named winner at each budget is really the column maximum
  * the specific accuracies quoted in prose match the table cells
  * cross-cutting claims (Wasserstein-IP never leads, band widths)

Run from the repo root:  python scripts/audit_prose_claims.py
"""
import re
import sys
from pathlib import Path

SEC = Path(__file__).resolve().parent.parent / "iclr2027_paper" / "sec"
# results live in experiments.tex; the cross-dataset summary is in appendix.tex
TEX = "\n".join((SEC / f).read_text() for f in ("experiments.tex", "appendix.tex"))

# what the prose says, per table: budget -> method named as winner
PROSE_WINNERS = {
    "tab:caltech": {
        25: ["WASSALWITHSOFT"], 50: ["WASSALWITHSOFT"], 100: ["WASSALWITHSOFT"],
        175: ["WASSALWITHSOFT"], 200: ["WASSALWITHSOFT"],
    },
    "tab:stl10": {
        25: ["WASSALWITHSOFT"], 50: ["WASSAL"], 100: ["WASSAL"], 125: ["WASSAL"],
        150: ["WASSAL"], 175: ["WASSAL"], 200: ["WASSAL"],
    },
    "tab:cifar10": {
        25: ["UHERDING"], 50: ["WASSALWITHSOFT"], 100: ["RANDOM"],
        125: ["MAXHERDING"], 150: ["WASSAL"], 175: ["LEASTCONF"], 200: ["BADGE"],
    },
    "tab:svhn": {
        25: ["WASSAL"], 50: ["GRADMATCH-TSS"], 100: ["MARGIN"],
        175: ["LEASTCONF"], 200: ["LEASTCONF"],
    },
    "tab:pneumo": {
        20: ["UHERDING"], 30: ["GRADMATCH-TSS"], 40: ["GRADMATCH-TSS"],
        50: ["CORESET"], 60: ["BADGE"], 70: ["WASSAL"], 80: ["UHERDING"],
        90: ["MAXHERDING"], 100: ["BADGE"],
    },
}

RULES = ("toprule", "midrule", "bottomrule", "cmidrule", "addlinespace")


def clean(cell: str) -> float:
    """'\\textbf{24.31} \\' -> 24.31"""
    cell = cell.replace(r"\textbf", "").replace(r"\underline", "")
    cell = cell.replace("{", "").replace("}", "")
    cell = cell.replace("%", "").strip().rstrip("\\").strip()
    return float(cell)


def norm(name: str) -> str:
    name = name.strip()
    name = name.replace(r"\modelwithsoft", "WASSALWITHSOFT")
    name = name.replace(r"\model", "WASSAL")
    name = name.replace("{", "").replace("}", "").replace("\\", "")
    return re.sub(r"\s+", "", name).upper()


def parse_table(tex: str, label: str):
    """Return (budgets, {METHOD: [values...]}) for the table carrying `label`."""
    m = re.search(re.escape(r"\label{" + label + "}"), tex)
    if not m:
        return None, None
    # the label sits between \caption and \begin{tabular}, so search forward
    start = tex.find(r"\begin{tabular", m.start())
    end = tex.find(r"\bottomrule", m.start())
    if start == -1 or end == -1 or end < start:
        return None, None
    block = tex[start:end]

    header = re.search(r"Strategy\s*&\s*(.+?)\\\\", block, re.S)
    budgets = [x.strip() for x in header.group(1).split("&")]

    rows = {}
    for raw in block.split(r"\\"):
        line = raw.strip()
        # a chunk can start with a \midrule left over from the previous row
        for rule in RULES:
            line = line.replace(rule, " ")
        line = line.strip()
        # the first chunk also carries \begin{tabular}{...}Strategy, drop it
        if not line or "Strategy" in line or "begin{tabular" in line:
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != len(budgets) + 1:
            continue
        name = norm(cells[0])
        try:
            vals = [float(clean(c)) for c in cells[1:]]
        except ValueError:
            continue
        rows[name] = vals
    return budgets, rows


def main() -> int:
    tex = TEX
    problems = []

    for label in ("tab:caltech", "tab:stl10", "tab:cifar10", "tab:svhn", "tab:pneumo"):
        budgets, rows = parse_table(tex, label)
        if not rows:
            print(f"[FAIL] {label}: could not parse")
            problems.append(label)
            continue
        winners = PROSE_WINNERS[label]
        ok = True
        for i, b in enumerate(budgets):
            best = max(rows.items(), key=lambda kv: kv[1][i])
            try:
                claimed = winners.get(int(b))
            except ValueError:
                claimed = None
            if claimed is None:
                continue
            if best[0] not in claimed:
                print(f"[FAIL] {label} b{b}: table max = {best[0]} "
                      f"{best[1][i]:.2f}, prose names {claimed}")
                problems.append(f"{label}@{b}")
                ok = False
        if ok:
            print(f"[ok]   {label}: {len(budgets)} budgets, named winners match column maxima")

    # ---- quoted accuracies -------------------------------------------------
    quotes = [
        ("tab:caltech", "WASSALWITHSOFT", 0, 20.69, "Caltech b25"),
        ("tab:caltech", "WASSALWITHSOFT", -1, 27.40, "Caltech b200"),
        ("tab:stl10", "WASSAL", 1, 23.88, "STL b50"),
        ("tab:stl10", "WASSAL", -1, 27.66, "STL b200"),
        ("tab:stl10", "WASSALWITHSOFT", 0, 23.80, "STL b25 withsoft"),
        ("tab:cifar10", "WASSALWITHSOFT", 1, 35.97, "CIFAR b50 withsoft"),
        ("tab:cifar10", "WASSAL", 4, 41.67, "CIFAR b150"),
        ("tab:cifar10", "UHERDING", 0, 32.79, "CIFAR b25 UHerding"),
        ("tab:cifar10", "RANDOM", 2, 39.38, "CIFAR b100 Random"),
        ("tab:cifar10", "MAXHERDING", 3, 40.23, "CIFAR b125 MaxHerding"),
        ("tab:cifar10", "LEASTCONF", 5, 42.51, "CIFAR b175 LeastConf"),
        ("tab:cifar10", "BADGE", 6, 42.99, "CIFAR b200 BADGE"),
        ("tab:svhn", "WASSAL", 0, 28.85, "SVHN b25"),
        ("tab:svhn", "GRADMATCH-TSS", 1, 38.22, "SVHN b50"),
        ("tab:svhn", "MARGIN", 2, 48.86, "SVHN b100"),
        ("tab:svhn", "LEASTCONF", 3, 64.86, "SVHN b175"),
        ("tab:svhn", "LEASTCONF", 4, 67.99, "SVHN b200"),
        ("tab:pneumo", "UHERDING", 0, 86.86, "Pneumo b20"),
        ("tab:pneumo", "GRADMATCH-TSS", 1, 86.70, "Pneumo b30"),
        ("tab:pneumo", "GRADMATCH-TSS", 2, 87.29, "Pneumo b40"),
        ("tab:pneumo", "CORESET", 3, 85.79, "Pneumo b50"),
        ("tab:pneumo", "BADGE", 4, 86.38, "Pneumo b60"),
        ("tab:pneumo", "WASSAL", 5, 85.79, "Pneumo b70"),
        ("tab:pneumo", "UHERDING", 6, 86.70, "Pneumo b80"),
        ("tab:pneumo", "MAXHERDING", 7, 86.06, "Pneumo b90"),
        ("tab:pneumo", "BADGE", 8, 86.22, "Pneumo b100"),
    ]
    cache = {}
    print()
    bad = 0
    for label, method, idx, expect, tag in quotes:
        if label not in cache:
            cache[label] = parse_table(tex, label)
        _, rows = cache[label]
        if method not in rows:
            print(f"[FAIL] {tag}: method {method} not in {label}")
            bad += 1
            continue
        got = rows[method][idx]
        if abs(got - expect) > 0.005:
            print(f"[FAIL] {tag}: table {got:.2f} vs prose {expect:.2f}")
            bad += 1
    if bad == 0:
        print(f"[ok]   all {len(quotes)} quoted accuracies match the tables")

    # ---- cross-cutting claims ---------------------------------------------
    print()
    leads = []
    band_lo, band_hi = 1e9, -1e9
    for label in PROSE_WINNERS:
        _, rows = parse_table(tex, label)
        for name, vals in rows.items():
            for v in vals:
                band_lo, band_hi = min(band_lo, v), max(band_hi, v)
        if "WASSERSTEIN-IP" in rows:
            wip = rows["WASSERSTEIN-IP"]
            for i in range(len(wip)):
                if wip[i] == max(v[i] for v in rows.values()):
                    leads.append((label, i))
    if leads:
        print(f"[FAIL] 'Wasserstein-IP does not lead a budget' -> leads {leads}")
        problems.append("wip-leads")
    else:
        print("[ok]   'Wasserstein-IP does not lead a budget' holds in all 5 tables")

    # ---- summary table against per-budget tables --------------------------
    summary_cols, summary = parse_table(tex, "tab:summary")
    if summary:
        errs = 0
        cols = list(summary_cols)
        sources = {
            "CIFAR-10": "tab:cifar10",
            "SVHN": "tab:svhn",
            "PneumoniaMNIST": "tab:pneumo",
            "STL-10": "tab:stl10",
            "Caltech-101": "tab:caltech",
        }
        for ci, colname in enumerate(cols):
            src = sources.get(colname)
            if not src:
                print(f"[FAIL] summary has unexpected column {colname!r}")
                problems.append(f"summary-col:{colname}")
                continue
            _, rows = parse_table(tex, src)
            for name, vals in rows.items():
                if name not in summary:
                    continue
                expect = sum(vals) / len(vals)
                got = summary[name][ci]
                if abs(got - expect) > 0.011:
                    print(f"[FAIL] summary {name}/{colname}: {got} "
                          f"vs per-budget mean {expect:.2f}")
                    errs += 1
                    problems.append(f"summary:{name}/{colname}")
        if errs == 0:
            print("[ok]   summary table = mean of the per-budget tables for every cell")

    print()
    if problems:
        print(f"PROSE AUDIT: {len(problems)} PROBLEM(S) -> {problems}")
        return 1
    print("PROSE AUDIT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

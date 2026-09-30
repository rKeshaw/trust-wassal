"""Build paper-style result tables from the `*_full` results trees.

For every (strategy, budget) cell, reads the result JSONs of the seeds that
actually exist on disk, and reports the mean final-round test accuracy
(last entry of `test_acc`) across those seeds, with sample std when 2+ seeds
exist. Output: one Markdown file per dataset under tutorials/tables/ plus a
combined CSV of all cells.

Notes
-----
- Seeds on disk as of 2026-09-22: CIFAR-10 and Caltech-101 have exp2/exp3/exp4;
  SVHN, STL-10 and PneumoniaMNIST have exp2 only. The table header records
  which seeds were found per dataset.
- The `exp1/`, `pneumo_2epoch_gpu3_test/` and `verify1/` trees are excluded:
  they use different configs / a different seed and are not part of the paper
  matrix.
- `test_acc` is stored in percent already (0-100); no rescaling is done.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trust.utils.paths import RESULTS_DIR

BASELINES = ["badge", "glister", "gradmatch-tss", "coreset",
             "us", "leastconf", "margin", "random"]
MAIN_STRATEGIES = ["WASSAL_WITHSOFT", "WASSAL"] + BASELINES
WITHSOFT_BASELINES = [b + "_withsoft" for b in BASELINES]
MODERN_BASELINES = ["typiclust", "probcover", "dcom", "alfamargin",
                    "maxherding", "uherding", "wassersteinip"]

DATASETS = {
    "cifar10": {
        "root": "cifar10_full/cifar10", "rounds": 10,
        "budgets": [25, 50, 100, 125, 150, 175, 200],
    },
    "svhn": {
        "root": "svhn_full/svhn", "rounds": 10,
        "budgets": [25, 50, 100, 175, 200],
    },
    "pneumoniamnist": {
        "root": "pneumo_full/pneumoniamnist", "rounds": 10,
        "budgets": [20, 30, 40, 50, 60, 70, 80, 90, 100],
    },
    "stl10": {
        "root": "stl10_full/stl10", "rounds": 10,
        "budgets": [25, 50, 100, 125, 150, 175, 200],
    },
    "caltech101": {
        "root": "caltech_full/caltech101", "rounds": 8,
        "budgets": [25, 50, 100, 175, 200],
    },
}


def load_cell(base: Path, strategy: str, budget: int):
    """Return {exp: final_acc} for a cell; missing seeds are simply absent."""
    out = {}
    bdir = base / strategy / str(budget)
    if not bdir.exists():
        return out
    for expdir in sorted(bdir.iterdir()):
        if not expdir.is_dir():
            continue
        jsons = list(expdir.glob("*.json"))
        if not jsons:
            continue
        try:
            data = json.loads(jsons[0].read_text())
        except json.JSONDecodeError:
            continue
        ta = data.get("test_acc", [])
        if ta:
            out[expdir.name] = ta[-1]
    return out


def fmt(vals):
    if not vals:
        return "--"
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return f"{mean:.2f}"
    std = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return f"{mean:.2f} ± {std:.2f}"


def mean_of(vals):
    return sum(vals) / len(vals) if vals else None


def build_table(base: Path, budgets, strategies):
    """Return {strategy: {budget: [per-seed finals]}}."""
    table = {}
    for strat in strategies:
        row = {}
        for b in budgets:
            cell = load_cell(base, strat, b)
            if cell:
                row[b] = [cell[k] for k in sorted(cell)]
        table[strat] = row
    return table


def render_markdown(ds, cfg, table, all_strategies):
    seeds = sorted({e for row in table.values() for v in row.values()
                    for e in []})  # placeholder, recomputed below
    budgets = cfg["budgets"]
    seed_set = set()
    for strat in all_strategies:
        for b in budgets:
            bdir = Path(base_dir) / strat / str(b)
            if bdir.exists():
                seed_set |= {d.name for d in bdir.iterdir() if d.is_dir()}
    seed_str = "/".join(sorted(seed_set)) if seed_set else "none"

    lines = []
    lines.append(f"## {ds} (rounds{cfg['rounds']})")
    lines.append("")
    lines.append(f"Seeds on disk: **{seed_str}**. "
                 "Metric: mean final-round test accuracy (%) over the seeds "
                 "on disk; ± is the sample std (shown only when 2+ seeds "
                 "exist). `--` = no run on disk.")
    lines.append("")
    header = "| Strategy | " + " | ".join(str(b) for b in budgets) + " |"
    sep = "|---" * (len(budgets) + 1) + "|"
    lines += [header, sep]

    # best value per budget across the strategies being rendered (first
    # number in each cell, i.e. the mean)
    best_per_budget = {}
    for strat in all_strategies:
        for b, vals in table[strat].items():
            m = mean_of(vals)
            if m is not None and (b not in best_per_budget or m > best_per_budget[b]):
                best_per_budget[b] = m

    for strat in all_strategies:
        cells = []
        for b in budgets:
            vals = table[strat].get(b, [])
            text = fmt(vals)
            if vals and mean_of(vals) >= best_per_budget.get(b, 1e9) - 1e-9:
                text = f"**{text}**"
            cells.append(text)
        lines.append(f"| {strat} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    out_dir = RESULTS_DIR.parent / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    global base_dir
    csv_rows = []
    for ds, cfg in DATASETS.items():
        base_dir = RESULTS_DIR / cfg["root"] / "classimb" / f"rounds{cfg['rounds']}"
        if not base_dir.exists():
            print(f"[skip] {base_dir} does not exist")
            continue
        strategies = [s for s in MAIN_STRATEGIES + WITHSOFT_BASELINES
                      + MODERN_BASELINES
                      if (base_dir / s).exists()]
        table = build_table(base_dir, cfg["budgets"], strategies)
        md = render_markdown(ds, cfg, table, strategies)
        md_path = out_dir / f"{ds}_table.md"
        md_path.write_text(md)
        print(md)
        print(f"[ok] wrote {md_path}")

        for strat in strategies:
            for b, vals in table[strat].items():
                m = mean_of(vals)
                std = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 \
                    if len(vals) > 1 else ""
                csv_rows.append({
                    "dataset": ds, "strategy": strat, "budget": b,
                    "n_seeds": len(vals),
                    "mean_final_acc": f"{m:.2f}" if m is not None else "",
                    "std_final_acc": f"{std:.2f}" if std != "" else "",
                    "per_seed": ";".join(f"{v:.2f}" for v in vals),
                })

    csv_path = out_dir / "all_datasets_cells.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "strategy", "budget", "n_seeds",
            "mean_final_acc", "std_final_acc", "per_seed"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[ok] wrote {csv_path} ({len(csv_rows)} cells)")


if __name__ == "__main__":
    main()

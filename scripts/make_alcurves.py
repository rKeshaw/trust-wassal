"""Regenerate images/alcurves.pdf for the ICLR paper from current results.

Reads tutorials/tables/all_datasets_cells.csv (produced by make_tables.py) and
plots mean final accuracy versus budget for our two methods against the
recent optimal-transport / herding baselines, on the four primary datasets.
The full set of methods is in the paper tables; the plot keeps the reading
easy by showing only the WASSAL family and the SOTA comparisons.

Run:  python scripts/make_alcurves.py
"""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "tutorials" / "tables" / "all_datasets_cells.csv"
OUT = ROOT / "iclr2027_paper" / "images" / "alcurves.pdf"

DS = [
    ("caltech101", "Caltech-101", [25, 50, 100, 175, 200]),
    ("cifar10", "CIFAR-10", list(range(25, 201, 25))),
    ("svhn", "SVHN", [25, 50, 100, 175, 200]),
    ("stl10", "STL-10", list(range(25, 201, 25))),
]

PLOT = [
    ("WASSAL_WITHSOFT", "WASSAL+Soft", "#d62728", "-", "o"),
    ("WASSAL", "WASSAL", "#1f77b4", "-", "s"),
    ("wassersteinip", "Wasserstein-IP", "#2ca02c", "--", "^"),
    ("maxherding", "MaxHerding", "#9467bd", "-.", "D"),
    ("uherding", "UHerding", "#8c564b", "-.", "v"),
]


def load():
    data = {}
    for row in csv.DictReader(open(CSV)):
        if row["dataset"] not in {d[0] for d in DS}:
            continue
        data[(row["dataset"], row["strategy"], int(row["budget"]))] = \
            float(row["mean_final_acc"])
    return data


def main():
    data = load()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 7,
        "axes.linewidth": 0.6,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.titlesize": 7.5,
    })
    fig, axes = plt.subplots(1, 4, figsize=(5.5, 2.15))
    for ax, (ds, title, budgets) in zip(axes, DS):
        for strat, name, color, ls, marker in PLOT:
            xs = [b for b in budgets if (ds, strat, b) in data]
            ys = [data[(ds, strat, b)] for b in xs]
            if not xs:
                continue
            lw = 1.6 if strat in ("WASSAL", "WASSAL_WITHSOFT") else 0.9
            ax.plot(xs, ys, color=color, linestyle=ls, marker=marker,
                    markersize=2.6, linewidth=lw, label=name)
        ax.set_title(title)
        ax.set_xlabel("Budget")
        ax.grid(True, linewidth=0.3, alpha=0.4)
    axes[0].set_ylabel("Mean final acc. (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print(f"[ok] wrote {OUT}")


if __name__ == "__main__":
    main()

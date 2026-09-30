"""Show all score tables with colors: everything in blue, per-budget maximum
in bold bright blue. Run in a real terminal (colors will not survive log files).

Usage: python scripts/show_scores.py
"""
import glob
import json
import statistics as st

BLUE = "\033[34m"
BRIGHT = "\033[1;94m"  # bold + bright blue = best per budget
HDR = "\033[1;36m"
END = "\033[0m"

CONFIG = {
    "CIFAR-10": ("tutorials/results/cifar10_full/cifar10/classimb/rounds10",
                 [25, 50, 100, 125, 150, 175, 200]),
    "SVHN": ("tutorials/results/svhn_full/svhn/classimb/rounds10",
             [25, 50, 100, 175, 200]),
    "PneumoniaMNIST": ("tutorials/results/pneumo_full/pneumoniamnist/classimb/rounds10",
                       [20, 30, 40, 50, 60, 70, 80, 90, 100]),
    "STL-10": ("tutorials/results/stl10_full/stl10/classimb/rounds10",
               [25, 50, 100, 125, 150, 175, 200]),
    "Caltech-101": ("tutorials/results/caltech_full/caltech101/classimb/rounds8",
                    [25, 50, 100, 175, 200]),
}
METHODS = ["WASSAL_WITHSOFT", "WASSAL", "badge", "glister", "gradmatch-tss",
           "coreset", "us", "leastconf", "margin", "random",
           "typiclust", "probcover", "dcom", "alfamargin",
           "maxherding", "uherding", "wassersteinip"]
MODERN = {"typiclust", "probcover", "dcom", "alfamargin",
          "maxherding", "uherding", "wassersteinip"}


def finals(base, m, b):
    out = []
    for f in glob.glob(f"{base}/{m}/{b}/*/*.json"):
        try:
            out.append(json.load(open(f))["test_acc"][-1])
        except Exception:
            pass
    return out


for ds, (base, budgets) in CONFIG.items():
    data = {}
    for m in METHODS:
        row = {}
        for b in budgets:
            v = finals(base, m, b)
            if v:
                row[b] = (st.mean(v), st.stdev(v) if len(v) > 1 else None, len(v))
        if row:
            data[m] = row
    best = {b: max(data[m][b][0] for m in data if b in data[m]) for b in budgets}

    print(f"\n{HDR}=== {ds} ==={END}")
    print(f"{HDR}{'method':<18}" + "".join(f"b{b:<13}" for b in budgets) + END)
    for m in METHODS:
        if m not in data:
            continue
        tag = " *" if m in MODERN else ""
        line = ""
        for b in budgets:
            if b in data[m]:
                mu, sd, n = data[m][b]
                txt = f"{mu:.1f}" + (f"±{sd:.1f}" if sd else "")
                cell = f"{txt}({n})"
                color = BRIGHT if abs(mu - best[b]) < 1e-9 else BLUE
                line += f"{color}{cell:<13}{END}"
            else:
                line += f"{BLUE}{'--':<13}{END}"
        print(f"{BLUE}{m + tag:<18}{END}{line}")

print(f"\n{BLUE}* = modern 2022-2024 method{END}; "
      f"{BRIGHT}bold-bright = best per budget{END}; (n) = seeds; "
      "mean final-round test accuracy (%)")

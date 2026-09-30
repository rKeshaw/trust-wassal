#!/usr/bin/env python3
"""
Diagnostic sweep over tutorials/results/*_full result JSONs.

For every (dataset, strategy, budget, run) cell, extract:
  - test_acc trajectory (rounds 0..9)
  - sel_per_cls: per-round per-class number of selected queries
  - all_class_acc: per-round per-class test accuracy

Aggregates printed:
  A. Best-round vs final-round accuracy (per strategy, budget).
  B. Selection-balance diagnostics for WASSAL variants:
       zero-class rounds (classes that got 0 queries),
       share of queries received by top-3 classes,
       max class share.
  C. Same for a strong baseline (badge) for contrast.
"""
import json
import os
import glob
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "tutorials", "results")
DATASETS = ["cifar10_full", "svhn_full", "stl10_full", "caltech101_full", "pneumoniamnist_full"]

FOCUS = {
    "wassal": ["WASSAL_WITHSOFT", "WASSAL"],
    "baseline": ["badge", "random", "glister"],
}


def cells(dataset):
    out = []
    for strat_dir in sorted(glob.glob(os.path.join(ROOT, dataset, "*", "*", "rounds10", "*"))):
        sf = os.path.basename(strat_dir)
        for bud_dir in sorted(glob.glob(os.path.join(strat_dir, "*"))):
            for jf in glob.glob(os.path.join(bud_dir, "*", "*.json")):
                run = os.path.basename(os.path.dirname(jf))
                out.append((sf, int(os.path.basename(bud_dir)), run, jf))
    return out


def agg(vals):
    v = np.array(vals, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def main():
    for ds in DATASETS:
        print("=" * 100)
        print("DATASET:", ds)
        print("=" * 100)
        table = {}
        for sf, bud, run, jf in cells(ds):
            try:
                d = json.load(open(jf))
            except Exception:
                continue
            accs = d.get("test_acc") or []
            spc = d.get("sel_per_cls") or []
            aca = d.get("all_class_acc") or []
            if not accs:
                continue
            table.setdefault(sf, {}).setdefault(bud, []).append(
                {"run": run, "accs": accs, "spc": spc, "aca": aca}
            )

        # ---- A. final vs best ----
        print("\n-- A. final-round vs best-round test acc (mean over seeds; fmt: final | best | argmax-round) --")
        strat_order = sorted(table.keys())
        budgets = sorted({b for sf in table for b in table[sf]})
        header = "strategy".ljust(24) + "".join(str(b).ljust(22) for b in budgets)
        print(header)
        for sf in strat_order:
            row = sf.ljust(24)
            for b in budgets:
                runs = table[sf].get(b, [])
                if not runs:
                    row += "-".ljust(22)
                    continue
                fins, bests, args_ = [], [], []
                for r in runs:
                    fins.append(r["accs"][-1])
                    bests.append(max(r["accs"]))
                    args_.append(int(np.argmax(r["accs"])))
                fm, fs = agg(fins)
                bm, _ = agg(bests)
                am = agg(args_)[0]
                row += f"{fm:5.1f}±{fs:3.1f}|{bm:5.1f}|r{am:<4.1f}    "
            print(row)

        # ---- B/C. selection balance ----
        print("\n-- B/C. per-round selection balance (mean over rounds&seeds): zc=classes w/ 0 sel, top3=max share of 3 classes, mx=single max --")
        for sf in strat_order:
            zc_l, top3_l, mx_l, ent_l = [], [], [], []
            for b in table[sf]:
                for r in table[sf][b]:
                    for round_sel in r["spc"]:
                        s = np.array(round_sel, dtype=float)
                        if s.sum() <= 0:
                            continue
                        p = s / s.sum()
                        zc_l.append((s == 0).sum())
                        top3_l.append(np.sort(p)[-3:].sum())
                        mx_l.append(p.max())
                        ent_l.append(-(p * np.log(p + 1e-12)).sum())
            if not zc_l:
                continue
            first_round_sel = next(iter(table[sf].values()))[0]['spc'][0]
            ncls = len(first_round_sel)
            print(
                f"  {sf.ljust(24)} zc={np.mean(zc_l):5.2f}/{ncls} "
                f"top3={np.mean(top3_l):.2f} mx={np.mean(mx_l):.2f} H={np.mean(ent_l):.2f} (n_rounds={len(zc_l)})"
            )

        # ---- per-class acc gap: target classes ----
        print("\n-- D. final per-class test acc (mean over seeds, last round; aca lists) --")
        for sf in strat_order:
            for b in sorted(table[sf])[:2]:
                runs = table[sf][b]
                finals = [r["aca"][-1] for r in runs if r["aca"]]
                if not finals:
                    continue
                m = np.mean(np.array(finals, dtype=float), axis=0)
                print(f"  {sf.ljust(24)} bud={b:<4} mean={m.mean():5.1f} min={m.min():5.1f} max={m.max():5.1f} std={m.std():4.1f} classes<int10: {np.round(m,0).astype(int).tolist()}")


if __name__ == "__main__":
    main()

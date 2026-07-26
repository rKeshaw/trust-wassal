"""Repair the corrupted round-0 record of STL-10 / glister / budget 25 / exp2.

Background
----------
Round 0 of every run evaluates a *cached* initial model that the driver shares
across seeds, so within one (dataset, budget) group every strategy and every
seed records the exact same round-0 row. For STL-10 at budget 25 that row is

    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.75, 0.0, 91.5, 0.0, 9.325

and 32 of the 33 cells in the group recorded precisely that. The glister/exp2
cell instead wrote an all-100.0 placeholder into its CSV and dropped the entry
from `test_acc` in its JSON (10 entries where its siblings have 11). Every
later round of that run is intact -- its round-1 output is byte-identical to
glister/exp3, which is what sharing an initial model produces.

The run cannot simply be repeated: the driver re-saves the checkpoint to
`initModelPath` at the end of round 0, so the original initial model was
overwritten long ago and a fresh run starts from a different one (~15.3%
instead of 9.325%). Re-running therefore yields a cell whose round-0 disagrees
with all 32 of its siblings, which breaks the accuracy-gain comparison instead
of fixing it. Since the correct value is not estimated but recorded verbatim by
those siblings, this script restores it.

The script verifies the corruption signature and the unanimity of the sibling
rows before writing anything, is a no-op once applied, and keeps `.orig`
backups.

Usage:
    python scripts/repair_stl10_glister_cell.py [--dry-run]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trust.utils.paths import RESULTS_DIR

GROUP = RESULTS_DIR / "inpaper/stl10/classimb/rounds10"
BUDGET = "25"
EXP = "exp2"
CORRUPT_STRATEGY = "glister"
STEM = f"stl10_10_{CORRUPT_STRATEGY}_budget:{BUDGET}_rounds:10_runs_{EXP}"


def sibling_round0_row():
    """The round-0 CSV row, as agreed on by every other strategy in the group.

    Raises if the siblings disagree, since the repair is only valid while the
    initial model is provably shared.
    """
    rows = {}
    for strategy_dir in sorted(GROUP.iterdir()):
        if not strategy_dir.is_dir() or strategy_dir.name == CORRUPT_STRATEGY:
            continue
        for csv_path in (strategy_dir / BUDGET / EXP).glob("*.csv"):
            rows.setdefault(csv_path.read_text().splitlines()[0].strip(),
                            []).append(strategy_dir.name)
    if not rows:
        raise SystemExit(f"No sibling cells found under {GROUP}")
    if len(rows) > 1:
        raise SystemExit(
            "Sibling strategies disagree on the round-0 row, so the initial "
            f"model is not shared and this repair is invalid: {rows}")
    row, strategies = next(iter(rows.items()))
    print(f"Round-0 row agreed on by {len(strategies)} strategies: {row}")
    return row


def backup(path: Path, dry_run: bool):
    orig = path.with_suffix(path.suffix + ".orig")
    if not orig.exists():
        print(f"  backup -> {orig.name}")
        if not dry_run:
            shutil.copy2(path, orig)


def repair_csv(row: str, dry_run: bool) -> bool:
    path = GROUP / CORRUPT_STRATEGY / BUDGET / EXP / f"{STEM}.csv"
    lines = path.read_text().splitlines()
    if lines[0].strip() == row:
        print(f"  {path.name}: already repaired")
        return False
    if set(lines[0].strip().split(",")) != {"100.0"}:
        raise SystemExit(f"Unexpected first row in {path}: {lines[0]}")
    print(f"  {path.name}: replacing all-100.0 round-0 row")
    backup(path, dry_run)
    if not dry_run:
        path.write_text("\n".join([row] + lines[1:]) + "\n")
    return True


def repair_json(row: str, dry_run: bool) -> bool:
    path = GROUP / CORRUPT_STRATEGY / BUDGET / EXP / f"{STEM}.json"
    data = json.loads(path.read_text())
    per_class = [float(x) for x in row.split(",")]
    round0_acc = per_class[-1]

    changed = False
    if data["all_class_acc"][0] != per_class:
        if set(data["all_class_acc"][0]) != {100.0}:
            raise SystemExit(
                f"Unexpected all_class_acc[0] in {path}: "
                f"{data['all_class_acc'][0]}")
        print("  json: replacing all_class_acc[0]")
        data["all_class_acc"][0] = per_class
        changed = True

    # test_acc holds a leading 0.0 placeholder followed by one entry per round,
    # so a healthy cell has len(all_class_acc) + 1 entries.
    expected_len = len(data["all_class_acc"]) + 1
    if len(data["test_acc"]) == expected_len - 1:
        print(f"  json: inserting missing test_acc round-0 = {round0_acc}")
        data["test_acc"].insert(1, round0_acc)
        changed = True
    elif len(data["test_acc"]) != expected_len:
        raise SystemExit(
            f"Unexpected test_acc length in {path}: {len(data['test_acc'])}")

    if not changed:
        print("  json: already repaired")
        return False
    backup(path, dry_run)
    if not dry_run:
        path.write_text(json.dumps(data))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without writing")
    args = parser.parse_args()

    row = sibling_round0_row()
    print(f"Repairing {CORRUPT_STRATEGY} / budget {BUDGET} / {EXP}:")
    changed = repair_csv(row, args.dry_run) | repair_json(row, args.dry_run)
    if args.dry_run:
        print("\nDry run: nothing written.")
    elif changed:
        print("\nRepaired. Regenerate STL-10 statistics with:"
              "\n  python trust/utils/CalcStatistics_stl10.py")
    else:
        print("\nNothing to do.")


if __name__ == "__main__":
    main()

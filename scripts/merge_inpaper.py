"""Assemble the unified `inpaper` results tree.

STL-10 results were produced under the `onlywassal` (WASSAL strategies) and
`onlyal` (AL baselines) experiment folders, and Caltech-101 under
`onlywassal`. This script copies them into `tutorials/results/inpaper/` so
all five datasets share one layout for the paper tables.

Existing files in `inpaper` are never overwritten, so rerun outputs that were
written directly to `inpaper` take precedence.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trust.utils.paths import RESULTS_DIR

# (source experiment folder, dataset, rounds folder)
MERGE_SOURCES = [
    ("onlywassal", "stl10", "rounds10"),
    ("onlyal", "stl10", "rounds10"),
    ("onlywassal", "caltech101", "rounds8"),
    ("onlywassal", "cifar10", "rounds10"),
    ("onlyal", "cifar10", "rounds10"),
    ("onlywassal", "svhn", "rounds10"),
    ("onlywassal", "pneumoniamnist", "rounds10"),
]


def merge(src_root: Path, dst_root: Path) -> tuple[int, int]:
    copied = skipped = 0
    for src in sorted(src_root.rglob("*")):
        if not src.is_file():
            continue
        dst = dst_root / src.relative_to(src_root)
        if dst.exists():
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied, skipped


def main():
    for experiment, dataset, rounds in MERGE_SOURCES:
        src_root = RESULTS_DIR / experiment / dataset / "classimb" / rounds
        dst_root = RESULTS_DIR / "inpaper" / dataset / "classimb" / rounds
        if not src_root.exists():
            print(f"[skip] {src_root} does not exist")
            continue
        copied, skipped = merge(src_root, dst_root)
        print(f"[ok] {experiment}/{dataset}: copied {copied} files, "
              f"kept {skipped} existing files in inpaper")


if __name__ == "__main__":
    main()

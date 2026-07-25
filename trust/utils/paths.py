"""Central location for repository paths.

All scripts derive locations from REPO_ROOT instead of hardcoding
machine-specific absolute paths, so the code runs on any checkout.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "tutorials" / "results"
# Drivers and prepare_*.py use datadir="data" relative to the repo root.
DATA_DIR = REPO_ROOT / "data"

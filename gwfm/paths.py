"""All artifact paths resolve relative to this checkout or --work-dir."""

import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("GWFM_WORK_DIR", REPOSITORY)).resolve()
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SEEDS = (11, 23, 37)
BLOCKS = {
    "Modal-1": slice(0, 16),
    "Modal-2": slice(16, 32),
    "Modal-3": slice(32, 48),
    "full": slice(0, 48),
}
TARGETS = ["Respiratory burden", "Heat-related burden"]


def encoder_dir(model, seed):
    return ROOT / "checkpoints" / model / f"seed{seed}"

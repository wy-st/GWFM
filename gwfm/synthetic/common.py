import hashlib
import json
from pathlib import Path
import numpy as np


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def split_regions(group, seed):
    rng = np.random.default_rng(seed)
    result = np.empty(len(group), dtype=np.int64)
    for g in np.unique(group):
        ids = rng.permutation(np.flatnonzero(group == g))
        if len(ids) < 10:
            raise ValueError(
                "Each group needs at least 10 regions for 70/10/20 splitting"
            )
        n_test = max(1, round(0.2 * len(ids)))
        n_val = max(1, round(0.1 * len(ids)))
        result[ids[:n_test]] = 2
        result[ids[n_test : n_test + n_val]] = 1
        result[ids[n_test + n_val :]] = 0
    return result

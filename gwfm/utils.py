from collections import defaultdict

METRICS = ("mae", "rmse", "nmae", "nrmse", "r2")
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(y, p, scale):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    assert y.shape == p.shape and y.ndim == 1 and len(y) > 0
    assert (
        np.isfinite(y).all()
        and np.isfinite(p).all()
        and np.isfinite(scale)
        and scale > 0
    )
    e = p - y
    mse = float(np.mean(e * e))
    mae = float(np.mean(abs(e)))
    var = float(np.var(y))
    return dict(
        n_events=len(y),
        mae=mae,
        rmse=float(np.sqrt(mse)),
        nmae=mae / scale,
        nrmse=float(np.sqrt(mse) / scale),
        r2=1 - mse / var if len(y) >= 2 and var > 0 else None,
    )


def aggregate(rows):
    # Macro across features, never average unlike physical units for MAE/RMSE.
    group = defaultdict(list)
    keys = ("method", "seed", "epochs", "scenario", "pattern", "missing_count", "level")
    for r in rows:
        for mod in (r["modality"], "all"):
            group[tuple(r[k] for k in keys) + (mod,)].append(r)
    output = []
    for values, rs in group.items():
        out = dict(zip(keys + ("modality",), values))
        out["n_features"] = len(rs)
        out.update(
            n_events_min=min(r["n_events"] for r in rs),
            n_events_max=max(r["n_events"] for r in rs),
            n_nodes_min=min(r["n_nodes"] for r in rs),
            n_nodes_max=max(r["n_nodes"] for r in rs),
        )
        for metric in ("nmae", "nrmse", "r2"):
            valid = [r[metric] for r in rs if r[metric] is not None]
            out[metric] = float(np.mean(valid)) if valid else None
            out[metric + "_features"] = len(valid)
        output.append(out)
    return output


def summarize(rows, feature=False):
    keys = [
        "method",
        "epochs",
        "scenario",
        "pattern",
        "missing_count",
        "level",
        "modality",
    ]
    if feature:
        keys += ["feature", "unit"]
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    output = []
    for key, rs in groups.items():
        out = dict(zip(keys, key))
        out["seeds"] = len(rs)
        for metric in METRICS if feature else ("nmae", "nrmse", "r2"):
            vals = [r[metric] for r in rs if r[metric] is not None]
            out[metric + "_mean"] = float(np.mean(vals)) if vals else None
            out[metric + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        for metric in (
            ("n_events", "n_nodes", "n_families")
            if feature
            else (
                "n_features",
                "n_events_min",
                "n_events_max",
                "n_nodes_min",
                "n_nodes_max",
            )
        ):
            assert len({r[metric] for r in rs}) == 1, (key, metric)
            out[metric] = rs[0][metric]
        output.append(out)
    return output


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to export to {path}")
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def moran(x, edges, row_standardized=False):
    z = np.asarray(x) - np.mean(x)
    weights = (
        1.0 / np.bincount(edges[0], minlength=len(x))[edges[0]]
        if row_standardized
        else np.ones(edges.shape[1])
    )
    return float(
        len(x)
        / weights.sum()
        * (weights * z[edges[0]] * z[edges[1]]).sum()
        / np.square(z).sum()
    )

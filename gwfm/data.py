"""Model-facing observations only; isolate parent families before preprocessing."""

from pathlib import Path

import numpy as np
import torch


def load_graph(
    directory, scenario="complementary", graph_rule="distance", distance_bandwidths=None
):
    """Load observed attributes and graph metadata; never load health labels or oracle truth."""
    if scenario != "complementary":
        raise ValueError(
            "Only the reported complementary observation scenario is supported"
        )
    if graph_rule != "distance":
        raise ValueError("The final release uses the documented distance graph")
    directory = Path(directory)
    with np.load(directory / f"{scenario}.npz", allow_pickle=False) as f:
        g = {
            k: f[k]
            for k in (
                "x",
                "observed",
                "node_type",
                "node_id",
                "parent",
                "edge_index",
                "modality_slices",
            )
        }
    x, obs = g["x"], g["observed"]
    if x.ndim != 2 or obs.ndim != 2 or obs.shape[0] != len(x) or obs.dtype != bool:
        raise ValueError("Expected x[N,F] and boolean observed[N,M]")
    n = len(x)
    for key in ("node_type", "node_id", "parent"):
        if g[key].shape != (n,) or not np.issubdtype(g[key].dtype, np.integer):
            raise ValueError(f"Invalid {key}")
    nr = int((g["node_type"] == 0).sum())
    if not 0 < nr < n or not np.array_equal(
        g["node_type"], np.r_[np.zeros(nr), np.ones(n - nr)]
    ):
        raise ValueError("Expected regions first, then subregions, with types 0/1")
    parent = g["parent"][nr:]
    if (g["parent"][:nr] != -1).any() or (parent < 0).any() or (parent >= nr).any():
        raise ValueError("Invalid parent row indices")
    if (np.bincount(parent, minlength=nr) == 0).any() or len(
        np.unique(g["node_id"])
    ) != n:
        raise ValueError("Every region needs children and node IDs must be unique")
    slices = g.pop("modality_slices")
    if slices.shape != (obs.shape[1], 2) or not np.issubdtype(slices.dtype, np.integer):
        raise ValueError("Invalid modality slices")
    if (
        slices[0, 0] != 0
        or slices[-1, 1] != x.shape[1]
        or np.any(slices[:, 1] <= slices[:, 0])
        or not np.array_equal(slices[1:, 0], slices[:-1, 1])
    ):
        raise ValueError("Modality slices must partition the features contiguously")
    feature_modality = np.repeat(np.arange(len(slices)), slices[:, 1] - slices[:, 0])
    visible = obs[:, feature_modality]
    if not np.isfinite(x[visible]).all():
        raise ValueError("Observed features must be finite")
    g["x"] = np.where(visible, x, 0).astype(np.float32)
    if not np.isfinite(g["x"]).all():
        raise ValueError("Features exceed float32 range")
    edges = g["edge_index"]
    if (
        edges.ndim != 2
        or edges.shape[0] != 2
        or not np.issubdtype(edges.dtype, np.integer)
        or (edges < 0).any()
        or (edges >= n).any()
    ):
        raise ValueError("Invalid edge indices")
    with np.load(directory / "geometry.npz", allow_pickle=False) as f:
        if not np.array_equal(f["parent"], parent):
            raise ValueError("Geometry and graph parent mappings differ")
        area = f["child_area"]
        if area.shape != (n - nr,) or not np.isfinite(area).all() or (area <= 0).any():
            raise ValueError("Expected positive finite child polygon areas")
    with np.load(directory / "split.npz", allow_pickle=False) as f:
        split = f["split"]
        if (
            not np.array_equal(f["region_id"], g["node_id"][:nr])
            or split.shape != (nr,)
            or not np.isin(split, [0, 1, 2]).all()
            or set(split.tolist()) != {0, 1, 2}
        ):
            raise ValueError(
                "Expected matching region IDs and nonempty train/val/test splits"
            )
    split = np.r_[split, split[parent]]
    if graph_rule == "distance":
        from .models.spatial_graph import distance_graph, fit_bandwidths

        with np.load(directory / f"{scenario}.npz", allow_pickle=False) as f:
            pos = f["pos"]
        if pos.shape != (n, 2) or not np.isfinite(pos).all():
            raise ValueError("Finite node coordinates required for distance graph")
        g["pos"] = pos.astype(np.float32)
        xy = pos * 200.0
        bandwidths = (
            fit_bandwidths(xy, g["node_type"], split == 0)
            if distance_bandwidths is None
            else distance_bandwidths
        )
        e, weights, types = distance_graph(xy, g["node_type"], bandwidths)
        g["distance_bandwidths"] = np.asarray(bandwidths, dtype=float)
        up = np.stack([np.arange(nr, n), parent])
        edges = np.concatenate([e.numpy(), up, up[::-1]], axis=1)
        g.update(
            edge_weight=np.r_[weights.numpy(), np.ones(2 * (n - nr), np.float32)],
            edge_type=np.r_[types.numpy(), np.full(n - nr, 4), np.full(n - nr, 5)],
        )
    g.update(
        edge_index=edges,
        feature_modality=feature_modality,
        child_weight=np.r_[np.zeros(nr), area].astype(np.float32),
        split=split,
    )
    return {k: torch.from_numpy(v) for k, v in g.items()}


def subset_graph(g, code):
    """Keep complete parent families and only edges within this partition."""
    keep = g["split"] == code
    ids = keep.nonzero().flatten()
    if not len(ids):
        raise ValueError("Empty graph partition")
    remap = torch.full((len(keep),), -1, dtype=torch.long)
    remap[ids] = torch.arange(len(ids))
    edge_keys = ("edge_index", "edge_weight", "edge_type")
    fixed_keys = ("feature_modality", "distance_bandwidths")
    out = {k: v[ids].clone() for k, v in g.items() if k not in edge_keys + fixed_keys}
    edges = g["edge_index"]
    edge_keep = keep[edges].all(dim=0)
    out["edge_index"] = remap[edges[:, edge_keep]]
    for key in edge_keys[1:]:
        if key in g:
            out[key] = g[key][edge_keep].clone()
    for key in fixed_keys:
        if key in g:
            out[key] = g[key].clone()
    children = out["node_type"] == 1
    out["parent"][children] = remap[out["parent"][children]]
    if (out["parent"][children] < 0).any():
        raise ValueError("A subregion was separated from its parent")
    return out


def fit_standardizer(g):
    """Fit one mean and standard deviation per attribute from visible training nodes."""
    visible = g["observed"][:, g["feature_modality"]]
    count = visible.sum(0)
    if (count == 0).any():
        raise ValueError("Each feature needs at least one observed training value")
    x = g["x"].double()
    mean = torch.where(visible, x, 0).sum(0) / count
    var = torch.where(visible, (x - mean).square(), 0).sum(0) / count
    return mean.float(), var.sqrt().clamp_min(1e-6).float()


def standardize(g, mean, std):
    """Apply training statistics and keep every unavailable input at zero."""
    visible = g["observed"][:, g["feature_modality"]]
    return dict(g, x=torch.where(visible, ((g["x"] - mean) / std).clamp(-5, 5), 0))


def sample_mask(observed, node_type, rate, generator=None, strategy="independent"):
    """Hide only observed modalities: mixed masks for training, fixed-rate masks for validation."""
    if not 0 < rate < 1:
        raise ValueError("Mask rate must be between zero and one")
    if strategy == "independent":
        mask = torch.rand(observed.shape, generator=generator) < rate
    elif strategy == "mixed":
        rates = torch.stack(
            [1 - observed[node_type == level].float().mean(0) for level in (0, 1)]
        )
        empirical = torch.rand(observed.shape, generator=generator) < rates[node_type]
        # Explicitly expose single-, multi-, and all-modality gaps, as well as usual coverage.
        ranks = torch.rand(observed.shape, generator=generator).argsort(1).argsort(1)
        count = torch.randint(
            1, observed.shape[1] + 1, (len(observed), 1), generator=generator
        )
        stress = ranks < count
        mask = torch.where(
            torch.rand((len(observed), 1), generator=generator) < 0.5, empirical, stress
        )
    else:
        raise ValueError(f"Unknown mask strategy: {strategy}")
    mask &= observed
    # Avoid empty evaluation cells in small partitions, without masking natural gaps.
    for level in (0, 1):
        for m in range(observed.shape[1]):
            ids = ((node_type == level) & observed[:, m]).nonzero().flatten()
            if len(ids) and not mask[ids, m].any():
                mask[ids[torch.randint(len(ids), (1,), generator=generator)], m] = True
    return mask

"""Coordinate graph construction; no outcomes or oracle features are accessed."""

import numpy as np
from scipy.spatial import cKDTree
import torch


def distance_graph(pos_km, node_type, bandwidths, epsilon=0.1):
    """STGCN-style thresholded Gaussian; all supplied coordinates are in kilometers."""
    pos = np.asarray(pos_km, dtype=float)
    levels = np.asarray(node_type)
    bandwidths = np.asarray(bandwidths, dtype=float)
    if (
        pos.shape != (len(levels), 2)
        or not np.isfinite(pos).all()
        or not np.isin(levels, [0, 1]).all()
        or bandwidths.shape != (2,)
        or not np.isfinite(bandwidths).all()
        or (bandwidths <= 0).any()
        or not 0 < epsilon < 1
    ):
        raise ValueError(
            "Finite coordinates, two positive bandwidths and 0 < epsilon < 1 required"
        )
    all_edges, all_weights, all_types = [], [], []
    for level in (0, 1):
        ids = np.flatnonzero(levels == level)
        sigma = bandwidths[level]
        pairs = cKDTree(pos[ids]).query_pairs(
            sigma * np.sqrt(-np.log(epsilon)), output_type="ndarray"
        )
        pairs = ids[pairs]
        pairs = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
        squared = np.square(pos[pairs[:, 0]] - pos[pairs[:, 1]]).sum(1)
        all_edges.append(pairs.T)
        all_weights.append(np.exp(-squared / sigma**2).astype(np.float32))
        all_types.append(np.full(len(pairs), level, dtype=np.int64))
    return tuple(
        torch.from_numpy(np.concatenate(items, axis=1 if i == 0 else 0))
        for i, items in enumerate((all_edges, all_weights, all_types))
    )


def fit_bandwidths(pos_km, node_type, training, neighbors=6):
    result = []
    for level in (0, 1):
        xy = np.asarray(pos_km)[np.asarray(training) & (np.asarray(node_type) == level)]
        if len(xy) <= neighbors:
            raise ValueError(
                "Not enough training locations to estimate the distance bandwidth"
            )
        distances = cKDTree(xy).query(xy, k=neighbors + 1)[0][:, -1]
        sigma = float(np.median(distances))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Degenerate spatial coordinates")
        result.append(sigma)
    return result

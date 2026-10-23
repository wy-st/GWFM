"""Standalone equal-weight kNN regression using two coordinates only."""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor


def coordinate_knn(pos, values, visible, query_pos, neighbors, fallback):
    pos, values, visible, query_pos = map(np.asarray, (pos, values, visible, query_pos))
    fallback = np.asarray(fallback)
    if (
        values.ndim != 2
        or pos.shape != (len(values), 2)
        or visible.shape != (len(values),)
        or visible.dtype != bool
        or query_pos.ndim != 2
        or query_pos.shape[1] != 2
        or not np.isfinite(pos).all()
        or not np.isfinite(query_pos).all()
        or not np.isfinite(values[visible]).all()
        or fallback.shape != (values.shape[1],)
        or not np.isfinite(fallback).all()
        or not isinstance(neighbors, int)
        or neighbors < 1
    ):
        raise ValueError(
            "Expected finite 2D coordinates, observed donor values, and a training-only fallback"
        )
    if not len(query_pos) or not visible.any():
        return np.broadcast_to(fallback, (len(query_pos), values.shape[1])).copy()
    model = KNeighborsRegressor(
        n_neighbors=min(neighbors, int(visible.sum())),
        weights="uniform",
        metric="euclidean",
    )
    inference = model.fit(pos[visible], values[visible]).predict(query_pos)
    if not np.isfinite(inference).all():
        raise RuntimeError("Non-finite coordinate-kNN inference")
    return inference

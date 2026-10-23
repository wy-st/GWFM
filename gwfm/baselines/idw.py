"""Final IDW baseline with explicit donor availability and training-mean fallback."""

import numpy as np
from scipy.spatial import cKDTree


def interpolate(
    pos, values, visible, query_pos, method, neighbors, *, power=2.0, fallback=None
):
    pos, values, visible, query_pos = map(np.asarray, (pos, values, visible, query_pos))
    if (
        visible.dtype != bool
        or visible.shape != values.shape
        or values.ndim != 1
        or pos.shape != (len(values), 2)
        or query_pos.ndim != 2
        or query_pos.shape[1] != 2
        or not np.isfinite(pos).all()
        or not np.isfinite(query_pos).all()
        or not np.isfinite(values[visible]).all()
        or neighbors < 1
        or method != "idw"
        or power <= 0
    ):
        raise ValueError("Invalid interpolation input or visible donor value")
    if not len(query_pos):
        return np.empty(0)
    if not visible.any():
        if fallback is None or not np.isfinite(fallback):
            raise ValueError(
                "No donors; an explicit training-only fallback is required"
            )
        return np.full(len(query_pos), fallback)
    donor_pos, donor_values = pos[visible], values[visible]
    k = min(int(neighbors), len(donor_values))
    distance, index = cKDTree(donor_pos).query(query_pos, k=list(range(1, k + 1)))
    exact = distance <= 1e-12
    inference = np.empty(len(query_pos))
    matched = exact.any(1)
    if matched.any():
        inference[matched] = (donor_values[index[matched]] * exact[matched]).sum(
            1
        ) / exact[matched].sum(1)
    pending = np.flatnonzero(~matched)
    for start in range(0, len(pending), 128):
        rows = pending[start : start + 128]
        idx = index[rows]
        w = (distance[rows, :1] / distance[rows]) ** power
        w /= w.sum(1, keepdims=True)
        inference[rows] = (w * donor_values[idx]).sum(1)
    if not np.isfinite(inference).all():
        raise RuntimeError("Nonfinite interpolation inference")
    return inference

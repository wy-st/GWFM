"""Report maps summarize errors across seeds, not errors of an ensemble inference."""

import numpy as np
import pandas as pd


def summarize_runs(frame):
    if frame.empty or frame.duplicated(["method", "seed", "node_id", "feature"]).any():
        raise ValueError(
            "Nonempty, unique method/seed/node/feature inferences required"
        )
    if not np.isfinite(frame[["true_value", "inference"]]).all().all():
        raise ValueError("Finite targets and inferences required")
    nodes, metrics = [], []
    for (feature, method), group in frame.groupby(["feature", "method"], sort=False):
        reference = group[group.seed.eq(group.seed.iloc[0])].sort_values("node_id")
        y, ids = reference.true_value.to_numpy(), reference.node_id.to_numpy()
        if len(y) < 2 or np.var(y) == 0:
            raise ValueError("R2 requires at least two nonconstant targets")
        errors, r2s = [], []
        for _, run in group.groupby("seed"):
            run = run.sort_values("node_id")
            np.testing.assert_array_equal(run.node_id, ids)
            np.testing.assert_allclose(run.true_value, y, rtol=0, atol=1e-10)
            error = np.abs(run.inference.to_numpy() - y)
            errors.append(error)
            r2s.append(1 - np.mean(error**2) / np.var(y))
        error = np.mean(errors, axis=0)
        nodes.append(
            pd.DataFrame(
                dict(
                    feature=feature,
                    method=method,
                    node_id=ids,
                    true_value=y,
                    mean_absolute_error=error,
                    n_seeds=len(errors),
                )
            )
        )
        metrics.append(
            dict(
                feature=feature,
                method=method,
                n=len(y),
                seeds=len(errors),
                r2=np.mean(r2s),
                mae=error.mean(),
            )
        )
    return pd.concat(nodes, ignore_index=True), pd.DataFrame(metrics)

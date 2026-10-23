"""Coordinate GPR: train-only kernel estimation, explicit visible-donor conditioning."""

import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel


def observed_data(pos, values, visible):
    pos, values, visible = map(np.asarray, (pos, values, visible))
    if (
        values.ndim != 1
        or pos.shape != (len(values), 2)
        or visible.shape != values.shape
        or visible.dtype != bool
        or not np.isfinite(pos).all()
        or not np.isfinite(values[visible]).all()
    ):
        raise ValueError(
            "Finite 2D coordinates, one target and an explicit Boolean observation mask required"
        )
    return pos[visible], values[visible]


def fit_kernel(pos, values, visible, family, seed=11):
    x, y = observed_data(pos, values, visible)
    if len(y) < 2 or family not in ("RBF", "Matern32"):
        raise ValueError("At least two training observations and RBF/Matern32 required")
    mean, scale = float(y.mean()), max(float(y.std()), 1e-12)
    spatial = (
        RBF(0.1, (0.005, 2.0)) if family == "RBF" else Matern(0.1, (0.005, 2.0), nu=1.5)
    )
    kernel = ConstantKernel(1.0, (0.001, 1000.0)) * spatial + WhiteKernel(
        0.05, (1e-5, 10.0)
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=False,
        n_restarts_optimizer=2,
        random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gp.fit(x, (y - mean) / scale)
    if not np.isfinite(gp.kernel_.theta).all():
        raise FloatingPointError("Nonfinite fitted kernel")
    info = dict(
        family=family,
        mean=mean,
        scale=scale,
        n_train=len(y),
        amplitude=float(gp.kernel_.k1.k1.constant_value),
        length_scale=float(gp.kernel_.k1.k2.length_scale),
        noise=float(gp.kernel_.k2.noise_level),
        log_marginal_likelihood=float(gp.log_marginal_likelihood_value_),
        warnings=[str(w.message) for w in caught],
    )
    return gp.kernel_, info


def coordinate_gpr(pos, values, visible, query_pos, kernel, mean, scale):
    """Return predictive mean/std; kernel and target scaler remain frozen."""
    x, y = observed_data(pos, values, visible)
    query_pos = np.asarray(query_pos)
    if (
        query_pos.ndim != 2
        or query_pos.shape[1] != 2
        or not np.isfinite(query_pos).all()
        or not np.isfinite(mean)
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("Finite query coordinates and training target scaler required")
    if not len(query_pos):
        return np.empty(0), np.empty(0)
    gp = GaussianProcessRegressor(
        kernel=kernel, optimizer=None, alpha=1e-8, normalize_y=False
    )
    if len(y):
        gp.fit(x, (y - mean) / scale)
    pred, std = gp.predict(query_pos, return_std=True)
    pred, std = pred * scale + mean, std * scale
    if not np.isfinite(pred).all() or not np.isfinite(std).all() or (std < 0).any():
        raise FloatingPointError("Invalid GPR inference")
    return pred, std

"""Urbanicity-dependent Bernoulli missingness with calibrated expected coverage."""

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit


def urban_missing_mask(urbanicity, rates, slopes, rng):
    u, rates, slopes = map(
        lambda x: np.asarray(x, dtype=float), (urbanicity, rates, slopes)
    )
    if u.ndim != 1 or not len(u) or rates.ndim != 1 or slopes.shape != rates.shape:
        raise ValueError(
            "Require a node vector and matching modality rate/slope vectors"
        )
    if (
        not all(np.isfinite(x).all() for x in (u, rates, slopes))
        or np.any((rates <= 0) | (rates >= 1))
        or np.any(slopes < 0)
    ):
        raise ValueError(
            "Finite inputs, rates in (0,1) and nonnegative slopes required"
        )
    intercepts = []
    for rate, slope in zip(rates, slopes):
        shift = slope * u
        intercepts.append(
            brentq(
                lambda a: expit(a - shift).mean() - rate,
                float(shift.min()) - 40,
                float(shift.max()) + 40,
            )
        )
    intercepts = np.asarray(intercepts)
    probability = expit(intercepts[None, :] - u[:, None] * slopes[None, :])
    return rng.random(probability.shape) >= probability, probability, intercepts

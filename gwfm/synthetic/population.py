"""Population bookkeeping and semantically constrained synthetic attributes."""

import numpy as np
from scipy.special import expit, logit

from .geometry import settlement_density

FEATURES = [
    "temperature_c",
    "relative_humidity",
    "annual_precipitation_mm",
    "hot_day_fraction",
    "pm25_ug_m3",
    "no2_ug_m3",
    "ozone_ug_m3",
    "emission_index",
    "population_density_km2",
    "deprivation_index",
    "older_population_share",
    "healthcare_access",
]
UNITS = [
    "deg C",
    "fraction",
    "mm/year",
    "fraction",
    "ug/m3",
    "ug/m3",
    "ug/m3",
    "relative index",
    "people/km2",
    "index [0,1]",
    "fraction",
    "index [0,1]",
]
BOUNDED = [1, 3, 9, 10, 11]
POSITIVE = [2, 4, 5, 6, 7, 8]


def weighted_mean(values, parent, weights, n):
    values, weights = np.asarray(values), np.asarray(weights)
    if (
        len(values) != len(parent)
        or weights.shape != (len(parent),)
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("One finite positive aggregation weight per child is required")
    totals = np.bincount(parent, weights=weights, minlength=n)
    if (totals <= 0).any():
        raise ValueError("Every parent requires positive population/area")
    result = np.zeros((n,) + values.shape[1:])
    extra = (1,) * (values.ndim - 1)
    np.add.at(result, parent, values * weights.reshape((-1,) + extra))
    return result / totals.reshape((-1,) + extra)


def local_variation(seed, eta, parent, coefficients, column):
    """Mix a smooth field with parent-shared and child-specific real heterogeneity."""
    coefficients = np.asarray(coefficients, dtype=float)
    if (
        coefficients.shape != (12, 2)
        or not np.isfinite(coefficients).all()
        or (coefficients < 0).any()
    ):
        raise ValueError(
            "Expected 12 finite nonnegative [parent, child] variation scales"
        )
    center, scale = eta.mean(), eta.std()
    if scale <= 1e-12:
        raise ValueError("A nonconstant base field is required")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 54103, column]))
    regional = rng.normal(size=int(parent.max()) + 1)
    local = rng.normal(size=len(parent))
    a, b = coefficients[column]
    mixed = (eta - center) / scale + a * regional[parent] + b * local
    # Keep physical link-scale dispersion stable while changing spatial dependence.
    return center + scale * (mixed - mixed.mean()) / mixed.std()


def population_state(geometry, config, seed=None):
    """Piecewise-constant density on existing polygons, normalized to a total mass."""
    total, side = config["population_total"], config["domain_side_km"]
    if not np.isfinite([total, side]).all() or min(total, side) <= 0:
        raise ValueError("Positive finite population_total and domain_side_km required")
    parent, n = geometry["parent"], len(geometry["regions"])
    area = np.array([p.area for p in geometry["subregions"]]) * side**2
    # shortcut: one interior-point density per small polygon; use areal quadrature for coarser units.
    density = settlement_density(geometry["child_xy"])
    if seed is None:
        raise ValueError("A data seed is required for population heterogeneity")
    density = np.exp(
        local_variation(seed, np.log(density), parent, config["attribute_variation"], 8)
    )
    mass = area * density
    child_population = total * mass / mass.sum()
    region_population = np.bincount(parent, weights=child_population, minlength=n)
    region_area = np.bincount(parent, weights=area, minlength=n)
    all_area = np.r_[region_area, area]
    all_population = np.r_[region_population, child_population]
    density = all_population / all_area
    center, scale = np.log(density[n:]).mean(), np.log(density[n:]).std()
    if scale <= 1e-12:
        raise ValueError("Nonconstant population density is required")
    return {
        "population": all_population,
        "area_km2": all_area,
        "density_km2": density,
        "urbanicity": (np.log(density) - center) / scale,
        "urbanicity_center": center,
        "urbanicity_scale": scale,
    }


def decode(eta):
    x = eta.copy()
    x[:, 0] = 18 + 4 * eta[:, 0]
    x[:, BOUNDED] = expit(eta[:, BOUNDED])
    x[:, 10] = 0.05 + 0.40 * x[:, 10]
    x[:, POSITIVE] = np.exp(eta[:, POSITIVE])
    return x


def encode(x):
    eta = x.copy()
    eta[:, 0] = (x[:, 0] - 18) / 4
    eta[:, BOUNDED] = logit(x[:, BOUNDED])
    eta[:, 10] = logit((x[:, 10] - 0.05) / 0.40)
    eta[:, POSITIVE] = np.log(x[:, POSITIVE])
    return eta


def semantic_eta(state, geometry, population):
    n = len(geometry["regions"])
    h, p, v = state["child"].T
    u = population["urbanicity"][n:]
    moisture, services = state["common"][n:], state["infrastructure"][n:]
    return np.column_stack(
        [
            h + 0.12 * u,
            0.5 - 0.35 * h + 0.7 * moisture,
            6.6 - 0.15 * h + 0.35 * moisture,
            -3 + 0.7 * h + 0.15 * u,
            2.4 + 0.40 * p + 0.15 * u,
            2.6 + 0.45 * p + 0.30 * u,
            3.5 + 0.25 * h - 0.20 * p + 0.20 * moisture,
            0.50 * p + 0.20 * u,
            np.log(population["density_km2"][n:]),
            -0.4 + 0.75 * v + 0.1 * moisture,
            -0.7 + 0.45 * v - 0.12 * u + 0.25 * services,
            0.5 - 0.5 * v + 0.6 * u + 0.25 * services,
        ]
    )


def semantic_features(seed, state, geometry, config, population):
    """Separate urban density from socioeconomic factors; measure on valid link scales."""
    n, parent = len(geometry["regions"]), geometry["parent"]
    eta = semantic_eta(state, geometry, population)
    for j in range(12):
        if j != 8:  # Density already defines conserved population mass and U.
            eta[:, j] = local_variation(
                seed, eta[:, j], parent, config["attribute_variation"], j
            )
    child = decode(eta)
    child[:, 8] = population["density_km2"][n:]
    region = weighted_mean(child, parent, population["area_km2"][n:], n)
    region[:, 9:] = weighted_mean(child[:, 9:], parent, population["population"][n:], n)
    region[:, 8] = population["density_km2"][:n]
    underlying = np.vstack([region, child])
    eta_all = np.vstack([encode(region), eta])
    child_sd = config["child_observation_noise"]
    region_sd = config["region_observation_noise"]
    if not np.isfinite([child_sd, region_sd]).all() or min(child_sd, region_sd) < 0:
        raise ValueError("Observation noise scales must be finite and nonnegative")
    observation_sd = np.r_[np.full(n, region_sd), np.full(len(child), child_sd)][
        :, None
    ]
    # Log-concentration errors are smaller than logit-index errors; density has a fixed 10% log SD.
    observation_sd = observation_sd * np.array(
        [1, 1, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1]
    )
    observation_sd[:, 8] = 0.10
    rng = np.random.default_rng(np.random.SeedSequence([seed, 45102]))
    errors = rng.normal(size=eta_all.shape) * observation_sd
    # Mean-preserving lognormal measurement error for positive-valued measurements.
    errors[:, POSITIVE] -= 0.5 * observation_sd[:, POSITIVE] ** 2
    complete = decode(eta_all + errors)
    if not np.isfinite(complete).all():
        raise ValueError(
            "Nonfinite generated attributes; reduce latent or observation scales"
        )
    return {
        "complete": complete,
        "underlying": underlying,
        "observation_sd": observation_sd,
        "population": population,
    }

"""Versioned synthetic fields on an irregular hierarchy with explicit parent IDs."""

from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit

from .common import sha256, write_json, split_regions
from .geometry import make_geometry, save_geometry
from .missingness import urban_missing_mask
from .population import population_state, semantic_features, weighted_mean

MODALITIES = ["climate", "pollution", "sociodemographic"]


def group_mean(values, parent, n):
    result = np.zeros((n,) + values.shape[1:], dtype=float)
    np.add.at(result, parent, values)
    counts = np.bincount(parent, minlength=n)
    if (counts == 0).any():
        raise ValueError("Every region requires a child")
    return result / counts.reshape((n,) + (1,) * (values.ndim - 1))


def basis(xy, knots, length):
    """Finite process-convolution basis: BB' is PSD, diag(BB') is exactly one."""
    xy, knots = np.asarray(xy), np.asarray(knots)
    length = np.broadcast_to(np.asarray(length), (len(xy),))
    if (
        not np.isfinite(xy).all()
        or not np.isfinite(knots).all()
        or not np.isfinite(length).all()
        or np.any(length <= 0)
    ):
        raise ValueError("Finite coordinates and positive length scales required")
    delta = cdist(xy, knots, metric="sqeuclidean")
    log_weight = -0.5 * delta / length[:, None] ** 2
    log_weight -= log_weight.max(1, keepdims=True)
    values = np.exp(log_weight)
    return values / np.sqrt(np.square(values).sum(1, keepdims=True))


def environment(xy, phase):
    boundary = xy[:, 0] - 0.50 - 0.17 * np.sin(2 * np.pi * xy[:, 1] + phase)
    return boundary, expit(boundary / 0.10)


def feature_state(seed, config, geometry, diagnostics=True):
    """Generate the paper's spatial fields and measured attributes without outcomes."""
    n, nc = config["n_regions"], config["n_subregions"]
    rng_geo, rng_field, _, rng_noise, _, _ = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(6)
    ]
    centers = geometry["region_xy"]
    xy = np.vstack([centers, geometry["child_xy"]])
    phase = float(rng_geo.uniform(-np.pi, np.pi))
    axis = np.linspace(-0.35, 1.35, config["knot_grid"])
    a, b = np.meshgrid(axis, axis)
    knots = np.column_stack([a.ravel(), b.ravel()])
    signed, smooth_gate = environment(xy, phase)
    side = signed > 0
    gate = smooth_gate
    local_scale = 0.65 + 0.7 * gate
    fields, near_cov = [], []
    covariance_ids = np.linspace(0, n - 1, min(n, 120), dtype=int)
    field_scale = config.get("field_scale", 1.0)
    nugget_scale = config.get("nugget_scale", 1.0)
    trend_scale = config.get("mean_trend_scale", 1.0)
    if (
        not np.isfinite([field_scale, nugget_scale, trend_scale]).all()
        or field_scale <= 0
        or min(nugget_scale, trend_scale) < 0
    ):
        raise ValueError(
            "Finite positive field_scale and nonnegative noise/trend scales required"
        )
    for length in np.asarray(config["field_lengths"]) * field_scale:
        B = basis(
            xy,
            knots,
            length * local_scale,
        )
        fields.append(B @ rng_field.normal(size=len(knots)))
        (
            near_cov.append(B[covariance_ids] @ B[covariance_ids].T)
            if diagnostics
            else None
        )
    fields = np.column_stack(fields)
    common_B = basis(xy, knots, 0.22 * field_scale)
    common, nuisance, infrastructure = (
        common_B @ rng_field.normal(size=(len(knots), 3))
    ).T
    # Rows are normalized separately so changing local correlations does not inflate variance.
    loading = np.zeros((len(xy), 3, 3))
    loading[:, 0] = [1, 0.15, 0.10]
    loading[:, 1, 0] = 0.65 - 1.10 * gate
    loading[:, 1, 1] = 1
    loading[:, 1, 2] = 0.10
    loading[:, 2, 0] = -0.35 + 0.70 * gate
    loading[:, 2, 1] = 0.15
    loading[:, 2, 2] = 1
    loading /= np.linalg.norm(loading, axis=2, keepdims=True)
    scales = np.column_stack(
        [0.85 + 0.30 * gate, 1.15 - 0.30 * gate, 0.80 + 0.40 * gate]
    )
    means = trend_scale * np.column_stack(
        [0.50 * (2 * gate - 1), -0.40 * (2 * gate - 1), 0.35 * (2 * gate - 1)]
    )
    latent = means + scales * np.einsum("nij,nj->ni", loading, fields)
    latent += 0.18 * common[:, None]
    # Small-scale within-region variation has a location-dependent magnitude.
    micro_sd = np.column_stack(
        [0.18 + 0.10 * gate, 0.20 + 0.18 * gate, 0.22 - 0.08 * gate]
    )
    latent[n:] += rng_noise.normal(size=(nc, 3)) * micro_sd[n:] * nugget_scale
    child = latent[n:]
    state = {
        "knots": knots,
        "signed": signed,
        "gate": gate,
        "side": side,
        "local_scale": local_scale,
        "near_cov": near_cov,
        "covariance_ids": covariance_ids,
        "latent": latent,
        "child": child,
        "nuisance": nuisance,
        "infrastructure": infrastructure,
        "loading": loading,
        "scales": scales,
        "phase": phase,
        "common": common,
    }
    state.update(
        semantic_features(
            seed, state, geometry, config, population_state(geometry, config, seed)
        )
    )
    return state


def generate(seed: int, config: dict, out_dir: str) -> str:
    """Write the deterministic reported dataset and return its directory; reject existing outputs."""
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty output directory for data generation")
    required = dict(
        family="smooth",
        attribute_model="attribute_moran_v2",
        missing_mechanism="urbanicity_logistic",
        spatial_adjacency="rook",
        scenarios=["complementary"],
    )
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError(
            "Only the paper's smooth population-based dataset is supported"
        )
    out.mkdir(parents=True, exist_ok=True)
    family = config["family"]
    n, nc = config["n_regions"], config["n_subregions"]
    missing_rates = np.broadcast_to(
        np.asarray(config["missing_fraction"], dtype=float), (len(MODALITIES),)
    )
    if not np.isfinite(missing_rates).all() or np.any(
        (missing_rates < 0.1) | (missing_rates > 0.4)
    ):
        raise ValueError(
            "Each modality requires a missing fraction between 0.1 and 0.4"
        )
    if not 1 <= config["neighbors"] < n or config["knot_grid"] < 3:
        raise ValueError("Invalid neighborhood or convolution grid")
    rng_geo, rng_field, rng_beta, rng_noise, rng_mask, rng_y = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(6)
    ]
    geometry = make_geometry(config.get("geometry_seed", seed), config)
    geometry_report = save_geometry(geometry, out)
    centers, parent = geometry["region_xy"], geometry["parent"]
    xy = np.vstack([centers, geometry["child_xy"]])
    group = geometry["group"]
    state = feature_state(seed, config, geometry)
    knots = state["knots"]
    signed = state["signed"]
    gate = state["gate"]
    side = state["side"]
    local_scale = state["local_scale"]
    near_cov = state["near_cov"]
    covariance_ids = state["covariance_ids"]
    latent = state["latent"]
    complete = state["complete"]
    nuisance = state["nuisance"]
    loading = state["loading"]
    scales = state["scales"]
    phase = state["phase"]
    coefficient_fields = []
    for length in config["coefficient_lengths"]:
        B = basis(centers, knots, length)
        coefficient_fields.append(np.tanh(B @ rng_beta.normal(size=len(knots))))
    coefficient_fields = np.column_stack(coefficient_fields)
    base_beta = np.array([0.60, 0.60, 0.40, 0.50, 0.35])
    beta = np.broadcast_to(base_beta, (n, 5)).copy()
    beta *= 1 + config["coefficient_amplitude"] * coefficient_fields
    clean = state["underlying"][n:]
    h = (clean[:, 0] - 18) / 4
    p = (np.log(clean[:, 4]) - 2.4) / 0.4
    v = (clean[:, 9] - 0.4) / 0.2 + 0.4 * (clean[:, 10] - 0.18) / 0.08
    weights = state["population"]["population"][n:]

    def aggregate(values):
        return weighted_mean(values, parent, weights, n)

    hm, pm, vm = aggregate(np.column_stack([h, p, v])).T
    hv = aggregate(h * v)
    pstd = np.sqrt(np.maximum(0, aggregate(p**2) - pm**2))
    distance = cdist(centers, centers)
    # Outcome spillover uses a continuous kernel, independent of the learner's adjacency.
    W = np.exp(-0.5 * (distance / 0.09) ** 2)
    np.fill_diagonal(W, 0)
    W /= W.sum(1, keepdims=True)
    signal = np.column_stack(
        [
            beta[:, 0] * hm
            + beta[:, 1] * pm
            + beta[:, 2] * vm
            + 0.20 * (W @ pm)
            + 0.15 * nuisance[:n],
            beta[:, 3] * hv
            + beta[:, 1] * pm
            + beta[:, 4] * pstd
            + 0.15 * (W @ hm)
            + 0.15 * nuisance[:n],
        ]
    )
    noise_sd = config["noise_sd"] * (0.75 + 0.50 * gate[:n])
    noise = rng_y.normal(size=(n, 2)) * noise_sd[:, None]
    y = signal + noise
    population = state["population"]
    urbanicity = population["urbanicity"]
    center, scale = population["urbanicity_center"], population["urbanicity_scale"]
    slopes = np.asarray(config["missing_urbanicity_slopes"], dtype=float)
    rm, rp, ri = urban_missing_mask(urbanicity[:n], missing_rates, slopes, rng_mask)
    cm, cp, ci = urban_missing_mask(urbanicity[n:], missing_rates, slopes, rng_mask)
    # Generator-only diagnostics; no latent urbanicity or probabilities enter predictor graphs.
    np.savez_compressed(
        out / "missingness.npz",
        urbanicity=urbanicity,
        probability_complementary=np.vstack([rp, cp]),
        intercepts=np.stack([ri, ci]),
        slopes=slopes,
        urbanicity_center=center,
        urbanicity_scale=scale,
    )
    e = geometry["region_rook"].T
    spatial = np.vstack([e, e[:, ::-1]])
    children = np.arange(n, n + nc)
    edges = np.vstack(
        [
            spatial,
            np.column_stack([children, parent]),
            np.column_stack([parent, children]),
        ]
    )
    edge_type = np.r_[np.zeros(len(spatial), int), np.ones(nc, int), np.full(nc, 2)]
    split = split_regions(group, config["split_seed"])
    np.savez_compressed(out / "targets.npz", region_id=np.arange(n), y=y)
    np.savez_compressed(
        out / "split.npz", region_id=np.arange(n), split=split, group=group
    )
    # Oracle arrays are never included in a predictor graph or fitted preprocessing.
    np.savez_compressed(out / "population.npz", **state["population"])
    extra_truth = {
        "underlying": state["underlying"],
        "observation_sd": state["observation_sd"],
    }
    np.savez_compressed(
        out / "truth.npz",
        complete=complete,
        latent=latent,
        beta=beta,
        signal=signal,
        noise=noise,
        noise_sd=noise_sd,
        gate=gate,
        side=side,
        signed_boundary=signed,
        length_scales=np.array(config["field_lengths"])[None, :]
        * config.get("field_scale", 1.0)
        * local_scale[:, None],
        loading=loading,
        marginal_scale=scales,
        phase=phase,
        latent_covariance=np.array(near_cov),
        covariance_region_ids=covariance_ids,
        **extra_truth,
    )
    diagnostic = {}
    for scenario in config["scenarios"]:
        mask = np.vstack([rm, cm])
        realized_rates = np.stack([(~mask[:n]).mean(0), (~mask[n:]).mean(0)])
        if np.any((realized_rates < 0.1) | (realized_rates > 0.4)):
            raise ValueError(
                f"Realized missing rates outside assignment bounds: {realized_rates}"
            )
        x = np.where(np.repeat(mask, 4, 1), complete, 0)
        np.savez_compressed(
            out / f"{scenario}.npz",
            x=x,
            observed=mask,
            pos=xy,
            node_id=np.arange(len(xy)),
            node_type=np.r_[np.zeros(n, int), np.ones(nc, int)],
            parent=np.r_[np.full(n, -1), parent],
            group=np.r_[group, group[parent]],
            edge_index=edges.T.astype(np.int64),
            edge_type=edge_type,
            modality_slices=np.array([[0, 4], [4, 8], [8, 12]]),
        )
        child_avail = group_mean(mask[n:].astype(float), parent, n)
        diagnostic[scenario] = {
            "missing_fraction_by_level": realized_rates.tolist(),
            "child_available_given_parent_missing": (
                float(child_avail[~rm].mean()) if (~rm).any() else None
            ),
        }
        gradient = []
        for ids in [np.arange(n), np.arange(n, n + nc)]:
            bins = np.array_split(ids[np.argsort(urbanicity[ids], kind="stable")], 4)
            gradient.append([(~mask[part]).mean(0).tolist() for part in bins])
        diagnostic[scenario]["missing_by_urbanicity_quartile_low_to_high"] = gradient
    from .population import FEATURES as feature_names, UNITS

    extra_meta = {
        "feature_units": UNITS,
        "aggregation": "Area-weighted climate/pollution; population-weighted social indices; density = total population / total area; independent measurement noise after aggregation",
        "population_interpretation": "Synthetic population mass, not integer census counts; piecewise-constant interior-point settlement density; same centers as geometry",
        "urbanicity_interpretation": "Standardized log true population density, not a full urban-rural classification",
    }
    extra_meta.update(
        population_interpretation="Synthetic population mass; settlement gradient plus parent and child log-density heterogeneity; conserved across levels",
        autocorrelation="Per-attribute row-standardized rook Moran I on complete measurements; parameters frozen on unlabeled pilot seeds",
        local_variation="Parent-shared and child-specific effects are underlying heterogeneity, not measurement error",
    )
    write_json(
        out / "metadata.json",
        {
            "schema_version": 4,
            "geometry": geometry_report,
            "family": family,
            "seed": seed,
            "config": config,
            "n_regions": n,
            "n_subregions": nc,
            "features": feature_names,
            "modalities": MODALITIES,
            "target_names": ["Respiratory burden", "Heat-related burden"],
            "diagnostics": diagnostic,
            "files": {p.name: sha256(p) for p in sorted(out.glob("*.npz"))},
            **extra_meta,
        },
    )
    return str(out)

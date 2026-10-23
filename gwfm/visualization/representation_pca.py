"""Extract frozen GWFM branch representations for the report's regional PCA."""

import numpy as np

import torch

from ..data import load_graph, subset_graph, standardize
from ..models.model import encoder_from_config
from ..paths import ROOT as REPO, DATA


def principal_components(beta):
    beta = np.asarray(beta, dtype=np.float64)
    assert beta.ndim == 2 and beta.shape[1] == 48 and np.isfinite(beta).all()
    mean = beta.mean(0)
    centered = beta - mean
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    # Fix the otherwise arbitrary sign by the largest absolute loading, without inspecting maps.
    pivot = np.abs(components).argmax(1)
    components *= np.where(components[np.arange(48), pivot] >= 0, 1.0, -1.0)[:, None]
    scores = centered @ components.T
    variance = singular**2 / (len(beta) - 1)
    assert variance.sum() > 0
    ratio = variance / variance.sum()
    np.testing.assert_allclose(scores @ components + mean, beta, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        components @ components.T, np.eye(48), atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(scores.var(0, ddof=1), variance, atol=1e-12, rtol=1e-10)
    return scores, components, mean, ratio


@torch.no_grad()
def extract(model, raw, split):
    folder = (REPO / model["checkpoint"]).parent
    saved = torch.load(folder / "model.pt", weights_only=True)
    cfg = saved["config"]
    graph = load_graph(
        DATA, graph_rule="distance", distance_bandwidths=cfg["distance_bandwidths"]
    )
    np.testing.assert_array_equal(graph["split"].numpy(), split)
    enc = encoder_from_config(graph["feature_modality"], cfg).eval()
    enc.load_state_dict(saved["encoder_state"], strict=True)
    beta = np.empty((5400, 48), dtype=np.float32)
    hk = np.empty_like(beta)
    original = np.load(folder / "embeddings.npz")
    np.testing.assert_array_equal(original["node_id"], raw["node_id"])
    for code in (0, 1, 2):
        part = standardize(subset_graph(graph, code), saved["mean"], saved["std"])
        base, z, b = enc.representations(part, part["observed"])
        torch.testing.assert_close(b, 2 * enc.beta_map(base).sigmoid(), rtol=0, atol=0)
        value = enc.trunk_embedding(
            enc.k0, part, part["observed"], enc.topology(part, part["observed"])
        )
        torch.testing.assert_close(z, b * value, rtol=0, atol=0)
        np.testing.assert_allclose(
            z.numpy(), original["embedding"][split == code], atol=0, rtol=0
        )
        beta[split == code] = b.numpy()
        hk[split == code] = value.numpy()
    assert np.isfinite(beta).all() and (beta > 0).all() and (beta < 2).all()
    assert np.isfinite(hk).all()
    return {"beta": beta, "hk": hk}

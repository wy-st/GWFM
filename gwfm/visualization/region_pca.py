"""Combine the existing seed-11 region PCA maps of beta and h_k."""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

from ..paths import DATA
from ..utils import digest, moran
from .style import check_canvas


def render(source, destination):
    geometry = DATA / "regions.geojson"
    paths = [geometry, DATA / "geometry.npz", Path(__file__)] + [
        source / quantity / name
        for quantity in ("beta", "hk")
        for name in ("pca_region_seed11.npz", "spatial_diagnostics.csv")
    ]
    hashes = {path: digest(path) for path in paths}
    features = json.loads(geometry.read_text())["features"]
    assert len(features) == 600
    assert all(
        f["geometry"]["type"] == "Polygon" and len(f["geometry"]["coordinates"]) == 1
        for f in features
    )
    polygons = {
        f["properties"]["node_id"]: f["geometry"]["coordinates"][0] for f in features
    }
    assert set(polygons) == set(range(600))
    with np.load(DATA / "geometry.npz") as geometry_data:
        edges = geometry_data["region_rook"]
    edges = np.unique(np.c_[edges, edges[::-1]], axis=1)
    assert (
        (edges >= 0).all() and (edges < 600).all() and not (edges[0] == edges[1]).any()
    )
    assert np.bincount(edges[0], minlength=600).min() > 0
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 24,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 10.2))
    fig.subplots_adjust(
        left=0.075, right=0.995, top=0.92, bottom=0.012, wspace=0.035, hspace=0.24
    )
    panels = []
    for row, (quantity, label) in enumerate((("hk", r"$h_k$"), ("beta", r"$\beta$"))):
        with np.load(source / quantity / "pca_region_seed11.npz") as pca:
            ids, scores, ratios = (
                pca["node_id"],
                pca["scores"],
                pca["explained_variance_ratio"],
            )
        np.testing.assert_array_equal(ids, np.arange(600))
        assert scores.shape == (600, 48) and np.isfinite(scores).all()
        with (source / quantity / "spatial_diagnostics.csv").open() as file:
            previous = {
                int(r["pc"]): float(r["moran_i"])
                for r in csv.DictReader(file)
                if r["seed"] == "11" and r["level"] == "region"
            }
        for pc, ax in enumerate(axes[row]):
            values = scores[:, pc]
            moran_i = moran(values, edges, row_standardized=True)
            np.testing.assert_allclose(moran_i, previous[pc + 1], rtol=0, atol=1e-12)
            limit = float(np.abs(values).max())
            assert limit > 0
            collection = PolyCollection(
                [polygons[int(i)] for i in ids],
                array=values,
                cmap="RdBu_r",
                norm=Normalize(-limit, limit),
                edgecolors="white",
                linewidths=0.22,
            )
            ax.add_collection(collection)
            assert len(collection.get_paths()) == 600
            np.testing.assert_array_equal(collection.get_array(), values)
            ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(
                f"PC{pc + 1} ({ratios[pc]:.1%})\nMoran’s I = {moran_i:.3f}",
                fontsize=22,
                pad=10,
                linespacing=1.15,
            )
            panels.append(
                dict(
                    quantity=quantity,
                    pc=pc + 1,
                    explained_variance=float(ratios[pc]),
                    moran_i=moran_i,
                    vmin=-limit,
                    vmax=limit,
                    region_count=600,
                )
            )
        pos = axes[row, 0].get_position()
        fig.text(
            0.032, (pos.y0 + pos.y1) / 2, label, fontsize=32, ha="center", va="center"
        )
    assert len(fig.axes) == 6
    check_canvas(fig, "beta_hk_region_pca")
    # Panel titles must remain separated from their maps and from other titles.
    renderer = fig.canvas.get_renderer()
    titles = [ax.title.get_window_extent(renderer) for ax in axes.flat]
    for index, (ax, title) in enumerate(zip(axes.flat, titles)):
        assert title.y0 >= ax.get_window_extent(renderer).y1
        assert not any(title.overlaps(other) for other in titles[index + 1 :])
    fig.savefig(destination)
    plt.close(fig)
    for path, sha in hashes.items():
        assert digest(path) == sha
    return panels

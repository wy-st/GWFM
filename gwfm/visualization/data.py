"""Figure 1 and the descriptive source statistics used in Figure 2."""

import hashlib
import json

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from ..utils import moran
from ..data import sample_mask
from . import style as export

from ..paths import FIGURES, DATA, ROOT as REPO

OUT = FIGURES
FULL = DATA
URBAN = DATA / "observation_by_urbanicity.csv"
COLORS = ["#0072B2", "#009E73", "#8A66AD"]
TINTS = ["#DDEEF8", "#DCF3EC", "#EDE5F5"]
MODS = ["Climate", "Pollution", "Sociodemographic"]
NATURAL, ARTIFICIAL = "#DFE3E7", "#E69F00"
INK, MUTED = "#202C38", "#576570"
REP = [0, 4, 8]
CMAPS = [
    LinearSegmentedColormap.from_list(m, ["#F5F8FA", c]) for m, c in zip(MODS, COLORS)
]


def load():
    sources = [
        DATA / f
        for f in ["complementary.npz", "geometry.npz", "split.npz", "schema.json"]
    ]
    sources += [
        FULL / f for f in ["truth.npz", "regions.geojson", "moran_by_attribute.csv"]
    ]
    sources += [
        URBAN,
        FULL / "missingness.npz",
        FULL / "targets.npz",
        FULL / "metadata.json",
    ]
    hashes = {
        str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources
    }
    raw, geo = dict(np.load(sources[0])), dict(np.load(sources[1]))
    split = np.load(DATA / "split.npz")["split"]
    truth = np.load(FULL / "truth.npz")["complete"]
    schema = json.loads((DATA / "schema.json").read_text())
    assert raw["x"].shape == truth.shape == (5400, 12)
    observed = raw["observed"]
    np.testing.assert_array_equal(raw["node_id"], np.arange(5400))
    np.testing.assert_array_equal(raw["modality_slices"], [[0, 4], [4, 8], [8, 12]])
    np.testing.assert_allclose(
        raw["x"][observed.repeat(4, axis=1)], truth[observed.repeat(4, axis=1)]
    )
    node_split = np.r_[split, split[raw["parent"][len(split) :]]]
    ids = np.flatnonzero(node_split == 2)
    # Reproduce the original illustration only; this mask never selects downstream targets.
    illustration = sample_mask(
        torch.from_numpy(observed[ids]),
        torch.from_numpy(raw["node_type"][ids]),
        0.3,
        torch.Generator().manual_seed(20281020),
        "mixed",
    ).numpy()
    assert not (illustration & ~observed[ids]).any()
    region_ids = ids[ids < 600]
    np.testing.assert_array_equal(region_ids, np.flatnonzero(split == 2))
    hidden = np.zeros((600, 3), dtype=bool)
    hidden[region_ids] = illustration[ids < 600]
    obs = observed[:600]
    assert not (hidden & ~obs).any()
    assert not hidden[split != 2].any()
    state = np.where(~obs, 1, np.where(hidden, 2, 0))
    counts = np.array([[(state[:, m] == s).sum() for s in range(3)] for m in range(3)])
    np.testing.assert_array_equal(counts.sum(1), [600] * 3)
    np.testing.assert_array_equal(counts[:, 1], [77, 213, 145])
    np.testing.assert_array_equal(counts[:, 2], [34, 44, 37])
    features = json.loads((FULL / "regions.geojson").read_text())["features"]
    polygon_dict = {
        f["properties"]["node_id"]: np.asarray(f["geometry"]["coordinates"][0])
        for f in features
    }
    assert set(polygon_dict) == set(range(600))
    polygons = [polygon_dict[i] for i in range(600)]
    edges = np.column_stack([geo["region_rook"], geo["region_rook"][::-1]])
    mi = np.array([moran(truth[:600, j], edges, True) for j in range(12)])
    old = pd.read_csv(FULL / "moran_by_attribute.csv")
    old = old[
        (old.level == "region") & (old.weights == "rook") & (old.normalization == "row")
    ]
    np.testing.assert_allclose(
        mi, old.set_index("feature").loc[schema["features"], "moran_i"], atol=1e-12
    )
    pd.DataFrame(
        {"feature": schema["features"], "modality": np.repeat(MODS, 4), "moran_i": mi}
    ).to_csv(OUT / "source_data/moran_by_feature.csv", index=False)
    pd.DataFrame(
        counts,
        index=MODS,
        columns=["visible", "naturally_missing", "artificially_masked"],
    ).rename_axis("modality").to_csv(OUT / "source_data/mask_counts.csv")
    rows = []
    for i in range(600):
        for j, name in enumerate(schema["features"]):
            rows.append(
                dict(
                    node_id=i,
                    x=raw["pos"][i, 0],
                    y=raw["pos"][i, 1],
                    split=int(split[i]),
                    feature=name,
                    modality=MODS[j // 4],
                    state=["visible", "naturally_missing", "artificially_masked"][
                        state[i, j // 4]
                    ],
                    complete_diagnostic_value=truth[i, j],
                    visible_input_value=(
                        float(raw["x"][i, j]) if state[i, j // 4] == 0 else None
                    ),
                )
            )
    pd.DataFrame(rows).to_csv(OUT / "source_data/node_feature_states.csv", index=False)
    return raw, truth[:600], polygons, state, counts, region_ids, mi, hashes


def clean_map(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CCD3D9")
        spine.set_linewidth(0.6)


def value_map(ax, polygons, values, modality, state=None, norm=None):
    norm = norm or Normalize(values.min(), values.max())
    keep = np.ones(600, dtype=bool) if state is None else state == 0
    pc = PolyCollection(
        [p for p, k in zip(polygons, keep) if k],
        array=values[keep],
        cmap=CMAPS[modality],
        norm=norm,
        edgecolors="white",
        linewidths=0.22,
    )
    ax.add_collection(pc)
    if state is not None:
        for code, color, hatch, edge in [
            (1, NATURAL, "xx", "#9AA4AD"),
            (2, "#F9E2AF", "///", "#B7780A"),
        ]:
            ax.add_collection(
                PolyCollection(
                    [p for p, s in zip(polygons, state) if s == code],
                    facecolors=color,
                    edgecolors=edge,
                    linewidths=0.22,
                    hatch=hatch,
                )
            )
    clean_map(ax)
    return pc


def label(fig, x, y, text, size=14):
    fig.text(x, y, text, fontsize=size, fontweight="bold", color=INK, va="top")


def data_figure(raw, truth, polygons, state, counts):
    urban = pd.read_csv(URBAN)
    urban = urban[
        (urban.scenario == "complementary") & (urban.level == "region")
    ].copy()
    missing = np.load(FULL / "missingness.npz")
    bins = np.array_split(np.argsort(missing["urbanicity"][:600], kind="stable"), 4)
    assert len(urban) == 12 and all(len(ids) == 150 for ids in bins)
    for m, modality in enumerate(["climate", "pollution", "sociodemographic"]):
        rows = urban[urban.modality == modality].sort_values("urbanicity_quartile")
        np.testing.assert_array_equal(rows.urbanicity_quartile, [1, 2, 3, 4])
        np.testing.assert_array_equal(
            rows.observed_n, [raw["observed"][ids, m].sum() for ids in bins]
        )
        np.testing.assert_allclose(rows.observed_fraction, rows.observed_n / 150)
        np.testing.assert_allclose(
            rows.expected_observed_fraction,
            [(1 - missing["probability_complementary"][ids, m]).mean() for ids in bins],
        )
    urban.to_csv(OUT / "source_data/urbanicity_sampling.csv", index=False)
    targets = np.load(FULL / "targets.npz")
    np.testing.assert_array_equal(targets["region_id"], np.arange(600))
    target_values = targets["y"]
    assert target_values.shape == (600, 2) and np.isfinite(target_values).all()
    assert (
        json.loads((FULL / "metadata.json").read_text())["target_names"] == export.RISKS
    )
    target_truth = np.load(FULL / "truth.npz")
    np.testing.assert_allclose(
        target_values, target_truth["signal"] + target_truth["noise"]
    )
    pd.DataFrame(target_values, columns=export.RISKS).rename_axis("region_id").to_csv(
        OUT / "source_data/inference_targets.csv"
    )
    width, height = 7.2, 3.0
    map_side, map_bottom = 1.1, 1.7218
    summary_left, summary_width = 3.02, 2.3785
    fig = plt.figure(figsize=(width, height))
    label(fig, 0.04 / width, 2.962 / height, "a", 9)
    positions = [0.13, 1.63, 3.13]
    attribute_lines = [
        [
            "Temperature",
            "Relative\nhumidity",
            "Annual\nprecipitation",
            "Hot-day\nfraction",
        ],
        ["PM2.5", "NO2", "Ozone", "Emission\nindex"],
        [
            "Population\ndensity",
            "Deprivation\nindex",
            "Older population\nshare",
            "Healthcare\naccess",
        ],
    ]
    np.testing.assert_array_equal(
        [name.replace("\n", " ") for group in attribute_lines for name in group],
        export.LABELS,
    )
    attribute_texts = []
    for m, (left, j) in enumerate(zip(positions, REP)):
        fig.text(
            (left + map_side / 2) / width,
            2.962 / height,
            MODS[m],
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color=COLORS[m],
        )
        ax = fig.add_axes(
            [left / width, map_bottom / height, map_side / width, map_side / height]
        )
        np.testing.assert_allclose(
            ax.get_position().size * fig.get_size_inches(), [map_side, map_side]
        )
        vals = truth[:, j]
        # Visible observations define the color scale; hidden values never set it.
        visible_values = vals[state[:, m] == 0]
        norm = (
            LogNorm(visible_values.min(), visible_values.max())
            if j == 8
            else Normalize(visible_values.min(), visible_values.max())
        )
        pc = value_map(ax, polygons, vals, m, state[:, m], norm)
        cax = fig.add_axes(
            [
                (left + map_side + 0.0852) / width,
                map_bottom / height,
                0.0456 / width,
                map_side / height,
            ]
        )
        assert cax.get_position().x0 > ax.get_position().x1
        cb = fig.colorbar(pc, cax=cax, orientation="vertical")
        cb.solids.set_rasterized(False)
        cb.solids.set_edgecolor("face")
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(labelsize=7, length=1.5, pad=1)
        if j == 8:
            ticks = np.array([100, 300, 1000, 3000, 10000])
            ticks = ticks[(ticks >= norm.vmin) & (ticks <= norm.vmax)]
            cb.set_ticks(ticks, labels=[str(x) for x in ticks])
            cb.ax.minorticks_off()
        else:
            ticks = MaxNLocator(4).tick_values(norm.vmin, norm.vmax)
            cb.set_ticks(ticks[(ticks >= norm.vmin) & (ticks <= norm.vmax)])
        cb.set_label(
            ["Temperature (°C)", "PM2.5 (µg/m³)", "Density (people/km²; log)"][m],
            fontsize=7,
            labelpad=1,
        )
        band_top = map_bottom - 0.045
        band = plt.Rectangle(
            ((left - 0.015) / width, (band_top - 0.47) / height),
            1.42 / width,
            0.47 / height,
            transform=fig.transFigure,
            facecolor=TINTS[m],
            alpha=0.45,
            edgecolor="none",
        )
        fig.add_artist(band)
        fig.add_artist(
            plt.Rectangle(
                ((left - 0.015) / width, (band_top - 0.47) / height),
                0.018 / width,
                0.47 / height,
                transform=fig.transFigure,
                facecolor=COLORS[m],
                edgecolor="none",
            )
        )
        for k, name in enumerate(attribute_lines[m]):
            text = fig.text(
                (left + 0.035 + 0.75 * (k % 2)) / width,
                (band_top - 0.115 - 0.225 * (k // 2)) / height,
                name,
                ha="left",
                va="center",
                fontsize=7,
                color=INK,
                linespacing=1.15,
            )
            attribute_texts.append((text, band))
    handles = [
        Patch(facecolor="#6E9CAD", label="Observed"),
        Patch(
            facecolor=NATURAL,
            edgecolor="#9AA4AD",
            hatch="xx",
            label="Naturally\nmissing",
        ),
        Patch(
            facecolor="#F9E2AF",
            edgecolor="#B7780A",
            hatch="///",
            label="Artificially\nmasked",
        ),
    ]
    legend = fig.legend(
        handles=handles,
        loc="center right",
        bbox_to_anchor=(
            (summary_left + summary_width) / width,
            (map_bottom + map_side / 2) / height,
        ),
        ncol=1,
        fontsize=7,
        handlelength=1.2,
        handletextpad=0.4,
        labelspacing=0.85,
        borderpad=0,
        borderaxespad=0,
    )
    label(fig, 0.06 / width, 1.1128 / height, "b  Urbanicity sampling", 8.5)
    # Keep the lower chart sizes and leave space between the panels.
    ax = fig.add_axes([0.42 / width, 0.2806 / height, 2.2 / width, 0.684 / height])
    np.testing.assert_allclose(
        ax.get_position().size * fig.get_size_inches(), [2.2, 0.684]
    )
    for m, (modality, color, marker) in enumerate(
        zip(["climate", "pollution", "sociodemographic"], COLORS, ["o", "s", "^"])
    ):
        rows = urban[urban.modality == modality].sort_values("urbanicity_quartile")
        ax.plot(
            rows.urbanicity_quartile,
            100 * rows.expected_observed_fraction,
            "--",
            color=color,
            linewidth=0.8,
            alpha=0.65,
        )
        ax.plot(
            rows.urbanicity_quartile,
            100 * rows.observed_fraction,
            color=color,
            marker=marker,
            linewidth=1.1,
            markersize=3.2,
            label=MODS[m],
        )
    ax.set(
        xlim=(0.8, 4.2),
        ylim=(0, 103),
        xticks=[1, 2, 3, 4],
        yticks=[0, 50, 100],
        xlabel="Urbanicity quartile (low → high)",
        ylabel="Observed (%)",
    )
    ax.tick_params(labelsize=7, length=2, pad=1.5)
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)
    ax.xaxis.labelpad = 1.5
    ax.yaxis.labelpad = 1.5
    ax.legend(
        loc="lower right",
        fontsize=7,
        frameon=False,
        labelspacing=0.15,
        borderpad=0.1,
        handlelength=1.4,
    )
    ax.grid(axis="y", color="#E6EAED", linewidth=0.4)
    ax.set_axisbelow(True)
    label(fig, (summary_left - 0.05) / width, 1.1128 / height, "c  Mask summary", 8.5)
    ax = fig.add_axes(
        [summary_left / width, 0.2996 / height, summary_width / width, 0.646 / height]
    )
    np.testing.assert_allclose(
        ax.get_position().size * fig.get_size_inches(), [2.3785, 0.646]
    )
    for m in range(3):
        offset = 0
        for s, c in enumerate([COLORS[m], NATURAL, "#F9E2AF"]):
            n = counts[m, s]
            ax.barh(
                m,
                n,
                left=offset,
                height=0.52,
                color=c,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.text(
                offset + n / 2,
                m,
                str(n),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if s == 0 else INK,
                fontweight="bold",
            )
            offset += n
        ax.text(0, m - 0.39, MODS[m], fontsize=7.2, color=INK)
    ax.set(xlim=(0, 600), ylim=(2.6, -0.65), yticks=[], xticks=[0, 300, 600])
    ax.set_xlabel("Regions", fontsize=7, labelpad=1.5)
    ax.tick_params(labelsize=7, length=2, pad=1.5)
    for side in ["left", "right", "top"]:
        ax.spines[side].set_visible(False)
    fig.canvas.draw()
    legend_box = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    np.testing.assert_allclose(legend_box.x1, ax.get_position().x1 * width)
    assert (
        legend_box.x0
        > cb.ax.yaxis.label.get_window_extent()
        .transformed(fig.dpi_scale_trans.inverted())
        .x1
    )
    assert legend_box.y0 > map_bottom and legend_box.y1 < map_bottom + map_side
    for i, (text, band) in enumerate(attribute_texts):
        box = text.get_window_extent()
        band_box = band.get_window_extent()
        assert band_box.x0 < box.x0 < box.x1 < band_box.x1
        assert band_box.y0 < box.y0 < box.y1 < band_box.y1
        assert all(
            not box.overlaps(other.get_window_extent())
            for other, _ in attribute_texts[:i]
        )
    label(fig, 5.49 / width, 2.962 / height, "d", 8.5)
    target_norm = Normalize(target_values.min(), target_values.max())
    for k, (bottom, title) in enumerate(
        zip([map_bottom, 0.34], ["Respiratory burden", "Heat-related burden"])
    ):
        fig.text(
            (5.6 + map_side / 2) / width,
            (bottom + map_side + 0.1402) / height,
            title,
            ha="center",
            va="top",
            fontsize=7.8,
            fontweight="bold",
            color="#A34B1E",
        )
        ax = fig.add_axes(
            [5.6 / width, bottom / height, map_side / width, map_side / height]
        )
        np.testing.assert_allclose(
            ax.get_position().size * fig.get_size_inches(), [map_side, map_side]
        )
        pc = PolyCollection(
            polygons,
            array=target_values[:, k],
            cmap="YlOrRd",
            norm=target_norm,
            edgecolors="white",
            linewidths=0.22,
        )
        ax.add_collection(pc)
        clean_map(ax)
        cax = fig.add_axes(
            [
                (5.6 + map_side + 0.0852) / width,
                bottom / height,
                0.0456 / width,
                map_side / height,
            ]
        )
        cb = fig.colorbar(pc, cax=cax, orientation="vertical")
        cb.solids.set_rasterized(False)
        cb.solids.set_edgecolor("face")
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(labelsize=7, length=1.5, pad=1)
        ticks = MaxNLocator(4).tick_values(target_norm.vmin, target_norm.vmax)
        cb.set_ticks(ticks[(ticks >= target_norm.vmin) & (ticks <= target_norm.vmax)])
        cb.set_label("Score (sim.)", fontsize=7, labelpad=1)
    return fig


def heterogeneity_statistics(raw):
    """Audit the numbers printed in the final schematic without redrawing it."""
    x, y = raw["x"][:600, 0], raw["x"][:600, 4]
    observed = raw["observed"][:600, 0] & raw["observed"][:600, 1]
    rows, points = [], []
    for name, zone in (
        ("West", raw["pos"][:600, 0] < 0.4),
        ("East", raw["pos"][:600, 0] > 0.6),
    ):
        ids = np.flatnonzero(observed & zone)
        slope, intercept = np.polyfit(x[ids], y[ids], 1)
        rows.append(
            dict(
                zone=name,
                n=len(ids),
                pearson_r=np.corrcoef(x[ids], y[ids])[0, 1],
                ols_slope=slope,
                ols_intercept=intercept,
            )
        )
        points.extend(
            dict(node_id=int(i), zone=name, temperature_c=x[i], pm25_ug_m3=y[i])
            for i in ids
        )
    assert [r["n"] for r in rows] == [155, 114]
    np.testing.assert_array_equal(
        np.round([r["pearson_r"] for r in rows], 3), [0.473, -0.315]
    )
    pd.DataFrame(rows).to_csv(
        OUT / "source_data/local_relationship_summary.csv", index=False
    )
    pd.DataFrame(points).to_csv(
        OUT / "source_data/local_relationship_points.csv", index=False
    )
    return rows


def build(destination):
    (OUT / "source_data").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "font.weight": "normal",
            "text.color": INK,
            "axes.labelcolor": INK,
            "hatch.linewidth": 0.35,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
        }
    )
    raw, truth, polygons, state, counts, ids, mi, hashes = load()
    fig = data_figure(raw, truth, polygons, state, counts)
    export.check_canvas(fig, "Figure1_data")
    assert len(fig.axes) == 12  # Five maps, five colorbars, and two summary charts.
    fig.savefig(destination)
    plt.close(fig)
    summaries = heterogeneity_statistics(raw)
    for relative, sha in hashes.items():
        assert hashlib.sha256((REPO / relative).read_bytes()).hexdigest() == sha
    return dict(
        source_hashes=hashes,
        illustration_seed=20281020,
        used_for_downstream_selection=False,
        counts=counts.tolist(),
        health_targets=export.RISKS,
        correlations=summaries,
        moran_range=[float(mi.min()), float(mi.max())],
        panels=["a", "b", "c", "d"],
    )

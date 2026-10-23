"""Export only manuscript Figures 1–5 and Table 1 from the frozen final run."""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
import pymupdf
from threadpoolctl import threadpool_limits

from gwfm.paths import DATA, TARGETS, FIGURES, REPOSITORY
from gwfm.current import inputs
from gwfm.utils import digest, save_json, moran
from gwfm.visualization import composite as c, ablation as a, region_pca as r
from gwfm.visualization import representation_pca as rep
from gwfm.visualization.report_statistics import summarize_runs

from gwfm.paths import ROOT

CURRENT = ROOT / "results/current"
PROBES = ROOT / "results/figure_probes"
OUT = None  # Assigned to a temporary directory while composing PDF panels.
SOURCE = FIGURES / "source_data"
MM = 72 / 25.4
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 12,
    }
)


def statistics():
    f = pd.concat(
        [
            pd.read_csv(CURRENT / "feature_metrics_by_seed.csv"),
            pd.read_csv(PROBES / "feature_metrics_by_seed.csv"),
        ],
        ignore_index=True,
    )
    h = pd.concat(
        [
            pd.read_csv(CURRENT / "inference/test_metrics.csv"),
            pd.read_csv(PROBES / "health_metrics_by_seed.csv"),
        ],
        ignore_index=True,
    )
    assert not f.duplicated(["method", "seed", "feature"]).any()
    assert not h.duplicated(["method", "seed", "target"]).any()
    assert len(f) == 432 and len(h) == 72
    f.to_csv(SOURCE / "attribute_metrics_by_seed.csv", index=False)
    h.to_csv(SOURCE / "health_metrics_by_seed.csv", index=False)
    rows = pd.concat(
        [
            f[["method", "seed", "feature", "r2"]].rename(columns={"feature": "key"}),
            f.groupby(["method", "seed"], sort=False)
            .r2.mean()
            .reset_index()
            .assign(key="all"),
            h[["method", "seed", "target", "r2"]].rename(columns={"target": "key"}),
            h.groupby(["method", "seed"], sort=False)
            .r2.mean()
            .reset_index()
            .assign(key="overall"),
        ],
        ignore_index=True,
    )
    summary = (
        rows.groupby(["method", "key"], sort=False)
        .agg(mean=("r2", "mean"), sd=("r2", "std"), n=("seed", "size"))
        .reset_index()
    )
    keys = c.b.NAMES + ["all"] + TARGETS + ["overall"]
    table = summary.pivot(index="key", columns="method", values="mean").loc[
        keys, c.m.ORDER
    ]
    sd = summary.pivot(index="key", columns="method", values="sd").loc[keys, c.m.ORDER]
    assert table.shape == (16, 11) and np.isfinite(table.to_numpy()).all()
    for method in c.m.ORDER:
        assert (
            summary.loc[summary.method.eq(method), "n"]
            .eq(1 if method in c.METHODS[:3] else 3)
            .all()
        )
        np.testing.assert_allclose(
            table.loc["all", method], table.loc[c.b.NAMES, method].mean(), atol=1e-12
        )
        np.testing.assert_allclose(
            table.loc["overall", method], table.loc[TARGETS, method].mean(), atol=1e-12
        )
    table.rename(columns=c.b.display).to_csv(SOURCE / "table1_r2_mean.csv")
    sd.rename(columns=c.b.display).to_csv(SOURCE / "table1_r2_sd.csv")
    summary.to_csv(SOURCE / "all_method_summary.csv", index=False)
    per_category = (
        f.groupby(["method", "seed", "modality"], sort=False).r2.mean().reset_index()
    )
    categories = (
        per_category.groupby(["method", "modality"], sort=False)
        .agg(mean=("r2", "mean"), sd=("r2", "std"), n=("seed", "size"))
        .reset_index()
    )
    categories.to_csv(SOURCE / "category_statistics.csv", index=False)
    return table, summary, categories


def check_manuscript_values(table, summary, metrics, pca):
    """Compare recomputed numbers with the final manuscript at its printed precision."""
    ref = json.loads((REPOSITORY / "report/figure_reference.json").read_text())
    np.testing.assert_array_equal(
        table.loc[ref["row_order"], ref["table1_methods"]].round(3),
        ref["table1_r2_3dp"],
    )
    ablations = summary.pivot(index="key", columns="method", values="mean")
    np.testing.assert_array_equal(
        ablations.loc[ref["row_order"], ref["ablation_methods"]].round(3),
        ref["ablation_r2_3dp"],
    )
    for feature in ("Heat-related burden", "temperature_c"):
        rows = metrics.loc[metrics.feature.eq(feature)].set_index("method")
        rows = rows.loc[ref["figure4_maps"]["methods"]]
        expected = ref["figure4_maps"][feature]
        assert rows.n.eq(expected["n"]).all()
        for metric in ("r2", "mae"):
            np.testing.assert_array_equal(
                rows[metric].round(3), expected[metric + "_3dp"]
            )
    for quantity in ("hk", "beta"):
        rows = [p for p in pca if p["quantity"] == quantity]
        np.testing.assert_array_equal(
            np.round([100 * p["explained_variance"] for p in rows], 1),
            ref["pca"][quantity]["variance_percent_1dp"],
        )
        np.testing.assert_array_equal(
            np.round([p["moran_i"] for p in rows], 3),
            ref["pca"][quantity]["moran_3dp"],
        )
    changes = {}
    for name, key in (("inference_r2", "overall"), ("interpolation_r2", "all")):
        changes[name] = round(
            100 * (table.loc[key, "GWFM"] / table.loc[key, "PDFM"] - 1), 1
        )
    for name, key in (
        ("heat_mae_reduction", "Heat-related burden"),
        ("temperature_mae_reduction", "temperature_c"),
    ):
        rows = metrics.loc[metrics.feature.eq(key)].set_index("method")
        changes[name] = round(
            100 * (1 - rows.loc["GWFM", "mae"] / rows.loc["PDFM", "mae"]), 1
        )
    assert changes == ref["reported_relative_changes_percent"], changes
    return dict(
        table1_cells=176,
        ablation_cells=64,
        map_statistics=20,
        pca_panels=6,
        relative_changes=changes,
    )


def map_data(table):
    frames = [
        pd.read_csv(CURRENT / "inference/test_inferences.csv").rename(
            columns={"target": "feature"}
        ),
        pd.read_csv(CURRENT / "test_inferences.csv"),
    ]
    values = pd.concat(frames, ignore_index=True).rename(
        columns={"region_id": "node_id"}
    )
    values = values.loc[
        values.method.isin(c.METHODS)
        & values.feature.isin([TARGETS[1], "temperature_c"])
    ]
    nodes, metrics = summarize_runs(values)
    ranges = {}
    for feature in (TARGETS[1], "temperature_c"):
        maximum = float(
            nodes.loc[nodes.feature.eq(feature), "mean_absolute_error"].max()
        )
        ranges[feature] = dict(
            vmin=0.0,
            vmax=float(np.nextafter(maximum, np.inf)),
            shared_methods=c.METHODS,
        )
    metrics["vmin"] = 0.0
    metrics["vmax"] = [ranges[f]["vmax"] for f in metrics.feature]
    for row in metrics.itertuples():
        assert row.n == (120 if row.feature == TARGETS[1] else 110)
        np.testing.assert_allclose(
            row.r2, table.loc[row.feature, row.method], atol=1e-12
        )
    values.to_csv(SOURCE / "map_inferences_by_seed.csv", index=False)
    nodes.to_csv(SOURCE / "map_mean_absolute_error.csv", index=False)
    metrics.to_csv(SOURCE / "map_metrics.csv", index=False)
    save_json(SOURCE / "map_ranges.json", ranges)
    return nodes, metrics


def export_document(doc, stem):
    page = doc[0]
    assert len(doc) == 1
    if doc.name != str(OUT / f"{stem}.pdf"):
        doc.save(OUT / f"{stem}.pdf", garbage=4, deflate=True)
    page.get_pixmap(dpi=300, alpha=False).save(FIGURES / f"{stem}.png")


def spaced_table(draw, *args):
    original = Axes.text

    def text(ax, x, y, s, *other, **kwargs):
        if "\n" in str(s):
            kwargs["linespacing"] = max(1.2, kwargs.get("linespacing", 1.2))
        return original(ax, x, y, s, *other, **kwargs)

    # Slightly wider leading prevents the inherited multiline PDF glyph boxes touching.
    with patch.object(Axes, "text", text):
        return draw(*args)


def quantitative_figures(table, summary, categories):
    c.ROOT, c.ASSETS, c.SOURCE = OUT, OUT / "assets", SOURCE
    c.ASSETS.mkdir(parents=True, exist_ok=True)
    spaced_table(c.table, table.rename(columns=c.b.display))
    with pymupdf.open(OUT / "table1.pdf") as doc:
        export_document(doc, "Table1_performance")
    nodes, metrics = map_data(table)
    core = summary.loc[summary.method.isin(c.METHODS)]
    health = core.loc[core.key.isin(TARGETS + ["overall"])]
    attributes = core.loc[core.key.isin(c.b.NAMES)]
    cats = categories.loc[categories.method.isin(c.METHODS)].rename(
        columns={"modality": "key"}
    )
    c.bar(
        health,
        TARGETS + ["overall"],
        ["Respiratory-\nrelated burden", "Heat-related\nburden", "Overall"],
        "inference_bar",
    )
    c.bar(cats, c.b.CATEGORIES, c.b.CATEGORY_LABELS, "interpolation_bar")
    c.radar(
        health,
        TARGETS + ["overall"],
        ["Respiratory-related\nburden", "Heat-related\nburden", "Overall\n(mean)"],
        (0, 1),
        "inference_radar",
    )
    radar_lower = min(-0.25, float(np.floor(attributes["mean"].min() * 4) / 4))
    c.radar(
        attributes, c.b.NAMES, c.b.RADAR_LABELS, (radar_lower, 1), "interpolation_radar"
    )
    original_save = c.save_panel

    def save_vector_panel(fig, name):
        for ax in fig.axes:
            for collection in ax.collections:
                collection.set_rasterized(False)
                if isinstance(collection, QuadMesh):
                    collection.set_edgecolor("face")
        return original_save(fig, name)

    with patch.object(c, "save_panel", save_vector_panel):
        c.maps(nodes, metrics)
    c.legends()
    placements = [
        ("inference_bar", (4, 17.3, 124, 89.3)),
        ("inference_radar", (126, 7.5, 246, 89.5)),
        ("maps_heat", (248, 8, 556, 92)),
        ("interpolation_bar", (4, 103.3, 124, 175.3)),
        ("interpolation_radar", (126, 93.5, 246, 175.5)),
        ("maps_temperature", (248, 94, 556, 178)),
        ("bar_legend", (4, 178, 124, 192)),
        ("radar_legend", (126, 178, 246, 192)),
    ]
    doc = pymupdf.open()
    page = doc.new_page(width=560 * MM, height=194 * MM)
    for name, rect in placements:
        with pymupdf.open(c.ASSETS / f"{name}.pdf") as panel:
            target = pymupdf.Rect([v * MM for v in rect])
            np.testing.assert_allclose(
                [target.width, target.height],
                [panel[0].rect.width, panel[0].rect.height],
                atol=0.01,
            )
            page.show_pdf_page(target, panel, 0)
    for title, y in (
        ("Inference (Primary task)", 8),
        ("Interpolation (Auxiliary task)", 94),
    ):
        page.insert_text((4 * MM, y * MM), title, fontname="hebo", fontsize=17)
    for label, x, y in (
        ("a", 4, 17),
        ("b", 127, 17),
        ("c", 249, 17),
        ("d", 4, 103),
        ("e", 127, 103),
        ("f", 249, 103),
    ):
        page.insert_text((x * MM, y * MM), label, fontname="hebo", fontsize=14)
    assert page.get_text().count("R² =") == 10
    export_document(doc, "Figure4_performance")
    doc.close()
    return metrics


def pca_data():
    raw, _, split, _ = inputs()
    representations = rep.extract(
        dict(seed=11, checkpoint="checkpoints/GWFM/seed11/model.pt"), raw, split
    )
    geometry = np.load(DATA / "geometry.npz")["region_rook"]
    edges = np.unique(np.c_[geometry, geometry[::-1]], axis=1)
    verification = []
    for quantity in ("hk", "beta"):
        folder = SOURCE / "representations" / quantity
        folder.mkdir(parents=True, exist_ok=True)
        scores, components, mean, ratios = rep.principal_components(
            representations[quantity][:600]
        )
        np.savez_compressed(
            folder / "pca_region_seed11.npz",
            node_id=np.arange(600),
            scores=scores,
            components=components,
            mean=mean,
            explained_variance_ratio=ratios,
        )
        np.savez_compressed(
            folder / "natural_representation_seed11.npz",
            node_id=np.arange(600),
            values=representations[quantity][:600],
        )
        diagnostics = [
            dict(
                seed=11,
                level="region",
                pc=j + 1,
                explained_variance=ratios[j],
                moran_i=moran(scores[:, j], edges, True),
            )
            for j in range(3)
        ]
        pd.DataFrame(diagnostics).to_csv(
            folder / "spatial_diagnostics.csv", index=False
        )
        verification.extend([dict(quantity=quantity, **d) for d in diagnostics])
    save_json(
        SOURCE / "pca_reextraction_checks.json",
        dict(
            status="passed",
            seed=11,
            rows=600,
            no_random_mask=True,
            recomputed_from_checkpoint=True,
            panels=verification,
        ),
    )
    panels = r.render(SOURCE / "representations", OUT / "pca.pdf")
    save_json(SOURCE / "pca_panels.json", panels)
    return panels


def ablation_figure(summary):
    a.OUTPUT = OUT / "assets"
    keys = c.b.NAMES + ["all"] + TARGETS + ["overall"]
    methods = ["no_weighting", "no_modality", "no_prototype", "GWFM"]
    table = summary.pivot(index="key", columns="method", values="mean").loc[
        keys, methods
    ]
    table.to_csv(SOURCE / "ablation_r2_mean.csv")
    summary.pivot(index="key", columns="method", values="sd").loc[keys, methods].to_csv(
        SOURCE / "ablation_r2_sd.csv"
    )
    left = spaced_table(
        a.draw_table,
        table,
        c.b.LABELS + ["All 12 variables (mean)"] + c.m.INFERENCE_LABELS,
    )
    right = OUT / "pca.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=14.2 * 72, height=6.4 * 72)
    with pymupdf.open(left) as source:
        scale = (6.37 * 72 - 2) / source[0].rect.height
        x = (7.1 * 72 - source[0].rect.width * scale) / 2
        clip = pymupdf.Rect(
            0, 0, source[0].rect.width - a.REMOVED_WIDTH, source[0].rect.height
        )
        page.show_pdf_page(
            pymupdf.Rect(x, 2, x + clip.width * scale, 2 + clip.height * scale),
            source,
            0,
            clip=clip,
        )
        cells = [
            d["rect"]
            for d in source[0].get_drawings()
            if d["fill"] is not None
            and len(d["items"]) == 1
            and d["rect"].width < 100
            and d["rect"].height < 30
        ]
        assert len(cells) == 64
        top, bottom = (
            2 + min(v.y0 for v in cells) * scale,
            2 + max(v.y1 for v in cells) * scale,
        )
    with pymupdf.open(right) as source:
        boxes = [
            d["rect"]
            for d in source[0].get_drawings()
            if d["fill"] == (1.0, 1.0, 1.0)
            and len(d["items"]) == 1
            and 100 < d["rect"].width < source[0].rect.width / 2
            and abs(d["rect"].width - d["rect"].height) < 0.01
        ]
        assert len(boxes) == 6
        map_top, map_bottom = min(v.y0 for v in boxes), max(v.y1 for v in boxes)
        scale = (bottom - top) / (map_bottom - map_top)
        x, y = (
            page.rect.width - 1.44 - source[0].rect.width * scale,
            top - map_top * scale,
        )
        destination = pymupdf.Rect(
            x, y, x + source[0].rect.width * scale, y + source[0].rect.height * scale
        )
        assert page.rect.contains(destination)
        page.show_pdf_page(destination, source, 0)
        np.testing.assert_allclose(
            [y + map_top * scale, y + map_bottom * scale], [top, bottom], atol=1e-10
        )
    for label, xpos in (("a", 7), ("b", x + 2)):
        page.insert_text((xpos, 17), label, fontsize=12, fontname="hebo")
    assert page.get_text().count("Moran") == 6
    export_document(doc, "Figure5_ablation")
    doc.close()


def export_checks():
    pymupdf.TOOLS.set_small_glyph_heights(True)
    checks = []
    for name in (
        "Figure1_data",
        "Table1_performance",
        "Figure4_performance",
        "Figure5_ablation",
    ):
        with pymupdf.open(OUT / f"{name}.pdf") as doc:
            page = doc[0]
            required = {
                "Figure1_data": [
                    "Respiratory burden",
                    "Heat-related burden",
                    "Mask summary",
                    "Urbanicity sampling",
                ],
                "Table1_performance": ["Inference", "Interpolation", "Multi-", "Modal"],
                "Figure4_performance": [
                    "Heat-related burden",
                    "Temperature",
                    "Power scale",
                ],
                "Figure5_ablation": [
                    "Geographic",
                    "Modality",
                    "Prototype",
                    "PC1",
                    "PC2",
                    "PC3",
                ],
            }
            assert all(label in page.get_text() for label in required[name]), name
            spans = [
                s
                for block in page.get_text("dict")["blocks"]
                if "lines" in block
                for line in block["lines"]
                for s in line["spans"]
                if s["text"].strip()
            ]
            assert spans and min(s["size"] for s in spans) >= 5
            assert all(page.rect.contains(pymupdf.Rect(s["bbox"])) for s in spans)
            overlaps = []
            for i, s in enumerate(spans):
                for t in spans[i + 1 :]:
                    box = pymupdf.Rect(s["bbox"]) & pymupdf.Rect(t["bbox"])
                    if not box.is_empty and box.width > 0.3 and box.height > 0.3:
                        overlaps.append((s["text"], t["text"]))
            assert not overlaps, (name, overlaps)
            assert not page.get_images()
            checks.append(
                dict(
                    figure=name,
                    minimum_font_pt=min(s["size"] for s in spans),
                    editable_text=True,
                    vector_only=True,
                    text_overlaps=0,
                    all_text_inside=True,
                )
            )
    return checks


FINAL_FILES = [
    "Figure1_data.png",
    "Figure2_heterogeneity.jpg",
    "Figure3_framework.jpeg",
    "Figure4_performance.png",
    "Figure5_ablation.png",
    "Table1_performance.png",
]


def copy_manuscript_artwork():
    """Figures 2 and 3 are the final manuscript artwork, not new numerical plots."""
    reference = json.loads((REPOSITORY / "report/figure_reference.json").read_text())
    for name in FINAL_FILES[1:3]:
        source, destination = REPOSITORY / "figures" / name, FIGURES / name
        assert digest(source) == reference["static_artwork"][name]["sha256"]
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
    return reference


def main():
    global OUT
    from gwfm.visualization import data as data_plot

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    assert json.loads((PROBES / "verification.json").read_text())["status"] == "passed"
    assert (
        json.loads((CURRENT / "logs/verification.json").read_text())["status"]
        == "passed"
    )
    frozen = {
        str(p.relative_to(ROOT)): digest(p)
        for base in ("data", "checkpoints", "results/current", "results/figure_probes")
        for p in (ROOT / base).rglob("*")
        if p.is_file()
    }
    reference = copy_manuscript_artwork()
    with tempfile.TemporaryDirectory(
        prefix="gwfm-figures-"
    ) as directory, threadpool_limits(limits=2):
        OUT = Path(directory)
        table, summary, categories = statistics()
        metrics = quantitative_figures(table, summary, categories)
        pca = pca_data()
        numeric_checks = check_manuscript_values(table, summary, metrics, pca)
        ablation_figure(summary)
        data_checks = data_plot.build(OUT / "Figure1_data.pdf")
        with pymupdf.open(OUT / "Figure1_data.pdf") as doc:
            export_document(doc, "Figure1_data")
        checks = export_checks()
    for name, value in frozen.items():
        assert digest(ROOT / name) == value, name
    assert sorted(p.name for p in FIGURES.iterdir() if p.is_file()) == sorted(
        FINAL_FILES
    )
    save_json(
        SOURCE / "verification.json",
        dict(
            status="passed",
            final_files=FINAL_FILES,
            composition_pdf_checks=checks,
            delivery_format="One raster image per manuscript item; temporary vector PDFs are used for composition and layout checks.",
            data_figure=data_checks,
            manuscript_numeric_checks=numeric_checks,
            source_sha256=frozen,
            manuscript_source=reference["manuscript_source"],
            output_sha256={name: digest(FIGURES / name) for name in FINAL_FILES},
            intermediate_panels_retained=False,
            duplicate_versions=False,
        ),
    )
    print("Exported manuscript Figures 1–5 and Table 1; all checks passed.", flush=True)


if __name__ == "__main__":
    main()

"""Rebuild manuscript Table 1 and Figure 4 from the frozen local inferences."""

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, PowerNorm, to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator

from . import palette as m
from ..paths import FIGURES
from ..utils import save_json

b = m.b
METHODS = m.CORE
ROOT = FIGURES / "composite"
ASSETS = ROOT / "assets"
SOURCE = ROOT / "source_data"
MM = 72 / 25.4
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.weight": "bold",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def save_panel(fig, name):
    b.check_canvas(fig, name)
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        labels = [t for t in ax.get_xticklabels() if t.get_visible() and t.get_text()]
        for i, a in enumerate(labels):
            for other in labels[i + 1 :]:
                assert not a.get_window_extent(renderer).overlaps(
                    other.get_window_extent(renderer)
                ), (name, a.get_text(), other.get_text())
    fig.savefig(ASSETS / f"{name}.pdf")
    plt.close(fig)


def bar(frame, keys, labels, name):
    fig, ax = plt.subplots(figsize=(120 / 25.4, 72 / 25.4))
    fig.subplots_adjust(left=0.11, right=0.96, top=0.93, bottom=0.19)
    b.draw_bars(ax, frame, keys, labels, METHODS, [m.COLORS[x] for x in METHODS])
    ax.set_ylabel("")
    ax.set_yticks(np.arange(0, 1, 0.2))
    ax.tick_params(axis="both", labelsize=13, pad=3)
    for collection in ax.collections:
        collection.set_linewidth(0.85)
    for line in ax.lines:
        line.set_markeredgewidth(0.7)
        line.set_markersize(3)
    save_panel(fig, name)


def radar(frame, keys, labels, limits, name):
    fig = plt.figure(figsize=(120 / 25.4, 82 / 25.4))
    ax = fig.add_axes(
        [31 / 120, (82 - 42 - 29) / 82, 58 / 120, 58 / 82], projection="polar"
    )
    theta = np.linspace(0, 2 * np.pi, len(keys), endpoint=False)
    closed = np.r_[theta, theta[0]]
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(*limits)
    ticks = np.arange(0, 1.01, 0.25)
    ax.set_yticks(ticks)
    ax.set_yticklabels([])
    ax.set_xticks(theta)
    labels = [
        s.replace("Older population\nshare", "Older\npopulation\nshare") for s in labels
    ]
    ax.set_xticklabels(labels, fontsize=12.5)
    ax.tick_params(axis="x", pad=8)
    for angle, label in zip(theta, ax.get_xticklabels()):
        label.set_ha(
            "left"
            if 0.01 < angle < np.pi - 0.01
            else "right" if angle > np.pi + 0.01 else "center"
        )
    ax.grid(color="#d5dadd", lw=0.5)
    ax.spines["polar"].set_color("#879096")
    ax.spines["polar"].set_linewidth(0.55)
    if limits[0] < 0:
        angles = np.linspace(0, 2 * np.pi, 361)
        ax.plot(
            angles, np.zeros_like(angles), color="#7d858b", ls="--", lw=0.6, zorder=1
        )
    for method in METHODS:
        values = (
            frame[frame.method.eq(method)].set_index("key").loc[keys, "mean"].to_numpy()
        )
        assert (
            np.isfinite(values).all()
            and (values >= limits[0]).all()
            and (values <= limits[1]).all()
        )
        line, color = np.r_[values, values[0]], m.RADAR_COLORS[method]
        ax.fill(closed, line, color=color, alpha=0.10, lw=0, zorder=2)
        ax.plot(
            closed,
            line,
            color=color,
            lw=1.4 if method == "GWFM" else 1.05,
            ls=":" if method == "KNN" else "-",
            zorder=4,
        )
    assert len(ax.patches) == 5 and not ax.collections
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = ax.get_window_extent(renderer)
    cx, cy, radius = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, box.width / 2
    for label in ax.get_xticklabels():
        extent = label.get_window_extent(renderer)
        dx = max(extent.x0 - cx, 0, cx - extent.x1)
        dy = max(extent.y0 - cy, 0, cy - extent.y1)
        assert np.hypot(dx, dy) > radius + 1, (
            name,
            label.get_text(),
            "label intrudes into radar",
        )
    save_panel(fig, name)


def maps(nodes, metrics):
    geo = json.loads(b.GEOMETRY.read_text())["features"]
    polygons = {
        g["properties"]["node_id"]: np.asarray(g["geometry"]["coordinates"][0])
        for g in geo
    }
    assert len(polygons) == 600
    for feature, label, unit, cmap, name in (
        (
            "Heat-related burden",
            "Heat-related burden\n(n = 120)",
            "score",
            "Reds",
            "maps_heat",
        ),
        ("temperature_c", "Temperature\n(n = 110)", "°C", "Greens", "maps_temperature"),
    ):
        fig = plt.figure(figsize=(308 / 25.4, 84 / 25.4))
        axes = []
        for i, method in enumerate(METHODS):
            ax = fig.add_axes([(10 + i * 58.5) / 308, 13 / 84, 57 / 308, 57 / 84])
            axes.append(ax)
            data = nodes[
                nodes.feature.eq(feature) & nodes.method.eq(method)
            ].sort_values("node_id")
            score = metrics[
                metrics.feature.eq(feature) & metrics.method.eq(method)
            ].iloc[0]
            ids, errors = (
                data.node_id.to_numpy(int),
                data.mean_absolute_error.to_numpy(),
            )
            assert len(ids) == score.n and not data.node_id.duplicated().any()
            np.testing.assert_allclose(errors.mean(), score.mae, atol=1e-12)
            # The manuscript uses the same power scale for all five heat-error maps.
            norm = (
                PowerNorm(1.5, score.vmin, score.vmax, clip=False)
                if feature == "Heat-related burden"
                else Normalize(score.vmin, score.vmax, clip=False)
            )
            assert errors.min() >= norm.vmin and errors.max() <= norm.vmax
            ax.add_collection(
                PolyCollection(
                    list(polygons.values()),
                    facecolors="#e4e7eb",
                    edgecolors="white",
                    linewidths=0.195,
                )
            )
            pc = PolyCollection(
                [polygons[j] for j in ids],
                array=errors,
                cmap=cmap,
                norm=norm,
                edgecolors="white",
                linewidths=0.286,
            )
            ax.add_collection(pc)
            ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[], yticks=[])
            ax.set_title(
                f"{method}\nR² = {score.r2:.3f} | MAE = {score.mae:.3f}",
                fontsize=13,
                pad=6,
            )
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#c5cbd0")
                spine.set_linewidth(0.65)
            if i == 0:
                ax.set_ylabel(label, fontsize=12, labelpad=3)
        cax = fig.add_axes([10 / 308, 8 / 84, 208 / 308, 2 / 84])
        cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
        cb.solids.set_rasterized(False)
        cb.solids.set_edgecolor("face")
        cb.ax.tick_params(labelsize=12.5, width=0.65, length=2.6, pad=2.6)
        ticks = MaxNLocator(4).tick_values(norm.vmin, norm.vmax)
        cb.set_ticks(ticks[(ticks >= norm.vmin) & (ticks <= norm.vmax)])
        description = fig.text(
            230 / 308,
            9 / 84,
            f"Mean absolute error ({unit})",
            fontsize=13,
            ha="left",
            va="center",
        )
        if feature == "Heat-related burden":
            fig.text(
                230 / 308,
                3 / 84,
                "Power scale (γ = 1.5)",
                fontsize=10,
                ha="left",
                va="center",
            )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        titles = [ax.title.get_window_extent(renderer) for ax in axes]
        assert all(a.x1 + 1 < b.x0 for a, b in zip(titles, titles[1:]))
        assert cax.get_window_extent(renderer).y1 < min(
            ax.get_window_extent(renderer).y0 for ax in axes
        )
        assert (
            description.get_window_extent(renderer).x0
            > cax.get_window_extent(renderer).x1
        )
        save_panel(fig, name)


def table(values):
    data = {"columns": list(values.columns), "rows": []}
    names = dict(
        zip(b.NAMES + ["all"] + b.RISKS + ["overall"], m.LABELS + m.INFERENCE_LABELS)
    )
    for key in b.RISKS + ["overall"] + b.NAMES + ["all"]:
        row = values.loc[key].to_numpy()
        cmap = m.ORANGE if key in b.RISKS + ["overall"] else m.GREEN
        relative = (
            (row - row.min()) / np.ptp(row)
            if np.ptp(row) > 1e-12
            else np.full(len(row), 0.5)
        )
        colors = [cmap(x) for x in relative]
        data["rows"].append(
            dict(
                key=key,
                label=names[key],
                values=row.tolist(),
                fills=[to_hex(x)[1:] for x in colors],
                text_colors=[m.text_color(x)[1:] for x in colors],
            )
        )
    width, height, left, cell_width, row_height = 280, 145, 78, 18, 6.8
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis("off")
    ax.text(
        4, 10, "Inference (Primary task)", fontsize=14, va="center", color="#a23d0d"
    )
    ax.text(
        4,
        47.5,
        "Interpolation (Auxiliary task)",
        fontsize=14,
        va="center",
        color="#075633",
    )
    headers, numbers, labels = [], [], []
    for col, method in enumerate(data["columns"]):
        header = (
            method.replace(" (", "\n")
            .replace(")", "")
            .replace("Multi-Modal", "Multi-\nModal")
        )
        headers.append(
            ax.text(
                left + (col + 0.5) * cell_width,
                10,
                header,
                fontsize=12,
                ha="center",
                va="center",
                linespacing=1.05,
                color="#075633" if method.startswith("GWFM") else "#172126",
            )
        )
    for i, row in enumerate(data["rows"]):
        y = 19.5 + i * row_height if i < 3 else 52.8 + (i - 3) * row_height
        labels.append(
            ax.text(
                left - 2,
                y + row_height / 2,
                row["label"],
                fontsize=12,
                ha="right",
                va="center",
            )
        )
        for col, (value, color, foreground) in enumerate(
            zip(row["values"], row["fills"], row["text_colors"])
        ):
            x = left + col * cell_width
            ax.add_patch(
                Rectangle(
                    (x, y),
                    cell_width,
                    row_height,
                    facecolor="#" + color,
                    edgecolor="none",
                )
            )
            numbers.append(
                ax.text(
                    x + cell_width / 2,
                    y + row_height / 2,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="#" + foreground,
                )
            )
        if i in (6, 10, 14):
            ax.plot(
                [4, left + 11 * cell_width],
                [y + row_height] * 2,
                color="#b7c4ba",
                lw=0.5,
            )
    for start, count in ((19.5, 3), (52.8, 13)):
        for col in (3, 7):
            ax.plot(
                [left + col * cell_width] * 2,
                [start, start + count * row_height],
                color="white",
                lw=1.3,
            )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    cell_pixels = cell_width / 25.4 * fig.dpi
    row_pixels = row_height / 25.4 * fig.dpi
    for t in headers + numbers:
        assert t.get_window_extent(renderer).width < 0.94 * cell_pixels, t.get_text()
    for t in numbers + labels:
        assert t.get_window_extent(renderer).height < 0.88 * row_pixels, t.get_text()
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text():
            assert fig.bbox.contains(
                t.get_window_extent(renderer).x0, t.get_window_extent(renderer).y0
            )
            assert fig.bbox.contains(
                t.get_window_extent(renderer).x1, t.get_window_extent(renderer).y1
            )
    assert len(numbers) == len(ax.patches) == 176
    fig.savefig(ROOT / "table1.pdf")
    plt.close(fig)
    save_json(SOURCE / "table1.json", data)


def legends():
    for name in ("bar", "radar"):
        fig = plt.figure(figsize=(120 / 25.4, 14 / 25.4))
        handles = (
            [Rectangle((0, 0), 1, 1, color=m.COLORS[x]) for x in METHODS]
            if name == "bar"
            else [
                Line2D(
                    [0],
                    [0],
                    color=m.RADAR_COLORS[x],
                    lw=1.5,
                    ls=":" if x == "KNN" else "-",
                )
                for x in METHODS
            ]
        )
        fig.legend(
            handles,
            METHODS,
            loc="center",
            ncol=3,
            fontsize=12.5,
            columnspacing=1.4,
            handlelength=1.6,
            handletextpad=0.6,
        )
        save_panel(fig, name + "_legend")

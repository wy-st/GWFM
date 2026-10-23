"""Place primary inference above auxiliary interpolation beside the aligned PCA maps."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ..paths import FIGURES
from .style import check_canvas
from .palette import GREEN, ORANGE, text_color

OUTPUT = FIGURES / "ablation"
REMOVED_WIDTH = 7.1 * 72 * 0.98 * 0.965 / 7


def draw_table(table, labels):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.weight": "bold",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(7.1, 6.35))
    fig.subplots_adjust(
        left=0.015, right=0.995 - 0.98 * 0.965 / 7, top=0.995, bottom=0.015
    )
    ax.set(xlim=(0, 7 - 0.965), ylim=(-0.65, 18.4))
    ax.axis("off")
    left, width = 2.12, 0.965
    top, gap, row_height = 16.05, 1.5, 0.9375
    interpolation_top = top - 3 * row_height - gap
    titles = [
        "w/o\nGeographic\nweighting",
        "w/o\nModality\nperception",
        "w/o\nPrototype\nregularization",
        "GWFM\n(Full)",
    ]
    ax.text(
        0.25,
        17.95,
        "Inference (Primary task)",
        color="#a23d0d",
        fontsize=13,
        va="center",
    )
    for col, title in enumerate(titles):
        ax.text(
            left + width * (col + 0.5),
            16.85,
            title,
            fontsize=10,
            color="#075633" if col == 3 else "#182126",
            ha="center",
            va="center",
            linespacing=1.08,
        )
    display_rows = list(range(13, 16)) + list(range(13))
    assert len(set(display_rows)) == len(table) == 16
    for row in display_rows:
        values = table.iloc[row].to_numpy()
        y = (
            top - (row - 13 + 0.5) * row_height
            if row >= 13
            else interpolation_top - (row + 0.5) * row_height
        )
        cmap = ORANGE if row >= 13 else GREEN
        spread = np.ptp(values)
        relative = (
            (values - values.min()) / spread if spread > 1e-12 else np.full(4, 0.5)
        )
        for col, value in enumerate(values):
            # Match the reference: color encodes relative R² within each variable.
            color = cmap(float(relative[col]))
            ax.add_patch(
                Rectangle(
                    (left + width * col, y - row_height / 2),
                    width,
                    row_height,
                    facecolor=color,
                    edgecolor="none",
                )
            )
            ax.text(
                left + width * (col + 0.5),
                y,
                f"{value:.3f}",
                fontsize=12.8,
                ha="center",
                va="center",
                color=text_color(color),
            )
        label = labels[row]
        if row == 13:
            label = "Respiratory-related\nburden"
        ax.text(
            left - 0.10,
            y,
            label,
            fontsize=11.5 if row == 13 else 12,
            ha="right",
            va="center",
            linespacing=1.0,
        )
        if row in (3, 7, 11):
            ax.plot(
                [0.05, 6.95 - 0.965], [y - row_height / 2] * 2, color="#cbd7d0", lw=0.55
            )
    ax.text(
        0.25,
        interpolation_top + gap / 2,
        "Interpolation (Auxiliary task)",
        color="#075633",
        fontsize=13,
        va="center",
    )
    check_canvas(fig, "final_ablation_table")
    renderer = fig.canvas.get_renderer()
    boxes = [text.get_window_extent(renderer) for text in ax.texts]
    for i, box in enumerate(boxes):
        for j in range(i):
            assert not box.overlaps(boxes[j]), (
                ax.texts[i].get_text(),
                ax.texts[j].get_text(),
            )
    numbers = [
        text for text in ax.texts if text.get_text().replace(".", "", 1).isdigit()
    ]
    assert len(numbers) == len(ax.patches) == 64
    for text, cell in zip(numbers, ax.patches):
        box, slot = text.get_window_extent(renderer), cell.get_window_extent(renderer)
        assert slot.contains(box.x0, box.y0) and slot.contains(box.x1, box.y1)
    stem = OUTPUT / "ablation_r2"
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    return stem.with_suffix(".pdf")

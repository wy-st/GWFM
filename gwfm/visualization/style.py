"""Shared labels, plot formatting, and bounds checks for final manuscript figures."""

import json

import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..paths import DATA

GEOMETRY = DATA / "regions.geojson"
CORE = ["KNN", "IDW", "GPR", "PDFM", "GWFM"]
BLOCKS = [f"{model} (Modal-{m})" for model in ("PDFM", "GWFM") for m in (1, 2, 3)]
ORDER = CORE[:3] + BLOCKS[:3] + ["PDFM"] + BLOCKS[3:] + ["GWFM"]
COLORS = {
    "KNN": "#7d8993",
    "IDW": "#75b39d",
    "GPR": "#dfa078",
    "PDFM": "#899bc4",
    "GWFM": "#bb679d",
}
LINESTYLES = {"KNN": ":", "IDW": "--", "GPR": "-.", "PDFM": "-", "GWFM": "-"}
BLOCK_COLORS = ["#5a91b7", "#d3a067", "#75a893", "#b7669e"]
CATEGORIES = ["climate", "pollution", "sociodemographic"]
CATEGORY_LABELS = ["Climate", "Pollution", "Sociodemographic"]
NAMES = json.loads((DATA / "schema.json").read_text())["features"]
LABELS = [
    "Temperature",
    "Relative humidity",
    "Annual precipitation",
    "Hot-day fraction",
    "PM2.5",
    "NO2",
    "Ozone",
    "Emission index",
    "Population density",
    "Deprivation index",
    "Older population share",
    "Healthcare access",
]
RADAR_LABELS = [
    "Temperature",
    "Relative\nhumidity",
    "Annual\nprecipitation",
    "Hot-day\nfraction",
    "PM2.5",
    "NO2",
    "Ozone",
    "Emission\nindex",
    "Population\ndensity",
    "Deprivation\nindex",
    "Older population\nshare",
    "Healthcare\naccess",
]
RISKS = ["Respiratory burden", "Heat-related burden"]
RISK_LABELS = ["Respiratory health\nburden", "Heat-related health\nburden"]
UNITS = json.loads((DATA / "schema.json").read_text())["units"]
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 13,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.facecolor": "white",
    }
)
CHECKS = []


def display(method):
    return method + " (Multi-Modal)" if method in ("PDFM", "GWFM") else method


def check_canvas(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text():
            b = t.get_window_extent(renderer)
            # Text outside the figure would be lost with the fixed-size export.
            assert (
                b.x0 >= -1
                and b.y0 >= -1
                and b.x1 <= fig.bbox.width + 1
                and b.y1 <= fig.bbox.height + 1
            ), (name, t.get_text(), b)
    CHECKS.append(
        dict(
            figure=name,
            axes=len(fig.axes),
            canvas_text_inside=True,
            width_inches=fig.get_figwidth(),
            height_inches=fig.get_figheight(),
        )
    )


def draw_bars(ax, frame, keys, labels, methods, colors):
    width = 0.78 / len(methods)
    x = np.arange(len(keys))
    bounds = [0.0]
    for i, method in enumerate(methods):
        d = frame[frame.method == method].set_index("key").loc[keys]
        mean = d["mean"].to_numpy()
        sd = d["sd"].fillna(0).to_numpy()
        pos = x + (i - (len(methods) - 1) / 2) * width
        ax.bar(
            pos, mean, width=width * 0.94, color=colors[i], edgecolor="none", zorder=3
        )
        finite = d["sd"].notna().to_numpy()
        if finite.any():
            ax.errorbar(
                pos[finite],
                mean[finite],
                yerr=sd[finite],
                fmt="none",
                ecolor="#44494c",
                elinewidth=1.4,
                capsize=2.4,
                zorder=5,
            )
        bounds.extend((mean - sd).tolist())
        bounds.extend((mean + sd).tolist())
    lo, hi = min(bounds), max(bounds)
    pad = 0.10 * max(hi - lo, 0.3)
    ax.set_ylim(lo - pad, hi + pad)
    # Explicitly omit locator ticks outside the axes; canvas checks inspect their text too.
    ticks = MaxNLocator(nbins=5).tick_values(lo - pad, hi + pad)
    ax.set_yticks(ticks[(ticks >= lo - pad) & (ticks <= hi + pad)])
    ax.set_xticks(x, labels, fontsize=12.5)
    ax.set_ylabel("R²", fontsize=15)
    ax.axhline(0, color="#6b7277", lw=0.85, zorder=2)
    ax.yaxis.grid(True, color="#eceff1", lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=12)

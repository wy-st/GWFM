"""Shared colors, labels, and readable table text for the report figures."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from . import style as b

LABELS = b.LABELS + ["All 12 variables (mean)"]
INFERENCE_TARGETS = b.RISKS + ["overall"]
CORE = ["KNN", "IDW", "GPR", "PDFM", "GWFM"]
ORDER = (
    CORE[:3]
    + [f"PDFM (Modal-{i})" for i in (1, 2, 3)]
    + ["PDFM"]
    + [f"GWFM (Modal-{i})" for i in (1, 2, 3)]
    + ["GWFM"]
)
COLORS = {
    "KNN": "#7d8993",
    "IDW": "#75b39d",
    "GPR": "#dfa078",
    "PDFM": "#899bc4",
    "GWFM": "#bb679d",
}
RADAR_COLORS = {
    "KNN": "#777777",
    "IDW": "#ff7f0e",
    "GPR": "#2ca02c",
    "PDFM": "#d62728",
    "GWFM": "#007fba",
}
INFERENCE_LABELS = [
    "Respiratory-related burden",
    "Heat-related burden",
    "Overall inference",
]
GREEN = LinearSegmentedColormap.from_list(
    "attribute_contrast", ["#f5faf5", "#d8ead9", "#87c18a", "#38995b", "#075633"]
)
ORANGE = LinearSegmentedColormap.from_list(
    "outcome_contrast", ["#fff8ef", "#fee0b5", "#f6ad68", "#df762c", "#a23d0d"]
)
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    }
)


def text_color(rgba):
    rgb = np.asarray(rgba[:3])
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = float(linear @ [0.2126, 0.7152, 0.0722])
    black, white = (luminance + 0.05) / 0.05, 1.05 / (luminance + 0.05)
    assert max(black, white) >= 4.5
    return "#000000" if black >= white else "#ffffff"

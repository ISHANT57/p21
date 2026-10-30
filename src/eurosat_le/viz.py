"""Shared Matplotlib style and colours for report figures.

Categorical hues are assigned in this fixed order and always follow the same
entity (e.g. a model keeps its colour in every figure). Never cycle them.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

# Blue <-> red with a neutral grey midpoint (polarity, e.g. z-scores).
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#104281", "#3987e5", "#f0efec", "#e34948", "#9e2b2b"]
)
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#fcfcfb", *SEQUENTIAL_BLUE])


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "legend.frameon": False,
        "font.size": 9,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })

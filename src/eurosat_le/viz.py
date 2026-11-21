"""Shared Matplotlib style and colours for report figures.

Categorical hues are assigned in this fixed order and always follow the same
entity (e.g. a model keeps its colour in every figure). Never cycle them.

Used by `eurosat_le.figures` (Phase 7 report figures) and `scripts/inspect_dataset.py`. This module only
defines colour constants, two colormaps and `apply_style`; it does not draw anything itself.

Design decisions:
- All colours are module constants so that a figure never hard-codes its own palette.
- The "Agg" backend is selected at import time (before pyplot is imported), so figures can be written to
  PNG on machines without a display (Colab, CI, a plain terminal). This is why the pyplot import below
  carries a `noqa: E402` marker.
- The page/figure background is SURFACE (an off-white), and text uses the TEXT_* greys, never a series
  colour, so labels stay readable whatever hue a line has.
"""

# Lets type hints use the modern syntax on all supported Python versions.
from __future__ import annotations

# matplotlib is the plotting library; its backend must be chosen before pyplot is imported.
import matplotlib

# Use the "Agg" backend: it draws straight into image files and needs no screen (works on Colab and servers).
matplotlib.use("Agg")
# pyplot is the drawing interface (used here to change global style settings); LinearSegmentedColormap builds
# our own smooth colour scales from a short list of colours.
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# ======================================================================
# CONCEPTS USED IN THIS FILE
# ======================================================================
# matplotlib: the Python plotting library used for every figure. pyplot is its drawing interface.
# backend ('Agg'): the part of matplotlib that does the drawing. 'Agg' writes straight to image files and needs no
#   screen, so figures can be made on Colab, servers or in a terminal. It must be chosen before pyplot is imported,
#   hence the '# noqa: E402' markers (the linter would complain about an import after code).
# rcParams: matplotlib's global settings dictionary (colours, line width, font size, dpi). apply_style changes it once
#   so all figures look the same. It is process-wide state.
# categorical palette (CATEGORICAL): distinct hues for separate things, such as the models. A model always gets the same
#   colour (figures.MODEL_COLOR zips the model list with this list).
# sequential palette (SEQUENTIAL_BLUE, SEQUENTIAL): a light-to-dark ramp for magnitudes, used for confusion matrices and
#   per-class F1 heatmaps.
# diverging palette (DIVERGING): blue to red with a grey middle, for values on both sides of zero. Used by
#   scripts/inspect_dataset.py (z-scores), not by figures.py.
# LinearSegmentedColormap.from_list: builds a smooth colour scale by blending between the listed colours.
# hex colour: '#rrggbb', the red, green and blue amounts in base 16.
# text colours vs series colours: titles and labels use the grey TEXT_* tokens, never a series colour, so text stays
#   readable. Figures also add markers and direct labels, so colour is not the only cue.
# spines and grid: spines are the frame lines around the axes (the top and right ones are hidden); the light grid is
#   drawn behind the data (axisbelow).
# dpi and bbox_inches='tight': dots per inch of the saved image (200) and trimming of the empty border so legends
#   outside the axes are not cut off.
# 'from __future__ import annotations': lets type hints use the modern syntax on older Python 3 versions.

# ======================================================================
# Q&A ABOUT THIS FILE (file-level viva questions)
# ======================================================================
# Q: What is this file for?
#    A: It defines the shared colours (as module constants), two custom colour scales and apply_style, so every report
#       figure has the same look. It draws nothing itself.
# Q: Where does it fit in the pipeline?
#    A: figures.py (Phase 7 report figures), scripts/evaluate.py and scripts/inspect_dataset.py import it and call
#       apply_style() before drawing.
# Q: Why is matplotlib.use('Agg') called at import time?
#    A: So that figures can be saved as PNG on machines without a display. The backend must be set before pyplot is
#       imported, and importing this module early guarantees that.
# Q: Why keep colours in constants instead of in each figure?
#    A: A figure never hard-codes its own palette, so a model keeps one colour everywhere and the look can be changed in
#       one place.

# Series hues in assignment order; `figures.MODEL_COLOR` zips the model list with this list.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Light-to-dark blue ramp for magnitudes (confusion matrices, per-class F1 heatmaps).
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# Near-black for titles and the most important text.
TEXT_PRIMARY = "#0b0b0b"
# Medium grey for axis labels, tick labels and direct line labels (never a series colour).
TEXT_SECONDARY = "#52514e"
# Very light grey for grid lines and axis edges.
GRID = "#e4e3df"
# Off-white background used for the figure and the axes.
SURFACE = "#fcfcfb"

# Blue <-> red with a neutral grey midpoint (polarity, e.g. z-scores).
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#104281", "#3987e5", "#f0efec", "#e34948", "#9e2b2b"]
)
# Smooth colour scale from the off-white background up to dark blue, for values from low to high.
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#fcfcfb", *SEQUENTIAL_BLUE])


# Q: What does this function do?
#    A: It writes the report look into matplotlib.rcParams: off-white background, no top and right frame lines, a light
#       grid behind the data, grey axis text, bold titles, no legend frame, font size 9 and 200 dpi tight-bbox saving.
# Q: Why is it called at the start of every figure function?
#    A: rcParams are global settings that matplotlib reads when a figure is created, so the style must be installed
#       before drawing; calling it each time guarantees the same look whichever figure is made first.
# Q: What are the inputs and what does it return?
#    A: None and nothing; it only changes matplotlib's global settings (process-wide state).
# Q: Why bbox 'tight'?
#    A: Some legends are placed outside the axes; tight mode enlarges the saved area so they are not cropped.
def apply_style() -> None:
    """Install the report look into `matplotlib.rcParams` (global, process-wide state).

    Call once at the start of every figure function. Sets the off-white background, top/right spines
    off, a light grid drawn behind the data, grey axis text, bold titles, no legend frame, and saves at
    200 dpi with a tight bounding box (so legends placed outside the axes are not cropped).

    Takes no arguments and returns nothing; it only changes matplotlib's global settings.
    """
    # Update many matplotlib defaults at once, so every figure made afterwards looks the same.
    plt.rcParams.update({
        # Off-white background for the whole figure, for the plotting area and for the saved image file.
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        # Axis line, axis label and title colours (title is bold and darkest).
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titleweight": "bold",
        # Hide the top and right frame lines for a cleaner look.
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Show a light grid, drawn behind the data so it does not hide it.
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        # Grey tick labels.
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        # Default line thickness and marker size for curves.
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        # No box around legends.
        "legend.frameon": False,
        # Small default text size suited to a two-column report.
        "font.size": 9,
        # Save images at 200 dots per inch and trim the empty border (keeps outside legends from being cut off).
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })

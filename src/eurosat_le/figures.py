"""Report figures built from the Phase 6 tables (results/summary/eval/*.csv). Phase 7.

Every function takes plain DataFrames and an output path, and returns the path, or
None when the data for that figure is not there yet (so the script can run on a
partial set of results). Nothing here reads run folders or trains anything.

Colour rule: a model keeps its colour (and marker) in every figure. Text uses text
tokens, never the series colour. The green is below 3:1 contrast on the surface, so
every line is also direct-labelled and has its own marker shape.

Pipeline position: Phase 7. `scripts/evaluate.py` (Phase 6) writes the CSV tables under
results/summary/eval/; `scripts/make_figures.py` reads them with pandas and calls the functions below,
which write PNGs (normally into results/figures/). Tables are built by `eurosat_le.tables`.

Inputs (columns the functions rely on):
- `groups` (groups.csv): model, modality, fraction, variant, test_f1_mean/_std, test_acc_mean/_std.
- `pairs` (compare_*_pairs.csv): one row per seed, with model, modality, fraction and delta_f1.
- `per_class` (per_class.csv): variant, model, modality, fraction, class, f1_mean.
- confusion matrices: already row-normalised [10, 10] arrays, averaged over the seeds.

Design decisions:
- Every score is mean +- std over the 3 seeds; the paired-gain figures also show one dot per seed.
- A model keeps one colour and one marker everywhere (MODEL_COLOR / MODEL_MARKER); colours come from
  `viz.CATEGORICAL` in the order of MODELS. Series are also labelled directly, so colour alone is not
  the only cue.
- Each function returns the written path, or None when its data is empty, so partial results still
  produce whichever figures are possible. Only models/modalities/fractions present in the data are drawn.
"""

# Lets type hints such as "Path | None" be written the modern way on all supported Python versions.
from __future__ import annotations

# Path is the type of the output file locations.
from pathlib import Path

# numpy does the small numeric helpers (sorting, arrays); pandas holds the evaluation tables.
import numpy as np
import pandas as pd

# viz holds the shared colours and the plot style; CLASSES is the ordered list of the ten class names.
from eurosat_le import viz
from eurosat_le.constants import CLASSES

# ======================================================================
# CONCEPTS USED IN THIS FILE
# ======================================================================
# matplotlib figure, axes, subplots: a figure is the whole image; axes are one plot panel inside it; plt.subplots(1, n,
#   sharey=True) makes n panels side by side that share the y scale so they compare directly.
# groups / pairs / per_class DataFrames: groups.csv (mean and std per setting), compare_*_pairs.csv (one row per matched
#   seed) and per_class.csv (per-class F1). All are written earlier by scripts/evaluate.py.
# label fraction and label efficiency: the share of labelled training images used (1%, 10%, 100%). A label-efficiency
#   curve shows how the score changes as more labels are available.
# log-scaled x axis: each tick is a factor of 10 from the last, so 1%, 10% and 100% are evenly spaced (_frac_ticks).
# macro-F1 and accuracy: macro-F1 averages the per-class F1 (harmonic mean of precision and recall) with equal weight;
#   accuracy is the share of correct predictions. metric_columns chooses which columns are plotted.
# mean ± std over seeds and error bars: the mean over repeated runs (seeds) is the point, the sample std is the vertical
#   bar (ax.errorbar). A NaN std (one seed) is drawn as 0 with fillna(0).
# paired difference per seed: for the same seed, the score of one setup minus the other (delta_f1). rgb_vs_ms_gain uses
#   13-band minus RGB; pseudo_label_gain uses student minus teacher.
# bar plus dots: a pale bar is the mean difference over seeds and one dot per seed shows the spread. Above the zero line
#   the first-named setup helps, below it hurts.
# pseudo-labelling (pl, plb), teacher and student: a teacher labels unlabelled images and a student trains on them;
#   'plb' is the class-balanced version, drawn with hatching.
# confusion matrix: a table of true class (rows) against predicted class (columns). Row normalisation divides each row
#   by its sum so every row adds to 1 (100 percent) and classes of different size compare fairly. The diagonal is the
#   per-class recall.
# per-class F1 heatmap: a colour grid with one row per land-cover class and one column per model / input (imshow); built
#   with pivot_table(index='class', columns='col', values='f1_mean') and reindex(CLASSES).
# colour scale limits (vmin, vmax) and colormap: vmin / vmax fix which values get the lightest and darkest colour;
#   viz.SEQUENTIAL is the light-to-dark blue map. The confusion matrices use 0..1 in every panel.
# colour rule: a model keeps one colour (MODEL_COLOR) and one marker shape (MODEL_MARKER) in every figure, lines are
#   also labelled directly, and text uses grey tokens, so colour is not the only cue. The green has below 3:1 contrast
#   on the background, which is why markers and labels are added.
# direct labels and _spread: the model name is written at the end of its line instead of only in a legend; _spread
#   nudges label heights apart so they do not overlap.
# NaN: pandas' 'not a number' marker for a missing value (a one-seed std, a class with no data).
# np.argsort, np.full, np.nanmin: numpy helpers to get the sort order of values, make an array of one repeated value
#   (the x position of the dots) and take a minimum that ignores NaN.
# 'return path or None' pattern: each figure function returns the written path, or None when its data is empty, so the
#   script can run on partial results.
# lazy import of pyplot: 'import matplotlib.pyplot as plt' is placed inside the functions. viz (imported at the top)
#   already selected the non-interactive 'Agg' backend, so figures save without a display.
# dpi and closing figures: the style saves at 200 dpi; _save closes each figure with plt.close to free memory.
# 'from __future__ import annotations': lets type hints like 'Path | None' work on older Python 3 versions.

# ======================================================================
# Q&A ABOUT THIS FILE (file-level viva questions)
# ======================================================================
# Q: What is this file for?
#    A: It draws the report figures (label-efficiency curves, RGB-vs-13-band gain, pseudo-label gain, confusion
#       matrices, per-class F1 heatmap) from the evaluation tables, and defines the shared names, colours and markers of
#       the models.
# Q: Where does it fit in the pipeline?
#    A: Phase 7. scripts/evaluate.py (Phase 6) writes results/summary/eval/*.csv; scripts/make_figures.py loads them
#       with pandas and calls the functions here, which write PNG files (normally into results/figures). Tables are
#       built by eurosat_le.tables, which also reuses the names defined here.
# Q: Does this file read run folders or train anything?
#    A: No. Every function takes plain DataFrames (or arrays) and an output path, so it only plots results that already
#       exist.
# Q: Why does every figure function return a path or None?
#    A: None means 'no data for this figure yet'. The script can then skip it with a message and still produce the other
#       figures from a partial set of results.

# The model names in the fixed order used everywhere (this order decides colours and table rows).
MODELS = ["scratch_cnn", "resnet50", "vit_small", "vit_scratch"]
# Human-readable model names for legends, titles and tables.
MODEL_LABEL = {"scratch_cnn": "Scratch CNN", "resnet50": "ResNet-50", "vit_small": "ViT-Small",
               "vit_scratch": "ViT-Tiny (scratch)"}
# One fixed colour per model (the first colours of the shared palette, in MODELS order).
MODEL_COLOR = dict(zip(MODELS, viz.CATEGORICAL))
# One fixed marker shape per model, so lines are told apart even without colour.
MODEL_MARKER = {"scratch_cnn": "o", "resnet50": "s", "vit_small": "^", "vit_scratch": "D"}
# Readable names for the two kinds of input images (3 colour bands, or all 13 satellite bands).
MODALITY_LABEL = {"rgb": "RGB (3 bands)", "ms": "13 bands"}
# Readable labels for the share of labelled training images.
FRACTION_LABEL = {0.01: "1%", 0.1: "10%", 1.0: "100%"}
# Readable names for the two pseudo-labelling variants.
VARIANT_LABEL = {"pl": "pseudo-label (global τ)", "plb": "pseudo-label (class-balanced)"}


# Q: What does this function do?
#    A: It returns the names of the mean and std columns in groups.csv for a metric: ('test_f1_mean', 'test_f1_std') for
#       'f1' and ('test_acc_mean', 'test_acc_std') for anything else.
# Q: Why is it a function used by both figures and tables?
#    A: Both need the same column names; having one place avoids typing mistakes and keeps them consistent.
def metric_columns(metric: str) -> tuple[str, str]:
    """Names of the mean and std columns of groups.csv for a metric.

    `metric` is "f1" for test macro-F1; any other value means test accuracy. Returns
    ("test_f1_mean", "test_f1_std") or ("test_acc_mean", "test_acc_std").
    """
    # Choose the short metric name used inside the column names.
    name = "f1" if metric == "f1" else "acc"
    # Build the two column names from it.
    return f"test_{name}_mean", f"test_{name}_std"


# Q: What does this function do?
#    A: It lists the models that occur in a table's 'model' column, in the fixed MODELS order.
# Q: Why walk through MODELS instead of using the order in the data?
#    A: The order of rows in a CSV can change. The fixed order keeps colours, legends and bars in the same sequence in
#       every figure.
# Q: What are the input and output?
#    A: A DataFrame with a 'model' column in; a list of model name strings out.
def _models_present(df: pd.DataFrame) -> list[str]:
    """Models found in `df["model"]`, in the fixed MODELS order (not the order of appearance).

    Args:
        df: any table with a "model" column.
    Returns:
        The model names that occur in the table, sorted by the fixed MODELS order.
    """
    # Walk through the fixed list so the order never depends on how the rows happen to be arranged.
    return [m for m in MODELS if m in set(df["model"])]


# Q: What does this function do?
#    A: It sets the x axis of a panel to a log scale with one tick per label fraction, written as 1%, 10%, 100%.
# Q: Why a log scale?
#    A: The fractions grow by a factor of 10 each time, so on a log axis they are evenly spaced and the small fractions
#       are not squeezed against the left edge.
# Q: Why minorticks_off, and what does it return?
#    A: A log axis would add small in-between ticks that mean nothing here. The function returns nothing; it changes the
#       axes in place.
def _frac_ticks(ax, fractions) -> None:
    """Log-scaled x axis with one tick per label fraction, shown as 1% / 10% / 100%.

    Args:
        ax: the matplotlib axes to change.
        fractions: the label fractions (may repeat) that need a tick.
    Returns:
        Nothing; the axes are changed in place.
    """
    # Unique fractions, smallest first.
    fr = sorted(set(fractions))
    # A log scale spaces 1%, 10% and 100% evenly.
    ax.set_xscale("log")
    # Put a tick at each fraction and write it as a readable percentage.
    ax.set_xticks(fr, [FRACTION_LABEL.get(round(f, 4), f"{f:.0%}") for f in fr])
    # Remove the small in-between ticks that a log axis would otherwise draw.
    ax.minorticks_off()


# Q: What does this function do?
#    A: It takes wanted label heights and moves them apart (at least `gap` between neighbours) so the direct labels at
#       the ends of the lines do not overlap. Heights are returned in the original order.
# Q: Why is 'top' needed?
#    A: Pushing labels up could move the highest one out of the plot. If it goes above 'top' it is pinned there and the
#       labels below it are pushed down instead.
# Q: Why process the labels from lowest to highest?
#    A: Sorting with argsort lets each label be compared only with the next higher one; a label that is too close is
#       moved up to exactly gap above its lower neighbour.
def _spread(ys: list[float], gap: float, top: float | None = None) -> list[float]:
    """Nudge label heights apart so neighbouring labels do not overlap.

    `ys` are the desired label heights (data units) and `gap` the minimum spacing. Labels are processed
    from lowest to highest and pushed UP where they are too close. If `top` is given and the pushing
    would lift the highest label above it, that label is held at `top` and the ones below are pushed
    DOWN instead, so labels stay inside the axes. Returns the adjusted heights in the original input order.
    """
    # Indices of the labels from the lowest height to the highest.
    order = np.argsort(ys)
    # Working copy of the heights as floats.
    out = np.array(ys, dtype=float)
    # Compare each label with the next higher one.
    for a, b in zip(order[:-1], order[1:]):
        # Too close: push the higher label up so the two are exactly `gap` apart.
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    # If the pushing moved the highest label above the allowed top, pin it at the top...
    if top is not None and len(order) and out[order[-1]] > top:
        out[order[-1]] = top
        # Walk down from the highest label; each one must be at least `gap` below the one above it.
        for a, b in zip(order[::-1][:-1], order[::-1][1:]):
            out[b] = min(out[b], out[a] - gap)
    # Give the heights back as a plain list, in the original order of the inputs.
    return list(out)


# Q: What does this function do?
#    A: It creates the output folder if needed, saves the figure to the path and closes the figure.
# Q: Why close the figure?
#    A: Several figures are made in one run; closing frees the memory and avoids matplotlib's warning about too many
#       open figures.
# Q: Why return the path?
#    A: So each figure function can end with 'return _save(fig, path)' and callers get the written path.
def _save(fig, path: Path) -> Path:
    """Create the parent folder, write the figure with the saved rcParams (see viz.apply_style), close it.

    Args:
        fig: the matplotlib figure to save.
        path: where to write the image file.
    Returns:
        The same path, so figure functions can end with "return _save(...)".
    """
    # Make sure the output folder exists before writing into it.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write the image (dpi and tight border come from the style settings).
    fig.savefig(path)
    # pyplot is imported here, only when a figure is actually saved.
    import matplotlib.pyplot as plt

    # Close the figure to free its memory (many figures are made in one run).
    plt.close(fig)
    return path


# Q: What does this function do?
#    A: It draws the test score against the label fraction for the supervised runs: one panel per input type (RGB, 13
#       bands), one line per model, with error bars for the std over seeds, direct labels and a shared legend.
# Q: Why do the panels share the y axis and why are the labels sometimes left out?
#    A: A shared y axis lets the two panels be compared directly. If the line ends are so close that a label would have
#       to move more than half a gap away from its own line, the label would sit next to a different line and mislead,
#       so no direct labels are written there; the legend and the marker shapes still identify the lines.
# Q: Why is a NaN std replaced by 0, and what is the gap based on?
#    A: A single seed has no std (NaN); 0 draws no bar instead of failing. The gap is 6.5 percent of the visible y
#       range, so it works for scores spanning 0.5-1.0 or only 0.97-0.99.
# Q: What are the inputs and what is returned?
#    A: groups (supervised rows are used), the output path and metric ('f1' or accuracy). It returns the path, or None
#       if there are no supervised rows.
def label_efficiency(groups: pd.DataFrame, path: Path, metric: str = "f1") -> Path | None:
    """Test score against label fraction: one line per model, RGB and 13-band side by side.

    Args:
        groups: groups.csv as a DataFrame; only rows with variant == "supervised" are used.
        path: output PNG.
        metric: "f1" for test macro-F1, anything else for test accuracy.

    Returns the path, or None if there are no supervised rows. One panel per modality present (shared
    y axis); error bars are the std over seeds, drawn as 0 when only one seed exists (the std is NaN).
    """
    # pyplot is imported inside the function so importing this module stays cheap.
    import matplotlib.pyplot as plt

    # Only supervised runs belong in this figure.
    g = groups[groups["variant"] == "supervised"]
    # No supervised data yet: no figure.
    if g.empty:
        return None
    # Apply the shared report look before drawing.
    viz.apply_style()
    # Models in the data (fixed order) and the input types in the data (RGB first).
    mods = _models_present(g)
    mods_in = [m for m in ("rgb", "ms") if m in set(g["modality"])]
    # One panel per input type, side by side, sharing the y axis so the panels can be compared directly.
    fig, axes = plt.subplots(1, len(mods_in), figsize=(3.4 * len(mods_in) + 1.4, 3.1), sharey=True, squeeze=False)
    # Word used in the y-axis label, and the names of the mean and std columns to plot.
    name = "macro-F1" if metric == "f1" else "accuracy"
    mean_col, std_col = metric_columns(metric)
    # Draw one panel per input type.
    for ax, mod in zip(axes[0], mods_in):
        # The last point of every line (model, x, y), used afterwards to place the direct labels.
        ends = []
        # One line per model.
        for m in mods:
            # This model's rows for this input type, ordered from the fewest to the most labels.
            d = g[(g["model"] == m) & (g["modality"] == mod)].sort_values("fraction")
            # This model was not run on this input type.
            if d.empty:
                continue
            # Draw the mean score as a line with markers; error bars show the std over seeds (NaN counts as 0).
            ax.errorbar(d["fraction"], d[mean_col],
                        yerr=d[std_col].fillna(0),
                        color=MODEL_COLOR[m], marker=MODEL_MARKER[m], capsize=2.5, elinewidth=1.2,
                        markeredgecolor=viz.SURFACE, markeredgewidth=1.2, label=MODEL_LABEL[m])
            # Remember where the line ends so it can be labelled.
            ends.append((m, d["fraction"].iloc[-1], d[mean_col].iloc[-1]))
        # Log x axis with ticks 1% / 10% / 100%.
        _frac_ticks(ax, g["fraction"])
        # Widen the right edge so the direct labels written after the last point fit inside the axes.
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] * 2.4)
        # The spacing is a share of the visible y range (6.5%, about one 7.5 pt label height on this
        # figure size), so it works whether the scores span 0.5-1.0 or only 0.97-0.99. Labels are kept
        # inside the axes (top = 2% of the range below the upper edge).
        lo, hi = ax.get_ylim()
        gap = 0.065 * (hi - lo)
        # Where each label would like to be (at the end of its line) and where it can be without overlapping.
        want = [e[2] for e in ends]
        ys = _spread(want, gap=gap, top=hi - 0.02 * (hi - lo))
        # A label that had to move more than half a gap from its own line end would sit next to a
        # different line and mislead. When the line ends are that crowded (all models near 99% at 100%
        # labels), write no direct labels in this panel: the legend below names every line, and each
        # line also has its own marker.
        if all(abs(y - w) <= 0.5 * gap for y, w in zip(ys, want)):
            # Write each model's name just to the right of its line end, at the adjusted height.
            for (m, x, _), y in zip(ends, ys):
                ax.text(x * 1.12, y, MODEL_LABEL[m], color=viz.TEXT_SECONDARY, fontsize=7.5, va="center")
        # Panel title (the input type) and the shared x-axis caption.
        ax.set_title(MODALITY_LABEL[mod])
        ax.set_xlabel("labelled training images (share of the 18,899)")
    # The y-axis label is written once, on the first panel only.
    axes[0][0].set_ylabel(f"test {name} (mean ± std, 3 seeds)")
    # Take the line names and symbols from the first panel to build one legend for the whole figure.
    handles, labels = axes[0][0].get_legend_handles_labels()
    # Shared legend below the panels (negative y = outside the axes); position tuned by eye.
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.17), fontsize=8)
    # Write the PNG and return its path.
    return _save(fig, path)


# Q: What does this function do?
#    A: It shows how much the 13-band input improves test macro-F1 over RGB: for each label fraction and model a pale
#       bar (mean over seeds) with one dot per seed (13-band minus RGB).
# Q: Why show one dot per seed and a zero line?
#    A: The dots show how much the seeds disagree, so a small gain with large spread is not over-read. The zero line
#       separates 'extra bands help' (above) from 'RGB is better' (below).
# Q: Why only the supervised pairs?
#    A: If the table has a 'variant' column, the supervised rows are used so pseudo-label students do not mix into the
#       comparison.
# Q: What are the inputs and what is returned?
#    A: pairs (compare_rgb_vs_ms_pairs.csv, with delta_f1) and an output path. It returns the path or None if there are
#       no rows.
def rgb_vs_ms_gain(pairs: pd.DataFrame, path: Path) -> Path | None:
    """Gain in test macro-F1 from the other 9-10 bands, by label fraction. One dot per seed.

    Args:
        pairs: compare_rgb_vs_ms_pairs.csv (one row per matched seed); `delta_f1` = F1(13-band) - F1(RGB).
            If it has a "variant" column only the supervised rows are used.
        path: output PNG.

    Returns the path, or None if there are no rows. Each bar is the mean over seeds, each dot one seed;
    bars are grouped by label fraction, with one bar per model inside a group.
    """
    # pyplot is imported inside the function so importing this module stays cheap.
    import matplotlib.pyplot as plt

    # Use supervised pairs only (when the table has a variant column at all).
    p = pairs[pairs["variant"] == "supervised"] if "variant" in pairs else pairs
    # No pairs yet: no figure.
    if p.empty:
        return None
    # Apply the shared report look before drawing.
    viz.apply_style()
    # Models in the data (fixed order) and the label fractions in the data (smallest first).
    mods = _models_present(p)
    fracs = sorted(p["fraction"].unique())
    # A single panel for all models and fractions.
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    width = 0.7 / len(mods)  # a group of bars spans about 0.7 of the unit spacing between fractions
    # Loop over models (bar position inside a group) and label fractions (group position).
    for j, m in enumerate(mods):
        for i, f in enumerate(fracs):
            # The per-seed differences for this model at this fraction.
            d = p[(p["model"] == m) & (p["fraction"] == f)]
            # This combination was not run.
            if d.empty:
                continue
            x = i + (j - (len(mods) - 1) / 2) * (width + 0.03)  # bars centred on the fraction tick, 0.03 apart
            # A pale bar for the mean gain over seeds; only the first group gets a legend entry per model.
            ax.bar(x, d["delta_f1"].mean(), width=width, color=MODEL_COLOR[m], alpha=0.35,
                   label=MODEL_LABEL[m] if i == 0 else None)
            # One dot per seed on top of the bar, so the spread between seeds is visible.
            ax.scatter(np.full(len(d), x), d["delta_f1"], s=16, color=MODEL_COLOR[m], edgecolor=viz.SURFACE,
                       linewidth=0.8, zorder=3)
    # A zero line: above it the 13-band input helps, below it RGB is better.
    ax.axhline(0, color=viz.TEXT_SECONDARY, linewidth=0.8)
    # Label each group of bars with its label fraction.
    ax.set_xticks(range(len(fracs)), [FRACTION_LABEL.get(round(f, 4), f"{f:.0%}") for f in fracs])
    # Axis captions and title.
    ax.set_xlabel("labelled training images")
    ax.set_ylabel("test macro-F1: 13 bands − RGB")
    ax.set_title("What the extra bands add")
    # Legend (one entry per model) placed below the axes.
    ax.legend(ncol=len(mods), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))  # below the axes
    # Write the PNG and return its path.
    return _save(fig, path)


# Q: What does this function do?
#    A: It shows how much each pseudo-label student improves over its own teacher: one panel per input type, one bar per
#       (variant, model, fraction) with a dot for every seed (student minus teacher).
# Q: Why are class-balanced bars hatched, and why is the colour the model's colour?
#    A: The hatching separates the 'plb' bars from the plain 'pl' bars without a new colour, and a model keeps its
#       colour in every figure.
# Q: What happens when a variant's table is missing?
#    A: Variants whose table is None or empty are dropped; if none remain the function returns None.
# Q: What does the zero line mean?
#    A: Above it the student beats its teacher; below it the student is worse.
def pseudo_label_gain(pairs: dict[str, pd.DataFrame], path: Path) -> Path | None:
    """Gain in test macro-F1 of each pseudo-label student over its own teacher; one dot per seed.

    `pairs` maps a variant ('pl', 'plb') to its compare_supervised_vs_<variant>_pairs table.
    `delta_f1` there is F1(student) - F1(teacher) for the same seed. Variants whose table is None or
    empty are dropped; returns None if none remain. One panel per modality; one bar per
    (variant, model, fraction) slot, in the order global-tau variant first, then class-balanced
    (hatched). Each bar is the mean over seeds and each dot one seed.
    """
    # pyplot is imported inside the function so importing this module stays cheap.
    import matplotlib.pyplot as plt

    # Drop variants whose pair table is missing or empty.
    pairs = {k: v for k, v in pairs.items() if v is not None and not v.empty}
    # Nothing left: no figure.
    if not pairs:
        return None
    # Apply the shared report look before drawing.
    viz.apply_style()
    # The input types that occur in at least one variant (RGB first); one panel each.
    mods_in = [m for m in ("rgb", "ms") if any(m in set(v["modality"]) for v in pairs.values())]
    fig, axes = plt.subplots(1, len(mods_in), figsize=(3.4 * len(mods_in) + 0.6, 3.1), sharey=True, squeeze=False)
    # Draw one panel per input type.
    for ax, mod in zip(axes[0], mods_in):
        # x positions of the bars, their tick labels, and k = the next free bar position.
        slots, labels, k = [], [], 0
        # Bars are laid out variant by variant, then model by model, then fraction by fraction.
        for variant, p in pairs.items():
            for m in _models_present(p):
                for f in sorted(p["fraction"].unique()):
                    # Per-seed student-minus-teacher differences for this exact combination.
                    d = p[(p["modality"] == mod) & (p["model"] == m) & (p["fraction"] == f)]
                    # This combination was not run.
                    if d.empty:
                        continue
                    # Class-balanced bars are hatched so they differ from the plain pseudo-label bars.
                    hatch = "//" if variant == "plb" else None
                    # A pale bar for the mean gain over seeds.
                    ax.bar(k, d["delta_f1"].mean(), width=0.62, color=MODEL_COLOR[m], alpha=0.35, hatch=hatch,
                           edgecolor=MODEL_COLOR[m] if hatch else "none", linewidth=0)
                    # One dot per seed on top of the bar.
                    ax.scatter(np.full(len(d), k), d["delta_f1"], s=16, color=MODEL_COLOR[m], edgecolor=viz.SURFACE,
                               linewidth=0.8, zorder=3)
                    # Remember the bar position and write its tick label: model, fraction and (if balanced) a note.
                    slots.append(k)
                    labels.append(f"{MODEL_LABEL[m]}\n{FRACTION_LABEL.get(round(f, 4), f'{f:.0%}')}"
                                  + ("\nbalanced" if variant == "plb" else ""))
                    k += 1
        # A zero line: above it the student beats its teacher, below it the student is worse.
        ax.axhline(0, color=viz.TEXT_SECONDARY, linewidth=0.8)
        # Put the labels under the bars (small font because there are many bars) and title the panel.
        ax.set_xticks(slots, labels, fontsize=6.5)
        ax.set_title(MODALITY_LABEL[mod])
    # The y-axis label is written once, on the first panel only.
    axes[0][0].set_ylabel("test macro-F1: student − teacher")
    # Write the PNG and return its path.
    return _save(fig, path)


# Q: What does this function do?
#    A: It draws one row-normalised confusion matrix per panel (for example RGB and 13-band of the best model) side by
#       side, with the percentage written in each cell.
# Q: Why row-normalised and with one shared 0..1 colour scale?
#    A: Dividing each row by its total shows the share of each true class, so classes with different sizes compare
#       fairly; the same colour scale in all panels makes the panels directly comparable.
# Q: Why are small cells left blank and some texts white?
#    A: Cells under 0.5 percent would show a meaningless '0', so they are left empty. Text is white on dark cells (above
#       0.55) and near-black on light cells so it stays readable.
# Q: What are the inputs and what is returned?
#    A: mats (title -> 10 x 10 matrix), the output path and a figure title. It returns the path, or None if mats is
#       empty.
def confusion_grid(mats: dict[str, np.ndarray], path: Path, title: str) -> Path | None:
    """Row-normalised confusion matrices side by side, shared colour scale 0..1.

    Args:
        mats: panel title -> [10, 10] matrix (rows = true class, columns = predicted, each row sums to 1).
        path: output PNG.
        title: figure-level title.

    Returns the path, or None if `mats` is empty. Cells show the percentage rounded to an integer;
    cells below 0.5% are left blank, and the text is white on dark cells (> 0.55) for contrast.
    """
    # pyplot is imported inside the function so importing this module stays cheap.
    import matplotlib.pyplot as plt

    # No matrices: no figure.
    if not mats:
        return None
    # Apply the shared report look before drawing.
    viz.apply_style()
    # One panel per matrix, side by side.
    n = len(mats)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), squeeze=False)
    # Will hold the last drawn image; the colour bar below needs one.
    im = None
    # Draw each matrix into its own panel.
    for ax, (name, cm) in zip(axes[0], mats.items()):
        # Show the matrix as coloured squares; the same 0..1 scale in every panel makes them comparable.
        im = ax.imshow(cm, cmap=viz.SEQUENTIAL, vmin=0, vmax=1)
        # No grid lines over an image.
        ax.grid(False)
        # Predicted class names along the bottom, tilted so they do not overlap.
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=60, ha="right", fontsize=6.5)
        # The class names are written once, on the first panel; on the others they would overlap the
        # numbers of the panel to their left.
        ax.set_yticks(range(len(CLASSES)), CLASSES if ax is axes[0][0] else [], fontsize=6.5)
        # Axis caption and panel title (the model and input type).
        ax.set_xlabel("predicted")
        ax.set_title(name, fontsize=9)
        # Write the percentage inside each cell: loop over true classes (rows) and predicted classes (columns).
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                # Leave cells below 0.5% empty, because they would show a meaningless "0".
                if cm[i, j] >= 0.005:
                    # Text colour is white on dark cells and near-black on light cells so it stays readable.
                    ax.text(j, i, f"{cm[i, j] * 100:.0f}", ha="center", va="center", fontsize=5.5,
                            color="white" if cm[i, j] > 0.55 else viz.TEXT_PRIMARY)
    # The "true" label is written once, on the first panel.
    axes[0][0].set_ylabel("true")
    # Overall title above all panels.
    fig.suptitle(title, fontsize=9)
    # One colour bar shared by all panels, explaining the colour scale.
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.7, label="share of the true class")
    # Write the PNG and return its path.
    return _save(fig, path)


# Q: What does this function do?
#    A: It draws a heatmap of the per-class test F1 (mean over seeds) with classes down and one column per model and
#       input type, with the value written in every cell.
# Q: Why does the colour scale start just below the lowest value instead of at 0?
#    A: Most classes are near-perfect, so a scale from 0 would show almost one colour. Starting at the minimum minus
#       0.05 (not below 0) keeps small differences visible; the printed numbers give the exact levels.
# Q: Why is pivot_table used, and what does the 'multi' flag do?
#    A: pivot_table turns the long table (one row per class, model, input) into a class-by-column grid. With several
#       models each column heading shows model and input; with one model the name goes into the title.
# Q: What are the inputs and what is returned?
#    A: per_class (supervised rows at `fraction` are used), the exact fraction value and an output path. It returns the
#       path, or None if no rows match.
def per_class_heatmap(per_class: pd.DataFrame, fraction: float, path: Path) -> Path | None:
    """Per-class test F1 (mean over seeds): classes down, model x input across.

    Args:
        per_class: per_class.csv as a DataFrame; supervised rows at `fraction` only.
        fraction: label fraction to show (exact float match, e.g. 1.0 or 0.01).
        path: output PNG.

    Returns the path, or None if no rows match. Columns are ordered by model, then RGB before 13-band;
    a missing combination simply has no column. The colour scale starts just below the lowest value
    shown (min - 0.05, floored at 0) rather than at 0, so small differences between near-perfect
    classes stay visible; read the printed numbers, not the colour, for absolute levels.
    """
    # pyplot is imported inside the function so importing this module stays cheap.
    import matplotlib.pyplot as plt

    # Keep supervised rows at the requested label fraction.
    d = per_class[(per_class["variant"] == "supervised") & (per_class["fraction"] == fraction)]
    # No matching rows: no figure.
    if d.empty:
        return None
    # Apply the shared report look before drawing.
    viz.apply_style()
    # Models present in the data, in the fixed order.
    models = _models_present(d)
    # With one model its name goes into the title and the columns say only the input. With several
    # models each column says both (two lines).
    multi = len(models) > 1
    # Add a "col" column holding the column heading for each row (optional model name + input type).
    d = d.assign(col=lambda x: (x["model"].map(MODEL_LABEL) + "\n" if multi else "")
                 + x["modality"].map({"rgb": "RGB", "ms": "13-band"}))
    # The wanted column order: by model, with RGB before 13-band.
    order = [f"{MODEL_LABEL[m]}\n{mo}" if multi else mo for m in models for mo in ("RGB", "13-band")]
    # Rearrange into a grid: one row per class (in the fixed class order), one column per model / input.
    piv = d.pivot_table(index="class", columns="col", values="f1_mean").reindex(CLASSES)
    # Put the columns in the wanted order, dropping combinations that do not exist.
    piv = piv[[c for c in order if c in piv.columns]]
    # Figure width grows with the number of columns.
    fig, ax = plt.subplots(figsize=(1.3 * piv.shape[1] + 2.4, 4.0))
    # aspect="auto" lets the cells be wider than tall, so neighbouring column labels do not touch.
    im = ax.imshow(piv.to_numpy(), cmap=viz.SEQUENTIAL, vmin=max(0.0, np.nanmin(piv.to_numpy()) - 0.05), vmax=1,
                   aspect="auto")
    # No grid lines over an image.
    ax.grid(False)
    # Column headings along the bottom and class names down the left side.
    ax.set_xticks(range(piv.shape[1]), piv.columns, fontsize=7)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=7.5)
    lo = im.norm.vmin + 0.6 * (1 - im.norm.vmin)  # above this value the cell is dark enough for white text
    # Write the F1 value inside every cell: loop over classes (rows) and columns.
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            # Empty cells (a class without data) get no number.
            if not np.isnan(v):
                # White text on dark cells, near-black on light cells, for readability.
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > lo else viz.TEXT_PRIMARY)
    # With a single model, its name goes into the title instead of into every column heading.
    who = "" if multi else f"{MODEL_LABEL[models[0]]}, "
    ax.set_title(f"Per-class test F1, {who}{FRACTION_LABEL.get(round(fraction, 4), f'{fraction:.0%}')} labels")
    # Colour bar explaining the scale.
    fig.colorbar(im, ax=ax, shrink=0.8, label="F1 (mean of 3 seeds)")
    # Write the PNG and return its path.
    return _save(fig, path)

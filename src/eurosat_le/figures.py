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

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eurosat_le import viz
from eurosat_le.constants import CLASSES

MODELS = ["scratch_cnn", "resnet50", "vit_small", "vit_scratch"]
MODEL_LABEL = {"scratch_cnn": "Scratch CNN", "resnet50": "ResNet-50", "vit_small": "ViT-Small",
               "vit_scratch": "ViT-Tiny (scratch)"}
MODEL_COLOR = dict(zip(MODELS, viz.CATEGORICAL))
MODEL_MARKER = {"scratch_cnn": "o", "resnet50": "s", "vit_small": "^", "vit_scratch": "D"}
MODALITY_LABEL = {"rgb": "RGB (3 bands)", "ms": "13 bands"}
FRACTION_LABEL = {0.01: "1%", 0.1: "10%", 1.0: "100%"}
VARIANT_LABEL = {"pl": "pseudo-label (global τ)", "plb": "pseudo-label (class-balanced)"}


def metric_columns(metric: str) -> tuple[str, str]:
    """Names of the mean and std columns of groups.csv for a metric.

    `metric` is "f1" for test macro-F1; any other value means test accuracy. Returns
    ("test_f1_mean", "test_f1_std") or ("test_acc_mean", "test_acc_std").
    """
    name = "f1" if metric == "f1" else "acc"
    return f"test_{name}_mean", f"test_{name}_std"


def _models_present(df: pd.DataFrame) -> list[str]:
    """Models found in `df["model"]`, in the fixed MODELS order (not the order of appearance)."""
    return [m for m in MODELS if m in set(df["model"])]


def _frac_ticks(ax, fractions) -> None:
    """Log-scaled x axis with one tick per label fraction, shown as 1% / 10% / 100%."""
    fr = sorted(set(fractions))
    ax.set_xscale("log")
    ax.set_xticks(fr, [FRACTION_LABEL.get(round(f, 4), f"{f:.0%}") for f in fr])
    ax.minorticks_off()


def _spread(ys: list[float], gap: float) -> list[float]:
    """Nudge label heights apart so neighbouring labels do not overlap.

    `ys` are the desired label heights (data units) and `gap` the minimum spacing. Labels are processed
    from lowest to highest and only ever pushed UP (never down), so a label can end up above its
    line end. Returns the adjusted heights in the original input order.
    """
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return list(out)


def _save(fig, path: Path) -> Path:
    """Create the parent folder, write the figure with the saved rcParams (see viz.apply_style), close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def label_efficiency(groups: pd.DataFrame, path: Path, metric: str = "f1") -> Path | None:
    """Test score against label fraction: one line per model, RGB and 13-band side by side.

    Args:
        groups: groups.csv as a DataFrame; only rows with variant == "supervised" are used.
        path: output PNG.
        metric: "f1" for test macro-F1, anything else for test accuracy.

    Returns the path, or None if there are no supervised rows. One panel per modality present (shared
    y axis); error bars are the std over seeds, drawn as 0 when only one seed exists (the std is NaN).
    """
    import matplotlib.pyplot as plt

    g = groups[groups["variant"] == "supervised"]
    if g.empty:
        return None
    viz.apply_style()
    mods = _models_present(g)
    mods_in = [m for m in ("rgb", "ms") if m in set(g["modality"])]
    fig, axes = plt.subplots(1, len(mods_in), figsize=(3.4 * len(mods_in) + 1.4, 3.1), sharey=True, squeeze=False)
    name = "macro-F1" if metric == "f1" else "accuracy"
    for ax, mod in zip(axes[0], mods_in):
        ends = []
        for m in mods:
            d = g[(g["model"] == m) & (g["modality"] == mod)].sort_values("fraction")
            if d.empty:
                continue
            ax.errorbar(d["fraction"], d[f"test_{'f1' if metric == 'f1' else 'acc'}_mean"],
                        yerr=d[f"test_{'f1' if metric == 'f1' else 'acc'}_std"].fillna(0),
                        color=MODEL_COLOR[m], marker=MODEL_MARKER[m], capsize=2.5, elinewidth=1.2,
                        markeredgecolor=viz.SURFACE, markeredgewidth=1.2, label=MODEL_LABEL[m])
            ends.append((m, d["fraction"].iloc[-1], d[f"test_{'f1' if metric == 'f1' else 'acc'}_mean"].iloc[-1]))
        _frac_ticks(ax, g["fraction"])
        # Widen the right edge so the direct labels written after the last point fit inside the axes.
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] * 2.4)
        # gap is in score units (0-1 scale); the value was tuned by eye so 7.5 pt labels do not collide.
        ys = _spread([e[2] for e in ends], gap=0.035)
        for (m, x, _), y in zip(ends, ys):
            ax.text(x * 1.12, y, MODEL_LABEL[m], color=viz.TEXT_SECONDARY, fontsize=7.5, va="center")
        ax.set_title(MODALITY_LABEL[mod])
        ax.set_xlabel("labelled training images (share of the 18,899)")
    axes[0][0].set_ylabel(f"test {name} (mean ± std, 3 seeds)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    # Shared legend below the panels (negative y = outside the axes); position tuned by eye.
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.17), fontsize=8)
    return _save(fig, path)


def rgb_vs_ms_gain(pairs: pd.DataFrame, path: Path) -> Path | None:
    """Gain in test macro-F1 from the other 9-10 bands, by label fraction. One dot per seed.

    Args:
        pairs: compare_rgb_vs_ms_pairs.csv (one row per matched seed); `delta_f1` = F1(13-band) - F1(RGB).
            If it has a "variant" column only the supervised rows are used.
        path: output PNG.

    Returns the path, or None if there are no rows. Each bar is the mean over seeds, each dot one seed;
    bars are grouped by label fraction, with one bar per model inside a group.
    """
    import matplotlib.pyplot as plt

    p = pairs[pairs["variant"] == "supervised"] if "variant" in pairs else pairs
    if p.empty:
        return None
    viz.apply_style()
    mods = _models_present(p)
    fracs = sorted(p["fraction"].unique())
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    width = 0.7 / len(mods)  # a group of bars spans about 0.7 of the unit spacing between fractions
    for j, m in enumerate(mods):
        for i, f in enumerate(fracs):
            d = p[(p["model"] == m) & (p["fraction"] == f)]
            if d.empty:
                continue
            x = i + (j - (len(mods) - 1) / 2) * (width + 0.03)  # bars centred on the fraction tick, 0.03 apart
            ax.bar(x, d["delta_f1"].mean(), width=width, color=MODEL_COLOR[m], alpha=0.35,
                   label=MODEL_LABEL[m] if i == 0 else None)
            ax.scatter(np.full(len(d), x), d["delta_f1"], s=16, color=MODEL_COLOR[m], edgecolor=viz.SURFACE,
                       linewidth=0.8, zorder=3)
    ax.axhline(0, color=viz.TEXT_SECONDARY, linewidth=0.8)
    ax.set_xticks(range(len(fracs)), [FRACTION_LABEL.get(round(f, 4), f"{f:.0%}") for f in fracs])
    ax.set_xlabel("labelled training images")
    ax.set_ylabel("test macro-F1: 13 bands − RGB")
    ax.set_title("What the extra bands add")
    ax.legend(ncol=len(mods), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))  # below the axes
    return _save(fig, path)


def pseudo_label_gain(pairs: dict[str, pd.DataFrame], path: Path) -> Path | None:
    """Gain in test macro-F1 of each pseudo-label student over its own teacher; one dot per seed.

    `pairs` maps a variant ('pl', 'plb') to its compare_supervised_vs_<variant>_pairs table.
    `delta_f1` there is F1(student) - F1(teacher) for the same seed. Variants whose table is None or
    empty are dropped; returns None if none remain. One panel per modality; one bar per
    (variant, model, fraction) slot, in the order global-tau variant first, then class-balanced
    (hatched). Each bar is the mean over seeds and each dot one seed.
    """
    import matplotlib.pyplot as plt

    pairs = {k: v for k, v in pairs.items() if v is not None and not v.empty}
    if not pairs:
        return None
    viz.apply_style()
    mods_in = [m for m in ("rgb", "ms") if any(m in set(v["modality"]) for v in pairs.values())]
    fig, axes = plt.subplots(1, len(mods_in), figsize=(3.4 * len(mods_in) + 0.6, 3.1), sharey=True, squeeze=False)
    for ax, mod in zip(axes[0], mods_in):
        slots, labels, k = [], [], 0
        for variant, p in pairs.items():
            for m in _models_present(p):
                for f in sorted(p["fraction"].unique()):
                    d = p[(p["modality"] == mod) & (p["model"] == m) & (p["fraction"] == f)]
                    if d.empty:
                        continue
                    hatch = "//" if variant == "plb" else None
                    ax.bar(k, d["delta_f1"].mean(), width=0.62, color=MODEL_COLOR[m], alpha=0.35, hatch=hatch,
                           edgecolor=MODEL_COLOR[m] if hatch else "none", linewidth=0)
                    ax.scatter(np.full(len(d), k), d["delta_f1"], s=16, color=MODEL_COLOR[m], edgecolor=viz.SURFACE,
                               linewidth=0.8, zorder=3)
                    slots.append(k)
                    labels.append(f"{MODEL_LABEL[m]}\n{FRACTION_LABEL.get(round(f, 4), f'{f:.0%}')}"
                                  + ("\nbalanced" if variant == "plb" else ""))
                    k += 1
        ax.axhline(0, color=viz.TEXT_SECONDARY, linewidth=0.8)
        ax.set_xticks(slots, labels, fontsize=6.5)
        ax.set_title(MODALITY_LABEL[mod])
    axes[0][0].set_ylabel("test macro-F1: student − teacher")
    return _save(fig, path)


def confusion_grid(mats: dict[str, np.ndarray], path: Path, title: str) -> Path | None:
    """Row-normalised confusion matrices side by side, shared colour scale 0..1.

    Args:
        mats: panel title -> [10, 10] matrix (rows = true class, columns = predicted, each row sums to 1).
        path: output PNG.
        title: figure-level title.

    Returns the path, or None if `mats` is empty. Cells show the percentage rounded to an integer;
    cells below 0.5% are left blank, and the text is white on dark cells (> 0.55) for contrast.
    """
    import matplotlib.pyplot as plt

    if not mats:
        return None
    viz.apply_style()
    n = len(mats)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), squeeze=False)
    im = None
    for ax, (name, cm) in zip(axes[0], mats.items()):
        im = ax.imshow(cm, cmap=viz.SEQUENTIAL, vmin=0, vmax=1)
        ax.grid(False)
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=60, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=6.5)
        ax.set_xlabel("predicted")
        ax.set_title(name, fontsize=9)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                if cm[i, j] >= 0.005:
                    ax.text(j, i, f"{cm[i, j] * 100:.0f}", ha="center", va="center", fontsize=5.5,
                            color="white" if cm[i, j] > 0.55 else viz.TEXT_PRIMARY)
    axes[0][0].set_ylabel("true")
    fig.suptitle(title, fontsize=9)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.7, label="share of the true class")
    return _save(fig, path)


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
    import matplotlib.pyplot as plt

    d = per_class[(per_class["variant"] == "supervised") & (per_class["fraction"] == fraction)]
    if d.empty:
        return None
    viz.apply_style()
    d = d.assign(col=lambda x: x["model"].map(MODEL_LABEL) + "\n" + x["modality"].map({"rgb": "RGB", "ms": "13-band"}))
    order = [f"{MODEL_LABEL[m]}\n{mo}" for m in _models_present(d) for mo in ("RGB", "13-band")]
    piv = d.pivot_table(index="class", columns="col", values="f1_mean").reindex(CLASSES)
    piv = piv[[c for c in order if c in piv.columns]]
    fig, ax = plt.subplots(figsize=(0.95 * piv.shape[1] + 2.2, 4.0))
    im = ax.imshow(piv.to_numpy(), cmap=viz.SEQUENTIAL, vmin=max(0.0, np.nanmin(piv.to_numpy()) - 0.05), vmax=1)
    ax.grid(False)
    ax.set_xticks(range(piv.shape[1]), piv.columns, fontsize=7)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=7.5)
    lo = im.norm.vmin + 0.6 * (1 - im.norm.vmin)  # above this value the cell is dark enough for white text
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > lo else viz.TEXT_PRIMARY)
    ax.set_title(f"Per-class test F1, {FRACTION_LABEL.get(round(fraction, 4), f'{fraction:.0%}')} labels")
    fig.colorbar(im, ax=ax, shrink=0.8, label="F1 (mean of 3 seeds)")
    return _save(fig, path)

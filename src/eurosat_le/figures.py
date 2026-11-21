"""Report figures built from the Phase 6 tables (results/summary/eval/*.csv). Phase 7.

Every function takes plain DataFrames and an output path, and returns the path, or
None when the data for that figure is not there yet (so the script can run on a
partial set of results). Nothing here reads run folders or trains anything.

Colour rule: a model keeps its colour (and marker) in every figure. Text uses text
tokens, never the series colour. The green is below 3:1 contrast on the surface, so
every line is also direct-labelled and has its own marker shape.
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


def _models_present(df: pd.DataFrame) -> list[str]:
    return [m for m in MODELS if m in set(df["model"])]


def _frac_ticks(ax, fractions) -> None:
    fr = sorted(set(fractions))
    ax.set_xscale("log")
    ax.set_xticks(fr, [FRACTION_LABEL.get(round(f, 4), f"{f:.0%}") for f in fr])
    ax.minorticks_off()


def _spread(ys: list[float], gap: float) -> list[float]:
    """Nudge label heights apart so neighbouring labels do not overlap."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return list(out)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def label_efficiency(groups: pd.DataFrame, path: Path, metric: str = "f1") -> Path | None:
    """Test score against label fraction: one line per model, RGB and 13-band side by side."""
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
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] * 2.4)
        ys = _spread([e[2] for e in ends], gap=0.035)
        for (m, x, _), y in zip(ends, ys):
            ax.text(x * 1.12, y, MODEL_LABEL[m], color=viz.TEXT_SECONDARY, fontsize=7.5, va="center")
        ax.set_title(MODALITY_LABEL[mod])
        ax.set_xlabel("labelled training images (share of the 18,899)")
    axes[0][0].set_ylabel(f"test {name} (mean ± std, 3 seeds)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.17), fontsize=8)
    return _save(fig, path)


def rgb_vs_ms_gain(pairs: pd.DataFrame, path: Path) -> Path | None:
    """Gain in test macro-F1 from the other 9-10 bands, by label fraction. One dot per seed."""
    import matplotlib.pyplot as plt

    p = pairs[pairs["variant"] == "supervised"] if "variant" in pairs else pairs
    if p.empty:
        return None
    viz.apply_style()
    mods = _models_present(p)
    fracs = sorted(p["fraction"].unique())
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    width = 0.7 / len(mods)
    for j, m in enumerate(mods):
        for i, f in enumerate(fracs):
            d = p[(p["model"] == m) & (p["fraction"] == f)]
            if d.empty:
                continue
            x = i + (j - (len(mods) - 1) / 2) * (width + 0.03)
            ax.bar(x, d["delta_f1"].mean(), width=width, color=MODEL_COLOR[m], alpha=0.35,
                   label=MODEL_LABEL[m] if i == 0 else None)
            ax.scatter(np.full(len(d), x), d["delta_f1"], s=16, color=MODEL_COLOR[m], edgecolor=viz.SURFACE,
                       linewidth=0.8, zorder=3)
    ax.axhline(0, color=viz.TEXT_SECONDARY, linewidth=0.8)
    ax.set_xticks(range(len(fracs)), [FRACTION_LABEL.get(round(f, 4), f"{f:.0%}") for f in fracs])
    ax.set_xlabel("labelled training images")
    ax.set_ylabel("test macro-F1: 13 bands − RGB")
    ax.set_title("What the extra bands add")
    ax.legend(ncol=len(mods), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    return _save(fig, path)


def pseudo_label_gain(pairs: dict[str, pd.DataFrame], path: Path) -> Path | None:
    """Gain in test macro-F1 of each pseudo-label student over its own teacher; one dot per seed.

    `pairs` maps a variant ('pl', 'plb') to its compare_supervised_vs_<variant>_pairs table.
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
    """Row-normalised confusion matrices side by side, shared colour scale 0..1."""
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
    """Per-class test F1 (mean over seeds): classes down, model x input across."""
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
    lo = im.norm.vmin + 0.6 * (1 - im.norm.vmin)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > lo else viz.TEXT_PRIMARY)
    ax.set_title(f"Per-class test F1, {FRACTION_LABEL.get(round(fraction, 4), f'{fraction:.0%}')} labels")
    fig.colorbar(im, ax=ax, shrink=0.8, label="F1 (mean of 3 seeds)")
    return _save(fig, path)

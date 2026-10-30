"""Inspect EuroSAT and compute train-only normalisation statistics.

Run after download_data.py and make_splits.py:
    python scripts/inspect_dataset.py

Writes:
    splits/norm_stats.json                       per-channel mean/std from the TRAIN split only
    results/dataset_inspection/summary.json      counts, shapes, dtypes, value ranges, duplicates
    results/dataset_inspection/class_counts.csv
    results/dataset_inspection/samples_rgb.png   5 random patches per class
    results/dataset_inspection/class_counts.png
    results/dataset_inspection/spectral_signatures.png  per-class band means (z-scored)
"""

import hashlib
import json

import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

from eurosat_le import viz
from eurosat_le.constants import CLASSES, DATASETS, MS_BANDS, RAW_DIR, RESULTS_DIR, SPLITS_DIR
from eurosat_le.data.splits import check_integrity, load_split

OUT = RESULTS_DIR / "dataset_inspection"


class RunningStats:
    """Per-channel mean/std over all pixels, accumulated in float64."""

    def __init__(self, c):
        self.n, self.s, self.ss = 0, np.zeros(c), np.zeros(c)

    def add(self, x):  # x: (H, W, C)
        x = x.reshape(-1, x.shape[-1]).astype(np.float64)
        self.n += len(x)
        self.s += x.sum(0)
        self.ss += (x**2).sum(0)

    def result(self):
        mean = self.s / self.n
        return mean, np.sqrt(self.ss / self.n - mean**2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    viz.apply_style()
    check_integrity(SPLITS_DIR)
    df = load_split(SPLITS_DIR)
    summary = {"n_samples": len(df), "classes": CLASSES}

    counts = df.pivot_table(index="class", columns="split", values="id", aggfunc="count").reindex(CLASSES)
    counts = counts[["train", "val", "test"]]
    counts["total"] = counts.sum(axis=1)
    counts.to_csv(OUT / "class_counts.csv")
    summary["class_counts"] = counts.to_dict(orient="index")
    print(counts.to_string())

    # ---- RGB ---------------------------------------------------------------
    rgb_dir = RAW_DIR / DATASETS["rgb"]["folder"]
    rgb_stats, shapes, hashes = RunningStats(3), set(), {}
    for sid, split in tqdm(zip(df["id"], df["split"]), total=len(df), desc="RGB"):
        path = rgb_dir / f"{sid}.jpg"
        img = np.asarray(Image.open(path).convert("RGB"))
        shapes.add((img.shape, str(img.dtype)))
        hashes.setdefault(hashlib.md5(img.tobytes()).hexdigest(), []).append((sid, split))
        if split == "train":
            rgb_stats.add(img / 255.0)
    dups = [v for v in hashes.values() if len(v) > 1]
    cross = [v for v in dups if len({s for _, s in v}) > 1]
    summary["rgb"] = {
        "shapes": sorted(map(str, shapes)),
        "exact_pixel_duplicate_groups": len(dups),
        "duplicate_groups_spanning_splits": len(cross),
        "duplicate_examples": dups[:10],
    }
    mean, std = rgb_stats.result()
    norm = {"rgb": {"scale": "pixel/255", "mean": mean.round(6).tolist(), "std": std.round(6).tolist()}}
    print(f"RGB shapes {shapes}; duplicate groups {len(dups)} (across splits: {len(cross)})")

    # ---- 13-band MS ----------------------------------------------------------
    ms_dir = RAW_DIR / DATASETS["ms"]["folder"]
    if ms_dir.is_dir():
        ms_stats, ms_shapes = RunningStats(13), set()
        bmin, bmax = np.full(13, np.inf), np.full(13, -np.inf)
        class_sum = np.zeros((len(CLASSES), 13))
        class_n = np.zeros(len(CLASSES))
        for sid, split, label in tqdm(zip(df["id"], df["split"], df["label"]), total=len(df), desc="MS"):
            x = tifffile.imread(ms_dir / f"{sid}.tif")
            ms_shapes.add((x.shape, str(x.dtype)))
            bmin = np.minimum(bmin, x.reshape(-1, 13).min(0))
            bmax = np.maximum(bmax, x.reshape(-1, 13).max(0))
            if split == "train":
                ms_stats.add(x)
                class_sum[label] += x.reshape(-1, 13).mean(0)
                class_n[label] += 1
        mean, std = ms_stats.result()
        norm["ms"] = {"scale": "raw DN (reflectance x 10000)", "bands": MS_BANDS,
                      "mean": mean.round(3).tolist(), "std": std.round(3).tolist()}
        summary["ms"] = {"shapes": sorted(map(str, ms_shapes)),
                         "band_min": dict(zip(MS_BANDS, bmin.tolist())),
                         "band_max": dict(zip(MS_BANDS, bmax.tolist()))}
        print(f"MS shapes {ms_shapes}")
        plot_spectral(class_sum / class_n[:, None], mean, std)
    else:
        print("MS data not found; skipping 13-band statistics.")

    norm["computed_on"] = "train split only (splits/split.csv)"
    (SPLITS_DIR / "norm_stats.json").write_text(json.dumps(norm, indent=2) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    plot_samples(df, rgb_dir)
    plot_counts(counts)
    print(f"Wrote {OUT} and {SPLITS_DIR / 'norm_stats.json'}")


def plot_samples(df, rgb_dir, per_class=5, seed=0):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(per_class * 1.1, len(CLASSES) * 1.1))
    for r, cls in enumerate(CLASSES):
        ids = df.loc[df["class"] == cls, "id"].to_numpy()
        for c, sid in enumerate(rng.choice(ids, per_class, replace=False)):
            ax = axes[r, c]
            ax.imshow(Image.open(rgb_dir / f"{sid}.jpg"))
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
        axes[r, 0].set_ylabel(cls, rotation=0, ha="right", va="center", fontsize=8, color=viz.TEXT_PRIMARY)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(OUT / "samples_rgb.png")
    plt.close(fig)


def plot_counts(counts):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    y = np.arange(len(CLASSES))[::-1]
    ax.barh(y, counts["total"], height=0.6, color=viz.CATEGORICAL[0])
    for yi, v in zip(y, counts["total"]):
        ax.text(v + 30, yi, f"{v:,}", va="center", fontsize=8, color=viz.TEXT_SECONDARY)
    ax.set_yticks(y, CLASSES)
    ax.set_xlabel("Images")
    ax.set_title("EuroSAT images per class")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, counts["total"].max() * 1.12)
    fig.savefig(OUT / "class_counts.png")
    plt.close(fig)


def plot_spectral(class_means, band_mean, band_std):
    """Heatmap of per-class mean reflectance, z-scored per band against the train set."""
    import matplotlib.pyplot as plt

    # Show bands in wavelength order (B8A between B08 and B09), not file order.
    order = [MS_BANDS.index(b) for b in sorted(MS_BANDS, key=lambda b: (int(b[1:].rstrip("A")), b))]
    z = ((class_means - band_mean) / band_std)[:, order]
    lim = np.abs(z).max()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    im = ax.imshow(z, cmap=viz.DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(13), [MS_BANDS[i] for i in order], rotation=45)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Class mean vs. all-train mean (std units)")
    cb.outline.set_visible(False)
    ax.set_title("Spectral signature by class (train split)")
    fig.savefig(OUT / "spectral_signatures.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

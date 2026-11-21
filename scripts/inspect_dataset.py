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

Pipeline position: runs once after make_splits.py. Its most important output is
splits/norm_stats.json, which every dataset uses to standardise pixels (see
eurosat_le/data/datasets.py).

Design decisions:
    - Normalisation statistics are computed from the TRAIN split only (split.csv), never from
      validation or test images. Labels are not used. They do include train images that count as
      "unlabelled" in the 1% / 10% settings; that is not label leakage (see docs/DATA.md).
    - RGB statistics are on pixel/255; 13-band statistics are on raw DN (uint16), matching the scale
      factors in eurosat_le.data.datasets.SCALE. The two must stay consistent.
    - The 13 MS channels are read in file order, with B8A stored LAST (constants.MS_BANDS). The
      plot reorders them by wavelength for display only.
    - It also hashes the decoded pixels of every RGB image to report exact duplicates, including
      duplicates that fall into different splits (a possible train/test leak).
"""

# hashlib fingerprints images to find duplicates; json writes the summary files.
import hashlib
import json

# numpy does the array maths; tifffile reads the 13-band GeoTIFFs; PIL reads the RGB JPEGs; tqdm shows progress bars.
import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

# viz holds the shared plot style and colours.
from eurosat_le import viz
# Class names, dataset description, MS band names and the important folders.
from eurosat_le.constants import CLASSES, DATASETS, MS_BANDS, RAW_DIR, RESULTS_DIR, SPLITS_DIR
# check_integrity verifies the split files are unchanged; load_split reads split.csv.
from eurosat_le.data.splits import check_integrity, load_split

# ==================== CONCEPTS USED IN THIS FILE ====================
# - normalisation / standardisation: rescaling each channel to mean 0 and standard deviation 1 ((x - mean) / std) so
#   training is stable. The mean and std per channel are computed here and saved to splits/norm_stats.json.
# - train-only statistics (no leakage): mean/std come only from the train split, never from validation or test, so no
#   information from held-out data enters training. Labels are not used for them.
# - mean, population std, E[x^2] - E[x]^2: RunningStats keeps the pixel count, sum and sum of squares; mean = sum/n and
#   std = sqrt(sum_sq/n - mean^2). float64 keeps this subtraction accurate for large uint16 values.
# - running (streaming) statistics: adding one image at a time so the whole dataset never has to be in memory.
# - RGB vs 13-band MS, DN, uint16, scale: RGB statistics are on pixel/255 (0..1); MS statistics are on raw digital
#   numbers (reflectance x 10000, stored as uint16). The scales must match those used in eurosat_le.data.datasets.SCALE.
# - Sentinel-2 bands (B01..B12, B8A): the 13 spectral bands of the satellite; the files store them in order B01..B12
#   with B8A last (constants.MS_BANDS). The spectral plot reorders them by wavelength for display only.
# - GeoTIFF / tifffile / PIL: tifffile reads the 13-band .tif files into arrays (64, 64, 13); PIL (Pillow) decodes the
#   RGB .jpg files.
# - hash fingerprint (MD5) for duplicate detection: md5 of the decoded pixel bytes of each RGB image; images with the
#   same fingerprint are exact duplicates. It is only a duplicate fingerprint, not a security check.
# - train/test leakage via duplicates: if identical images fall into different splits (for example train and test), the
#   test score would be too optimistic; 'cross' counts such duplicate groups.
# - class counts (pivot_table): counts of images per class and per split, laid out as a table (rows = classes, columns =
#   train/val/test).
# - z-score and heatmap: z = (class mean - overall mean) / overall std per band; shown as a colour grid with a diverging
#   colour map (blue below average, red above), symmetric around zero.
# - spectral signature: how a class's average reflectance changes across wavelength bands; different land-use classes
#   have different signatures, which is why 13 bands can help.
# - matplotlib (plt, fig, ax), savefig, close: plotting library; each plot function draws a figure, saves it as PNG and
#   closes it to free memory. It is imported inside the functions so the rest of the script works without it.
# - numpy default_rng(seed) / choice: a seeded random generator; with a fixed seed the same 5 images per class are drawn
#   every time, so the figure is repeatable.
# - tqdm: shows a progress bar while looping over 27,000 images.
# - check_integrity: recomputes the SHA-256 of the split files and compares with manifest.json, so 'train' below really
#   means the committed, unmodified split.
# - json: summary.json and norm_stats.json are written as JSON text files.
# - if __name__ == '__main__': runs main() only when executed as a script, not when imported.
# ====================================================================
#
# ---- Q&A: THIS FILE ----
# Q: What is this file for?
# A: It inspects the dataset (counts, shapes, value ranges, duplicates, plots) and, most importantly, computes the
#    per-channel mean/std on the train split and saves them to splits/norm_stats.json, which every dataset uses to
#    standardise pixels.
# Q: Where does it fit in the pipeline?
# A: After download_data.py and make_splits.py, before build_cache.py and training. It runs once.
# Q: What does it read and write?
# A: Reads splits/split.csv, data/raw/EuroSAT_RGB/*.jpg and (if present) data/raw/EuroSAT_MS/*.tif. Writes
#    splits/norm_stats.json and, in results/dataset_inspection/, summary.json, class_counts.csv and the PNG figures
#    (samples_rgb, class_counts, spectral_signatures).
# Q: Why are the statistics computed only from the training images?
# A: Using validation or test images would let information from held-out data influence training (leakage). Train images
#    that count as unlabelled in the 1%/10% settings are included, but no labels are used, so that is not label leakage.

OUT = RESULTS_DIR / "dataset_inspection"  # folder for all outputs of this script


# ---- Q&A: RunningStats ----
# Q: What is RunningStats?
# A: A small accumulator that computes the per-channel mean and standard deviation over all pixels of many images,
#    keeping only three running totals: the number of pixels, the sum, and the sum of squares.
# Q: Why not load all images and call np.mean / np.std?
# A: The 13-band data is about 2.9 GB; running sums need only a few numbers in memory and give the same result.
# Q: Which standard deviation does it give?
# A: The population std (divides by the number of pixels), computed as sqrt(E[x^2] - E[x]^2) in float64 for accuracy.
class RunningStats:
    """Per-channel mean/std over all pixels, accumulated in float64.

    Keeps only running sums (count, sum, sum of squares), so the whole dataset never has to be in
    memory. The result is the population std (divides by the number of pixels), computed as
    sqrt(E[x^2] - E[x]^2); float64 keeps that subtraction accurate for the uint16 MS values.
    """

    # ---- Q&A: __init__ ----
    # Q: What does __init__ do?
    # A: It starts an empty accumulator: n = 0 pixels, and per-channel sum s and sum of squares ss as arrays of zeros.
    # Q: What is the input?
    # A: c, the number of channels: 3 for RGB, 13 for multispectral.
    def __init__(self, c):
        """Start an empty accumulator for `c` channels (3 for RGB, 13 for multispectral)."""
        # n = number of pixels seen; s = per-channel sum of values; ss = per-channel sum of squared values.
        self.n, self.s, self.ss = 0, np.zeros(c), np.zeros(c)

    # ---- Q&A: add ----
    # Q: What does add() do?
    # A: It adds one image to the totals: reshapes it to a list of pixels with C values each, converts to float64, and
    #    increases n, s and ss.
    # Q: Why convert to float64?
    # A: To avoid overflow and rounding errors when summing squares of large numbers, such as uint16 values up to about
    #    10,000 for the MS bands.
    # Q: What is the input?
    # A: An array of shape (H, W, C) of any numeric type. It returns nothing; it updates the object.
    def add(self, x):  # x: (H, W, C)
        """Add one image (H, W, C), any numeric dtype; every pixel counts equally."""
        # Flatten to a list of pixels with C values each, in float64 so the sums do not overflow or lose precision.
        x = x.reshape(-1, x.shape[-1]).astype(np.float64)
        # Count the pixels, and add this image's per-channel sums and sums of squares to the totals.
        self.n += len(x)
        self.s += x.sum(0)
        self.ss += (x**2).sum(0)

    # ---- Q&A: result ----
    # Q: What does result() return?
    # A: A pair (mean, std), each an array with one value per channel.
    # Q: How is the std obtained from the totals?
    # A: From the identity variance = E[x^2] - (E[x])^2, with E[x] = sum/n and E[x^2] = sum_sq/n, then a square root.
    def result(self):
        """Return (mean, std), each an array with one value per channel."""
        # mean = sum / count; std from the identity var = E[x^2] - (E[x])^2.
        mean = self.s / self.n
        return mean, np.sqrt(self.ss / self.n - mean**2)


# ---- Q&A: main ----
# Q: What does main() do?
# A: It checks the split files are unchanged, loads the split table, counts images per class and split, reads every RGB
#    image (collecting shape info, duplicate fingerprints and train-only mean/std), does the same for the 13-band images
#    if present (also band min/max and class means), writes norm_stats.json and summary.json, and draws the three
#    figures.
# Q: Why does it call check_integrity first?
# A: So that 'train' refers to the committed, unmodified split; otherwise the statistics could be computed on the wrong
#    images.
# Q: Why are RGB values divided by 255 but MS values left as raw DN?
# A: That matches the scale factors used when datasets are loaded (SCALE in eurosat_le.data.datasets); the statistics
#    and the loading code must use the same scale or the standardisation would be wrong.
# Q: Why is the MS part optional?
# A: The 13-band data is large; if its folder is missing the script prints a message and skips those statistics.
def main() -> None:
    """Inspect the data on disk and write norm_stats.json plus the inspection outputs (see module docstring)."""
    # Make sure the output folder exists and switch on the project's plot style.
    OUT.mkdir(parents=True, exist_ok=True)
    viz.apply_style()
    check_integrity(SPLITS_DIR)  # make sure "train" below refers to the committed, unmodified split
    # Table of all images with their class, label and split (train/val/test).
    df = load_split(SPLITS_DIR)
    # The summary dictionary that is saved as summary.json at the end.
    summary = {"n_samples": len(df), "classes": CLASSES}

    # Count images per class and split (rows follow the CLASSES order).
    counts = df.pivot_table(index="class", columns="split", values="id", aggfunc="count").reindex(CLASSES)
    # Put the columns in a fixed order and add a total per class.
    counts = counts[["train", "val", "test"]]
    counts["total"] = counts.sum(axis=1)
    # Save the table, add it to the summary and show it on screen.
    counts.to_csv(OUT / "class_counts.csv")
    summary["class_counts"] = counts.to_dict(orient="index")
    print(counts.to_string())

    # ---- RGB ---------------------------------------------------------------
    # Folder with the RGB JPEG images.
    rgb_dir = RAW_DIR / DATASETS["rgb"]["folder"]
    # Accumulator for 3 channels, the set of distinct image shapes/dtypes seen, and pixel-hash -> images with it.
    rgb_stats, shapes, hashes = RunningStats(3), set(), {}
    # Go through every image together with its split name, with a progress bar.
    for sid, split in tqdm(zip(df["id"], df["split"]), total=len(df), desc="RGB"):
        path = rgb_dir / f"{sid}.jpg"
        # Decode the JPEG into an array of shape (H, W, 3).
        img = np.asarray(Image.open(path).convert("RGB"))
        # Record the image's shape and data type (expected to be the same for every image).
        shapes.add((img.shape, str(img.dtype)))
        # MD5 of the decoded pixels (not the file) is only a duplicate fingerprint, not a security check.
        # Group images by fingerprint; groups with more than one member are exact duplicates.
        hashes.setdefault(hashlib.md5(img.tobytes()).hexdigest(), []).append((sid, split))
        if split == "train":  # leakage guard: statistics use train images only
            # Scale pixel values from 0..255 to 0..1 before accumulating.
            rgb_stats.add(img / 255.0)
    # Fingerprints shared by more than one image: the duplicate groups.
    dups = [v for v in hashes.values() if len(v) > 1]
    # Duplicate groups whose members are in more than one split would be train/test leakage.
    cross = [v for v in dups if len({s for _, s in v}) > 1]
    # Store the RGB findings in the summary (only the first 10 duplicate examples, to keep the file small).
    summary["rgb"] = {
        "shapes": sorted(map(str, shapes)),
        "exact_pixel_duplicate_groups": len(dups),
        "duplicate_groups_spanning_splits": len(cross),
        "duplicate_examples": dups[:10],
    }
    # Final per-channel mean and std of the train images, rounded and converted to plain lists for JSON.
    mean, std = rgb_stats.result()
    norm = {"rgb": {"scale": "pixel/255", "mean": mean.round(6).tolist(), "std": std.round(6).tolist()}}
    print(f"RGB shapes {shapes}; duplicate groups {len(dups)} (across splits: {len(cross)})")

    # ---- 13-band MS ----------------------------------------------------------
    # Folder with the 13-band GeoTIFFs; this part is skipped when they have not been downloaded.
    ms_dir = RAW_DIR / DATASETS["ms"]["folder"]
    if ms_dir.is_dir():
        # Accumulator for 13 channels and the set of distinct shapes/dtypes seen.
        ms_stats, ms_shapes = RunningStats(13), set()
        # Running per-band minimum and maximum over all pixels (start at +inf / -inf so any value replaces them).
        bmin, bmax = np.full(13, np.inf), np.full(13, -np.inf)
        # Sum of per-image band means for each class, and the number of train images per class (for class means).
        class_sum = np.zeros((len(CLASSES), 13))
        class_n = np.zeros(len(CLASSES))
        # Go through every image with its split and class number, with a progress bar.
        for sid, split, label in tqdm(zip(df["id"], df["split"], df["label"]), total=len(df), desc="MS"):
            x = tifffile.imread(ms_dir / f"{sid}.tif")  # (64, 64, 13) uint16, channels in file order (B8A last)
            ms_shapes.add((x.shape, str(x.dtype)))
            # Update the per-band min and max with this image's values.
            bmin = np.minimum(bmin, x.reshape(-1, 13).min(0))
            bmax = np.maximum(bmax, x.reshape(-1, 13).max(0))
            if split == "train":  # leakage guard: statistics and class means use train images only
                # Add this image to the normalisation statistics.
                ms_stats.add(x)
                # Add its per-band mean to its class total (the average is taken later by dividing by class_n).
                class_sum[label] += x.reshape(-1, 13).mean(0)
                class_n[label] += 1
        # Final per-band mean and std over the train images.
        mean, std = ms_stats.result()
        # Record them in raw digital-number units, together with the band order they refer to.
        norm["ms"] = {"scale": "raw DN (reflectance x 10000)", "bands": MS_BANDS,
                      "mean": mean.round(3).tolist(), "std": std.round(3).tolist()}
        # Add the MS findings (shapes and value range per band) to the summary.
        summary["ms"] = {"shapes": sorted(map(str, ms_shapes)),
                         "band_min": dict(zip(MS_BANDS, bmin.tolist())),
                         "band_max": dict(zip(MS_BANDS, bmax.tolist()))}
        print(f"MS shapes {ms_shapes}")
        # Draw the per-class spectral heatmap (class means = class_sum / class_n).
        plot_spectral(class_sum / class_n[:, None], mean, std)
    else:
        print("MS data not found; skipping 13-band statistics.")

    # Note in the file how the statistics were obtained, then save the normalisation statistics and the summary.
    norm["computed_on"] = "train split only (splits/split.csv)"
    (SPLITS_DIR / "norm_stats.json").write_text(json.dumps(norm, indent=2) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    # Draw the remaining two figures.
    plot_samples(df, rgb_dir)
    plot_counts(counts)
    print(f"Wrote {OUT} and {SPLITS_DIR / 'norm_stats.json'}")


# ---- Q&A: plot_samples ----
# Q: What does plot_samples() do?
# A: It saves samples_rgb.png: a grid with one row per class and per_class (5) randomly chosen RGB patches per row.
# Q: Why a fixed seed?
# A: So the same patches are drawn every time and the figure is repeatable.
# Q: What are the inputs?
# A: df (the split table), rgb_dir (folder of JPEGs), per_class and seed. It returns None.
def plot_samples(df, rgb_dir, per_class=5, seed=0):
    """Save a grid of `per_class` randomly chosen RGB patches per class (fixed seed, so the figure is repeatable)."""
    # matplotlib is imported here so the rest of the script works without it until a plot is drawn.
    import matplotlib.pyplot as plt

    # Random generator with a fixed seed -> the same images are picked every time.
    rng = np.random.default_rng(seed)
    # One row per class, `per_class` columns, each cell about 1.1 inches wide.
    fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(per_class * 1.1, len(CLASSES) * 1.1))
    for r, cls in enumerate(CLASSES):
        # All image ids that belong to this class.
        ids = df.loc[df["class"] == cls, "id"].to_numpy()
        # Draw `per_class` different ids at random and show each one in its own cell.
        for c, sid in enumerate(rng.choice(ids, per_class, replace=False)):
            ax = axes[r, c]
            ax.imshow(Image.open(rgb_dir / f"{sid}.jpg"))
            # Hide axis ticks, grid lines and the frame so only the picture remains.
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
        # Write the class name to the left of the row.
        axes[r, 0].set_ylabel(cls, rotation=0, ha="right", va="center", fontsize=8, color=viz.TEXT_PRIMARY)
    # Make the gaps between pictures small, then save the figure and free its memory.
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(OUT / "samples_rgb.png")
    plt.close(fig)


# ---- Q&A: plot_counts ----
# Q: What does plot_counts() do?
# A: It saves class_counts.png: a horizontal bar chart of the total number of images per class, with the exact count
#    written next to each bar.
# Q: Why is it useful?
# A: It shows at a glance how many images each class has, i.e. whether the classes are balanced or some are rarer than
#    others.
# Q: What is the input?
# A: The counts table made in main(); it uses its 'total' column. It returns None.
def plot_counts(counts):
    """Save a horizontal bar chart of the total number of images per class (from the class_counts table)."""
    # matplotlib is imported here so the rest of the script works without it until a plot is drawn.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    # Bar positions, reversed so the first class is drawn at the top.
    y = np.arange(len(CLASSES))[::-1]
    ax.barh(y, counts["total"], height=0.6, color=viz.CATEGORICAL[0])
    # Print the exact count (with thousands separator) just to the right of each bar.
    for yi, v in zip(y, counts["total"]):
        ax.text(v + 30, yi, f"{v:,}", va="center", fontsize=8, color=viz.TEXT_SECONDARY)
    # Label the bars with the class names and add axis label and title.
    ax.set_yticks(y, CLASSES)
    ax.set_xlabel("Images")
    ax.set_title("EuroSAT images per class")
    # No horizontal grid lines; leave 12% free space on the right so the count labels fit.
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, counts["total"].max() * 1.12)
    # Save the figure and free its memory.
    fig.savefig(OUT / "class_counts.png")
    plt.close(fig)


# ---- Q&A: plot_spectral ----
# Q: What does plot_spectral() do?
# A: It saves spectral_signatures.png: a heatmap of each class's mean value in each of the 13 bands, expressed as a
#    z-score against the train-set mean and std of that band.
# Q: Why reorder the bands by wavelength?
# A: The files store B8A last, but physically it lies between B08 and B09; sorting makes the picture follow the
#    spectrum. The reordering is for display only; the data and norm_stats keep file order.
# Q: Why a symmetric colour scale?
# A: vmin = -lim and vmax = +lim make zero (the average) the neutral colour, so red clearly means 'above average' and
#    blue 'below average'.
# Q: What are the inputs?
# A: class_means (n_classes x 13, file order), band_mean and band_std (train statistics). It returns None.
def plot_spectral(class_means, band_mean, band_std):
    """Heatmap of per-class mean reflectance, z-scored per band against the train set.

    `class_means` is (n_classes, 13) in file order; `band_mean` / `band_std` are the train-split statistics
    per band, so each cell shows how far a class's mean is from the overall mean, in std units.
    """
    # matplotlib is imported here so the rest of the script works without it until a plot is drawn.
    import matplotlib.pyplot as plt

    # Show bands in wavelength order (B8A between B08 and B09), not file order.
    # `order` = for each band in wavelength order, its column position in the file-order arrays.
    order = [MS_BANDS.index(b) for b in sorted(MS_BANDS, key=lambda b: (int(b[1:].rstrip("A")), b))]
    # z-score: (class mean - overall mean) / overall std, with the columns reordered by wavelength.
    z = ((class_means - band_mean) / band_std)[:, order]
    # Largest absolute z-value, used to make the colour scale symmetric around zero.
    lim = np.abs(z).max()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    # Draw the heatmap with a diverging colour map (blue-ish below the mean, red-ish above).
    im = ax.imshow(z, cmap=viz.DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    # Label the columns with band names (in wavelength order) and the rows with class names.
    ax.set_xticks(range(13), [MS_BANDS[i] for i in order], rotation=45)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    # Remove grid lines and the frame for a clean heatmap.
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    # Add a colour bar explaining the scale.
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Class mean vs. all-train mean (std units)")
    cb.outline.set_visible(False)
    ax.set_title("Spectral signature by class (train split)")
    # Save the figure and free its memory.
    fig.savefig(OUT / "spectral_signatures.png")
    plt.close(fig)


# Run main() only when the file is executed as a script, not when it is imported.
if __name__ == "__main__":
    main()

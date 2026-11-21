"""PyTorch datasets over the fixed EuroSAT split, for RGB or 13-band input.

Pixels are scaled and standardised per channel with the train-only statistics
in splits/norm_stats.json, then the optional transform is applied.
Labelled datasets yield (image, label). Unlabelled datasets yield only the
image (or the tuple returned by the transform, e.g. weak/strong views); they
never read the true labels.

Pipeline position: sits between the packed cache (eurosat_le.data.cache) / split files
(eurosat_le.data.splits) and the DataLoader factory (eurosat_le.data.loaders).

Design decisions:
    - Pixels are scaled (RGB / 255, MS left in raw DN) and then standardised per channel
      with the mean/std from splits/norm_stats.json. Those statistics come from the TRAIN
      split only (scripts/inspect_dataset.py), so no information from validation or test
      images enters the preprocessing. Augmentation runs after normalisation.
    - Datasets are described by a list of sample ids, mapped to rows of split.csv (which
      are also the row numbers of the packed array). One split file therefore serves
      both RGB and MS input, and the same scene is used for both.
    - Output is always 64x64. Models that need 224 px upsample on the GPU.
    - The unlabelled dataset hides labels at construction time (labels=None), so the
      semi-supervised code has no way to read them by accident.
"""

# Allow modern type-hint syntax on older Python versions.
from __future__ import annotations

# json: read norm_stats.json; Path: file paths.
import json
from pathlib import Path

# numpy: raw pixel arrays; torch: tensors; Dataset: the PyTorch base class that DataLoader understands.
import numpy as np
import torch
from torch.utils.data import Dataset

# Project helpers: default folders, packed-cache access, and split/subset readers.
from eurosat_le.constants import RAW_DIR, SPLITS_DIR
from eurosat_le.data.cache import CACHE_DIR, check_cache, read_raw
from eurosat_le.data.splits import load_split, load_subset

# RGB uint8 -> [0, 1]; MS stays in raw DN. Standardisation happens afterwards with the stats below.
SCALE = {"rgb": 1 / 255.0, "ms": 1.0}  # must match the scale used for norm_stats.json


class NpySource:
    """Row access to a packed .npy cache, opened lazily so each worker maps its own copy.

    Indexing with a row number returns that row's (C, 64, 64) array (uint8 or uint16, unscaled).
    The file is opened on first access, not in __init__, and the open array is dropped when the
    object is pickled. This lets DataLoader workers each open their own memory map instead of
    receiving a copy of a multi-GB array.
    """

    def __init__(self, path: Path, in_memory: bool = False):
        """Remember where the .npy file is; the file itself is not opened yet.

        Args:
            path: location of the packed .npy cache file.
            in_memory: if True, load the whole array into RAM on first access; otherwise memory-map it.
        """
        # Store the settings; _arr stays None until the first row is requested (lazy opening).
        self.path, self.in_memory, self._arr = Path(path), in_memory, None

    def __getitem__(self, row: int) -> np.ndarray:
        """Return one image (C, 64, 64) from the cache, opening the file on first use.

        Args:
            row: row number in the array (same as the row number in split.csv).
        """
        if self._arr is None:
            # First access in this process: open the file (RAM copy or read-only memory map).
            self._arr = np.load(self.path, mmap_mode=None if self.in_memory else "r")
        # Slice out the requested image.
        return self._arr[row]

    def __getstate__(self):  # never pickle the array itself into worker processes
        """Called when the object is pickled (sent to a worker): copy everything except the open array."""
        # Same attributes as before, but with the array replaced by None so workers re-open it themselves.
        return {**self.__dict__, "_arr": None}


class RawSource:
    """Row access that reads the original JPEG/GeoTIFF files (slow; fallback only).

    `all_ids` must be the full id list in split.csv order so that row numbers line up with
    NpySource. Useful to check that the cache and the raw files agree.
    """

    def __init__(self, modality: str, all_ids: list[str], raw_dir: Path = RAW_DIR):
        """Store what is needed to find the raw files.

        Args:
            modality: "rgb" or "ms".
            all_ids: every sample id in split.csv order (so list position = row number).
            raw_dir: folder with the extracted images.
        """
        # Keep the arguments for use in __getitem__.
        self.modality, self.all_ids, self.raw_dir = modality, all_ids, raw_dir

    def __getitem__(self, row: int) -> np.ndarray:
        """Read and return the original image file for row number `row` as a (C, H, W) array."""
        # Translate the row number to a sample id, then read that image file from disk.
        return read_raw(self.modality, self.all_ids[row], self.raw_dir)


class EuroSATDataset(Dataset):
    """Map-style dataset over a subset of the packed samples.

    Args:
        source: object whose [row] returns a (C, H, W) array (NpySource or RawSource).
        rows: row numbers into the source, one per item (i.e. split.csv row numbers).
        labels: integer labels aligned with `rows`, or None for an unlabelled dataset
            (then __getitem__ returns only the image / transform output).
        mean, std: per-channel statistics, in units after `scale` has been applied.
        scale: multiplier applied to raw pixels before standardising (see SCALE).
        transform: optional callable on the normalised float tensor. It may return a
            tuple (e.g. TwoViews gives weak and strong views).
        ids: sample ids aligned with `rows`; stored for bookkeeping only.
    """

    def __init__(self, source, rows, labels, mean, std, scale: float, transform=None, ids=None):
        """Store everything needed to produce items; no image is read here (see the class docstring for args)."""
        # Object that returns raw pixel arrays by row number.
        self.source = source
        # Which rows of the source belong to this dataset, as integers.
        self.rows = np.asarray(rows, dtype=np.int64)
        # Labels as integers, or None when this dataset must not expose labels.
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        # Reshape mean/std from (C,) to (C, 1, 1) so they broadcast over the height and width of an image.
        self.mean = torch.as_tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.as_tensor(std, dtype=torch.float32).view(-1, 1, 1)
        # Factor that brings raw pixel values to the range the statistics were computed in.
        self.scale = scale
        # Optional augmentation applied after normalisation.
        self.transform = transform
        # Sample ids, kept only for reference.
        self.ids = ids

    def __len__(self) -> int:
        """Number of items in this dataset (DataLoader uses this to know the epoch length)."""
        return len(self.rows)

    def load(self, i: int) -> torch.Tensor:
        """Normalised (C, H, W) float tensor for item i, without augmentation."""
        # Fetch the raw pixels of item i, convert them to float32, and wrap them as a torch tensor.
        x = torch.from_numpy(np.asarray(self.source[self.rows[i]], dtype=np.float32))
        # Scale to the working range, then standardise each channel: (value - mean) / std.
        return (x * self.scale - self.mean) / self.std

    def __getitem__(self, i: int):
        """Return item i: (image, label) if labelled, otherwise just the image (or the transform's tuple).

        Args:
            i: position in this dataset (0 .. len-1), not a split.csv row number.
        """
        # Load the normalised image.
        x = self.load(i)
        if self.transform is not None:
            # Augment (random flips, crops, ...); with TwoViews this becomes a (weak, strong) tuple.
            x = self.transform(x)
        if self.labels is None:
            # Unlabelled dataset: never return a label.
            return x
        # Labelled dataset: return the label as a plain Python int.
        return x, int(self.labels[i])


def load_norm_stats(modality: str, splits_dir: Path = SPLITS_DIR) -> tuple[list[float], list[float]]:
    """Per-channel (mean, std) lists for "rgb" or "ms" from splits/norm_stats.json.

    The values were computed on the training split only, in the units given by SCALE.

    Args:
        modality: "rgb" or "ms".
        splits_dir: folder containing norm_stats.json.

    Returns:
        A pair (mean, std), each a list with one number per channel.
    """
    # Read the JSON file and pick the entry for this input type.
    stats = json.loads((splits_dir / "norm_stats.json").read_text())[modality]
    return stats["mean"], stats["std"]


def make_dataset(modality: str, ids: list[str], labelled: bool = True, transform=None, labels=None,
                 source: str = "cache", in_memory: bool = False, splits_dir: Path = SPLITS_DIR,
                 cache_dir: Path = CACHE_DIR, raw_dir: Path = RAW_DIR) -> EuroSATDataset:
    """Dataset over `ids` (subset of split.csv).

    labelled=False hides labels. `labels` overrides the true labels (used for
    pseudo-labels); it must be aligned with `ids`.

    Args:
        modality: "rgb" or "ms".
        ids: sample ids (any subset of split.csv, in the order wanted); every id must
            exist in split.csv, otherwise a KeyError is raised.
        transform: optional augmentation applied after normalisation.
        source: "cache" (packed .npy, checked against the current split.csv) or "raw"
            (original image files; slow).
        in_memory: load the whole cache into RAM instead of memory-mapping it.

    Items are returned in the order of `ids`.

    Returns:
        An EuroSATDataset ready to be given to a DataLoader.
    """
    # Load split.csv (id, class, label, split for all 27,000 samples).
    split = load_split(splits_dir)
    # Map each sample id to its row number in split.csv (= its row in the packed array).
    row_of = {sid: r for r, sid in enumerate(split["id"])}
    # Row numbers for the requested ids, in the requested order (KeyError for an unknown id).
    rows = [row_of[s] for s in ids]
    if labelled and labels is None:
        # True labels come from split.csv. For the semi-supervised pool this branch is skipped
        # (labelled=False), so its ground-truth labels are never read.
        labels = split["label"].to_numpy()[rows]
    if not labelled:
        labels = None  # takes precedence over `labels`: an unlabelled dataset never carries labels
    if source == "cache":
        # Make sure the packed array matches the current split.csv before using it.
        check_cache(modality, splits_dir, cache_dir)
        # Pixel source that reads rows from the packed .npy file.
        src = NpySource(cache_dir / f"{modality}.npy", in_memory=in_memory)
    elif source == "raw":
        # Slow fallback: read the original image files one by one.
        src = RawSource(modality, split["id"].tolist(), raw_dir)
    else:
        raise ValueError(f"unknown source {source!r}")
    # Train-only per-channel statistics for standardising pixels.
    mean, std = load_norm_stats(modality, splits_dir)
    # Bundle everything into a Dataset; list(ids) stores a copy of the id list.
    return EuroSATDataset(src, rows, labels, mean, std, SCALE[modality], transform, ids=list(ids))


def split_ids(name: str, splits_dir: Path = SPLITS_DIR) -> list[str]:
    """Ids of the "train", "val" or "test" split, in split.csv order.

    Args:
        name: which split to return: "train", "val" or "test".
        splits_dir: folder containing split.csv.
    """
    # Read split.csv.
    split = load_split(splits_dir)
    # Keep only the rows whose split column equals `name` and return their ids as a Python list.
    return split.loc[split["split"] == name, "id"].tolist()


def labelled_ids(fraction: float, seed: int, splits_dir: Path = SPLITS_DIR) -> list[str]:
    """Ids of the labelled training subset for this fraction and subset seed (always a subset of train).

    Args:
        fraction: labelled fraction of train (0.01, 0.1 or 1.0).
        seed: which independent subset draw to use (0, 1 or 2).
        splits_dir: folder containing the subset files.
    """
    # Read the stored subset file for this (fraction, seed).
    return load_subset(fraction, seed, splits_dir)


def unlabelled_ids(fraction: float, seed: int, splits_dir: Path = SPLITS_DIR) -> list[str]:
    """Train ids NOT in the labelled subset (the semi-supervised pool).

    Val and test ids are never part of the pool. Use make_dataset(..., labelled=False) on
    these ids so that their labels stay hidden during semi-supervised training.
    """
    # Set of labelled ids, so that membership tests below are fast.
    labelled = set(labelled_ids(fraction, seed, splits_dir))
    # Every train id that is not labelled belongs to the unlabelled pool (split.csv order is kept).
    return [s for s in split_ids("train", splits_dir) if s not in labelled]

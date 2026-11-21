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

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

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
        self.path, self.in_memory, self._arr = Path(path), in_memory, None

    def __getitem__(self, row: int) -> np.ndarray:
        if self._arr is None:
            self._arr = np.load(self.path, mmap_mode=None if self.in_memory else "r")
        return self._arr[row]

    def __getstate__(self):  # never pickle the array itself into worker processes
        return {**self.__dict__, "_arr": None}


class RawSource:
    """Row access that reads the original JPEG/GeoTIFF files (slow; fallback only).

    `all_ids` must be the full id list in split.csv order so that row numbers line up with
    NpySource. Useful to check that the cache and the raw files agree.
    """

    def __init__(self, modality: str, all_ids: list[str], raw_dir: Path = RAW_DIR):
        self.modality, self.all_ids, self.raw_dir = modality, all_ids, raw_dir

    def __getitem__(self, row: int) -> np.ndarray:
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
        self.source = source
        self.rows = np.asarray(rows, dtype=np.int64)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.mean = torch.as_tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.as_tensor(std, dtype=torch.float32).view(-1, 1, 1)
        self.scale = scale
        self.transform = transform
        self.ids = ids

    def __len__(self) -> int:
        return len(self.rows)

    def load(self, i: int) -> torch.Tensor:
        """Normalised (C, H, W) float tensor for item i, without augmentation."""
        x = torch.from_numpy(np.asarray(self.source[self.rows[i]], dtype=np.float32))
        return (x * self.scale - self.mean) / self.std

    def __getitem__(self, i: int):
        x = self.load(i)
        if self.transform is not None:
            x = self.transform(x)
        if self.labels is None:
            return x
        return x, int(self.labels[i])


def load_norm_stats(modality: str, splits_dir: Path = SPLITS_DIR) -> tuple[list[float], list[float]]:
    """Per-channel (mean, std) lists for "rgb" or "ms" from splits/norm_stats.json.

    The values were computed on the training split only, in the units given by SCALE.
    """
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
    """
    split = load_split(splits_dir)
    row_of = {sid: r for r, sid in enumerate(split["id"])}
    rows = [row_of[s] for s in ids]
    if labelled and labels is None:
        # True labels come from split.csv. For the semi-supervised pool this branch is skipped
        # (labelled=False), so its ground-truth labels are never read.
        labels = split["label"].to_numpy()[rows]
    if not labelled:
        labels = None  # takes precedence over `labels`: an unlabelled dataset never carries labels
    if source == "cache":
        check_cache(modality, splits_dir, cache_dir)
        src = NpySource(cache_dir / f"{modality}.npy", in_memory=in_memory)
    elif source == "raw":
        src = RawSource(modality, split["id"].tolist(), raw_dir)
    else:
        raise ValueError(f"unknown source {source!r}")
    mean, std = load_norm_stats(modality, splits_dir)
    return EuroSATDataset(src, rows, labels, mean, std, SCALE[modality], transform, ids=list(ids))


def split_ids(name: str, splits_dir: Path = SPLITS_DIR) -> list[str]:
    """Ids of the "train", "val" or "test" split, in split.csv order."""
    split = load_split(splits_dir)
    return split.loc[split["split"] == name, "id"].tolist()


def labelled_ids(fraction: float, seed: int, splits_dir: Path = SPLITS_DIR) -> list[str]:
    """Ids of the labelled training subset for this fraction and subset seed (always a subset of train)."""
    return load_subset(fraction, seed, splits_dir)


def unlabelled_ids(fraction: float, seed: int, splits_dir: Path = SPLITS_DIR) -> list[str]:
    """Train ids NOT in the labelled subset (the semi-supervised pool).

    Val and test ids are never part of the pool. Use make_dataset(..., labelled=False) on
    these ids so that their labels stay hidden during semi-supervised training.
    """
    labelled = set(labelled_ids(fraction, seed, splits_dir))
    return [s for s in split_ids("train", splits_dir) if s not in labelled]

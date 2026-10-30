"""PyTorch datasets over the fixed EuroSAT split, for RGB or 13-band input.

Pixels are scaled and standardised per channel with the train-only statistics
in splits/norm_stats.json, then the optional transform is applied.
Labelled datasets yield (image, label). Unlabelled datasets yield only the
image (or the tuple returned by the transform, e.g. weak/strong views); they
never read the true labels.
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

SCALE = {"rgb": 1 / 255.0, "ms": 1.0}  # must match the scale used for norm_stats.json


class NpySource:
    """Row access to a packed .npy cache, opened lazily so each worker maps its own copy."""

    def __init__(self, path: Path, in_memory: bool = False):
        self.path, self.in_memory, self._arr = Path(path), in_memory, None

    def __getitem__(self, row: int) -> np.ndarray:
        if self._arr is None:
            self._arr = np.load(self.path, mmap_mode=None if self.in_memory else "r")
        return self._arr[row]

    def __getstate__(self):  # never pickle the array itself into worker processes
        return {**self.__dict__, "_arr": None}


class RawSource:
    """Row access that reads the original JPEG/GeoTIFF files (slow; fallback only)."""

    def __init__(self, modality: str, all_ids: list[str], raw_dir: Path = RAW_DIR):
        self.modality, self.all_ids, self.raw_dir = modality, all_ids, raw_dir

    def __getitem__(self, row: int) -> np.ndarray:
        return read_raw(self.modality, self.all_ids[row], self.raw_dir)


class EuroSATDataset(Dataset):
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
    stats = json.loads((splits_dir / "norm_stats.json").read_text())[modality]
    return stats["mean"], stats["std"]


def make_dataset(modality: str, ids: list[str], labelled: bool = True, transform=None, labels=None,
                 source: str = "cache", in_memory: bool = False, splits_dir: Path = SPLITS_DIR,
                 cache_dir: Path = CACHE_DIR, raw_dir: Path = RAW_DIR) -> EuroSATDataset:
    """Dataset over `ids` (subset of split.csv).

    labelled=False hides labels. `labels` overrides the true labels (used for
    pseudo-labels); it must be aligned with `ids`.
    """
    split = load_split(splits_dir)
    row_of = {sid: r for r, sid in enumerate(split["id"])}
    rows = [row_of[s] for s in ids]
    if labelled and labels is None:
        labels = split["label"].to_numpy()[rows]
    if not labelled:
        labels = None
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
    split = load_split(splits_dir)
    return split.loc[split["split"] == name, "id"].tolist()


def labelled_ids(fraction: float, seed: int, splits_dir: Path = SPLITS_DIR) -> list[str]:
    return load_subset(fraction, seed, splits_dir)


def unlabelled_ids(fraction: float, seed: int, splits_dir: Path = SPLITS_DIR) -> list[str]:
    """Train ids NOT in the labelled subset (the semi-supervised pool)."""
    labelled = set(labelled_ids(fraction, seed, splits_dir))
    return [s for s in split_ids("train", splits_dir) if s not in labelled]

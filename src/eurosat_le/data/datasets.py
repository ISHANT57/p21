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

# ====================================================================================================
# CONCEPTS USED IN THIS FILE
# - PyTorch Dataset (map-style): a class with __len__ (how many items) and __getitem__ (item number i) that a
#   DataLoader can batch. EuroSATDataset is one.
# - Normalisation / standardisation (mean, std): each channel is rescaled to (value - mean) / std so values are
#   centred on 0 with spread about 1, which makes training easier. Mean and std are stored per channel in
#   splits/norm_stats.json.
# - Train-only statistics: mean/std are computed on the training split only (scripts/inspect_dataset.py). Using
#   val/test images would leak information from them into preprocessing.
# - SCALE: RGB uint8 pixels are multiplied by 1/255 to get [0, 1]; MS pixels stay in raw DN (scale 1.0). The scale
#   must match what was used when norm_stats.json was computed.
# - Broadcasting / view(-1, 1, 1): mean and std have one number per channel; reshaping them to (C, 1, 1) lets PyTorch
#   subtract/divide them from every pixel of a (C, H, W) image.
# - Packed .npy cache / memmap / in_memory: NpySource reads rows from data/cache/<modality>.npy, either memory-mapped
#   (lazy, low RAM) or fully in RAM. uint8 for RGB, uint16 for MS.
# - Lazy opening and pickling: the array is opened on first access and dropped when the object is pickled
#   (__getstate__), so each DataLoader worker process opens its own map instead of being sent a multi-GB array.
# - Raw source (fallback): RawSource reads the original JPEG/GeoTIFF files through cache.read_raw. It is slow and
#   meant for checking that cache and raw files agree.
# - Row number / sample id: an id like 'Forest/Forest_12' names an image; its row number is its position in split.csv,
#   which is also its row in the cache array.
# - Labelled vs unlabelled dataset: a labelled dataset returns (image, label); an unlabelled one (labels=None) returns
#   only the image or the transform's output. Used for semi-supervised learning.
# - Unlabelled pool: training ids that are not in the labelled subset (unlabelled_ids). Their true labels are never
#   read.
# - Pseudo-labels: labels guessed by a model for unlabelled images; make_dataset(labels=...) can override the true
#   labels with them.
# - Augmentation / transform: random changes (flips, crops, noise) applied to the normalised tensor after loading; may
#   return a tuple (weak view, strong view) via TwoViews.
# - float32 / torch.from_numpy: the raw pixels are converted to a 32-bit float NumPy array, which torch.from_numpy
#   then wraps as a PyTorch tensor (sharing that array's memory).
# ====================================================================================================
#
# FILE-LEVEL Q&A
# Q: What is this file for?
# A: It turns the split files and the packed image cache into PyTorch Dataset objects that return normalised image
#    tensors (and labels when allowed).
# Q: Where does it fit in the pipeline?
# A: Between data.cache / data.splits (which produce the array and ids) and data.loaders (which wraps datasets in
#    DataLoaders).
# Q: What does it read and write?
# A: It reads splits/split.csv, the subset files, splits/norm_stats.json and the cache .npy (or raw images). It writes
#    nothing.
# Q: Why are labels hidden for the unlabelled dataset at construction time?
# A: So semi-supervised code cannot read true labels by accident; with labels=None there are none to read.
# Q: Why always 64x64 output?
# A: It keeps CPU work small; models that need 224 pixels upsample on the GPU.

# Q: What is SCALE?
# A: A dictionary with the multiplier applied to raw pixels before standardising: 1/255 for RGB (uint8 to 0..1) and
#    1.0 for MS (kept in raw DN). It must match the scale used to compute norm_stats.json.
# RGB uint8 -> [0, 1]; MS stays in raw DN. Standardisation happens afterwards with the stats below.
SCALE = {"rgb": 1 / 255.0, "ms": 1.0}  # must match the scale used for norm_stats.json


# Q: What does this class do?
# A: It gives row access to the packed .npy cache: source[row] returns one (C, 64, 64) image (uint8 or uint16,
#    unscaled).
# Q: Why is the file opened lazily and dropped when pickled?
# A: DataLoader workers are separate processes that receive a pickled copy of the dataset. Dropping the open array
#    means each worker opens its own memory map instead of receiving a copy of a multi-GB array.
# Q: What is the difference between in_memory True and False?
# A: True loads the whole array into RAM on first access (faster, more memory); False memory-maps it read-only (little
#    RAM, pages loaded on demand).
class NpySource:
    """Row access to a packed .npy cache, opened lazily so each worker maps its own copy.

    Indexing with a row number returns that row's (C, 64, 64) array (uint8 or uint16, unscaled).
    The file is opened on first access, not in __init__, and the open array is dropped when the
    object is pickled. This lets DataLoader workers each open their own memory map instead of
    receiving a copy of a multi-GB array.
    """

    # Q: What does this constructor do?
    # A: It only stores the path and the in_memory flag and sets _arr to None; the file is not opened until the first
    #    row is requested.
    # Q: Why not open the file here?
    # A: So the object is cheap to pickle and each worker process can open the file itself.
    def __init__(self, path: Path, in_memory: bool = False):
        """Remember where the .npy file is; the file itself is not opened yet.

        Args:
            path: location of the packed .npy cache file.
            in_memory: if True, load the whole array into RAM on first access; otherwise memory-map it.
        """
        # Store the settings; _arr stays None until the first row is requested (lazy opening).
        self.path, self.in_memory, self._arr = Path(path), in_memory, None

    # Q: What does this method do?
    # A: It returns the image stored at row `row` of the cache, opening the file the first time it is called.
    # Q: What are the inputs and what does it return?
    # A: row (int, same as the row number in split.csv). It returns a (C, 64, 64) NumPy array of the cache's dtype,
    #    not yet scaled.
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

    # Q: What does this method do?
    # A: It tells pickle what to save when the object is copied into a worker process: all attributes, but with the
    #    open array replaced by None.
    # Q: Why is this needed?
    # A: Without it, the memory-mapped or loaded array would be copied into each worker, which is slow and wastes
    #    memory; with it each worker re-opens the file on its first access.
    def __getstate__(self):  # never pickle the array itself into worker processes
        """Called when the object is pickled (sent to a worker): copy everything except the open array."""
        # Same attributes as before, but with the array replaced by None so workers re-open it themselves.
        return {**self.__dict__, "_arr": None}


# Q: What does this class do?
# A: It gives the same row access as NpySource, but reads the original JPEG/GeoTIFF file for that row each time.
# Q: Why does it need the full id list in split.csv order?
# A: So that row numbers line up with NpySource: all_ids[row] is the id of that row. This lets us check that the cache
#    and the raw files agree.
# Q: Why is it only a fallback?
# A: Opening a separate file for every image is slow compared with the packed array.
class RawSource:
    """Row access that reads the original JPEG/GeoTIFF files (slow; fallback only).

    `all_ids` must be the full id list in split.csv order so that row numbers line up with
    NpySource. Useful to check that the cache and the raw files agree.
    """

    # Q: What does this constructor do?
    # A: It stores the modality, the full id list and the raw folder; no file is read.
    def __init__(self, modality: str, all_ids: list[str], raw_dir: Path = RAW_DIR):
        """Store what is needed to find the raw files.

        Args:
            modality: "rgb" or "ms".
            all_ids: every sample id in split.csv order (so list position = row number).
            raw_dir: folder with the extracted images.
        """
        # Keep the arguments for use in __getitem__.
        self.modality, self.all_ids, self.raw_dir = modality, all_ids, raw_dir

    # Q: What does this method do?
    # A: It looks up the id for the row number and reads that image file with cache.read_raw.
    # Q: What are the inputs and what does it return?
    # A: row (int). It returns a (C, H, W) array, uint8 for RGB or uint16 for MS, unscaled.
    def __getitem__(self, row: int) -> np.ndarray:
        """Read and return the original image file for row number `row` as a (C, H, W) array."""
        # Translate the row number to a sample id, then read that image file from disk.
        return read_raw(self.modality, self.all_ids[row], self.raw_dir)


# Q: What does this class do?
# A: It is the PyTorch Dataset used everywhere: for item i it loads the pixels, scales and standardises them,
#    optionally augments them, and returns (image, label) or just the image.
# Q: Why does it take rows, labels, mean/std and a source instead of reading files itself?
# A: It does not care where pixels come from (cache or raw files) or which subset it is. The same class serves train,
#    val, test and the unlabelled pool for both RGB and MS.
# Q: Why is augmentation applied after normalisation?
# A: The augmentations (e.g. cutout fill 0, additive noise) are defined in normalised units, where 0 is the band mean.
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

    # Q: What does this constructor do?
    # A: It stores the pixel source, the row numbers, the labels (or None), and the per-channel mean/std as float
    #    tensors reshaped to (C, 1, 1), plus the scale, transform and ids.
    # Q: Why are mean and std reshaped to (C, 1, 1)?
    # A: So they broadcast over height and width: each channel is shifted and scaled by its own number.
    # Q: Why can labels be None?
    # A: That makes an unlabelled dataset that cannot return a label.
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

    # Q: What does this method do?
    # A: It returns the number of items in the dataset (the number of rows).
    # Q: Why is it needed?
    # A: The DataLoader uses it to know the epoch length and to sample indices.
    def __len__(self) -> int:
        """Number of items in this dataset (DataLoader uses this to know the epoch length)."""
        return len(self.rows)

    # Q: What does this method do?
    # A: It returns the normalised (C, H, W) float32 tensor of item i without any augmentation.
    # Q: What exactly is the normalisation?
    # A: (raw value * scale - mean) / std, per channel, using the train-only statistics.
    # Q: Why is it a separate method from __getitem__?
    # A: Other code can get the clean, un-augmented image directly; __getitem__ adds the transform and the label on
    #    top.
    def load(self, i: int) -> torch.Tensor:
        """Normalised (C, H, W) float tensor for item i, without augmentation."""
        # Fetch the raw pixels of item i, convert them to float32, and wrap them as a torch tensor.
        x = torch.from_numpy(np.asarray(self.source[self.rows[i]], dtype=np.float32))
        # Scale to the working range, then standardise each channel: (value - mean) / std.
        return (x * self.scale - self.mean) / self.std

    # Q: What does this method do?
    # A: It returns item i: the normalised (and, if a transform is set, augmented) image, together with its integer
    #    label if the dataset is labelled.
    # Q: What are the inputs and what does it return?
    # A: i is the position in this dataset (0 .. len-1), not a split.csv row. Output is (image, label), or only the
    #    image / transform output (e.g. a (weak, strong) tuple) when labels is None.
    # Q: Why return the label as a plain int?
    # A: It is simple to batch: the DataLoader turns a list of ints into a tensor of labels.
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


# Q: What does this function do?
# A: It reads the per-channel mean and std for 'rgb' or 'ms' from splits/norm_stats.json.
# Q: Why were they computed on the training split only?
# A: Using validation or test images would leak information about them into preprocessing.
# Q: What are the inputs and what does it return?
# A: modality, splits_dir. It returns (mean, std), two lists with one number per channel (3 for RGB, 13 for MS), in
#    the units given by SCALE.
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


# Q: What does this function do?
# A: It builds an EuroSATDataset over a list of sample ids: finds their split.csv rows, decides which labels to use,
#    chooses the pixel source and loads the normalisation statistics.
# Q: How are labelled=False and labels= handled?
# A: labelled=True uses the true labels from split.csv unless `labels` overrides them (pseudo-labels, aligned with
#    ids). labelled=False always gives labels=None, even if `labels` was passed, so unlabelled data never carries
#    labels.
# Q: Why check the cache against split.csv?
# A: check_cache refuses a cache built for a different split.csv, which would pair pixels with the wrong labels.
# Q: What are the inputs and what does it return?
# A: modality, ids (must all exist in split.csv, else KeyError), transform, source ('cache' or 'raw'), in_memory and
#    folders. It returns an EuroSATDataset whose items follow the order of ids.
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


# Q: What does this function do?
# A: It returns the ids of the 'train', 'val' or 'test' split, in split.csv order.
# Q: What are the inputs and what does it return?
# A: name (which split) and splits_dir. It returns a list of id strings.
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


# Q: What does this function do?
# A: It returns the ids of the labelled training subset for a label fraction and a subset seed.
# Q: Is it always inside the training set?
# A: Yes; the subset files are built only from train ids and verified by splits.verify.
# Q: What are the inputs and what does it return?
# A: fraction (0.01, 0.1 or 1.0), seed (0, 1 or 2), splits_dir. It returns a list of ids read from splits/subsets/.
def labelled_ids(fraction: float, seed: int, splits_dir: Path = SPLITS_DIR) -> list[str]:
    """Ids of the labelled training subset for this fraction and subset seed (always a subset of train).

    Args:
        fraction: labelled fraction of train (0.01, 0.1 or 1.0).
        seed: which independent subset draw to use (0, 1 or 2).
        splits_dir: folder containing the subset files.
    """
    # Read the stored subset file for this (fraction, seed).
    return load_subset(fraction, seed, splits_dir)


# Q: What does this function do?
# A: It returns the training ids that are NOT in the labelled subset: the unlabelled pool for semi-supervised
#    learning.
# Q: Why are val and test ids excluded?
# A: Only training data may be used for learning, even without labels; val and test must stay untouched for fair
#    evaluation.
# Q: Why build a set of the labelled ids first?
# A: Membership tests on a set are fast; with a list, checking about 19,000 ids against the labelled ones would be
#    slow.
# Q: What are the inputs and what does it return?
# A: fraction, seed, splits_dir. It returns a list in split.csv order (empty when fraction is 1.0). Use
#    make_dataset(..., labelled=False) on it so labels stay hidden.
def unlabelled_ids(fraction: float, seed: int, splits_dir: Path = SPLITS_DIR) -> list[str]:
    """Train ids NOT in the labelled subset (the semi-supervised pool).

    Val and test ids are never part of the pool. Use make_dataset(..., labelled=False) on
    these ids so that their labels stay hidden during semi-supervised training.
    """
    # Set of labelled ids, so that membership tests below are fast.
    labelled = set(labelled_ids(fraction, seed, splits_dir))
    # Every train id that is not labelled belongs to the unlabelled pool (split.csv order is kept).
    return [s for s in split_ids("train", splits_dir) if s not in labelled]

"""Pack all EuroSAT images of one input type into a single .npy array.

Layout: data/cache/<modality>.npy with shape (27000, C, 64, 64), rows in the
order of splits/split.csv; uint8 for RGB, uint16 (raw DN) for MS. A sidecar
<modality>.json records the split.csv hash so a stale cache is detected.
Reading one array (memory-mapped or fully in RAM) is far faster than opening
27,000 small files every epoch, especially on Colab's 2-core machines.

Pipeline position: runs after download_data.py and make_splits.py (scripts/build_cache.py
is the entry point) and before training. eurosat_le.data.datasets reads the result.

Design decisions:
    - One packed .npy per input type, in the row order of splits/split.csv, so a sample's
      split.csv row number is also its row in the array (datasets.make_dataset relies on this).
    - The pixels are stored exactly as they come from the raw files (uint8 RGB, raw uint16
      DN for MS, no scaling), so the cache is a lossless copy. Scaling and normalisation
      are done later, at load time, with the train-only statistics.
    - MS channels keep the file order, i.e. B8A is stored LAST (see constants.MS_BANDS).
    - Images stay 64x64 here. Upsampling to 224 px for the ImageNet models happens on
      the GPU inside the model, so the cache stays small (RGB ~0.33 GB, MS ~2.9 GB).
    - The sidecar JSON holds the SHA-256 of split.csv. If the split ever changes, the row
      order would no longer match, so check_cache refuses to use the stale array.
"""

# Allow modern type-hint syntax on older Python versions.
from __future__ import annotations

# hashlib: SHA-256 fingerprint of split.csv; json: write/read the sidecar file; Path: file paths.
import hashlib
import json
from pathlib import Path

# numpy: the packed array; tifffile: read 13-band GeoTIFFs; PIL: read JPEGs; tqdm: progress bar.
import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

# Project constants (default folders, dataset descriptions) and the split-file helpers.
from eurosat_le.constants import DATA_DIR, DATASETS, RAW_DIR, SPLITS_DIR
from eurosat_le.data.splits import SPLIT_FILE, load_split

# Folder where the packed .npy files and their .json sidecars are stored.
CACHE_DIR = DATA_DIR / "cache"
# Number of channels (bands) per image for each input type.
CHANNELS = {"rgb": 3, "ms": 13}
# Pixel data type stored in the cache for each input type (8-bit for JPEG, 16-bit for the raw GeoTIFF values).
DTYPES = {"rgb": np.uint8, "ms": np.uint16}


def read_raw(modality: str, sample_id: str, raw_dir: Path = RAW_DIR) -> np.ndarray:
    """Read one image from the extracted dataset as a (C, H, W) array.

    Args:
        modality: "rgb" (JPEG, decoded to uint8) or "ms" (GeoTIFF, raw uint16 DN).
        sample_id: modality-independent id such as "AnnualCrop/AnnualCrop_1".
        raw_dir: folder that holds EuroSAT_RGB / EuroSAT_MS.

    Returns:
        C-contiguous array in channel-first order. For "ms" the channels are in file
        order (B8A last); no reordering or scaling is applied.
    """
    # Look up the folder name and file extension for this input type.
    spec = DATASETS[modality]
    # Build the full file path, e.g. <raw_dir>/EuroSAT_RGB/AnnualCrop/AnnualCrop_1.jpg.
    path = raw_dir / spec["folder"] / f"{sample_id}{spec['ext']}"
    if modality == "rgb":
        # Decode the JPEG and force 3 colour channels, then view it as a (H, W, 3) uint8 array.
        x = np.asarray(Image.open(path).convert("RGB"))
    else:
        x = tifffile.imread(path)  # (H, W, 13) uint16, no band names in the file
    # Both readers return (H, W, C); move channels first. ascontiguousarray makes the copy
    # that a transposed view needs before it is written into the memmap.
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def _split_hash(splits_dir: Path) -> str:
    """SHA-256 of split.csv, used to tie a cache to the exact row order it was built from.

    Args:
        splits_dir: folder that contains split.csv.

    Returns:
        The 64-character hexadecimal fingerprint of the file's bytes.
    """
    # Any change to split.csv (even a reordering) changes this fingerprint.
    return hashlib.sha256((splits_dir / SPLIT_FILE).read_bytes()).hexdigest()


def build_cache(modality: str, raw_dir: Path = RAW_DIR, splits_dir: Path = SPLITS_DIR,
                cache_dir: Path = CACHE_DIR, overwrite: bool = False) -> Path:
    """Build <cache_dir>/<modality>.npy (and its .json sidecar) from the raw files.

    If the array already exists and `overwrite` is False, nothing is rebuilt, but the
    stale-split check still runs (it raises if split.csv changed). Otherwise every image
    is read in split.csv order and written into the array; any image with an unexpected
    shape or dtype aborts the build. Returns the path of the .npy file.

    Args:
        modality: "rgb" or "ms".
        raw_dir: folder with the extracted images.
        splits_dir: folder with split.csv (defines the row order).
        cache_dir: folder where the .npy and .json files are written.
        overwrite: rebuild even if the cache already exists.
    """
    # Make sure the output folder exists (no error if it is already there).
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{modality}.npy"
    if out.exists() and not overwrite:
        # Nothing to build, but still make sure the existing cache matches the current split.csv.
        check_cache(modality, splits_dir, cache_dir)
        print(f"[{modality}] cache up to date: {out}")
        return out
    # Sample ids in split.csv order: row i of the array will hold the image of ids[i].
    ids = load_split(splits_dir)["id"].tolist()
    # Final array shape: (number of images, channels, height, width).
    shape = (len(ids), CHANNELS[modality], 64, 64)
    # Write to a temporary name and rename at the end, so an interrupted build never leaves
    # a half-filled array that looks like a finished cache.
    tmp = out.with_suffix(".tmp.npy")
    # Create the .npy file on disk as a memory-mapped array, so the (up to ~3 GB) data is written
    # straight to disk instead of having to fit in RAM.
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=DTYPES[modality], shape=shape)
    # Fill the array one image at a time, with a progress bar.
    for i, sid in enumerate(tqdm(ids, desc=f"cache {modality}")):
        # Read the raw image file as a (C, 64, 64) array.
        x = read_raw(modality, sid, raw_dir)
        # Sanity check: a wrongly sized or wrongly typed image would corrupt the cache, so stop at once.
        if x.shape != shape[1:] or x.dtype != arr.dtype:
            raise RuntimeError(f"{sid}: got {x.shape} {x.dtype}, expected {shape[1:]} {arr.dtype}")
        # Copy the image into row i of the array.
        arr[i] = x
    # Force all pending data out to disk.
    arr.flush()
    # Drop the memmap so the file is closed (required before renaming on Windows).
    del arr
    # Give the finished file its final name.
    tmp.rename(out)
    # The sidecar is written last: a cache without a sidecar is treated as unusable by check_cache.
    # Facts about the cache that later checks and readers can rely on (including the split.csv fingerprint).
    meta = {"modality": modality, "shape": shape, "dtype": np.dtype(DTYPES[modality]).name,
            "split_sha256": _split_hash(splits_dir)}
    # Save the metadata as readable, indented JSON next to the array.
    (cache_dir / f"{modality}.json").write_text(json.dumps(meta, indent=2) + "\n")
    # Report where the file is and how big it is (bytes -> GB).
    print(f"[{modality}] wrote {out} ({out.stat().st_size / 1e9:.2f} GB)")
    return out


def check_cache(modality: str, splits_dir: Path = SPLITS_DIR, cache_dir: Path = CACHE_DIR) -> None:
    """Raise RuntimeError if the cache was built for a different split.csv than the current one.

    This protects against mixing up rows: labels and ids come from split.csv, pixels from the cache.
    Raises FileNotFoundError if the sidecar JSON is missing (cache not built).

    Args:
        modality: "rgb" or "ms".
        splits_dir: folder with the current split.csv.
        cache_dir: folder with the cache files.
    """
    # Load the metadata that was saved when the cache was built.
    meta = json.loads((cache_dir / f"{modality}.json").read_text())
    # Compare the saved split.csv fingerprint with the current one; a difference means the rows no longer match.
    if meta["split_sha256"] != _split_hash(splits_dir):
        raise RuntimeError(f"{cache_dir / modality}.npy was built for a different split.csv; rebuild it")


def load_cache(modality: str, in_memory: bool = False, splits_dir: Path = SPLITS_DIR,
               cache_dir: Path = CACHE_DIR) -> np.ndarray:
    """Return the cached array, memory-mapped (default) or fully loaded into RAM.

    The array has shape (27000, C, 64, 64), rows in split.csv order. Memory-mapping
    keeps RAM use low; in_memory=True trades ~0.33 GB (RGB) / ~2.9 GB (MS) of RAM for faster access.

    Args:
        modality: "rgb" or "ms".
        in_memory: if True, read the whole array into RAM; if False, map the file lazily.
        splits_dir: folder with the current split.csv (for the staleness check).
        cache_dir: folder with the cache files.
    """
    # Refuse to continue if the cache was built from a different split.csv.
    check_cache(modality, splits_dir, cache_dir)
    # mmap_mode=None loads everything into RAM; "r" opens a read-only memory map (pages loaded on demand).
    return np.load(cache_dir / f"{modality}.npy", mmap_mode=None if in_memory else "r")

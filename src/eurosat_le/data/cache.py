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

# ====================================================================================================
# CONCEPTS USED IN THIS FILE
# - Cache (.npy): one big file that holds every image of one input type in a single NumPy array, so training reads one
#   file instead of opening 27,000 small ones each epoch. Written to data/cache/<modality>.npy.
# - .npy format: NumPy's own binary file format for one array (shape and dtype are stored in its header); it can be
#   re-opened with np.load.
# - Memory map (memmap, mmap_mode): the array stays on disk and the operating system loads only the parts that are
#   read, so RAM use stays low. open_memmap creates the file while building; np.load(mmap_mode='r') reads it lazily.
# - uint8 / uint16: unsigned integers of 8 bits (0..255, JPEG pixels) and 16 bits (0..65535, raw satellite values).
#   The cache keeps the original type: uint8 for RGB, uint16 for MS.
# - DN (digital number): the raw integer pixel value stored by the satellite product; the MS cache holds it unscaled.
#   Scaling and normalisation happen later, in data.datasets.
# - Shape (N, C, H, W) / channel-first: N images, C channels (3 for RGB, 13 for MS), height and width 64. PyTorch
#   wants channels first, but image readers give (H, W, C), hence the transpose in read_raw.
# - Contiguous array: an array whose values sit next to each other in memory; a transposed view is not, so
#   np.ascontiguousarray makes a real copy before it is written to the memmap.
# - EuroSAT RGB (JPEG) / MS (GeoTIFF): RGB images are read with Pillow and forced to 3 channels; 13-band GeoTIFFs are
#   read with tifffile. The MS channels keep the file order (B8A is last).
# - Sidecar JSON: a small <modality>.json file next to the array holding modality, shape, dtype and the SHA-256 of
#   split.csv. It is written last, so a cache without it is treated as unusable.
# - Stale cache: a cache built for an older split.csv; its row order would no longer match the labels. check_cache
#   compares the saved split.csv hash with the current one and refuses a mismatch.
# - SHA-256 hash: a fingerprint of a file's bytes; any change to split.csv (even re-ordering rows) changes it.
# - Atomic write (temp file then rename): the array is built under a temporary name and renamed when complete, so an
#   interrupted build never leaves a half-filled file that looks finished.
# - tqdm: a library that shows a progress bar while the loop runs.
# ====================================================================================================
#
# FILE-LEVEL Q&A
# Q: What is this file for?
# A: It packs all 27,000 EuroSAT images of one input type into one .npy array (rows in split.csv order) and provides
#    safe access to it.
# Q: Where does it fit in the pipeline?
# A: After download_data.py and make_splits.py, before training (entry point scripts/build_cache.py). data.datasets
#    reads the array it produces.
# Q: What does it read and write?
# A: It reads the raw JPEG/GeoTIFF files and splits/split.csv; it writes data/cache/<modality>.npy and
#    data/cache/<modality>.json.
# Q: Why a cache instead of reading the image files directly?
# A: Opening 27,000 small files every epoch is slow, especially on Colab's 2-core machines; one array is far faster.
#    It also stores raw pixels without scaling, so it is a lossless copy.
# Q: Why keep images at 64x64 in the cache?
# A: It keeps the files small (RGB about 0.33 GB, MS about 2.9 GB); upsampling to 224 pixels for the ImageNet models
#    happens later on the GPU.

# Q: What are CACHE_DIR, CHANNELS and DTYPES?
# A: CACHE_DIR is data/cache (where the files go). CHANNELS gives the number of channels per image (3 for rgb, 13 for
#    ms). DTYPES gives the stored pixel type: uint8 for JPEG pixels, uint16 for the raw GeoTIFF values, so no
#    information is lost.
# Folder where the packed .npy files and their .json sidecars are stored.
CACHE_DIR = DATA_DIR / "cache"
# Number of channels (bands) per image for each input type.
CHANNELS = {"rgb": 3, "ms": 13}
# Pixel data type stored in the cache for each input type (8-bit for JPEG, 16-bit for the raw GeoTIFF values).
DTYPES = {"rgb": np.uint8, "ms": np.uint16}


# Q: What does this function do?
# A: It reads one image from the extracted dataset and returns it as a channel-first (C, H, W) NumPy array.
# Q: Why is the array transposed?
# A: Both readers return (H, W, C), but PyTorch models expect (C, H, W). ascontiguousarray makes a proper copy so it
#    can be written into the memmap.
# Q: What are the inputs and what does it return?
# A: modality ('rgb' or 'ms'), sample_id (e.g. 'AnnualCrop/AnnualCrop_1'), raw_dir. It returns uint8 (RGB) or uint16
#    (MS) data with no scaling or band reordering; MS keeps file order with B8A last.
# Q: Why convert('RGB') for JPEGs?
# A: It guarantees exactly 3 colour channels whatever mode the file is in.
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


# Q: What does this function do?
# A: It returns the SHA-256 fingerprint of splits/split.csv.
# Q: Why is it needed?
# A: The cache rows are in split.csv order. Saving the fingerprint with the cache lets us detect later that split.csv
#    changed and the rows no longer match.
# Q: What are the inputs and what does it return?
# A: splits_dir (folder with split.csv). It returns a 64-character hex string.
def _split_hash(splits_dir: Path) -> str:
    """SHA-256 of split.csv, used to tie a cache to the exact row order it was built from.

    Args:
        splits_dir: folder that contains split.csv.

    Returns:
        The 64-character hexadecimal fingerprint of the file's bytes.
    """
    # Any change to split.csv (even a reordering) changes this fingerprint.
    return hashlib.sha256((splits_dir / SPLIT_FILE).read_bytes()).hexdigest()


# Q: What does this function do?
# A: It reads every image in split.csv order, writes them into one memory-mapped .npy array, and then writes a sidecar
#    JSON with metadata and the split.csv hash.
# Q: Why is the array written under a temporary name, and the JSON last?
# A: If the build is interrupted, no file with the final name exists and no sidecar exists, so a half-finished cache
#    is never used. check_cache treats a missing sidecar as 'not built'.
# Q: Why check each image's shape and dtype?
# A: A wrongly sized or wrongly typed image would silently corrupt the cache or be converted by NumPy; stopping at
#    once tells us which id is wrong.
# Q: What are the inputs and what does it return?
# A: modality, raw_dir, splits_dir, cache_dir, overwrite. It returns the Path of the .npy file. If the cache exists
#    and overwrite is False it rebuilds nothing but still runs the stale-split check.
# Q: Why use open_memmap rather than building a list in RAM?
# A: The MS array is about 2.9 GB; writing straight to disk avoids holding it all in memory.
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


# Q: What does this function do?
# A: It compares the split.csv hash saved in the cache's JSON with the current split.csv and raises RuntimeError if
#    they differ.
# Q: What would go wrong without it?
# A: Labels and ids come from split.csv but pixels come from the cache. If split.csv changed, row i would hold the
#    wrong image for its label and training would be silently wrong.
# Q: What are the inputs and what does it return?
# A: modality, splits_dir, cache_dir. It returns None; it raises FileNotFoundError if the JSON is missing (cache not
#    built).
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


# Q: What does this function do?
# A: It checks the cache is current and returns it as a NumPy array of shape (27000, C, 64, 64), memory-mapped by
#    default.
# Q: Why memory-map by default, and when use in_memory=True?
# A: Memory mapping uses little RAM. in_memory=True loads everything (about 0.33 GB for RGB, 2.9 GB for MS) for faster
#    access if there is enough memory.
# Q: What are the inputs and what does it return?
# A: modality, in_memory, splits_dir, cache_dir. It returns the array (read-only memmap, or a normal array if
#    in_memory).
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

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

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

from eurosat_le.constants import DATA_DIR, DATASETS, RAW_DIR, SPLITS_DIR
from eurosat_le.data.splits import SPLIT_FILE, load_split

CACHE_DIR = DATA_DIR / "cache"
CHANNELS = {"rgb": 3, "ms": 13}
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
    spec = DATASETS[modality]
    path = raw_dir / spec["folder"] / f"{sample_id}{spec['ext']}"
    if modality == "rgb":
        x = np.asarray(Image.open(path).convert("RGB"))
    else:
        x = tifffile.imread(path)  # (H, W, 13) uint16, no band names in the file
    # Both readers return (H, W, C); move channels first. ascontiguousarray makes the copy
    # that a transposed view needs before it is written into the memmap.
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def _split_hash(splits_dir: Path) -> str:
    """SHA-256 of split.csv, used to tie a cache to the exact row order it was built from."""
    return hashlib.sha256((splits_dir / SPLIT_FILE).read_bytes()).hexdigest()


def build_cache(modality: str, raw_dir: Path = RAW_DIR, splits_dir: Path = SPLITS_DIR,
                cache_dir: Path = CACHE_DIR, overwrite: bool = False) -> Path:
    """Build <cache_dir>/<modality>.npy (and its .json sidecar) from the raw files.

    If the array already exists and `overwrite` is False, nothing is rebuilt, but the
    stale-split check still runs (it raises if split.csv changed). Otherwise every image
    is read in split.csv order and written into the array; any image with an unexpected
    shape or dtype aborts the build. Returns the path of the .npy file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{modality}.npy"
    if out.exists() and not overwrite:
        check_cache(modality, splits_dir, cache_dir)
        print(f"[{modality}] cache up to date: {out}")
        return out
    ids = load_split(splits_dir)["id"].tolist()
    shape = (len(ids), CHANNELS[modality], 64, 64)
    # Write to a temporary name and rename at the end, so an interrupted build never leaves
    # a half-filled array that looks like a finished cache.
    tmp = out.with_suffix(".tmp.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=DTYPES[modality], shape=shape)
    for i, sid in enumerate(tqdm(ids, desc=f"cache {modality}")):
        x = read_raw(modality, sid, raw_dir)
        if x.shape != shape[1:] or x.dtype != arr.dtype:
            raise RuntimeError(f"{sid}: got {x.shape} {x.dtype}, expected {shape[1:]} {arr.dtype}")
        arr[i] = x
    arr.flush()
    del arr
    tmp.rename(out)
    # The sidecar is written last: a cache without a sidecar is treated as unusable by check_cache.
    meta = {"modality": modality, "shape": shape, "dtype": np.dtype(DTYPES[modality]).name,
            "split_sha256": _split_hash(splits_dir)}
    (cache_dir / f"{modality}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[{modality}] wrote {out} ({out.stat().st_size / 1e9:.2f} GB)")
    return out


def check_cache(modality: str, splits_dir: Path = SPLITS_DIR, cache_dir: Path = CACHE_DIR) -> None:
    """Raise RuntimeError if the cache was built for a different split.csv than the current one.

    This protects against mixing up rows: labels and ids come from split.csv, pixels from the cache.
    Raises FileNotFoundError if the sidecar JSON is missing (cache not built).
    """
    meta = json.loads((cache_dir / f"{modality}.json").read_text())
    if meta["split_sha256"] != _split_hash(splits_dir):
        raise RuntimeError(f"{cache_dir / modality}.npy was built for a different split.csv; rebuild it")


def load_cache(modality: str, in_memory: bool = False, splits_dir: Path = SPLITS_DIR,
               cache_dir: Path = CACHE_DIR) -> np.ndarray:
    """Return the cached array, memory-mapped (default) or fully loaded into RAM.

    The array has shape (27000, C, 64, 64), rows in split.csv order. Memory-mapping
    keeps RAM use low; in_memory=True trades ~0.33 GB (RGB) / ~2.9 GB (MS) of RAM for faster access.
    """
    check_cache(modality, splits_dir, cache_dir)
    return np.load(cache_dir / f"{modality}.npy", mmap_mode=None if in_memory else "r")

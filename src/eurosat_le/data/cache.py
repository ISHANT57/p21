"""Pack all EuroSAT images of one input type into a single .npy array.

Layout: data/cache/<modality>.npy with shape (27000, C, 64, 64), rows in the
order of splits/split.csv; uint8 for RGB, uint16 (raw DN) for MS. A sidecar
<modality>.json records the split.csv hash so a stale cache is detected.
Reading one array (memory-mapped or fully in RAM) is far faster than opening
27,000 small files every epoch, especially on Colab's 2-core machines.
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
    """Read one image from the extracted dataset as a (C, H, W) array."""
    spec = DATASETS[modality]
    path = raw_dir / spec["folder"] / f"{sample_id}{spec['ext']}"
    if modality == "rgb":
        x = np.asarray(Image.open(path).convert("RGB"))
    else:
        x = tifffile.imread(path)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def _split_hash(splits_dir: Path) -> str:
    return hashlib.sha256((splits_dir / SPLIT_FILE).read_bytes()).hexdigest()


def build_cache(modality: str, raw_dir: Path = RAW_DIR, splits_dir: Path = SPLITS_DIR,
                cache_dir: Path = CACHE_DIR, overwrite: bool = False) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{modality}.npy"
    if out.exists() and not overwrite:
        check_cache(modality, splits_dir, cache_dir)
        print(f"[{modality}] cache up to date: {out}")
        return out
    ids = load_split(splits_dir)["id"].tolist()
    shape = (len(ids), CHANNELS[modality], 64, 64)
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
    meta = {"modality": modality, "shape": shape, "dtype": np.dtype(DTYPES[modality]).name,
            "split_sha256": _split_hash(splits_dir)}
    (cache_dir / f"{modality}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[{modality}] wrote {out} ({out.stat().st_size / 1e9:.2f} GB)")
    return out


def check_cache(modality: str, splits_dir: Path = SPLITS_DIR, cache_dir: Path = CACHE_DIR) -> None:
    meta = json.loads((cache_dir / f"{modality}.json").read_text())
    if meta["split_sha256"] != _split_hash(splits_dir):
        raise RuntimeError(f"{cache_dir / modality}.npy was built for a different split.csv; rebuild it")


def load_cache(modality: str, in_memory: bool = False, splits_dir: Path = SPLITS_DIR,
               cache_dir: Path = CACHE_DIR) -> np.ndarray:
    """Return the cached array, memory-mapped (default) or fully loaded into RAM."""
    check_cache(modality, splits_dir, cache_dir)
    return np.load(cache_dir / f"{modality}.npy", mmap_mode=None if in_memory else "r")

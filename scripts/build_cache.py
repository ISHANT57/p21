"""Pack the extracted images into data/cache/{rgb,ms}.npy for fast training.

Usage:
    python scripts/build_cache.py --which all     # rgb ~0.33 GB, ms ~2.9 GB

Run after download_data.py and make_splits.py. Safe to re-run: an up-to-date cache is kept.

Why a cache: reading 27,000 small JPEG/GeoTIFF files every epoch is slow, especially on the
2-core Colab/Kaggle machines. One packed array per input type, with rows in the order of
splits/split.csv, can be memory-mapped or loaded into RAM. Values are stored unscaled (uint8 RGB,
raw uint16 DN for the 13 bands, B8A last); normalisation happens later at load time. A sidecar JSON
records the hash of split.csv, so a cache built for a different split is rejected. Resizing to 224 px
is NOT done here; it happens on the GPU inside the models, so the cache stays at 64x64.

Output: data/cache/rgb.npy, data/cache/ms.npy (+ .json sidecars). See eurosat_le/data/cache.py.
"""

import argparse

from eurosat_le.data.cache import build_cache, load_cache
from eurosat_le.data.splits import check_integrity


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--which", choices=["rgb", "ms", "all"], default="all")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    check_integrity()  # refuse to build a cache from split files that no longer match manifest.json
    for m in ["rgb", "ms"] if args.which == "all" else [args.which]:
        build_cache(m, overwrite=args.overwrite)
        arr = load_cache(m)  # memory-mapped read-back; also re-runs the stale-split check
        print(f"[{m}] {arr.shape} {arr.dtype}, first-row mean {arr[0].mean():.1f}")


if __name__ == "__main__":
    main()

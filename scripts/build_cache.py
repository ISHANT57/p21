"""Pack the extracted images into data/cache/{rgb,ms}.npy for fast training.

Usage:
    python scripts/build_cache.py --which all     # rgb ~0.33 GB, ms ~2.9 GB

Run after download_data.py. Safe to re-run: an up-to-date cache is kept.
"""

import argparse

from eurosat_le.data.cache import build_cache, load_cache
from eurosat_le.data.splits import check_integrity


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--which", choices=["rgb", "ms", "all"], default="all")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    check_integrity()
    for m in ["rgb", "ms"] if args.which == "all" else [args.which]:
        build_cache(m, overwrite=args.overwrite)
        arr = load_cache(m)
        print(f"[{m}] {arr.shape} {arr.dtype}, first-row mean {arr[0].mean():.1f}")


if __name__ == "__main__":
    main()

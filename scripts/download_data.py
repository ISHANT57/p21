"""Download EuroSAT.

Usage:
    python scripts/download_data.py --which rgb        # ~95 MB
    python scripts/download_data.py --which ms         # ~2.1 GB zip (HF mirror first, Zenodo fallback), ~2.9 GB extracted
    python scripts/download_data.py --which all --delete-zip
"""

import argparse
from pathlib import Path

from eurosat_le.constants import DATASETS, RAW_DIR
from eurosat_le.data.download import disk_free_gb, fetch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--which", choices=["rgb", "ms", "all"], default="all")
    p.add_argument("--raw-dir", default=str(RAW_DIR))
    p.add_argument("--mirror", choices=["zenodo", "hf"], help="Use only this mirror (default: try all).")
    p.add_argument("--delete-zip", action="store_true", help="Remove the zip after extraction to save disk.")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    names = list(DATASETS) if args.which == "all" else [args.which]
    print(f"Free disk at {raw_dir}: {disk_free_gb(raw_dir):.1f} GB")
    for name in names:
        fetch(name, raw_dir, keep_zip=not args.delete_zip, mirror=args.mirror)


if __name__ == "__main__":
    main()

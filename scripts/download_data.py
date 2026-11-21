"""Download EuroSAT.

Usage:
    python scripts/download_data.py --which rgb        # ~95 MB
    python scripts/download_data.py --which ms         # ~2.1 GB zip (HF mirror first, Zenodo fallback), ~2.9 GB extracted
    python scripts/download_data.py --which all --delete-zip

First step of the pipeline. Downloads the RGB (JPEG) and/or 13-band (GeoTIFF) EuroSAT archives,
checks each against its expected size and hash (SHA-256 for the pinned Hugging Face commit, MD5 for
Zenodo; see eurosat_le/constants.py), extracts it safely, and leaves
data/raw/EuroSAT_RGB/<Class>/*.jpg and data/raw/EuroSAT_MS/<Class>/*.tif. If a mirror fails or its
file does not match the hash, the next mirror is tried. If a dataset is already extracted, it is
skipped. Next steps: make_splits.py (or its --check), inspect_dataset.py, build_cache.py.
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
    names = list(DATASETS) if args.which == "all" else [args.which]  # keys of DATASETS: "rgb", "ms"
    print(f"Free disk at {raw_dir}: {disk_free_gb(raw_dir):.1f} GB")
    for name in names:
        fetch(name, raw_dir, keep_zip=not args.delete_zip, mirror=args.mirror)


if __name__ == "__main__":
    main()

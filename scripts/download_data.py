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

# argparse reads the command-line options; Path is used to handle folder paths in an OS-independent way.
import argparse
from pathlib import Path

# DATASETS describes the two EuroSAT versions (rgb and ms) and RAW_DIR is the default download folder.
from eurosat_le.constants import DATASETS, RAW_DIR
# disk_free_gb reports free disk space; fetch downloads, verifies and extracts one dataset.
from eurosat_le.data.download import disk_free_gb, fetch


def main() -> None:
    """Read the command-line options, then download and extract the chosen EuroSAT version(s).

    Takes no arguments (options come from sys.argv) and returns nothing. Files are written under --raw-dir.
    """
    # Build the argument parser; the module docstring doubles as the --help text (RawDescription keeps its layout).
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Which dataset to fetch: the RGB JPEGs, the 13-band GeoTIFFs, or both.
    p.add_argument("--which", choices=["rgb", "ms", "all"], default="all")
    # Folder the archives are downloaded and extracted into.
    p.add_argument("--raw-dir", default=str(RAW_DIR))
    # Optionally force a single download source instead of trying every mirror in turn.
    p.add_argument("--mirror", choices=["zenodo", "hf"], help="Use only this mirror (default: try all).")
    # Flag: when given, the zip file is deleted once it has been extracted, to save disk space.
    p.add_argument("--delete-zip", action="store_true", help="Remove the zip after extraction to save disk.")
    # Parse sys.argv into the args object.
    args = p.parse_args()

    # Turn the text path into a Path object.
    raw_dir = Path(args.raw_dir)
    # Make sure the folder exists before anything is written into it (parents=True also creates missing parents).
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Decide which datasets to fetch: all of them, or just the one that was asked for.
    names = list(DATASETS) if args.which == "all" else [args.which]  # keys of DATASETS: "rgb", "ms"
    # Show the free space first, because the 13-band download needs several GB.
    print(f"Free disk at {raw_dir}: {disk_free_gb(raw_dir):.1f} GB")
    # Download, verify and extract each chosen dataset (already extracted ones are skipped inside fetch).
    for name in names:
        fetch(name, raw_dir, keep_zip=not args.delete_zip, mirror=args.mirror)


# Run main() only when the file is executed as a script, not when it is imported.
if __name__ == "__main__":
    main()

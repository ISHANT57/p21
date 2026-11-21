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

# ==================== CONCEPTS USED IN THIS FILE ====================
# - argparse: standard-library module that turns the text typed after the script name (for example --which rgb) into a
#   Python object; used in main().
# - sys.argv / command-line option: the list of words typed after 'python scripts/download_data.py'; parse_args() reads
#   it. Options starting with -- are optional switches.
# - RawDescriptionHelpFormatter: argparse setting that prints the module docstring in --help exactly as written, without
#   re-wrapping its lines.
# - EuroSAT: a public satellite-image dataset: 27,000 small 64x64 patches from Sentinel-2, in 10 land-use classes
#   (Forest, River, ...).
# - RGB vs MS (multispectral): the RGB version is ordinary 3-colour JPEG pictures (~95 MB); the MS version has all 13
#   Sentinel-2 spectral bands stored as GeoTIFF files (~2.1 GB zip). --which picks one or both.
# - GeoTIFF: a TIFF image file that can hold many bands (channels) plus map information; the 13-band EuroSAT images use
#   it (.tif).
# - mirror: an alternative server that hosts a copy of the same file. Here: Hugging Face ('hf') and Zenodo. fetch()
#   tries them in turn; --mirror forces just one.
# - checksum / hash (SHA-256, MD5): a short fingerprint computed from a file's bytes; if the downloaded file's
#   fingerprint and size do not match the expected ones, the download is corrupt and is discarded. This is done inside
#   fetch(), not in this file.
# - zip archive / extraction: the dataset is downloaded as a .zip, then unpacked into data/raw/EuroSAT_RGB or
#   EuroSAT_MS; fetch() refuses members that would escape the target folder ('zip slip').
# - idempotent / skip if present: running the script again is harmless: a dataset that is already extracted is skipped
#   by fetch().
# - pathlib.Path: object for file paths that works on Windows and Linux alike; mkdir(parents=True, exist_ok=True)
#   creates the folder and any missing parents without failing if it exists.
# - store_true flag: an argparse option with no value (--delete-zip); it is False unless typed, True when typed.
#   keep_zip=not args.delete_zip converts it.
# - if __name__ == '__main__': runs main() only when the file is executed as a script, not when another file imports it.
# ====================================================================
#
# ---- Q&A: THIS FILE ----
# Q: What is this file for?
# A: It is the command-line entry point for step 1 of the pipeline: getting the EuroSAT images onto the disk. The real
#    work (download, verify, extract) is in eurosat_le.data.download.fetch.
# Q: How is it run and what does it write?
# A: python scripts/download_data.py --which rgb|ms|all [--raw-dir DIR] [--mirror zenodo|hf] [--delete-zip]. It creates
#    data/raw/EuroSAT_RGB/<Class>/*.jpg and/or data/raw/EuroSAT_MS/<Class>/*.tif; it reads nothing except the network.
# Q: What comes next in the pipeline?
# A: scripts/make_splits.py (create the train/val/test split), then inspect_dataset.py and build_cache.py.


# ---- Q&A: main ----
# Q: What does main() do?
# A: It reads the options, makes sure the download folder exists, prints the free disk space, and calls fetch() once for
#    each dataset requested ('rgb', 'ms', or both).
# Q: Why print the free disk space first?
# A: The 13-band dataset needs about 2.1 GB for the zip plus 2.9 GB extracted. Showing the free space lets the user
#    notice a full disk before a long download fails.
# Q: What are the inputs and what does it return?
# A: No arguments: the options come from the command line (sys.argv). It returns None; its only effect is files under
#    --raw-dir.
# Q: Why keep the logic in eurosat_le.data.download and not in this script?
# A: So the same tested functions can be reused (by notebooks and tests) and the script stays a thin wrapper that only
#    parses options.
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

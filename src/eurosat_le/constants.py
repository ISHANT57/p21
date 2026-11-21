"""Project-wide constants: class names, band metadata, download sources, default paths and seeds.

This module is the single source of truth that every other part of the pipeline
imports (download, index/splits, cache, datasets, training), so that no script
hard-codes a path, a class order or a seed on its own.

Contents:
    - Default directories (data/raw, splits/, results/), derived from the repo root.
    - CLASSES / CLASS_TO_IDX: the 10 EuroSAT classes in alphabetical order. The list
      index is the integer label used in splits/split.csv and in every model output,
      so reordering this list would silently change all labels.
    - MS_BANDS / MS_RGB_IDX: channel order of the 13-band GeoTIFFs. Important fact:
      the files store B8A LAST (not between B08 and B09 as in the usual Sentinel-2
      order). See docs/DATA.md for how this was verified. The first-layer adaptation
      of the pretrained models (docs/MODELS.md) relies on MS_RGB_IDX.
    - DATASETS: where to download each input type from. Each entry lists mirrors in
      the order they are tried, with the exact byte size and a fixed hash (MD5 for
      Zenodo, SHA-256 for the pinned Hugging Face commit), so a truncated or altered
      archive is rejected by eurosat_le.data.download.
    - SPLIT_SEED / SUBSET_SEEDS / LABEL_FRACTIONS: fixed seeds and label fractions
      that define the train/val/test partition and the nested labelled subsets
      (eurosat_le.data.splits). Changing them changes every experiment.
"""

# Path gives us object-style, OS-independent file paths.
from pathlib import Path

# ====================================================================================================
# CONCEPTS USED IN THIS FILE
# - EuroSAT: a public satellite-image dataset of 27,000 patches (64x64 pixels), each labelled with one of 10 land-use
#   classes. Here it is the data the whole project uses (CLASSES, DATASETS).
# - Sentinel-2 / band: Sentinel-2 is a European satellite that records light in 13 wavelength ranges called bands
#   (B01..B12, B8A). A 'band' is one such image layer. MS_BANDS lists their names.
# - RGB vs MS: RGB = 3 visible bands (red, green, blue) stored as JPEG; MS (multispectral) = all 13 bands stored as a
#   GeoTIFF. DATASETS has one entry for each.
# - GeoTIFF / JPEG: GeoTIFF is an image file that can hold many bands with 16-bit values (used for MS); JPEG is a
#   compressed 3-channel 8-bit picture (used for RGB). See the 'ext' key in DATASETS.
# - DN (digital number): the raw integer value a satellite sensor stores per pixel; the MS files keep it unscaled (see
#   MS_BANDS comment for typical DN values).
# - Band order / B8A: position of each band inside the file's channel axis. B8A is stored LAST in EuroSAT_MS, not
#   between B08 and B09, so MS_RGB_IDX = [3, 2, 1] points at B04, B03, B02.
# - Mirror: an alternative web address that serves the same file (Zenodo is the official one, Hugging Face is a faster
#   copy). 'mirrors' in DATASETS are tried in order.
# - Checksum (MD5 / SHA-256): a fixed-length fingerprint computed from a file's bytes; if one byte changes the
#   fingerprint changes. Zenodo gives MD5, the pinned Hugging Face commit gives SHA-256; 'size' is the exact byte
#   count.
# - Pinned commit: the Hugging Face URL contains one exact git commit id, so the file behind the URL can never change
#   silently (_HF).
# - Seed: a number that starts a random number generator so 'random' choices repeat exactly. SPLIT_SEED fixes the
#   train/val/test split; SUBSET_SEEDS fix the labelled subsets.
# - Label fraction / labelled subset: the share of the training set whose labels we are allowed to use (1%, 10%,
#   100%); this project studies how accuracy changes with fewer labels (LABEL_FRACTIONS).
# - pathlib.Path / __file__: Path gives OS-independent file paths; __file__ is this file's own location, from which
#   REPO_ROOT is derived so no path is hard-coded.
# ====================================================================================================
#
# FILE-LEVEL Q&A
# Q: What is this file for?
# A: It is the single place that holds fixed facts used everywhere: folder paths, the 10 class names, the 13 band
#    names, where to download the data and how to verify it, and the seeds and label fractions.
# Q: Where does it fit in the pipeline?
# A: Every other module imports from it (download, index, splits, cache, datasets, loaders, training), so nothing
#    hard-codes a path, a class order or a seed.
# Q: What does it read and write?
# A: Nothing on disk. It only defines Python values (paths are computed, not created); other modules create the
#    folders.
# Q: Why keep constants in one file?
# A: If the class order or a seed were repeated in several files they could drift apart and silently change labels or
#    splits. One definition keeps every experiment consistent.

# Q: What is REPO_ROOT and why is it computed this way?
# A: The absolute path of the project's top folder: this file is src/eurosat_le/constants.py, so parents[2] is two
#    levels up from its folder. Computing it from __file__ means scripts work from any starting directory.
# Repository root (src/eurosat_le/constants.py -> repo root is two levels above src/)
# Absolute path of the top folder of the repo; every other path below is built from it, so the code
# works no matter which directory the user starts a script from.
REPO_ROOT = Path(__file__).resolve().parents[2]
# Q: What are DATA_DIR, RAW_DIR, SPLITS_DIR, RESULTS_DIR?
# A: Default folders (data/, data/raw/, splits/, results/) built from REPO_ROOT. data/raw holds the extracted images,
#    splits/ holds the split files committed to git, results/ holds run outputs.
# Parent folder for all downloaded / generated data (used by the download and cache modules).
DATA_DIR = REPO_ROOT / "data"
# Where the extracted EuroSAT images live (written by data.download, read by data.index and data.cache).
RAW_DIR = DATA_DIR / "raw"
# Folder holding split.csv, subset lists and norm_stats.json (written by data.splits, read by almost everything).
SPLITS_DIR = REPO_ROOT / "splits"
# Folder where training runs write their metrics, logs and figures.
RESULTS_DIR = REPO_ROOT / "results"

# Q: What is CLASSES and why is the order important?
# A: The 10 land-use class names, alphabetical. A class's position in the list is its integer label in split.csv and
#    in model outputs, so reordering would silently change every label. Alphabetical is a fixed, easy-to-reproduce
#    rule.
# The 10 EuroSAT classes, sorted alphabetically. The index in this list is the
# integer label used everywhere in the project.
CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
# Q: What is CLASS_TO_IDX?
# A: A dictionary from class name to its integer label (built with enumerate over CLASSES). data.index uses it to give
#    each image a label from its folder name.
# Reverse lookup: class name -> integer label (used by data.index to label each image from its folder name).
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Q: What is MS_BANDS and why is B8A last?
# A: The names of the 13 channels in the order they appear in the GeoTIFF files. The files have no band names, so the
#    order was worked out from the data (typical mean values per channel, correlation with the RGB JPEGs, see the
#    comment). B8A really is stored last in this dataset.
# Sentinel-2 band order in the EuroSAT_MS GeoTIFFs. The files carry no band names;
# B8A is stored LAST. Evidence (scripts/inspect_dataset.py, train means): channel 9
# has mean ~12 DN (cirrus B10), channels 10/11 ~1800/1100 (SWIR B11/B12), channel 12
# ~2600 (NIR, tracks B07/B08), and channels 3/2/1 correlate >0.99 with the RGB JPEGs.
# (Used for documentation and sanity checks; the list position is the channel number.)
MS_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]
# Q: What is MS_RGB_IDX?
# A: The channel positions of red (B04), green (B03) and blue (B02) inside the 13 bands: [3, 2, 1]. It is used when
#    the first layer of a pretrained RGB model is adapted to 13-band input, so the model can find the three visible
#    channels.
# Indices of the red, green and blue bands inside MS_BANDS (B04, B03, B02). Used when adapting the
# first layer of pretrained RGB models to 13-band input.
MS_RGB_IDX = [3, 2, 1]

# Download sources, tried in order. Zenodo record 7711810 is the authors' release;
# the torchgeo Hugging Face mirror (pinned commit) holds the same images and is
# usually much faster. Both are verified by hash. After extraction the class
# folders are moved to data/raw/<folder>/<Class>/<Class>_<n><ext>.
# Q: What are _ZENODO and _HF?
# A: Base web addresses for the two download sources (leading underscore = internal helpers). _ZENODO is the authors'
#    official record 7711810; _HF is the torchgeo copy on Hugging Face, pinned to one commit so it can never change.
# Base URL of the official Zenodo record (private helper, only used to build the URLs below).
_ZENODO = "https://zenodo.org/records/7711810/files"
# Base URL of the Hugging Face mirror, pinned to one exact commit so the files can never change.
_HF = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/68508cef507a0dca26a913adda91d5fc60b28d89"
# Q: What is DATASETS and how is it used?
# A: A dictionary with one entry per input type ('rgb', 'ms'): the folder name, file extension and a list of mirrors,
#    each with URL, local zip name, expected hash and exact byte size. data.download fetches and verifies with it;
#    data.index and data.cache use the folder and extension to find files.
# Q: Why store both size and hash, and why are the mirrors in different order for rgb and ms?
# A: The size check is cheap and catches truncated downloads before the slower hash check catches altered ones. For
#    the 2 GB MS file Hugging Face is listed first because it is usually much faster; for the small RGB file the
#    official Zenodo copy comes first.
# Description of each input type, keyed by modality name ("rgb" or "ms"). Read by data.download
# (what to fetch), data.index (where to scan) and data.cache (how to open each file).
DATASETS = {
    "rgb": {
        # Folder name under data/raw/ after extraction.
        "folder": "EuroSAT_RGB",
        # File extension of each image (3-band JPEG).
        "ext": ".jpg",
        # Download sources in the order they are tried; each has the URL, the local zip name,
        # the expected hash (md5 or sha256) and the exact size in bytes used to verify the download.
        "mirrors": [
            {"name": "zenodo", "url": f"{_ZENODO}/EuroSAT_RGB.zip?download=1", "filename": "EuroSAT_RGB.zip",
             "md5": "f46e308c4d50d4bf32fedad2d3d62f3b", "size": 94_658_721},
            {"name": "hf", "url": f"{_HF}/EuroSAT.zip", "filename": "EuroSAT_hf.zip",
             "sha256": "8ebea626349354c5328b142b96d0430e647051f26efc2dc974c843f25ecf70bd", "size": 94_280_567},
        ],
    },
    "ms": {
        # Folder name under data/raw/ after extraction (13-band version).
        "folder": "EuroSAT_MS",
        # File extension of each image (13-band GeoTIFF).
        "ext": ".tif",
        # The Hugging Face mirror is listed first because it is usually much faster for this 2 GB file.
        "mirrors": [
            {"name": "hf", "url": f"{_HF}/EuroSATallBands.zip", "filename": "EuroSATallBands_hf.zip",
             "sha256": "751f070f9bffa2eed48b24ca2dd0b02959280c08837e8c9a5532a67ba611df59", "size": 2_067_725_275},
            {"name": "zenodo", "url": f"{_ZENODO}/EuroSAT_MS.zip?download=1", "filename": "EuroSAT_MS.zip",
             "md5": "091174add3c8e680a49244acf185b9f0", "size": 2_065_402_329},
        ],
    },
}

# Seeds. SPLIT_SEED fixes the train/val/test partition for every experiment.
# SUBSET_SEEDS give independent draws of the 1% / 10% labelled subsets.
# Q: What are SPLIT_SEED, SUBSET_SEEDS and LABEL_FRACTIONS?
# A: SPLIT_SEED = 42 fixes the train/val/test split for every experiment. SUBSET_SEEDS = [0, 1, 2] give three
#    independent draws of the labelled subsets so results can be averaged. LABEL_FRACTIONS = [0.01, 0.10, 1.00] are
#    the shares of the training set that keep labels (1%, 10%, 100%).
# Random seed for the stratified train/val/test split (used by data.splits.make_split).
SPLIT_SEED = 42
# One labelled-subset draw per seed, so results can be averaged over 3 independent draws.
SUBSET_SEEDS = [0, 1, 2]
# Fractions of the training set that keep labels: 1%, 10% and 100%.
LABEL_FRACTIONS = [0.01, 0.10, 1.00]

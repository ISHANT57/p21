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

from pathlib import Path

# Repository root (src/eurosat_le/constants.py -> repo root is two levels above src/)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = REPO_ROOT / "splits"
RESULTS_DIR = REPO_ROOT / "results"

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
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Sentinel-2 band order in the EuroSAT_MS GeoTIFFs. The files carry no band names;
# B8A is stored LAST. Evidence (scripts/inspect_dataset.py, train means): channel 9
# has mean ~12 DN (cirrus B10), channels 10/11 ~1800/1100 (SWIR B11/B12), channel 12
# ~2600 (NIR, tracks B07/B08), and channels 3/2/1 correlate >0.99 with the RGB JPEGs.
MS_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]
# Indices of the red, green and blue bands inside MS_BANDS (B04, B03, B02).
MS_RGB_IDX = [3, 2, 1]

# Download sources, tried in order. Zenodo record 7711810 is the authors' release;
# the torchgeo Hugging Face mirror (pinned commit) holds the same images and is
# usually much faster. Both are verified by hash. After extraction the class
# folders are moved to data/raw/<folder>/<Class>/<Class>_<n><ext>.
_ZENODO = "https://zenodo.org/records/7711810/files"
_HF = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/68508cef507a0dca26a913adda91d5fc60b28d89"
DATASETS = {
    "rgb": {
        "folder": "EuroSAT_RGB",
        "ext": ".jpg",
        "mirrors": [
            {"name": "zenodo", "url": f"{_ZENODO}/EuroSAT_RGB.zip?download=1", "filename": "EuroSAT_RGB.zip",
             "md5": "f46e308c4d50d4bf32fedad2d3d62f3b", "size": 94_658_721},
            {"name": "hf", "url": f"{_HF}/EuroSAT.zip", "filename": "EuroSAT_hf.zip",
             "sha256": "8ebea626349354c5328b142b96d0430e647051f26efc2dc974c843f25ecf70bd", "size": 94_280_567},
        ],
    },
    "ms": {
        "folder": "EuroSAT_MS",
        "ext": ".tif",
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
SPLIT_SEED = 42
SUBSET_SEEDS = [0, 1, 2]
LABEL_FRACTIONS = [0.01, 0.10, 1.00]

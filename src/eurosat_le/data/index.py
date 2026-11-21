"""Build a deterministic sample index for EuroSAT.

Every sample is identified by a modality-independent id such as
"AnnualCrop/AnnualCrop_1". The RGB file is <id>.jpg in EuroSAT_RGB and the
13-band file is <id>.tif in EuroSAT_MS, so one split file serves both inputs.

Pipeline position: used by scripts/make_splits.py (to build and check the split) and for the
RGB-vs-MS id check; the split itself is then stored in splits/split.csv.

Design decisions:
    - The index is sorted by (class index, file number), not by directory listing order.
      Directory order differs between operating systems; sorting makes the index, and hence
      the split built from it, identical on every machine.
    - The integer label of a sample is its class's position in constants.CLASSES.
"""

# Allow modern type-hint syntax on older Python versions.
from __future__ import annotations

# Path: file paths; pandas: the table (DataFrame) of samples that is returned.
from pathlib import Path

import pandas as pd

# Class list and label mapping, dataset descriptions (folder/extension) and the default raw folder.
from eurosat_le.constants import CLASS_TO_IDX, CLASSES, DATASETS, RAW_DIR


def _scan(folder: Path, ext: str) -> list[str]:
    """Return unsorted "<Class>/<stem>" ids for all files with extension `ext` in each class folder.

    Raises FileNotFoundError if any of the 10 class folders is missing.

    Args:
        folder: dataset folder such as data/raw/EuroSAT_RGB.
        ext: file extension to keep, in lowercase, including the dot (".jpg" or ".tif").
    """
    # Collected ids across all classes.
    ids = []
    for cls in CLASSES:
        # Folder of this class, e.g. EuroSAT_RGB/Forest.
        cls_dir = folder / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder {cls_dir}")
        # Add "<Class>/<file name without extension>" for every file with the right extension
        # (the extension test ignores case and skips stray files such as thumbnails or .DS_Store).
        ids += [f"{cls}/{p.stem}" for p in cls_dir.iterdir() if p.suffix.lower() == ext]
    return ids


def _sort_key(sample_id: str) -> tuple[int, int]:
    """Sort key (class index, numeric file suffix); numeric so that _2 sorts before _10.

    Args:
        sample_id: id like "AnnualCrop/AnnualCrop_1".
    """
    # Separate the class folder name from the file stem ("AnnualCrop_1").
    cls, stem = sample_id.split("/")
    # Class position first, then the number after the last underscore, compared as an integer.
    return CLASS_TO_IDX[cls], int(stem.rsplit("_", 1)[1])


def build_index(raw_dir: Path = RAW_DIR, modality: str = "rgb") -> pd.DataFrame:
    """Return a DataFrame (id, class, label) sorted by class then file number.

    `modality` only selects which folder/extension is scanned; the ids do not depend on it.

    Args:
        raw_dir: folder that holds EuroSAT_RGB / EuroSAT_MS.
        modality: "rgb" or "ms".
    """
    # Folder name and extension for this input type.
    spec = DATASETS[modality]
    # Find all samples and put them in the fixed, machine-independent order.
    ids = sorted(_scan(raw_dir / spec["folder"], spec["ext"]), key=_sort_key)
    # Start the table with the id column.
    df = pd.DataFrame({"id": ids})
    # Class name = the part of the id before the "/".
    df["class"] = df["id"].str.split("/").str[0]
    # Integer label = position of that class name in constants.CLASSES.
    df["label"] = df["class"].map(CLASS_TO_IDX).astype(int)
    return df


def check_modalities_match(raw_dir: Path = RAW_DIR) -> None:
    """Raise if the RGB and MS folders do not contain exactly the same sample ids.

    Args:
        raw_dir: folder that holds EuroSAT_RGB and EuroSAT_MS.
    """
    # Ids found in the RGB folder, as a set for easy comparison.
    rgb = set(build_index(raw_dir, "rgb")["id"])
    # Ids found in the MS folder.
    ms = set(build_index(raw_dir, "ms")["id"])
    # One split file is used for both inputs, so the two id sets must be identical.
    if rgb != ms:
        raise RuntimeError(
            f"RGB/MS id mismatch: {len(rgb - ms)} only in RGB, {len(ms - rgb)} only in MS "
            f"(e.g. {sorted(rgb ^ ms)[:5]})"
        )

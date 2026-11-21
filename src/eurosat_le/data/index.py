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

# ====================================================================================================
# CONCEPTS USED IN THIS FILE
# - Sample id: a text name for one image that does not depend on the file type, such as 'AnnualCrop/AnnualCrop_1'
#   (class folder / file name without extension). RGB file = <id>.jpg, MS file = <id>.tif.
# - Modality: the type of input image: 'rgb' (3-band JPEG) or 'ms' (13-band GeoTIFF). It only decides which folder and
#   extension are scanned.
# - EuroSAT classes and labels: the 10 land-use classes; the integer label of an image is the position of its class
#   name in constants.CLASSES (CLASS_TO_IDX).
# - Deterministic index / sort key: directory listing order differs between operating systems, so ids are sorted by
#   (class index, numeric file number). The resulting table, and any split built from it, is identical on every
#   machine.
# - Numeric vs text sorting: as text, 'x_10' would sort before 'x_2'; _sort_key converts the number to an int so that
#   _2 comes before _10.
# - DataFrame (pandas): a table with named columns; build_index returns one with columns id, class, label.
# - Set comparison: sets are unordered collections; '!=' tests whether two sets contain the same ids and '^' gives ids
#   found in only one of them. Used to compare the RGB and MS folders.
# ====================================================================================================
#
# FILE-LEVEL Q&A
# Q: What is this file for?
# A: It lists every EuroSAT image as a sample id with its class and integer label, in a fixed order, and can check
#    that the RGB and MS folders hold the same images.
# Q: Where does it fit in the pipeline?
# A: After the download. scripts/make_splits.py calls build_index to get the table that data.splits turns into
#    split.csv, and check_modalities_match to verify the two input types.
# Q: What does it read and write?
# A: It reads only file names in data/raw/EuroSAT_RGB and/or EuroSAT_MS (it never opens the images) and writes
#    nothing.


# Q: What does this function do?
# A: It walks through the 10 class folders and returns an id '<Class>/<file stem>' for every file with the wanted
#    extension.
# Q: Why does it compare extensions in lowercase and ignore other files?
# A: So '.JPG' counts as '.jpg', and stray files such as thumbnails or .DS_Store do not become samples.
# Q: What are the inputs and what does it return?
# A: folder (e.g. data/raw/EuroSAT_RGB) and ext (e.g. '.jpg'). It returns an unsorted list of id strings, and raises
#    FileNotFoundError if a class folder is missing.
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


# Q: What does this function do?
# A: It turns an id like 'AnnualCrop/AnnualCrop_1' into the pair (class index, file number) that is used for sorting.
# Q: Why is the file number converted to an int?
# A: Sorting the text would put '_10' before '_2'. Sorting numbers gives the natural order.
# Q: What are the inputs and what does it return?
# A: sample_id (string). It returns a tuple of two ints.
def _sort_key(sample_id: str) -> tuple[int, int]:
    """Sort key (class index, numeric file suffix); numeric so that _2 sorts before _10.

    Args:
        sample_id: id like "AnnualCrop/AnnualCrop_1".
    """
    # Separate the class folder name from the file stem ("AnnualCrop_1").
    cls, stem = sample_id.split("/")
    # Class position first, then the number after the last underscore, compared as an integer.
    return CLASS_TO_IDX[cls], int(stem.rsplit("_", 1)[1])


# Q: What does this function do?
# A: It builds the table of all samples with columns id, class and label, sorted by class and file number.
# Q: Why is the table sorted explicitly?
# A: The order of files from the disk differs between operating systems. A fixed order makes the table, and the random
#    split made from it, identical for everybody.
# Q: What are the inputs and what does it return?
# A: raw_dir (folder holding EuroSAT_RGB / EuroSAT_MS) and modality ('rgb' or 'ms'). It returns a pandas DataFrame.
# Q: Does the modality change the ids?
# A: No. It only selects which folder and extension are scanned; both give the same ids, which is why one split file
#    serves both input types.
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


# Q: What does this function do?
# A: It checks that the RGB folder and the MS folder contain exactly the same sample ids, and raises RuntimeError if
#    not.
# Q: Why is this check needed?
# A: One split.csv is used for both input types. If an image existed in only one version, the two models would not see
#    the same scenes and the split would not apply to both.
# Q: What are the inputs and what does it return?
# A: raw_dir (folder with both datasets). It returns None; the error message counts the ids found in only one folder
#    and shows up to 5 examples.
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

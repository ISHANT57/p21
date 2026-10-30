"""Build a deterministic sample index for EuroSAT.

Every sample is identified by a modality-independent id such as
"AnnualCrop/AnnualCrop_1". The RGB file is <id>.jpg in EuroSAT_RGB and the
13-band file is <id>.tif in EuroSAT_MS, so one split file serves both inputs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from eurosat_le.constants import CLASS_TO_IDX, CLASSES, DATASETS, RAW_DIR


def _scan(folder: Path, ext: str) -> list[str]:
    ids = []
    for cls in CLASSES:
        cls_dir = folder / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder {cls_dir}")
        ids += [f"{cls}/{p.stem}" for p in cls_dir.iterdir() if p.suffix.lower() == ext]
    return ids


def _sort_key(sample_id: str) -> tuple[int, int]:
    cls, stem = sample_id.split("/")
    return CLASS_TO_IDX[cls], int(stem.rsplit("_", 1)[1])


def build_index(raw_dir: Path = RAW_DIR, modality: str = "rgb") -> pd.DataFrame:
    """Return a DataFrame (id, class, label) sorted by class then file number."""
    spec = DATASETS[modality]
    ids = sorted(_scan(raw_dir / spec["folder"], spec["ext"]), key=_sort_key)
    df = pd.DataFrame({"id": ids})
    df["class"] = df["id"].str.split("/").str[0]
    df["label"] = df["class"].map(CLASS_TO_IDX).astype(int)
    return df


def check_modalities_match(raw_dir: Path = RAW_DIR) -> None:
    """Raise if the RGB and MS folders do not contain exactly the same sample ids."""
    rgb = set(build_index(raw_dir, "rgb")["id"])
    ms = set(build_index(raw_dir, "ms")["id"])
    if rgb != ms:
        raise RuntimeError(
            f"RGB/MS id mismatch: {len(rgb - ms)} only in RGB, {len(ms - rgb)} only in MS "
            f"(e.g. {sorted(rgb ^ ms)[:5]})"
        )

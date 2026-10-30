"""Tests for archive handling in the download module (no network)."""

import zipfile

import pytest

from eurosat_le.constants import CLASSES
from eurosat_le.data.download import _find_class_root, _safe_extract


def test_find_class_root_nested(tmp_path):
    root = tmp_path / "ds" / "images" / "tif"
    for c in CLASSES:
        (root / c).mkdir(parents=True)
    assert _find_class_root(tmp_path) == root


def test_find_class_root_missing_class(tmp_path):
    for c in CLASSES[:-1]:
        (tmp_path / c).mkdir()
    with pytest.raises(RuntimeError):
        _find_class_root(tmp_path)


def test_safe_extract_rejects_path_traversal(tmp_path):
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(RuntimeError):
        _safe_extract(zip_path, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()

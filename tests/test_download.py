"""Tests for archive handling in the download module (no network)."""

# zipfile: build a small fake zip archive for the safety test.
import zipfile

# pytest: check that the code raises the expected error.
import pytest

# CLASSES: the 10 EuroSAT class names (folder names the downloader looks for).
from eurosat_le.constants import CLASSES
# The two helpers under test: find the folder holding the class folders, and extract a zip safely.
from eurosat_le.data.download import _find_class_root, _safe_extract


def test_find_class_root_nested(tmp_path):
    """The class folders may sit several directories deep; the finder must return the folder containing them."""
    # A fake download layout: ds/images/tif/<one folder per class>. tmp_path is a throwaway temp folder.
    root = tmp_path / "ds" / "images" / "tif"
    for c in CLASSES:
        (root / c).mkdir(parents=True)
    # Searching from the top must locate the deep folder that holds all 10 class folders.
    assert _find_class_root(tmp_path) == root


def test_find_class_root_missing_class(tmp_path):
    """If one class folder is missing the archive is incomplete, so the finder must raise an error."""
    # Create every class folder except the last one.
    for c in CLASSES[:-1]:
        (tmp_path / c).mkdir()
    # No folder has all 10 classes, so a RuntimeError is expected (not a silent wrong result).
    with pytest.raises(RuntimeError):
        _find_class_root(tmp_path)


def test_safe_extract_rejects_path_traversal(tmp_path):
    """A malicious zip with a '../' path must be refused and must not write outside the target folder."""
    # Build a zip containing one file whose name tries to escape the extraction folder ("zip slip").
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../escape.txt", "x")
    # Extraction must refuse with a RuntimeError ...
    with pytest.raises(RuntimeError):
        _safe_extract(zip_path, tmp_path / "out")
    # ... and the escaping file must really not have been created next to the output folder.
    assert not (tmp_path / "escape.txt").exists()

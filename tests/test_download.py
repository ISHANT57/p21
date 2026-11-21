"""Tests for archive handling in the download module (no network)."""

# zipfile: build a small fake zip archive for the safety test.
import zipfile

# pytest: check that the code raises the expected error.
import pytest

# CLASSES: the 10 EuroSAT class names (folder names the downloader looks for).
from eurosat_le.constants import CLASSES
# The two helpers under test: find the folder holding the class folders, and extract a zip safely.
from eurosat_le.data.download import _find_class_root, _safe_extract


# ====================================================================================================
# CONCEPTS USED IN THIS FILE (study notes)
# ====================================================================================================
# - unit test: a small automatic check of one function on tiny made-up input. All three tests here are unit tests of
#     private helpers in data/download.py.
# - regression / safety test: test_safe_extract_rejects_path_traversal guards a security behaviour: if someone later
#     removes the path check, this test fails and warns us.
# - assert: a statement that must be true; if false, pytest marks the test as FAILED.
# - tmp_path (pytest fixture): a fixture is a helper that pytest builds and passes in as an argument. tmp_path is a
#     fresh, empty temporary folder (a pathlib.Path) for each test, deleted automatically later, so tests never
#     touch the real data/ folder.
# - pytest.raises: a context manager that passes only if the code inside raises the named error (here RuntimeError).
#     It tests that bad input fails loudly instead of being ignored.
# - fake / synthetic data: empty folders named like the 10 EuroSAT classes, and a one-file zip created in the test.
#     No real download and no network are used.
# - CLASSES: the list of the 10 EuroSAT class folder names (constants.py); the downloader looks for a folder that
#     contains all of them.
# - archive layout / nested folders: different mirrors wrap the class folders in different parent folders, so
#     _find_class_root searches the folder and its sub-folders (sorted order) for the right one.
# - zip slip / path traversal: a malicious zip can contain a member name like '../escape.txt' that, when extracted,
#     would write OUTSIDE the target folder. _safe_extract resolves every member path and refuses (RuntimeError) if
#     it is not inside the output folder; it checks all members before extracting anything.
# - Path / mkdir(parents=True): pathlib paths; mkdir(parents=True) also creates missing parent folders.

# ====================================================================================================
# Q&A ABOUT THIS FILE (viva practice)
# ====================================================================================================
# Q: Which part of the project does this file protect?
# A: The archive handling in eurosat_le/data/download.py: finding the folder with the class sub-folders after
#    extraction (_find_class_root) and extracting a zip safely (_safe_extract).
# Q: How do you run it?
# A: pytest tests/test_download.py from the repo root (with eurosat_le importable).
# Q: Why does it need no dataset and no internet?
# A: Everything is built inside tmp_path: empty class folders and a tiny zip. The functions under test only look at
#    folder names and zip member names, not at image content, and the real downloading functions are not called.

# Q: What does test_find_class_root_nested check?
# A: Even when the 10 class folders sit deep (ds/images/tif/<class>), _find_class_root(tmp_path) returns exactly
#    that deep folder.
# Q: Why does it matter?
# A: The archives wrap the class folders in differently named parent directories. fetch() renames the folder that
#    this function finds to data/raw/EuroSAT_RGB or EuroSAT_MS, so a wrong answer breaks the whole data layout.
# Q: How is the input built?
# A: In tmp_path, make ds/images/tif and create one empty folder per name in CLASSES inside it. The parent folders
#    ds and images do not contain the classes themselves, so only tif qualifies.
def test_find_class_root_nested(tmp_path):
    """The class folders may sit several directories deep; the finder must return the folder containing them."""
    # A fake download layout: ds/images/tif/<one folder per class>. tmp_path is a throwaway temp folder.
    root = tmp_path / "ds" / "images" / "tif"
    for c in CLASSES:
        (root / c).mkdir(parents=True)
    # Searching from the top must locate the deep folder that holds all 10 class folders.
    assert _find_class_root(tmp_path) == root


# Q: What does this test check?
# A: With only 9 of the 10 class folders present, _find_class_root raises RuntimeError.
# Q: Why does it matter?
# A: A broken or incomplete archive must stop the pipeline with a clear error rather than silently return a folder
#    with missing classes (which would give an incomplete dataset).
# Q: How is the input built?
# A: All class folders except the last one (CLASSES[:-1]) are created directly in tmp_path; no folder has all 10.
def test_find_class_root_missing_class(tmp_path):
    """If one class folder is missing the archive is incomplete, so the finder must raise an error."""
    # Create every class folder except the last one.
    for c in CLASSES[:-1]:
        (tmp_path / c).mkdir()
    # No folder has all 10 classes, so a RuntimeError is expected (not a silent wrong result).
    with pytest.raises(RuntimeError):
        _find_class_root(tmp_path)


# Q: What does this test check?
# A: A zip that contains a member named '../escape.txt' is refused with RuntimeError, and the file escape.txt is
#    really not created next to the output folder.
# Q: Why does it matter?
# A: This is the 'zip slip' attack: extracting such a member would write outside the target folder and could
#    overwrite important files. The download code must refuse it.
# Q: How is the malicious zip built?
# A: zipfile.ZipFile(..., 'w').writestr('../escape.txt', 'x') creates a one-file archive with the dangerous name.
#    The second assert makes sure nothing was written, not just that an error was raised.
# Q: Which function does it exercise?
# A: _safe_extract(zip_path, out_dir): it resolves each member path, checks is_relative_to(out_dir) for all of them,
#    and only then calls extractall.
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

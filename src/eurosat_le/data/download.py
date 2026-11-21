"""Download and extract EuroSAT (RGB and/or 13-band MS) with hash verification.

First step of the pipeline (entry point: scripts/download_data.py). The mirrors, sizes and
hashes for each input type are listed in eurosat_le.constants.DATASETS.

Design decisions:
    - Every archive is checked against its expected byte size and a fixed hash (SHA-256 for
      the Hugging Face mirror, MD5 for Zenodo) before it is extracted. Everyone therefore
      works on byte-identical data, and a corrupt or altered download is discarded.
    - Mirrors are tried in order; a network error or a bad hash falls through to the next one.
    - Downloads go to a ".part" file and are renamed only when complete.
    - Extraction refuses archive members that would land outside the target folder.
    - Extraction goes to a temporary folder first; the class folders are then renamed to
      data/raw/<EuroSAT_RGB | EuroSAT_MS>/<Class>/..., so the layout does not depend on the
      folder names inside each mirror's archive.
"""

# Allow modern type-hint syntax on older Python versions.
from __future__ import annotations

# hashlib: file fingerprints; shutil: delete/move folders and check disk space; sys: write to stderr;
# urllib.request: download over HTTP; zipfile: open and extract the archives; Path: file paths.
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# Class names (to recognise the right folder in an archive), mirror lists, and the default download folder.
from eurosat_le.constants import CLASSES, DATASETS, RAW_DIR

# ====================================================================================================
# CONCEPTS USED IN THIS FILE
# - EuroSAT RGB / MS: the dataset comes in two versions: EuroSAT_RGB (3-band JPEGs, ~95 MB zip) and EuroSAT_MS
#   (13-band GeoTIFFs, ~2 GB zip). fetch(name) takes 'rgb' or 'ms'.
# - Sentinel-2 bands / GeoTIFF / JPEG: the MS files are 13-band 16-bit GeoTIFFs from the Sentinel-2 satellite; the RGB
#   files are ordinary JPEGs. This file only downloads and unpacks them; it does not open the images.
# - Mirror: another web address serving the same file. fetch tries each mirror listed in constants.DATASETS in order
#   (Zenodo = official, Hugging Face = faster copy).
# - Checksum / hash (MD5, SHA-256): a fingerprint computed from all bytes of a file. If the download is corrupted or
#   altered the fingerprint differs. file_hash computes it; _verify compares it with the expected value.
# - hashlib / chunked reading: hashlib is Python's hashing library. The file is read in 1 MiB chunks so that even the
#   2 GB archive never has to fit in memory.
# - Size check: comparing the file's byte size with the expected size; it is cheap and catches truncated downloads
#   before the slow hash step.
# - .part file: while downloading, data goes to 'name.zip.part' and is renamed only when finished, so an interrupted
#   download is never mistaken for a complete file.
# - Streaming download / Content-Length: reading the HTTP response piece by piece and writing it to disk;
#   Content-Length is the total size the server announces, used only for the progress percentage.
# - Timeout: the download call gives up if the server is silent for 60 seconds (urlopen timeout=60).
# - Zip archive extraction safety ('zip slip'): a malicious zip can contain names like '../x' that write outside the
#   target folder. _safe_extract checks every member's resolved path is inside the target before extracting anything.
# - Temporary extraction folder: the zip is first unpacked to data/raw/_extract_<name>, the folder holding the 10
#   class folders is found and renamed to the standard place, and the rest is deleted.
# - Idempotent: running fetch twice has the same effect as running it once: if the target folder already exists and is
#   non-empty, nothing is downloaded.
# - for-else: a Python loop whose else part runs only if the loop finished without 'break'. Here it means every mirror
#   failed.
# ====================================================================================================
#
# FILE-LEVEL Q&A
# Q: What is this file for?
# A: It downloads the EuroSAT archives, checks that they are exactly the expected files, extracts them safely and puts
#    the class folders into data/raw/<EuroSAT_RGB or EuroSAT_MS>/<Class>/.
# Q: Where does it fit in the pipeline?
# A: It is the first step (entry point scripts/download_data.py). Afterwards data.index lists the images, data.splits
#    builds the split and data.cache packs the images.
# Q: What does it read and write?
# A: It reads the mirror list from constants.DATASETS and downloads from the internet. It writes the zip (and
#    optionally deletes it), a temporary _extract_<name> folder, and the final extracted folder under data/raw.
# Q: Why verify hashes?
# A: So that everyone works on byte-identical data; a truncated, corrupted or tampered archive is rejected and the
#    next mirror is tried.


# Q: What does this function do?
# A: It computes the hash (fingerprint) of a file, e.g. MD5 or SHA-256, and returns it as a hex string.
# Q: Why is the file read in chunks?
# A: The MS archive is about 2 GB. Reading 1 MiB at a time keeps memory use tiny; the final hash is the same as
#    hashing the whole file at once.
# Q: What are the inputs and what does it return?
# A: path (file), algo (a hashlib name such as 'md5' or 'sha256') and chunk (bytes per read). It returns the lowercase
#    hexadecimal digest string.
def file_hash(path: Path, algo: str, chunk: int = 1 << 20) -> str:
    """Hex digest of a file using hashlib algorithm `algo` ("md5", "sha256", ...), read in 1 MiB chunks.

    Args:
        path: file to hash.
        algo: hash algorithm name understood by hashlib.
        chunk: number of bytes read at a time (default 1 MiB).

    Returns:
        The hash as a lowercase hexadecimal string.
    """
    # Create an empty hash object for the chosen algorithm.
    h = hashlib.new(algo)
    # Open the file in binary mode.
    with open(path, "rb") as f:
        # Read 1 MiB at a time until the file ends, so a multi-GB file never has to fit in memory.
        while block := f.read(chunk):
            # Feed each block into the running hash.
            h.update(block)
    # Final fingerprint; if even one byte of the file differs, this string differs.
    return h.hexdigest()


# Q: What does this function do?
# A: It downloads one URL to a local file, streaming it to disk and printing progress about every 5%.
# Q: Why write to a .part file and rename at the end?
# A: If the download is interrupted, no file with the final name exists, so a half-downloaded zip is never mistaken
#    for a complete one.
# Q: What are the inputs and what does it return?
# A: url (where to download) and dest (final file path). It returns nothing; the result is the file at dest.
# Q: Why does the progress code check `total`?
# A: Some servers do not send Content-Length. Then total is 0 and no progress lines are printed, but the download
#    still works.
def _download(url: str, dest: Path) -> None:
    """Stream `url` to `dest`, printing progress about every 5%.

    Writes to "<dest>.part" and renames on success, so `dest` only exists for a finished download.

    Args:
        url: address of the file to download.
        dest: where the finished file should be saved.
    """
    # Temporary name ("file.zip.part") used while the download is in progress.
    tmp = dest.with_suffix(dest.suffix + ".part")
    # Tell the user what is being downloaded and where it goes (flush=True shows it immediately).
    print(f"Downloading {url}\n  -> {dest}", flush=True)
    # Open the connection (give up if the server is silent for 60 s) and the local output file.
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
        # Total size announced by the server (0 if it does not say), used only for the progress display.
        total = int(resp.headers.get("Content-Length", 0))
        # Bytes received so far, and the next progress fraction at which a line will be printed.
        done, next_report = 0, 0.0
        # Receive the data in 1 MiB blocks so a big file never has to fit in memory.
        while block := resp.read(1 << 20):
            # Write the block to disk.
            out.write(block)
            # Update the byte counter.
            done += len(block)
            if total and done / total >= next_report:
                # Print the percentage done and the total size in MB.
                print(f"  {done / total:6.1%} of {total / 1e6:.0f} MB", flush=True)
                # Wait for another 5% before printing again.
                next_report += 0.05
    # Download finished without error: give the file its real name.
    tmp.rename(dest)


# Q: What does this function do?
# A: It checks that a downloaded file has the exact expected byte size and the expected hash, and raises RuntimeError
#    if not.
# Q: Why is the size checked before the hash?
# A: The size check is instant and catches truncated downloads; hashing a 2 GB file takes several seconds, so it is
#    only done when the size is right.
# Q: What are the inputs and what does it return?
# A: path (the downloaded file) and mirror (a dict from constants.DATASETS with 'size' and either 'sha256' or 'md5').
#    It returns None; success means no exception.
# Q: Which hash is used?
# A: SHA-256 if the mirror entry has one (Hugging Face), otherwise MD5 (Zenodo).
def _verify(path: Path, mirror: dict) -> None:
    """Raise RuntimeError unless `path` has the byte size and hash listed in the `mirror` spec.

    The cheap size check runs first to catch truncated downloads before hashing the whole file.
    The mirror spec holds either a "sha256" or an "md5" entry; sha256 is preferred when present.

    Args:
        path: the downloaded file.
        mirror: one entry of DATASETS[...]["mirrors"] with the expected "size" and hash.
    """
    # Actual size of the file on disk, in bytes.
    size = path.stat().st_size
    if size != mirror["size"]:
        # Wrong size means the download was cut short or the wrong file was served.
        raise RuntimeError(f"incomplete download {path}: {size} bytes, expected {mirror['size']}")
    # Choose which hash the spec provides.
    algo = "sha256" if "sha256" in mirror else "md5"
    print(f"  verifying {algo} ...", flush=True)
    # Compute the real hash of the file (can take a few seconds for the 2 GB archive).
    digest = file_hash(path, algo)
    # A different hash means the content was damaged or altered.
    if digest != mirror[algo]:
        raise RuntimeError(f"{algo} mismatch for {path}: got {digest}, expected {mirror[algo]}")


# Q: What does this function do?
# A: It extracts a zip file into a folder, but first checks that no member would be written outside that folder.
# Q: What would go wrong without the check?
# A: A malicious or damaged archive could contain a name like '../../something' (a 'zip slip') and overwrite files
#    elsewhere on the computer.
# Q: What are the inputs and what does it return?
# A: zip_path (the archive) and out_dir (target folder). It returns nothing; it raises RuntimeError on an unsafe
#    member name.
# Q: Why check all members before extracting any?
# A: So that an unsafe archive is rejected before a single file has been written.
def _safe_extract(zip_path: Path, out_dir: Path) -> None:
    """Extract a zip, refusing members that would land outside out_dir.

    Args:
        zip_path: the archive to extract.
        out_dir: folder to extract into (created if missing).
    """
    # Absolute, symlink-free path, so the "is it inside?" test below is reliable.
    out_dir = out_dir.resolve()
    # Guard against "zip slip": names like "../x" or absolute paths that would write outside out_dir.
    # All members are checked before anything is extracted.
    with zipfile.ZipFile(zip_path) as zf:
        # Go through every file listed in the archive.
        for member in zf.infolist():
            # Where this member would end up on disk.
            target = (out_dir / member.filename).resolve()
            # If that is not inside out_dir, the archive is trying to write somewhere it should not.
            if not target.is_relative_to(out_dir):
                raise RuntimeError(f"Unsafe path in archive: {member.filename}")
        # All members passed the check, so extract everything.
        zf.extractall(out_dir)


# Q: What does this function do?
# A: It finds the folder inside the extracted archive that directly contains the 10 class folders (AnnualCrop, Forest,
#    ...).
# Q: Why is it needed?
# A: The two mirrors wrap the class folders in differently named parent directories. Searching for the folder that has
#    all 10 class names makes the final layout independent of those names.
# Q: What are the inputs and what does it return?
# A: root (the extraction folder). It returns the Path of the matching folder, or raises RuntimeError if none is
#    found.
# Q: Why are the sub-folders sorted?
# A: So that if more than one candidate exists the result is the same on every machine.
def _find_class_root(root: Path) -> Path:
    """Return the directory under `root` that directly contains all class folders.

    The archives wrap the class folders in differently named parent directories, so search
    `root` and its sub-directories (in sorted order) for one that has all 10 CLASSES.

    Args:
        root: the folder the archive was extracted into.

    Returns:
        The folder whose children are the 10 class folders.
    """
    # Candidates: root itself, then every folder below it in a fixed (sorted) order.
    for d in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        # Accept the first candidate that has a sub-folder for every class name.
        if all((d / c).is_dir() for c in CLASSES):
            return d
    # Nothing matched: the archive does not have the expected content.
    raise RuntimeError(f"No directory with all {len(CLASSES)} class folders found under {root}")


# Q: What does this function do?
# A: It makes sure one dataset ('rgb' or 'ms') is downloaded, verified and extracted to data/raw/<folder>, and returns
#    that folder.
# Q: What happens if a download fails or is corrupt?
# A: The error is printed, the bad zip is deleted and the next mirror is tried. Only if every mirror fails does it
#    raise RuntimeError.
# Q: What are the inputs and what does it return?
# A: name ('rgb'/'ms'), raw_dir (where to put things), keep_zip (keep or delete the zip afterwards), mirror (restrict
#    to one mirror, or None for all). It returns the Path of the folder that holds the class sub-directories.
# Q: Why does it skip work when the folder exists?
# A: So the script can be re-run safely and quickly (idempotent). Note that an existing folder is not re-verified.
def fetch(name: str, raw_dir: Path = RAW_DIR, keep_zip: bool = True, mirror: str | None = None) -> Path:
    """Ensure dataset `name` ("rgb" or "ms") is downloaded, verified and extracted.

    Tries each mirror in order (or only `mirror` if given). Returns the folder
    holding the class sub-directories.

    Idempotent: if the target folder already exists and is non-empty, nothing is downloaded and
    it is NOT re-verified. A zip left in raw_dir is re-verified rather than re-downloaded; a zip
    that fails verification is deleted before the next mirror is tried. If `keep_zip` is False
    the verified zip is removed after extraction. Raises RuntimeError if every mirror fails.

    Args:
        name: "rgb" or "ms".
        raw_dir: folder where archives and extracted data are placed.
        keep_zip: keep the downloaded zip after extraction (True) or delete it (False).
        mirror: restrict to one mirror by name ("zenodo" or "hf"); None means try all in order.
    """
    # Description of this dataset (folder name, mirrors, ...) from constants.
    spec = DATASETS[name]
    # Create the download folder if it does not exist yet.
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Final location of the extracted images.
    folder = raw_dir / spec["folder"]
    if folder.is_dir() and any(folder.iterdir()):
        # Already extracted earlier: nothing to do.
        print(f"[{name}] already extracted at {folder}")
        return folder

    # Mirrors to try: all of them, or only the one the caller asked for.
    mirrors = [m for m in spec["mirrors"] if mirror in (None, m["name"])]
    if not mirrors:
        raise ValueError(f"Unknown mirror {mirror!r} for {name}")
    for m in mirrors:
        # Local file name of this mirror's archive.
        zip_path = raw_dir / m["filename"]
        try:
            # Download only if we do not already have the zip (e.g. from an interrupted earlier run).
            if not zip_path.exists():
                _download(m["url"], zip_path)
            # Check size and hash; raises if the zip is incomplete or altered.
            _verify(zip_path, m)
            # Good archive: stop trying mirrors (this also skips the for-else error below).
            break
        except Exception as e:  # network error or bad hash: try the next mirror
            # Report the failure on stderr but keep going.
            print(f"[{name}] mirror {m['name']} failed: {e}", file=sys.stderr, flush=True)
            # Delete the bad/partial zip so it is not trusted or reused.
            zip_path.unlink(missing_ok=True)
    else:
        # The loop ended without `break`: every mirror failed.
        raise RuntimeError(f"All mirrors failed for {name}")

    print(f"[{name}] extracting ...", flush=True)
    # Temporary folder for extraction; any leftover from a previous failed run is removed first.
    tmp_dir = raw_dir / f"_extract_{name}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    # Unzip safely into the temporary folder.
    _safe_extract(zip_path, tmp_dir)
    # Find the folder holding the class folders and move/rename it to the standard location.
    _find_class_root(tmp_dir).rename(folder)
    # Remove what is left of the temporary extraction folder.
    shutil.rmtree(tmp_dir)
    if not keep_zip:
        # Free disk space by deleting the verified zip.
        zip_path.unlink()
    print(f"[{name}] ready at {folder}", flush=True)
    return folder


# Q: What does this function do?
# A: It returns the free disk space, in decimal gigabytes, of the drive that holds a given path.
# Q: Why is it needed?
# A: The MS download needs about 2.1 GB for the zip plus about 2.9 GB extracted; scripts/download_data.py prints the
#    free space so a full disk is noticed before downloading.
# Q: What are the inputs and what does it return?
# A: path (any existing file or folder on that drive). It returns a float number of GB.
def disk_free_gb(path: Path) -> float:
    """Free disk space (decimal GB) on the volume holding `path`.

    The MS version needs about 2.1 GB for the zip plus about 2.9 GB once extracted.

    Args:
        path: any existing file or folder on the volume to check.
    """
    # disk_usage returns (total, used, free) in bytes; divide free by 1e9 to get GB.
    return shutil.disk_usage(path).free / 1e9

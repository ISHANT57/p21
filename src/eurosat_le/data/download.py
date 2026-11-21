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


def disk_free_gb(path: Path) -> float:
    """Free disk space (decimal GB) on the volume holding `path`.

    The MS version needs about 2.1 GB for the zip plus about 2.9 GB once extracted.

    Args:
        path: any existing file or folder on the volume to check.
    """
    # disk_usage returns (total, used, free) in bytes; divide free by 1e9 to get GB.
    return shutil.disk_usage(path).free / 1e9

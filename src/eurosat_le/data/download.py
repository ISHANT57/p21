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

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from eurosat_le.constants import CLASSES, DATASETS, RAW_DIR


def file_hash(path: Path, algo: str, chunk: int = 1 << 20) -> str:
    """Hex digest of a file using hashlib algorithm `algo` ("md5", "sha256", ...), read in 1 MiB chunks."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Stream `url` to `dest`, printing progress about every 5%.

    Writes to "<dest>.part" and renames on success, so `dest` only exists for a finished download.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}\n  -> {dest}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done, next_report = 0, 0.0
        while block := resp.read(1 << 20):
            out.write(block)
            done += len(block)
            if total and done / total >= next_report:
                print(f"  {done / total:6.1%} of {total / 1e6:.0f} MB", flush=True)
                next_report += 0.05
    tmp.rename(dest)


def _verify(path: Path, mirror: dict) -> None:
    """Raise RuntimeError unless `path` has the byte size and hash listed in the `mirror` spec.

    The cheap size check runs first to catch truncated downloads before hashing the whole file.
    The mirror spec holds either a "sha256" or an "md5" entry; sha256 is preferred when present.
    """
    size = path.stat().st_size
    if size != mirror["size"]:
        raise RuntimeError(f"incomplete download {path}: {size} bytes, expected {mirror['size']}")
    algo = "sha256" if "sha256" in mirror else "md5"
    print(f"  verifying {algo} ...", flush=True)
    digest = file_hash(path, algo)
    if digest != mirror[algo]:
        raise RuntimeError(f"{algo} mismatch for {path}: got {digest}, expected {mirror[algo]}")


def _safe_extract(zip_path: Path, out_dir: Path) -> None:
    """Extract a zip, refusing members that would land outside out_dir."""
    out_dir = out_dir.resolve()
    # Guard against "zip slip": names like "../x" or absolute paths that would write outside out_dir.
    # All members are checked before anything is extracted.
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (out_dir / member.filename).resolve()
            if not target.is_relative_to(out_dir):
                raise RuntimeError(f"Unsafe path in archive: {member.filename}")
        zf.extractall(out_dir)


def _find_class_root(root: Path) -> Path:
    """Return the directory under `root` that directly contains all class folders.

    The archives wrap the class folders in differently named parent directories, so search
    `root` and its sub-directories (in sorted order) for one that has all 10 CLASSES.
    """
    for d in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        if all((d / c).is_dir() for c in CLASSES):
            return d
    raise RuntimeError(f"No directory with all {len(CLASSES)} class folders found under {root}")


def fetch(name: str, raw_dir: Path = RAW_DIR, keep_zip: bool = True, mirror: str | None = None) -> Path:
    """Ensure dataset `name` ("rgb" or "ms") is downloaded, verified and extracted.

    Tries each mirror in order (or only `mirror` if given). Returns the folder
    holding the class sub-directories.

    Idempotent: if the target folder already exists and is non-empty, nothing is downloaded and
    it is NOT re-verified. A zip left in raw_dir is re-verified rather than re-downloaded; a zip
    that fails verification is deleted before the next mirror is tried. If `keep_zip` is False
    the verified zip is removed after extraction. Raises RuntimeError if every mirror fails.
    """
    spec = DATASETS[name]
    raw_dir.mkdir(parents=True, exist_ok=True)
    folder = raw_dir / spec["folder"]
    if folder.is_dir() and any(folder.iterdir()):
        print(f"[{name}] already extracted at {folder}")
        return folder

    mirrors = [m for m in spec["mirrors"] if mirror in (None, m["name"])]
    if not mirrors:
        raise ValueError(f"Unknown mirror {mirror!r} for {name}")
    for m in mirrors:
        zip_path = raw_dir / m["filename"]
        try:
            if not zip_path.exists():
                _download(m["url"], zip_path)
            _verify(zip_path, m)
            break
        except Exception as e:  # network error or bad hash: try the next mirror
            print(f"[{name}] mirror {m['name']} failed: {e}", file=sys.stderr, flush=True)
            zip_path.unlink(missing_ok=True)
    else:
        raise RuntimeError(f"All mirrors failed for {name}")

    print(f"[{name}] extracting ...", flush=True)
    tmp_dir = raw_dir / f"_extract_{name}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    _safe_extract(zip_path, tmp_dir)
    _find_class_root(tmp_dir).rename(folder)
    shutil.rmtree(tmp_dir)
    if not keep_zip:
        zip_path.unlink()
    print(f"[{name}] ready at {folder}", flush=True)
    return folder


def disk_free_gb(path: Path) -> float:
    """Free disk space (decimal GB) on the volume holding `path`.

    The MS version needs about 2.1 GB for the zip plus about 2.9 GB once extracted.
    """
    return shutil.disk_usage(path).free / 1e9

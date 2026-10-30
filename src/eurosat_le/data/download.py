"""Download and extract EuroSAT (RGB and/or 13-band MS) with hash verification."""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from eurosat_le.constants import CLASSES, DATASETS, RAW_DIR


def file_hash(path: Path, algo: str, chunk: int = 1 << 20) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
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
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (out_dir / member.filename).resolve()
            if not target.is_relative_to(out_dir):
                raise RuntimeError(f"Unsafe path in archive: {member.filename}")
        zf.extractall(out_dir)


def _find_class_root(root: Path) -> Path:
    """Return the directory under `root` that directly contains all class folders."""
    for d in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        if all((d / c).is_dir() for c in CLASSES):
            return d
    raise RuntimeError(f"No directory with all {len(CLASSES)} class folders found under {root}")


def fetch(name: str, raw_dir: Path = RAW_DIR, keep_zip: bool = True, mirror: str | None = None) -> Path:
    """Ensure dataset `name` ("rgb" or "ms") is downloaded, verified and extracted.

    Tries each mirror in order (or only `mirror` if given). Returns the folder
    holding the class sub-directories.
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
    return shutil.disk_usage(path).free / 1e9

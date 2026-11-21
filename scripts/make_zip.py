"""Pack the project into a zip that works on any OS (use this on Windows; no bash needed).

Usage:
    python scripts/make_zip.py                      # dist/p21_project.zip, with git history, no data
    python scripts/make_zip.py --no-git             # code only
    python scripts/make_zip.py --with-cache         # also data/cache/*.npy (about 3.2 GB)
    python scripts/make_zip.py --out D:/p21.zip

Never included: .venv, data/raw, dist, __pycache__, results/runs (git-ignored outputs).
Uncommitted changes in the working tree ARE included, as they are on disk.

Where it fits: a packaging helper, not part of the ML pipeline. It exists so the project can be handed
over or uploaded (for example to Colab) from Windows without bash or zip installed; only the Python
standard library is used.

Design decisions:
- The archive has one top-level folder (--prefix, default p21), so unzipping gives a single clean folder.
- Files are collected with a plain walk of the working tree (not "git archive"), which is why
  uncommitted edits are included. The .git folder is added separately unless --no-git is given.
- The data folder is skipped as a whole; only data/cache/* is added back, and only with --with-cache.
- Model weights and bytecode (.pt, .pth, .ckpt, .pyc) are never included, wherever they are.
- Paths inside the zip use forward slashes (as_posix) so the archive unpacks the same on every OS.
"""

import argparse
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # scripts/ -> repository root
SKIP_DIRS = {".venv", "dist", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".mypy_cache"}
SKIP_TOP = {"data", ".git"}  # handled explicitly below


def skip(rel: Path) -> bool:
    """True if the repo-relative path `rel` must stay out of the zip.

    Skipped: any path under a SKIP_DIRS folder or an *.egg-info folder, anything under results/runs, and
    bytecode or checkpoint files by extension. (data/ and .git/ are handled by the caller, see `files`.)
    """
    parts = rel.parts
    if any(p in SKIP_DIRS or p.endswith(".egg-info") for p in parts):
        return True
    return parts[:2] == ("results", "runs") or rel.suffix in {".pyc", ".pt", ".pth", ".ckpt"}


def files(args):
    """Yield (absolute path, repo-relative path) for every file to archive, in sorted (stable) order.

    Order: working tree first, then .git (unless --no-git), then data/cache (only with --with-cache).
    """
    for path in sorted(REPO.rglob("*")):
        rel = path.relative_to(REPO)
        if not path.is_file() or rel.parts[0] in SKIP_TOP or skip(rel):
            continue
        yield path, rel
    if not args.no_git:
        for path in sorted((REPO / ".git").rglob("*")):
            if path.is_file():
                yield path, path.relative_to(REPO)
    if args.with_cache:
        for path in sorted((REPO / "data" / "cache").glob("*")):
            yield path, path.relative_to(REPO)


def main() -> None:
    """Parse the options, write the zip (deflate, level 6) and print its path, size and file count."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=REPO / "dist" / "p21_project.zip")
    p.add_argument("--no-git", action="store_true", help="leave out the .git history")
    p.add_argument("--with-cache", action="store_true", help="include data/cache/*.npy (about 3.2 GB)")
    p.add_argument("--prefix", default="p21", help="top-level folder inside the zip")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path, rel in files(args):
            zf.write(path, f"{args.prefix}/{rel.as_posix()}")
            n += 1
    print(f"{args.out}  ({args.out.stat().st_size / 1e6:.1f} MB, {n} files)")


if __name__ == "__main__":
    main()

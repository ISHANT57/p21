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

# argparse reads the command-line options; zipfile writes the archive (standard library, so no zip tool is needed).
import argparse
import zipfile
from pathlib import Path

# ==================== CONCEPTS USED IN THIS FILE ====================
# - argparse: standard-library module that reads command-line options (--out, --no-git, --with-cache, --prefix) into an
#   object; used in main().
# - zip archive / zipfile: a single compressed file that holds many files. The standard-library zipfile module writes
#   it, so no external zip program is needed (important on Windows).
# - deflate compression, level 6: the usual zip compression method; level 6 is a middle setting between speed (1) and
#   smallest size (9).
# - generator (yield): files() does not build a list; it hands out one (absolute path, relative path) pair at a time, so
#   the zip is written file by file.
# - pathlib.Path, rglob, parts, suffix: Path objects for file paths; rglob('*') lists everything below a folder
#   recursively; .parts splits a path into folder names; .suffix is the file extension.
# - repo-relative path: the path counted from the repository root (for example scripts/train.py); it is the name the
#   file gets inside the zip.
# - as_posix(): writes a path with forward slashes, so the archive unpacks identically on Windows, Linux and macOS.
# - top-level prefix: every file is stored under '<prefix>/', default p21, so unzipping creates one clean folder.
# - git history (.git) / git-ignored outputs: the .git folder holds the commit history and is included unless --no-git;
#   results/runs, data/raw and .venv are not tracked by git and are never packed.
# - cache (data/cache): the packed .npy arrays of all images (about 3.2 GB); only added back with --with-cache.
# - model weights / bytecode: .pt, .pth, .ckpt (saved model weights) and .pyc (compiled Python) are skipped everywhere
#   because they are large or regenerated.
# - __file__ / parents[1]: REPO is found from the location of this script: parents[0] is the scripts/ folder and
#   parents[1] is the repository root above it.
# - if __name__ == '__main__': runs main() only when executed as a script, not when imported.
# ====================================================================
#
# ---- Q&A: THIS FILE ----
# Q: What is this file for?
# A: It is a packaging helper, not part of the machine-learning pipeline: it makes one zip of the project so it can be
#    handed over or uploaded (for example to Colab) from Windows without bash or zip installed.
# Q: What does it read and write?
# A: It reads the files of the repository on disk (including uncommitted edits) and writes one zip, by default
#    dist/p21_project.zip.
# Q: How is it run?
# A: python scripts/make_zip.py [--no-git] [--with-cache] [--out PATH] [--prefix NAME].

REPO = Path(__file__).resolve().parents[1]  # scripts/ -> repository root
# Folder names that are never packed, wherever they appear in the path (environments, caches, build output).
SKIP_DIRS = {".venv", "dist", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".mypy_cache"}
SKIP_TOP = {"data", ".git"}  # handled explicitly below


# ---- Q&A: skip ----
# Q: What does skip() do?
# A: It answers one question: must this repo-relative path be left out of the zip? It returns True for anything inside
#    an ignored folder (.venv, dist, __pycache__, ...), inside an *.egg-info folder, under results/runs, or with the
#    extension .pyc, .pt, .pth or .ckpt.
# Q: Why are these files excluded?
# A: They are either generated (caches, bytecode, training outputs) or very large (model weights, virtual environment),
#    so including them would make the zip huge and useless to the receiver.
# Q: What are the input and the return value?
# A: Input: a Path relative to the repository root. Returns a bool (True = skip).
def skip(rel: Path) -> bool:
    """True if the repo-relative path `rel` must stay out of the zip.

    Skipped: any path under a SKIP_DIRS folder or an *.egg-info folder, anything under results/runs, and
    bytecode or checkpoint files by extension. (data/ and .git/ are handled by the caller, see `files`.)
    """
    # Split the path into its folder/file parts, e.g. ("results", "runs", "x", "y.pt").
    parts = rel.parts
    # Skip anything that lives inside an ignored folder or a Python package metadata (*.egg-info) folder.
    if any(p in SKIP_DIRS or p.endswith(".egg-info") for p in parts):
        return True
    # Also skip training outputs (results/runs) and compiled/model-weight files, whatever folder they are in.
    return parts[:2] == ("results", "runs") or rel.suffix in {".pyc", ".pt", ".pth", ".ckpt"}


# ---- Q&A: files ----
# Q: What does files() do?
# A: It yields, one by one, every file that goes into the zip as a pair (absolute path, repo-relative path), in sorted
#    order: first the working tree, then .git (unless --no-git), then data/cache (only with --with-cache).
# Q: Why sorted order?
# A: So the zip lists files in the same order every time it is built, which makes the archive reproducible and easier to
#    compare.
# Q: Why are data/ and .git/ handled separately?
# A: data/ is huge and should be skipped as a whole except its cache, and .git must be optional. The main loop skips
#    both (SKIP_TOP) and the code below adds them back only when requested.
# Q: What are the input and the return value?
# A: Input: the parsed args (uses args.no_git and args.with_cache). Output: a generator of (Path, Path) pairs, not a
#    list.
def files(args):
    """Yield (absolute path, repo-relative path) for every file to archive, in sorted (stable) order.

    Order: working tree first, then .git (unless --no-git), then data/cache (only with --with-cache).
    """
    # Walk the whole working tree in sorted order so the zip content order is the same on every run.
    for path in sorted(REPO.rglob("*")):
        # Path relative to the repository root, used both for the filters and as the name inside the zip.
        rel = path.relative_to(REPO)
        # Ignore folders themselves, the top-level data/ and .git/ folders (added separately), and skipped files.
        if not path.is_file() or rel.parts[0] in SKIP_TOP or skip(rel):
            continue
        yield path, rel
    # Add the git history, since the main walk skipped .git, unless the user asked for code only.
    if not args.no_git:
        for path in sorted((REPO / ".git").rglob("*")):
            if path.is_file():
                yield path, path.relative_to(REPO)
    # Add back only the cached numpy arrays from data/, and only on request because they are about 3 GB.
    if args.with_cache:
        for path in sorted((REPO / "data" / "cache").glob("*")):
            yield path, path.relative_to(REPO)


# ---- Q&A: main ----
# Q: What does main() do?
# A: It parses the options, creates the output folder, opens the zip with deflate compression and writes every file from
#    files() under '<prefix>/<relative path>', then prints the zip path, its size in MB and the number of files.
# Q: Why use a plain walk of the working tree instead of 'git archive'?
# A: git archive only packs committed files; the walk includes uncommitted edits exactly as they are on disk, and it
#    needs no git program for the code part.
# Q: Why as_posix() when writing the names?
# A: So the archive uses forward slashes and unpacks correctly on every operating system.
# Q: What are the inputs and what does it return?
# A: Command-line options only; it returns None and its output is the zip file.
def main() -> None:
    """Parse the options, write the zip (deflate, level 6) and print its path, size and file count."""
    # Build the argument parser; the module docstring doubles as the --help text.
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Where to write the zip file.
    p.add_argument("--out", type=Path, default=REPO / "dist" / "p21_project.zip")
    # Flag: leave the .git history out of the archive.
    p.add_argument("--no-git", action="store_true", help="leave out the .git history")
    # Flag: also pack the (large) cached arrays from data/cache.
    p.add_argument("--with-cache", action="store_true", help="include data/cache/*.npy (about 3.2 GB)")
    # Name of the single top-level folder inside the zip.
    p.add_argument("--prefix", default="p21", help="top-level folder inside the zip")
    # Parse sys.argv into the args object.
    args = p.parse_args()

    # Make sure the output folder exists before the zip file is created in it.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Counter of files written, shown in the final message.
    n = 0
    # Open the zip for writing with deflate compression (level 6 = a balance of speed and size).
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Add each file under "<prefix>/<relative path>"; as_posix gives forward slashes so it unzips on any OS.
        for path, rel in files(args):
            zf.write(path, f"{args.prefix}/{rel.as_posix()}")
            n += 1
    # Report where the zip is, how big it is (in MB) and how many files it holds.
    print(f"{args.out}  ({args.out.stat().st_size / 1e6:.1f} MB, {n} files)")


# Run main() only when the file is executed as a script, not when it is imported.
if __name__ == "__main__":
    main()

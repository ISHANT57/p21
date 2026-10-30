# Working on Windows

Tested layout: Windows 10/11, PowerShell. Use **Python 3.12** (3.13+ may not have every wheel yet). Install it from python.org and tick "Add python.exe to PATH", or use `py -3.12`.

## 1. Unpack and open

Unzip `p21_project.zip`. It creates a folder `p21` that holds the code **and the git history** (`.git`). Open that folder in PowerShell:

```powershell
cd C:\path\to\p21
git status          # should say: On branch main, nothing to commit
git log --oneline   # shows the project history
```

If Git says the folder has "dubious ownership", run `git config --global --add safe.directory C:/path/to/p21`.

## 2. Create the environment

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
# If PowerShell blocks the script:  Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
```

Install PyTorch **first**. Pick one:

```powershell
# CPU only (always works)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# NVIDIA GPU: take the exact command for your CUDA version from https://pytorch.org/get-started/locally/
```

Then the rest:

```powershell
pip install -r requirements.txt -e .
pytest -q
```

Without the dataset, 6 tests are skipped (they need `data/cache`). That is expected.

## 3. Get the data (about 2.2 GB download, 3.2 GB for the packed cache)

```powershell
python scripts/download_data.py --which all --delete-zip
python scripts/make_splits.py --check      # must print "Splits OK"; never run it without --check
python scripts/build_cache.py --which all
```

Or copy `data\cache` from the Linux laptop (`python scripts/make_zip.py --with-cache` there) and skip the download. The splits are committed, so both machines use the same 70/15/15 split.

## 4. Commands that differ from the README

| README (Linux) | Windows PowerShell |
|---|---|
| `source .venv/bin/activate` | `.venv\Scripts\Activate.ps1` |
| `python3 ...` | `python ...` |
| `bash scripts/make_code_zip.sh` | `python scripts/make_zip.py` |
| `\` at the end of a line | a backtick `` ` `` at the end of a line, or one long line |

All other commands (`python scripts/train.py ...`, `run_grid.py`, `evaluate.py`, `pytest`) are the same.

## 5. Things to know

- **Data loader workers.** Windows starts worker processes differently and is slower at it. If a run hangs or is slow to start, add `--set num_workers=0` (for `train.py`) or `num_workers=0` to `--set` in `run_grid.py`.
- **GPU training on Windows** works with the CUDA build of PyTorch. For the 54-run grid, Colab or Kaggle is still the plan unless your machine has a good NVIDIA GPU.
- **Line endings.** `.gitattributes` keeps every file in LF form. Do not convert files to CRLF in your editor. In VS Code set `"files.eol": "\n"`.
- **Long paths.** Keep the project near the drive root (for example `C:\p21`). Windows limits paths to 260 characters by default.
- **Antivirus / OneDrive.** Do not put the project, and especially `data\cache`, in a OneDrive-synced folder. Syncing multi-GB files slows everything down.
- **Git identity.** Set it once: `git config --global user.name "..."` and `git config --global user.email "..."`.

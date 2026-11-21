"""Regression test: `scripts/evaluate.py` must write the same tables before and after a code change.

The test builds small fake runs (same helper as tests/test_evaluate.py), runs the script with a fixed
`--n-boot`, and compares `groups.csv` and `compare_rgb_vs_ms.csv` with the files saved in tests/data/.
Floats are compared with an absolute tolerance of 1e-9; all other columns must be equal.

Refreshing the expected files ON PURPOSE (only when a result is meant to change, and say why in the PR):

    UPDATE_EXPECTED=1 pytest tests/test_evaluate_regression.py

then review `git diff tests/data/` and commit the new files.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from test_evaluate import make_run  # reuse the fake-run builder, so both test files make runs the same way

from eurosat_le.constants import REPO_ROOT

DATA = Path(__file__).parent / "data"  # expected tables, committed with the test
FILES = ["groups.csv", "compare_rgb_vs_ms.csv"]  # main results table + one paired comparison (uses the bootstrap)
N_BOOT = 50  # few resamples keep the test fast; the bootstrap seed is fixed, so the interval is reproducible


@pytest.fixture(scope="module")
def eval_out(tmp_path_factory):
    """Run scripts/evaluate.py once on fake runs and return its output folder (shared by all tests here)."""
    root = tmp_path_factory.mktemp("regression")
    runs = root / "runs"
    # Two seeds per setting, so groups.csv has a std and the RGB-vs-13-band comparison has 2 pairs.
    # make_run seeds its random generator from the run name, so the fake predictions are always the same.
    for seed in (0, 1):
        make_run(runs, "resnet50", "rgb", 0.01, seed, 0.80)
        make_run(runs, "resnet50", "ms", 0.01, seed, 0.88)
    out = root / "eval"
    # Run the real script in a separate process, exactly as a user would from the command line.
    res = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "evaluate.py"), "--runs-dir", str(runs),
                          "--out", str(out), "--n-boot", str(N_BOOT)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr  # show the script's error message if it crashed
    return out


@pytest.mark.parametrize("name", FILES)
def test_evaluation_tables_unchanged(eval_out, name):
    """The new table must equal the saved one (floats within 1e-9, everything else exactly)."""
    new = pd.read_csv(eval_out / name)
    if os.environ.get("UPDATE_EXPECTED"):
        # Refresh mode: overwrite the expected file with the current output (12 significant digits is
        # far below the 1e-9 tolerance, and keeps the files small and readable in a diff).
        DATA.mkdir(exist_ok=True)
        new.to_csv(DATA / name, index=False, float_format="%.12g")
    expected = pd.read_csv(DATA / name)
    # rtol=0 + atol=1e-9: an absolute tolerance only, so small and large values are held to the same bar.
    pd.testing.assert_frame_equal(new, expected, check_exact=False, rtol=0, atol=1e-9)

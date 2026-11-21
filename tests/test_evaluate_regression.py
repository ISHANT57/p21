"""Regression test: `scripts/evaluate.py` must write the same tables before and after a code change.

The test builds small fake runs (same helper as tests/test_evaluate.py), runs the script with a fixed
`--n-boot`, and compares `groups.csv` and `compare_rgb_vs_ms.csv` with the files saved in tests/data/.
Floats are compared with an absolute tolerance of 1e-9; all other columns must be equal.

Refreshing the expected files ON PURPOSE (only when a result is meant to change, and say why in the PR):

    UPDATE_EXPECTED=1 pytest tests/test_evaluate_regression.py

then review `git diff tests/data/` and commit the new files.
"""

# os: read the UPDATE_EXPECTED switch; subprocess/sys: run the evaluate script as its own process.
import os
import subprocess
import sys
# Path: locate the folder with the saved expected tables.
from pathlib import Path

# pandas: read the CSV tables and compare them; pytest: fixtures and parametrisation.
import pandas as pd
import pytest
from test_evaluate import make_run  # reuse the fake-run builder, so both test files make runs the same way

# REPO_ROOT: to find scripts/evaluate.py.
from eurosat_le.constants import REPO_ROOT

# ====================================================================================================
# CONCEPTS USED IN THIS FILE (study notes)
# ====================================================================================================
# - regression test: a test that checks that the output of a program has NOT changed after a code change
#     (refactoring, speed-up). It does not prove the numbers are right, only that they are the same as before.
# - golden (expected) file: the saved correct output, here tests/data/groups.csv and
#     tests/data/compare_rgb_vs_ms.csv, committed to git. The new output is compared with it.
# - end-to-end test: runs the real script scripts/evaluate.py as a separate process (subprocess) instead of calling
#     one function.
# - fixture (@pytest.fixture) and scope='module': a fixture builds shared input. With scope='module' it runs ONCE
#     for the whole file, so the (slow) script is not run again for every test case.
# - tmp_path_factory: a pytest fixture that creates temporary folders (mktemp). It is used instead of tmp_path
#     because tmp_path cannot be used by a module-scoped fixture.
# - pytest.mark.parametrize: runs the same test once per value in FILES: groups.csv and compare_rgb_vs_ms.csv (2
#     cases).
# - synthetic / fake runs: make_run (imported from test_evaluate.py) writes fake run folders with random predictions
#     for the real val/test ids; no model is trained.
# - determinism: everything is reproducible: fake predictions are seeded from the run name, and the bootstrap has a
#     fixed seed, so the same code always writes the same tables.
# - bootstrap / N_BOOT: resampling of test images to get a confidence interval; N_BOOT = 50 resamples keeps the test
#     fast (the real default is 1000).
# - absolute tolerance (atol) vs relative (rtol): floats may differ by at most 1e-9 in absolute terms (rtol=0), so
#     small and large numbers are held to the same bar. Text, column names and row count must match exactly.
# - pandas.testing.assert_frame_equal: pandas' function that compares two tables cell by cell and fails with a
#     readable difference.
# - UPDATE_EXPECTED environment variable: a deliberate switch: UPDATE_EXPECTED=1 pytest ... overwrites the golden
#     files with the new output. Only done when a result is meant to change (then review git diff tests/data).
# - import from a sibling test file: `from test_evaluate import make_run` works because pytest puts the tests folder
#     on the import path; it avoids duplicating the fake-run builder.

# ====================================================================================================
# Q&A ABOUT THIS FILE (viva practice)
# ====================================================================================================
# Q: Which part of the project does this file protect?
# A: The numbers that scripts/evaluate.py writes (groups.csv and compare_rgb_vs_ms.csv). If a refactor or speed-up
#    silently changes a result in the report, this test fails.
# Q: How do you run it, and how do you refresh the expected files?
# A: pytest tests/test_evaluate_regression.py. To refresh on purpose: UPDATE_EXPECTED=1 pytest
#    tests/test_evaluate_regression.py, then review git diff tests/data/ and commit (say why in the PR).
# Q: Why is no real data needed?
# A: Four fake runs (2 seeds x RGB/13-band, ResNet-50, 1% labels) are generated in a temporary folder. They need the
#    committed splits/split.csv for the real image ids but no images and no training.

DATA = Path(__file__).parent / "data"  # expected tables, committed with the test
FILES = ["groups.csv", "compare_rgb_vs_ms.csv"]  # main results table + one paired comparison (uses the bootstrap)
N_BOOT = 50  # few resamples keep the test fast; the bootstrap seed is fixed, so the interval is reproducible


# Q: What does the eval_out fixture do?
# A: Builds 4 fake runs (resnet50, rgb at 80% and ms at 88% accuracy, seeds 0 and 1), runs scripts/evaluate.py on
#    them with --n-boot 50, asserts exit code 0, and returns the output folder.
# Q: Why scope='module'?
# A: The script is slow compared with a unit test. Running it once and sharing the output between both parametrised
#    cases saves time.
# Q: Why two seeds?
# A: With two seeds groups.csv has a standard deviation, and the RGB-vs-13-band comparison has 2 pairs, so the
#    bootstrap column is exercised.
# scope="module": the script runs once and all tests in this file reuse its output.
@pytest.fixture(scope="module")
def eval_out(tmp_path_factory):
    """Run scripts/evaluate.py once on fake runs and return its output folder (shared by all tests here)."""
    # A temporary working folder that lives as long as the module-scoped fixture.
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


# Q: What does test_evaluation_tables_unchanged check?
# A: The table the script just wrote (groups.csv or compare_rgb_vs_ms.csv) equals the committed expected table:
#    numbers within 1e-9, everything else exactly.
# Q: What bug would make it fail?
# A: Any change in the maths (F1, ECE, bootstrap), in the grouping or pairing, in column names or order, or in the
#    number of rows.
# Q: What happens with UPDATE_EXPECTED set?
# A: The expected file is overwritten with the new output (12 significant digits, far below the tolerance) before
#    the comparison, so the test then passes. It is never overwritten by default.
# Q: Which functions does it exercise?
# A: The whole evaluate.py pipeline: load_runs, run_metrics, grouping over seeds, and compare/paired_bootstrap.
# One test case per table name in FILES.
@pytest.mark.parametrize("name", FILES)
def test_evaluation_tables_unchanged(eval_out, name):
    """The new table must equal the saved one (floats within 1e-9, everything else exactly)."""
    # The table the script just produced.
    new = pd.read_csv(eval_out / name)
    # Only when the developer asks for it (UPDATE_EXPECTED=1) the saved file is overwritten, never by default.
    if os.environ.get("UPDATE_EXPECTED"):
        # Refresh mode: overwrite the expected file with the current output (12 significant digits is
        # far below the 1e-9 tolerance, and keeps the files small and readable in a diff).
        DATA.mkdir(exist_ok=True)
        new.to_csv(DATA / name, index=False, float_format="%.12g")
    # The previously saved ("golden") table that the new one is compared with.
    expected = pd.read_csv(DATA / name)
    # rtol=0 + atol=1e-9: an absolute tolerance only, so small and large values are held to the same bar.
    # The test fails if any number moved by more than 1e-9, or any text/column/row differs.
    pd.testing.assert_frame_equal(new, expected, check_exact=False, rtol=0, atol=1e-9)

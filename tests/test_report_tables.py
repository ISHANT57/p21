"""Tests for scripts/make_report_tables.py (extra report tables from the logged results)."""

# importlib.util/sys: load the script file as a module; Path: find the scripts folder.
import importlib.util
import sys
from pathlib import Path

# numpy/pandas: NaN values and the fake runs table.
import numpy as np
import pandas as pd

# ====================================================================================================
# CONCEPTS USED IN THIS FILE (study notes)
# ====================================================================================================
# - unit test: a check of one function on small made-up input. All four tests here are unit tests of functions in
#     scripts/make_report_tables.py (loaded below as the module `mrt`).
# - assert: a statement that must be true; if false, pytest marks the test as FAILED.
# - synthetic / fake runs table: _runs() builds a three-row pandas DataFrame shaped like
#     results/logs/runs_from_logs.csv, so no real results file is needed.
# - importlib / loading a script as a module: scripts/ is not a package, so the file is loaded from its path
#     (spec_from_file_location, module_from_spec, exec_module) and put in sys.modules; the functions are then used
#     as mrt.all_runs(...).
# - pandas DataFrame: a table of rows and named columns. The report-table functions take the runs table and return a
#     new DataFrame of display strings.
# - display strings: table cells are formatted text, not numbers: 0.885 becomes '88.50' (percent, two decimals),
#     fraction 1.0 becomes '100%', a gain becomes '+2.4'.
# - supervised run vs pseudo-label student: a supervised teacher is trained on the labelled images only; its 'pl'
#     student is trained on the labelled images plus the teacher's confident pseudo-labels (threshold tau).
# - teacher / student change (Delta): student test macro-F1 minus the F1 of its OWN teacher (looked up by run name
#     in the 'teacher' column), in percentage points.
# - macro-F1 and accuracy: macro-F1 = average of the per-class F1 over the 10 classes; accuracy = share of correct
#     predictions.
# - mean +- std over seeds and (n=1): a cell shows 'mean ± std' over seeds. With only one seed the std is NaN
#     (undefined), and the cell must say '(n=1)' instead of inventing a spread such as '± 0'.
# - NaN (np.nan, pd.isna): 'not a number' = missing value; used for columns that do not apply (a supervised run has
#     no teacher or tau).
# - groupby / mean: training_cost groups the supervised runs by model and label fraction and averages the minutes.
# - None instead of an empty table: pl_details returns None when there are no students, so the caller can skip
#     writing that table.

# ====================================================================================================
# Q&A ABOUT THIS FILE (viva practice)
# ====================================================================================================
# Q: Which part of the project does this file protect?
# A: scripts/make_report_tables.py: the extra tables of the report (all runs, pseudo-label details, training cost)
#    and the helper _pm that formats mean and std. Wrong rounding or wrong teacher matching would put wrong numbers
#    straight into the paper.
# Q: How do you run it?
# A: pytest tests/test_report_tables.py from the repo root. Importing the file loads the script once.
# Q: Why is no dataset or results folder needed?
# A: Each function only formats a DataFrame, so three hand-made rows with known numbers are enough to predict the
#    exact output strings.

# Top folder of the repository (this file is in tests/, so go one level up).
REPO_ROOT = Path(__file__).resolve().parents[1]
# Load scripts/make_report_tables.py as the module `mrt` (scripts are not an importable package).
spec = importlib.util.spec_from_file_location("make_report_tables", REPO_ROOT / "scripts" / "make_report_tables.py")
mrt = importlib.util.module_from_spec(spec)
sys.modules["make_report_tables"] = mrt
spec.loader.exec_module(mrt)


# Q: What does _runs build?
# A: Three fake runs, all ResNet-50 RGB seed 0: a supervised 1% run (test F1 0.885, 5 min), a supervised 100% run
#    (F1 0.986, 43 min) and the pseudo-label student of the 1% run (F1 0.9095, tau 0.89, 15,935 of 18,708 pool
#    images pseudo-labelled).
# Q: Why is it enough?
# A: It contains both variants, two label fractions and a teacher link, so every behaviour under test (sorting,
#    teacher matching, exclusion of students from the cost table) shows up. Teacher columns are NaN for the
#    supervised rows, as in the real CSV.
def _runs() -> pd.DataFrame:
    """Two supervised runs and one student whose teacher is the first run."""
    # Columns shared by all three fake runs (ResNet-50, RGB, 1% labels); NaN where a column does not apply
    # to supervised runs (teacher, threshold, pseudo-label counts).
    base = dict(model="resnet50", modality="rgb", fraction=0.01, val_acc=0.9, val_f1=0.89, best_epoch=7, minutes=5.0,
                source="x.ipynb", teacher=np.nan, tau=np.nan, val_precision=np.nan, n_pseudo=np.nan, n_pool=np.nan)
    # Run 1: the supervised teacher, test macro-F1 88.5%.
    sup = {**base, "run_name": "resnet50_rgb_1pct_seed0", "seed": 0, "variant": "supervised", "test_acc": 0.89, "test_f1": 0.885}
    # Run 2: a supervised run with all labels (100%), which also took longer (43 minutes).
    sup2 = {**base, "run_name": "resnet50_rgb_100pct_seed0", "fraction": 1.0, "seed": 0, "variant": "supervised",
            "test_acc": 0.987, "test_f1": 0.986, "minutes": 43.0}
    # Run 3: the pseudo-label student of run 1 (F1 90.95%), with its threshold and pseudo-label counts.
    pl = {**base, "run_name": "resnet50_rgb_1pct_seed0_pl", "seed": 0, "variant": "pl", "test_acc": 0.912, "test_f1": 0.9095,
          "teacher": "resnet50_rgb_1pct_seed0", "tau": 0.89, "val_precision": 0.95, "n_pseudo": 15935, "n_pool": 18708}
    return pd.DataFrame([sup, sup2, pl])


# Q: What does this test check?
# A: all_runs returns 3 rows ordered supervised, supervised, pseudo-label (variants named in words), the 1% run's
#    'Test F1' is '88.50' and the 100% run's 'Labels' is '100%'.
# Q: Why does it matter?
# A: The appendix table must list every run once, in a readable order, with percentages in the same format as the
#    main tables.
def test_all_runs_lists_every_run_in_percent():
    """The all-runs table must have one row per run, readable variant names and scores shown in percent."""
    t = mrt.all_runs(_runs())
    # 3 rows in the order supervised, supervised, pseudo-label (variants are sorted supervised first).
    assert len(t) == 3 and list(t["Variant"]) == ["supervised", "supervised", "pseudo-label"]
    # 0.885 becomes "88.50" (percent, two decimals), and fraction 1.0 becomes "100%".
    assert t.loc[0, "Test F1"] == "88.50" and t.loc[1, "Labels"] == "100%"


# Q: What does this test check?
# A: pl_details gives one row for the one student: change '+2.4' (90.95 - 88.5), 'Pseudo-labelled' '15,935 /
#    18,708', share '85.2'. Given only supervised runs it returns None.
# Q: What bug would make it fail?
# A: Comparing the student with the wrong run: the table also contains a 100%-label run (F1 98.6) with the same
#    model and seed, which would give a negative change. The teacher must be found by the 'teacher' run name.
# Q: How is 85.2 obtained?
# A: n_pseudo / n_pool * 100 = 15,935 / 18,708 * 100 = 85.2 percent of the unlabelled pool received a pseudo-label.
def test_pl_details_change_is_against_the_own_teacher():
    """The pseudo-label table must measure each student against its OWN teacher run, not any other run."""
    t = mrt.pl_details(_runs())
    # One student -> one row. Its change is 90.95 - 88.5 = +2.4 points (teacher is the 1% run, not the 100% one).
    assert len(t) == 1 and t.loc[0, "$\\Delta$"] == "+2.4"  # 90.95 - 88.5
    # 15,935 of 18,708 pool images were pseudo-labelled, which is 85.2%.
    assert t.loc[0, "Pseudo-labelled"] == "15,935 / 18,708" and t.loc[0, "Share (%)"] == "85.2"
    # With no student runs there is nothing to show, so the function returns None instead of an empty table.
    assert mrt.pl_details(_runs().query("variant == 'supervised'")) is None  # no students -> no table


# Q: What does this test check?
# A: training_cost has two rows, labels '1%' and '100%', with mean times '5.0' and '43.0' minutes.
# Q: Why does it matter?
# A: The cost table is for supervised runs only. The student (variant pl) must not be averaged into the 1% group,
#    which would change 5.0.
def test_training_cost_means_per_model_and_fraction():
    """The training-cost table must average the training minutes per model and label fraction (supervised only)."""
    t = mrt.training_cost(_runs())
    # Two groups: 1% labels (5.0 min; the student is excluded) and 100% labels (43.0 min).
    assert list(t["Labels"]) == ["1%", "100%"] and list(t["Mean time (min)"]) == ["5.0", "43.0"]


# Q: What does this test check?
# A: _pm(0.8, NaN, sign=True) gives '+80.0 (n=1)' and _pm(0.8, 0.01) gives '80.0 ± 1.0'.
# Q: Why does it matter?
# A: With a single seed no spread was measured. Printing '± 0.0' would wrongly suggest that seeds agreed exactly;
#    '(n=1)' states honestly that there is only one run.
# Q: What do the arguments mean?
# A: mean and std are 0-1 values, scaled by 100 into percent with one decimal; sign=True forces a '+' or '-' in
#    front (used for differences).
def test_pm_never_invents_a_spread_for_one_seed():
    """With a single seed (std is NaN) the cell must say "(n=1)" and not show a made-up "± 0"."""
    # NaN std, with a forced "+" sign for differences: only the mean (in percent) and the note "(n=1)".
    assert mrt._pm(0.8, np.nan, sign=True) == "+80.0 (n=1)"
    # With a real std the cell shows "mean ± std" in percent.
    assert mrt._pm(0.8, 0.01) == "80.0 ± 1.0"

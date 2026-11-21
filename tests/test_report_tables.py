"""Tests for scripts/make_report_tables.py (extra report tables from the logged results)."""

# importlib.util/sys: load the script file as a module; Path: find the scripts folder.
import importlib.util
import sys
from pathlib import Path

# numpy/pandas: NaN values and the fake runs table.
import numpy as np
import pandas as pd

# Top folder of the repository (this file is in tests/, so go one level up).
REPO_ROOT = Path(__file__).resolve().parents[1]
# Load scripts/make_report_tables.py as the module `mrt` (scripts are not an importable package).
spec = importlib.util.spec_from_file_location("make_report_tables", REPO_ROOT / "scripts" / "make_report_tables.py")
mrt = importlib.util.module_from_spec(spec)
sys.modules["make_report_tables"] = mrt
spec.loader.exec_module(mrt)


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


def test_all_runs_lists_every_run_in_percent():
    """The all-runs table must have one row per run, readable variant names and scores shown in percent."""
    t = mrt.all_runs(_runs())
    # 3 rows in the order supervised, supervised, pseudo-label (variants are sorted supervised first).
    assert len(t) == 3 and list(t["Variant"]) == ["supervised", "supervised", "pseudo-label"]
    # 0.885 becomes "88.50" (percent, two decimals), and fraction 1.0 becomes "100%".
    assert t.loc[0, "Test F1"] == "88.50" and t.loc[1, "Labels"] == "100%"


def test_pl_details_change_is_against_the_own_teacher():
    """The pseudo-label table must measure each student against its OWN teacher run, not any other run."""
    t = mrt.pl_details(_runs())
    # One student -> one row. Its change is 90.95 - 88.5 = +2.4 points (teacher is the 1% run, not the 100% one).
    assert len(t) == 1 and t.loc[0, "$\\Delta$"] == "+2.4"  # 90.95 - 88.5
    # 15,935 of 18,708 pool images were pseudo-labelled, which is 85.2%.
    assert t.loc[0, "Pseudo-labelled"] == "15,935 / 18,708" and t.loc[0, "Share (%)"] == "85.2"
    # With no student runs there is nothing to show, so the function returns None instead of an empty table.
    assert mrt.pl_details(_runs().query("variant == 'supervised'")) is None  # no students -> no table


def test_training_cost_means_per_model_and_fraction():
    """The training-cost table must average the training minutes per model and label fraction (supervised only)."""
    t = mrt.training_cost(_runs())
    # Two groups: 1% labels (5.0 min; the student is excluded) and 100% labels (43.0 min).
    assert list(t["Labels"]) == ["1%", "100%"] and list(t["Mean time (min)"]) == ["5.0", "43.0"]


def test_pm_never_invents_a_spread_for_one_seed():
    """With a single seed (std is NaN) the cell must say "(n=1)" and not show a made-up "± 0"."""
    # NaN std, with a forced "+" sign for differences: only the mean (in percent) and the note "(n=1)".
    assert mrt._pm(0.8, np.nan, sign=True) == "+80.0 (n=1)"
    # With a real std the cell shows "mean ± std" in percent.
    assert mrt._pm(0.8, 0.01) == "80.0 ± 1.0"

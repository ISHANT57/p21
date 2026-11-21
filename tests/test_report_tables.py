"""Tests for scripts/make_report_tables.py (extra report tables from the logged results)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_report_tables", REPO_ROOT / "scripts" / "make_report_tables.py")
mrt = importlib.util.module_from_spec(spec)
sys.modules["make_report_tables"] = mrt
spec.loader.exec_module(mrt)


def _runs() -> pd.DataFrame:
    """Two supervised runs and one student whose teacher is the first run."""
    base = dict(model="resnet50", modality="rgb", fraction=0.01, val_acc=0.9, val_f1=0.89, best_epoch=7, minutes=5.0,
                source="x.ipynb", teacher=np.nan, tau=np.nan, val_precision=np.nan, n_pseudo=np.nan, n_pool=np.nan)
    sup = {**base, "run_name": "resnet50_rgb_1pct_seed0", "seed": 0, "variant": "supervised", "test_acc": 0.89, "test_f1": 0.885}
    sup2 = {**base, "run_name": "resnet50_rgb_100pct_seed0", "fraction": 1.0, "seed": 0, "variant": "supervised",
            "test_acc": 0.987, "test_f1": 0.986, "minutes": 43.0}
    pl = {**base, "run_name": "resnet50_rgb_1pct_seed0_pl", "seed": 0, "variant": "pl", "test_acc": 0.912, "test_f1": 0.9095,
          "teacher": "resnet50_rgb_1pct_seed0", "tau": 0.89, "val_precision": 0.95, "n_pseudo": 15935, "n_pool": 18708}
    return pd.DataFrame([sup, sup2, pl])


def test_all_runs_lists_every_run_in_percent():
    t = mrt.all_runs(_runs())
    assert len(t) == 3 and list(t["Variant"]) == ["supervised", "supervised", "pseudo-label"]
    assert t.loc[0, "Test F1"] == "88.50" and t.loc[1, "Labels"] == "100%"


def test_pl_details_change_is_against_the_own_teacher():
    t = mrt.pl_details(_runs())
    assert len(t) == 1 and t.loc[0, "$\\Delta$"] == "+2.4"  # 90.95 - 88.5
    assert t.loc[0, "Pseudo-labelled"] == "15,935 / 18,708" and t.loc[0, "Share (%)"] == "85.2"
    assert mrt.pl_details(_runs().query("variant == 'supervised'")) is None  # no students -> no table


def test_training_cost_means_per_model_and_fraction():
    t = mrt.training_cost(_runs())
    assert list(t["Labels"]) == ["1%", "100%"] and list(t["Mean time (min)"]) == ["5.0", "43.0"]


def test_pm_never_invents_a_spread_for_one_seed():
    assert mrt._pm(0.8, np.nan, sign=True) == "+80.0 (n=1)"
    assert mrt._pm(0.8, 0.01) == "80.0 ± 1.0"

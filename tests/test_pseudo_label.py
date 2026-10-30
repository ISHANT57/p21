"""Pseudo-labelling tests on synthetic teacher outputs (no GPU, no cache needed)."""

import json

import numpy as np
import pandas as pd
import pytest

from eurosat_le.constants import CLASSES
from eurosat_le.data.datasets import split_ids, unlabelled_ids
from eurosat_le.data.splits import load_split, load_subset
from eurosat_le.ssl.pseudo_label import (choose_threshold, make_pseudo_labels, quality, select, softmax,
                                         threshold_table)
from eurosat_le.ssl.self_training import student_config, student_suffix
from eurosat_le.train.config import load_config


def _probs(conf, pred, k=10):
    """Rows whose max prob is `conf` at class `pred`, rest spread evenly."""
    p = np.tile(((1 - conf) / (k - 1))[:, None], (1, k))
    p[np.arange(len(conf)), pred] = conf
    return p


def test_threshold_is_smallest_tau_meeting_target():
    # confident predictions are right, unconfident ones wrong
    conf = np.array([0.99, 0.95, 0.9, 0.8, 0.6, 0.55])
    labels = np.array([0, 1, 2, 3, 4, 5])
    pred = np.array([0, 1, 2, 3, 9, 9])
    tau, table = choose_threshold(_probs(conf, pred), labels, target_precision=1.0)
    assert tau == pytest.approx(0.61)  # first grid value above 0.60 -> only correct predictions remain
    assert table.set_index("tau").loc[0.5, "coverage"] == 1.0


def test_threshold_falls_back_to_most_precise():
    probs = _probs(np.full(4, 0.9), np.zeros(4, int))
    tau, _ = choose_threshold(probs, np.ones(4, int), target_precision=0.95)
    assert 0.5 <= tau <= 0.99


def test_select_respects_tau_and_orders_by_confidence():
    conf = np.array([0.6, 0.97, 0.8, 0.99])
    idx = select(_probs(conf, np.array([0, 1, 2, 3])), tau=0.75)
    assert idx.tolist() == [3, 1, 2]


def test_select_balanced_follows_prior():
    rng = np.random.default_rng(0)
    pred = np.r_[np.zeros(100, int), np.ones(20, int)]
    probs = _probs(rng.uniform(0.9, 1.0, 120), pred, k=2)
    idx = select(probs, 0.5, balance=True, class_prior=[1, 1])
    counts = np.bincount(probs[idx].argmax(1), minlength=2)
    assert counts.tolist() == [60, 20]  # class 0 capped at half of 120; class 1 keeps all it has


def test_select_balanced_never_zeroes_other_classes():
    pred = np.r_[np.zeros(50, int), np.ones(10, int)]  # class 2 has no confident predictions
    probs = _probs(np.full(60, 0.95), pred, k=3)
    counts = np.bincount(probs[select(probs, 0.5, balance=True, class_prior=[1, 1, 1])].argmax(1), minlength=3)
    assert counts.tolist() == [20, 10, 0]


def test_student_suffix():
    assert [student_suffix(s) for s in ("", "pl", "pl2")] == ["pl", "pl2", "pl3"]
    with pytest.raises(ValueError):
        student_suffix("tune_lr0.001")


def test_student_config_from_planned_teacher():
    t = load_config(None, model="resnet50", modality="ms", fraction=0.01, seed=1, lr=3e-4)
    s = student_config(t)
    assert s.run_name == "resnet50_ms_1pct_seed1_pl" and s.teacher == t.run_name and s.lr == 3e-4
    assert s.pseudo_labels == "pseudo_labels.csv"


def test_make_pseudo_labels_end_to_end(tmp_path):
    frac, seed = 0.01, 0
    pool = np.array(unlabelled_ids(frac, seed))
    split = load_split().set_index("id")
    truth = split.loc[pool, "label"].to_numpy()
    rng = np.random.default_rng(0)
    # synthetic teacher: right with high confidence on 70% of images, wrong and unsure elsewhere
    right = rng.random(len(pool)) < 0.7
    pred = np.where(right, truth, (truth + 1) % 10)
    conf = np.where(right, rng.uniform(0.9, 1.0, len(pool)), rng.uniform(0.3, 0.7, len(pool)))
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    np.savez(teacher / "preds_unlabelled.npz", logits=np.log(_probs(conf, pred)), labels=np.array([]), ids=pool)
    val = np.array(split_ids("val"))
    vt = split.loc[val, "label"].to_numpy()
    vr = rng.random(len(val)) < 0.7
    vpred = np.where(vr, vt, (vt + 1) % 10)
    vconf = np.where(vr, rng.uniform(0.9, 1.0, len(val)), rng.uniform(0.3, 0.7, len(val)))
    np.savez(teacher / "preds_val.npz", logits=np.log(_probs(vconf, vpred)), labels=vt, ids=val)

    student = tmp_path / "student"
    summary = make_pseudo_labels(teacher, student, frac, seed, target_precision=0.95)
    pl = pd.read_csv(student / "pseudo_labels.csv")
    assert summary["tau"] <= 0.9 and summary["val_precision_at_tau"] >= 0.95
    assert not set(pl["id"]) & set(load_subset(frac, seed)), "pseudo-labels must come from the unlabelled pool"
    assert set(pl["id"]) <= set(pool)
    assert (pl["confidence"] >= summary["tau"]).all()
    q = quality(student / "pseudo_labels.csv")
    # tau targets 95% precision on val, so the pool's pseudo-label accuracy should match it closely
    assert q["accuracy"] == pytest.approx(summary["val_precision_at_tau"], abs=0.02)
    assert set(q["per_class"]) == set(CLASSES)
    assert json.loads((student / "pseudo_label_selection.json").read_text())["n_selected"] == len(pl)


def test_softmax_rows_sum_to_one():
    p = softmax(np.random.default_rng(0).normal(size=(5, 10)) * 50)
    assert np.allclose(p.sum(1), 1) and np.isfinite(p).all()


def test_threshold_table_monotone_coverage():
    rng = np.random.default_rng(1)
    t = threshold_table(softmax(rng.normal(size=(200, 10)) * 3), rng.integers(0, 10, 200))
    assert (np.diff(t["coverage"]) <= 0).all()

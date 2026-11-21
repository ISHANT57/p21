"""Pseudo-labelling tests on synthetic teacher outputs (no GPU, no cache needed)."""

# json: read the selection summary file; numpy/pandas: build fake teacher outputs; pytest: approx and errors.
import json

import numpy as np
import pandas as pd
import pytest

# CLASSES: the 10 class names. The ids helpers give the real split's image ids; the split helpers give
# the real labels and the labelled subset (used to build realistic fake teacher outputs).
from eurosat_le.constants import CLASSES
from eurosat_le.data.datasets import split_ids, unlabelled_ids
from eurosat_le.data.splits import load_split, load_subset
# The pseudo-labelling functions under test: threshold choice, selection, creation and quality check.
from eurosat_le.ssl.pseudo_label import (choose_threshold, make_pseudo_labels, quality, select, softmax,
                                         threshold_table)
# Helpers that name and configure a student run from its teacher.
from eurosat_le.ssl.self_training import student_config, student_suffix
from eurosat_le.train.config import load_config


def _probs(conf, pred, k=10):
    """Rows whose max prob is `conf` at class `pred`, rest spread evenly."""
    # Every row first gets the leftover probability (1 - conf) shared equally among the k-1 other classes ...
    p = np.tile(((1 - conf) / (k - 1))[:, None], (1, k))
    # ... then the predicted class gets probability `conf`, so each row sums to 1 and argmax is `pred`.
    p[np.arange(len(conf)), pred] = conf
    return p


def test_threshold_is_smallest_tau_meeting_target():
    """The chosen confidence threshold must be the smallest one that reaches the target precision."""
    # confident predictions are right, unconfident ones wrong
    # Six validation images with the teacher's confidence; the last two are predicted wrongly (class 9).
    conf = np.array([0.99, 0.95, 0.9, 0.8, 0.6, 0.55])
    labels = np.array([0, 1, 2, 3, 4, 5])
    pred = np.array([0, 1, 2, 3, 9, 9])
    # Ask for 100% precision: only images the teacher got right may remain after thresholding.
    tau, table = choose_threshold(_probs(conf, pred), labels, target_precision=1.0)
    assert tau == pytest.approx(0.61)  # first grid value above 0.60 -> only correct predictions remain
    # At the lowest threshold (0.5) every image passes, so coverage is 100%.
    assert table.set_index("tau").loc[0.5, "coverage"] == 1.0


def test_threshold_falls_back_to_most_precise():
    """If no threshold can reach the target precision, a valid fallback threshold must still be returned."""
    # Four images, all predicted as class 0 with 0.9 confidence but all truly class 1: precision is always 0.
    probs = _probs(np.full(4, 0.9), np.zeros(4, int))
    tau, _ = choose_threshold(probs, np.ones(4, int), target_precision=0.95)
    # The target is unreachable, so the function must fall back to some grid value instead of failing.
    assert 0.5 <= tau <= 0.99


def test_select_respects_tau_and_orders_by_confidence():
    """select() must keep only images with confidence >= tau, listed from most to least confident."""
    # Four unlabelled images, each predicted as a different class with the given confidence.
    conf = np.array([0.6, 0.97, 0.8, 0.99])
    idx = select(_probs(conf, np.array([0, 1, 2, 3])), tau=0.75)
    # Image 0 (0.6) is dropped; the rest come out sorted by confidence: 0.99 (3), 0.97 (1), 0.8 (2).
    assert idx.tolist() == [3, 1, 2]


def test_select_balanced_follows_prior():
    """With balancing, a class may keep at most its labelled-set share of the selection, but never more than it has."""
    rng = np.random.default_rng(0)
    # 120 images: 100 predicted as class 0 and only 20 as class 1; all are highly confident (0.9-1.0).
    pred = np.r_[np.zeros(100, int), np.ones(20, int)]
    probs = _probs(rng.uniform(0.9, 1.0, 120), pred, k=2)
    # The labelled set is perfectly balanced (prior 1:1), so each class may take at most half the selection.
    idx = select(probs, 0.5, balance=True, class_prior=[1, 1])
    # Count how many selected images each class received.
    counts = np.bincount(probs[idx].argmax(1), minlength=2)
    assert counts.tolist() == [60, 20]  # class 0 capped at half of 120; class 1 keeps all it has


def test_select_balanced_never_zeroes_other_classes():
    """Balancing must not throw away other classes just because one class has no confident images."""
    # 60 confident images: 50 of class 0, 10 of class 1 and none of class 2.
    pred = np.r_[np.zeros(50, int), np.ones(10, int)]  # class 2 has no confident predictions
    probs = _probs(np.full(60, 0.95), pred, k=3)
    # Equal prior over 3 classes: each class cap is 20 images. Count what each class actually keeps.
    counts = np.bincount(probs[select(probs, 0.5, balance=True, class_prior=[1, 1, 1])].argmax(1), minlength=3)
    # Class 0 is cut from 50 to the cap of 20, class 1 keeps its 10, and the empty class 2 stays at 0.
    assert counts.tolist() == [20, 10, 0]


def test_student_suffix():
    """Each self-training round adds one: '' -> pl -> pl2 -> pl3; other run types are rejected."""
    # Suffix of a student whose teacher is a plain run (""), a first-round student ("pl") and a second round ("pl2").
    assert [student_suffix(s) for s in ("", "pl", "pl2")] == ["pl", "pl2", "pl3"]
    # A learning-rate tuning run is not a valid teacher, so asking for its student must raise an error.
    with pytest.raises(ValueError):
        student_suffix("tune_lr0.001")


def test_student_config_from_planned_teacher():
    """A student's config must copy the teacher's settings, get the right name and point at its teacher."""
    # A teacher config that only exists in memory (it has not been trained), with a non-default learning rate.
    t = load_config(None, model="resnet50", modality="ms", fraction=0.01, seed=1, lr=3e-4)
    s = student_config(t)
    # Name gets the "_pl" suffix, the teacher is recorded by name, and the learning rate is copied over.
    assert s.run_name == "resnet50_ms_1pct_seed1_pl" and s.teacher == t.run_name and s.lr == 3e-4
    # The student reads its training labels from this file inside its own run folder.
    assert s.pseudo_labels == "pseudo_labels.csv"


def test_make_pseudo_labels_end_to_end(tmp_path):
    """Build a fake teacher output, run the real pseudo-label creation, and check every promised property."""
    # Settings: 1% labelled images, subset seed 0. `pool` = ids of the images treated as unlabelled.
    frac, seed = 0.01, 0
    pool = np.array(unlabelled_ids(frac, seed))
    # The real labels of the pool images; the fake teacher "knows" them so we can control how good it is.
    split = load_split().set_index("id")
    truth = split.loc[pool, "label"].to_numpy()
    rng = np.random.default_rng(0)
    # synthetic teacher: right with high confidence on 70% of images, wrong and unsure elsewhere
    right = rng.random(len(pool)) < 0.7
    pred = np.where(right, truth, (truth + 1) % 10)
    conf = np.where(right, rng.uniform(0.9, 1.0, len(pool)), rng.uniform(0.3, 0.7, len(pool)))
    # Fake teacher run folder with its predictions on the unlabelled pool (no labels are stored for it).
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    np.savez(teacher / "preds_unlabelled.npz", logits=np.log(_probs(conf, pred)), labels=np.array([]), ids=pool)
    # Same kind of fake predictions on the validation set; the labels are stored, since the threshold is
    # chosen on validation data (the only place where true labels may be used).
    val = np.array(split_ids("val"))
    vt = split.loc[val, "label"].to_numpy()
    vr = rng.random(len(val)) < 0.7
    vpred = np.where(vr, vt, (vt + 1) % 10)
    vconf = np.where(vr, rng.uniform(0.9, 1.0, len(val)), rng.uniform(0.3, 0.7, len(val)))
    np.savez(teacher / "preds_val.npz", logits=np.log(_probs(vconf, vpred)), labels=vt, ids=val)

    # Create the pseudo-labels in a new student folder, asking for a threshold with 95% validation precision.
    student = tmp_path / "student"
    summary = make_pseudo_labels(teacher, student, frac, seed, target_precision=0.95)
    pl = pd.read_csv(student / "pseudo_labels.csv")
    # Threshold is at most 0.9 (the right images have confidence >= 0.9) and meets the precision target.
    assert summary["tau"] <= 0.9 and summary["val_precision_at_tau"] >= 0.95
    # No labelled image may appear among the pseudo-labels; that would leak real labels into training.
    assert not set(pl["id"]) & set(load_subset(frac, seed)), "pseudo-labels must come from the unlabelled pool"
    # Every pseudo-labelled image must belong to the unlabelled pool.
    assert set(pl["id"]) <= set(pool)
    # Every kept image has confidence at or above the chosen threshold.
    assert (pl["confidence"] >= summary["tau"]).all()
    # quality() compares pseudo-labels with the hidden true labels (analysis only, never used for selection).
    q = quality(student / "pseudo_labels.csv")
    # tau targets 95% precision on val, so the pool's pseudo-label accuracy should match it closely
    assert q["accuracy"] == pytest.approx(summary["val_precision_at_tau"], abs=0.02)
    # The quality report must list all 10 classes.
    assert set(q["per_class"]) == set(CLASSES)
    # The saved summary file must record the same number of selected images as the CSV has rows.
    assert json.loads((student / "pseudo_label_selection.json").read_text())["n_selected"] == len(pl)


def test_softmax_rows_sum_to_one():
    """Softmax must turn any logits into probabilities (rows sum to 1), even for very large logits."""
    # Random logits multiplied by 50 make them huge; a careless softmax would overflow to inf/NaN here.
    p = softmax(np.random.default_rng(0).normal(size=(5, 10)) * 50)
    # Each row sums to 1 and every value is a finite number.
    assert np.allclose(p.sum(1), 1) and np.isfinite(p).all()


def test_threshold_table_monotone_coverage():
    """Raising the threshold can only remove images, so coverage must never increase along the threshold grid."""
    rng = np.random.default_rng(1)
    # 200 random fake validation predictions and random labels; the values do not matter, only the trend.
    t = threshold_table(softmax(rng.normal(size=(200, 10)) * 3), rng.integers(0, 10, 200))
    # Differences between consecutive coverage values must all be <= 0 (never rising).
    assert (np.diff(t["coverage"]) <= 0).all()

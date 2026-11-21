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


# ====================================================================================================
# CONCEPTS USED IN THIS FILE (study notes)
# ====================================================================================================
# - unit test: a check of ONE function on small made-up input: the threshold, select, softmax, student-name and
#     student-config tests.
# - end-to-end (integration) test: test_make_pseudo_labels_end_to_end runs the real pseudo-label creation from fake
#     teacher files to the written CSV/JSON and checks every promised property.
# - assert: a statement that must be true; if false, pytest marks the test as FAILED.
# - tmp_path (pytest fixture): a fresh empty temporary folder per test; the fake teacher and student run folders are
#     created there.
# - pytest.raises / pytest.approx: raises: passes only if the named error occurs. approx: float comparison with a
#     tolerance.
# - synthetic (fake) teacher output: _probs builds probability tables by hand, and the end-to-end test makes a
#     'teacher' that is right on about 70% of images with high confidence (0.9-1.0) and wrong with low confidence
#     (0.3-0.7). No model or GPU is needed.
# - logits, softmax, probability: logits = raw model scores; softmax turns each row into probabilities that sum to 1
#     (computed stably by subtracting the row maximum so huge scores do not overflow).
# - confidence and prediction: confidence = the largest probability of an image; prediction = its argmax class.
# - pseudo-label / self-training / teacher and student: a supervised teacher predicts labels for unlabelled images;
#     confident predictions become pseudo-labels; a student is then trained on labelled + pseudo-labelled images.
# - threshold tau: an unlabelled image gets a pseudo-label only if its confidence >= tau. The grid TAU_GRID is 0.50,
#     0.51, ..., 0.99.
# - precision and coverage (threshold_table): for a tau: coverage = share of validation images that pass the
#     threshold; precision = share of those passing images that the teacher got right.
# - target precision: choose_threshold picks the SMALLEST tau whose validation precision reaches the target (e.g.
#     0.95), keeping the most pseudo-labels at acceptable quality; if none reaches it, the most precise tau (lowest
#     among ties) is used as a fallback.
# - validation-only selection / data leakage: tau is chosen from the teacher's VALIDATION predictions (true labels
#     allowed there). The true labels of the unlabelled pool and the test set must never be used for selection.
# - labelled subset vs unlabelled pool: the labelled subset (e.g. 1% of train, subset seed 0) keeps its true labels;
#     the rest of train is the unlabelled pool that receives pseudo-labels. They must not overlap.
# - class-balanced selection (balance=True, class_prior): each predicted class may keep at most ceil(prior share x
#     number of confident images) items, most confident first, where the prior is the class mix of the labelled
#     subset. A class with fewer confident images keeps them all.
# - quality() (analysis only): compares pseudo-labels with the hidden true labels to report accuracy and per-class
#     counts for the report; it never influences which images are selected.
# - preds_*.npz files: numpy archives written by a run: logits, labels, ids. The fake teacher folder has
#     preds_val.npz (with true labels) and preds_unlabelled.npz (labels empty).
# - monotone: never increasing/decreasing. Coverage can only fall as tau rises, so np.diff(coverage) <= 0.
# - student naming (student_suffix, student_config): a student gets the suffix pl (pl2, pl3 for further rounds) and
#     copies the teacher's settings; its config records the teacher's run name and 'pseudo_labels.csv'.

# ====================================================================================================
# Q&A ABOUT THIS FILE (viva practice)
# ====================================================================================================
# Q: Which part of the project does this file protect?
# A: eurosat_le/ssl/pseudo_label.py (softmax, threshold choice, selection, make_pseudo_labels, quality) and the
#    naming/config helpers of ssl/self_training.py. These decide which unlabelled images become training targets, so
#    a leakage bug here would invalidate the semi-supervised results.
# Q: How do you run it?
# A: pytest tests/test_pseudo_label.py from the repo root (with eurosat_le importable).
# Q: Why are no GPU, cache or trained model needed?
# A: The functions only work on probability arrays and on id lists. Teacher outputs are faked; the only real inputs
#    are the committed split files in splits/ (ids and true labels), which the end-to-end test reads.

# Q: What does _probs do?
# A: Builds a table of probabilities with k columns: for each row the predicted class pred gets probability conf and
#    the remaining 1 - conf is shared equally by the other k - 1 classes.
# Q: Why is it enough?
# A: Each row sums to 1 and its maximum is conf at class pred (as long as conf > 1/k, true in all tests), so the
#    confidence and prediction of every fake image are exactly the ones we choose.
def _probs(conf, pred, k=10):
    """Rows whose max prob is `conf` at class `pred`, rest spread evenly."""
    # Every row first gets the leftover probability (1 - conf) shared equally among the k-1 other classes ...
    p = np.tile(((1 - conf) / (k - 1))[:, None], (1, k))
    # ... then the predicted class gets probability `conf`, so each row sums to 1 and argmax is `pred`.
    p[np.arange(len(conf)), pred] = conf
    return p


# Q: What does this test check?
# A: With six validation images (the last two predicted wrongly with confidence 0.6 and 0.55) and target precision
#    1.0, choose_threshold returns tau = 0.61, and the table shows coverage 1.0 at tau = 0.5.
# Q: Why 0.61?
# A: At tau <= 0.60 the wrong 0.6 image still passes, so precision < 1. The first grid value above 0.60 is 0.61,
#    which keeps only the four correct images (confidence >= 0.8), precision 1.0.
# Q: Why does it matter?
# A: The smallest acceptable tau keeps the most pseudo-labels while the quality is still known to be good; a larger
#    tau would throw away usable labels, a smaller one would let wrong labels in.
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


# Q: What does this test check?
# A: If no tau can reach the target precision (0.95), choose_threshold still returns a tau inside the grid (0.5 to
#    0.99) instead of failing.
# Q: How is the impossible case built?
# A: Four images all predicted as class 0 with confidence 0.9 but all truly class 1: precision is 0 wherever images
#    pass. The function then takes the most precise tau (the lowest among ties).
def test_threshold_falls_back_to_most_precise():
    """If no threshold can reach the target precision, a valid fallback threshold must still be returned."""
    # Four images, all predicted as class 0 with 0.9 confidence but all truly class 1: precision is always 0.
    probs = _probs(np.full(4, 0.9), np.zeros(4, int))
    tau, _ = choose_threshold(probs, np.ones(4, int), target_precision=0.95)
    # The target is unreachable, so the function must fall back to some grid value instead of failing.
    assert 0.5 <= tau <= 0.99


# Q: What does this test check?
# A: select(probs, tau=0.75) keeps only images with confidence >= 0.75 and lists them from most to least confident:
#    [3, 1, 2] for confidences [0.6, 0.97, 0.8, 0.99].
# Q: Why does it matter?
# A: Selecting only confident predictions is the core of pseudo-labelling; the order (most confident first) is how
#    pseudo_labels.csv is written and how class caps are applied.
def test_select_respects_tau_and_orders_by_confidence():
    """select() must keep only images with confidence >= tau, listed from most to least confident."""
    # Four unlabelled images, each predicted as a different class with the given confidence.
    conf = np.array([0.6, 0.97, 0.8, 0.99])
    idx = select(_probs(conf, np.array([0, 1, 2, 3])), tau=0.75)
    # Image 0 (0.6) is dropped; the rest come out sorted by confidence: 0.99 (3), 0.97 (1), 0.8 (2).
    assert idx.tolist() == [3, 1, 2]


# Q: What does this test check?
# A: With balance=True and a 1:1 class prior, 100 confident images of class 0 and 20 of class 1 give 60 and 20
#    selected: class 0 is capped at half of 120, class 1 keeps all it has.
# Q: Why does it matter?
# A: Without balancing, easy classes could dominate the pseudo-labels. The cap follows the class mix of the labelled
#    subset, which is known, so using it is legitimate.
# Q: Why only two classes?
# A: k=2 makes the arithmetic visible: caps = ceil(0.5 x 120) = 60 per class.
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


# Q: What does this test check?
# A: With 3 classes and equal prior, 50 + 10 + 0 confident images give [20, 10, 0] selected: the big class is cut to
#    the cap 20, the small class keeps its 10, and the class with none stays empty.
# Q: What bug would it catch?
# A: A balancing rule that rescales everything to the weakest class (or divides by zero) and throws away other
#    classes because one class has no confident image. The caps are only upper limits.
def test_select_balanced_never_zeroes_other_classes():
    """Balancing must not throw away other classes just because one class has no confident images."""
    # 60 confident images: 50 of class 0, 10 of class 1 and none of class 2.
    pred = np.r_[np.zeros(50, int), np.ones(10, int)]  # class 2 has no confident predictions
    probs = _probs(np.full(60, 0.95), pred, k=3)
    # Equal prior over 3 classes: each class cap is 20 images. Count what each class actually keeps.
    counts = np.bincount(probs[select(probs, 0.5, balance=True, class_prior=[1, 1, 1])].argmax(1), minlength=3)
    # Class 0 is cut from 50 to the cap of 20, class 1 keeps its 10, and the empty class 2 stays at 0.
    assert counts.tolist() == [20, 10, 0]


# Q: What does this test check?
# A: student_suffix maps '' to 'pl', 'pl' to 'pl2', 'pl2' to 'pl3'; a learning-rate tuning suffix raises ValueError.
# Q: Why does it matter?
# A: Run folders are named after the suffix, so students of students must not overwrite earlier runs, and a tuning
#    run (validation-only) must not become a teacher.
def test_student_suffix():
    """Each self-training round adds one: '' -> pl -> pl2 -> pl3; other run types are rejected."""
    # Suffix of a student whose teacher is a plain run (""), a first-round student ("pl") and a second round ("pl2").
    assert [student_suffix(s) for s in ("", "pl", "pl2")] == ["pl", "pl2", "pl3"]
    # A learning-rate tuning run is not a valid teacher, so asking for its student must raise an error.
    with pytest.raises(ValueError):
        student_suffix("tune_lr0.001")


# Q: What does this test check?
# A: student_config(teacher_config) gives run_name 'resnet50_ms_1pct_seed1_pl', teacher = the teacher's run name, lr
#    copied (3e-4) and pseudo_labels = 'pseudo_labels.csv'.
# Q: Why does it matter?
# A: A student must use the same model, fraction, seed and hyper-parameters as its teacher, so the evaluation can
#    compare the pair; only the training set differs.
# Q: What is a 'planned' teacher?
# A: A TrainConfig object created in memory that has not been trained, so no files are read; run_grid.py uses this
#    to list students before their teachers finish.
def test_student_config_from_planned_teacher():
    """A student's config must copy the teacher's settings, get the right name and point at its teacher."""
    # A teacher config that only exists in memory (it has not been trained), with a non-default learning rate.
    t = load_config(None, model="resnet50", modality="ms", fraction=0.01, seed=1, lr=3e-4)
    s = student_config(t)
    # Name gets the "_pl" suffix, the teacher is recorded by name, and the learning rate is copied over.
    assert s.run_name == "resnet50_ms_1pct_seed1_pl" and s.teacher == t.run_name and s.lr == 3e-4
    # The student reads its training labels from this file inside its own run folder.
    assert s.pseudo_labels == "pseudo_labels.csv"


# Q: What does this test check?
# A: make_pseudo_labels on a fake teacher writes pseudo_labels.csv and a summary with: tau <= 0.9 and validation
#    precision >= 0.95; no pseudo-labelled id is in the labelled subset; all ids come from the unlabelled pool;
#    every confidence >= tau; pool accuracy from quality() within 0.02 of the validation precision; all 10 classes
#    in the quality report; n_selected in the JSON equals the CSV rows.
# Q: Why does it matter?
# A: It proves the leakage safeguards: tau comes from validation only, labelled images are never pseudo-labelled,
#    and the measured pseudo-label accuracy on the hidden labels matches what validation promised.
# Q: How is the fake teacher built?
# A: Pool = real unlabelled ids for 1% labels, seed 0. Truth = real labels. The teacher is right on about 70% with
#    confidence 0.9-1.0 and otherwise wrong (class + 1) with confidence 0.3-0.7. Logits are np.log of the
#    probabilities. The same recipe is used for the real validation ids. Both are saved as npz files.
# Q: Which functions does it exercise?
# A: make_pseudo_labels (which calls load_subset, softmax, choose_threshold, select) and quality, plus
#    datasets.unlabelled_ids and the split helpers.
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


# Q: What does this test check?
# A: softmax of random logits multiplied by 50 has rows that sum to 1 and contains only finite numbers.
# Q: Why does it matter?
# A: exp() of a large number overflows to infinity. The implementation subtracts the row maximum first, which gives
#    the same probabilities but is numerically stable.
def test_softmax_rows_sum_to_one():
    """Softmax must turn any logits into probabilities (rows sum to 1), even for very large logits."""
    # Random logits multiplied by 50 make them huge; a careless softmax would overflow to inf/NaN here.
    p = softmax(np.random.default_rng(0).normal(size=(5, 10)) * 50)
    # Each row sums to 1 and every value is a finite number.
    assert np.allclose(p.sum(1), 1) and np.isfinite(p).all()


# Q: What does this test check?
# A: Along the threshold grid, coverage never increases (all consecutive differences <= 0).
# Q: Why does it matter?
# A: A higher tau can only remove images, never add them. A violation would reveal a bug in the table, which
#    choose_threshold uses to pick tau.
# Q: Why random data?
# A: Only the trend matters, not the values; 200 random predictions are enough.
def test_threshold_table_monotone_coverage():
    """Raising the threshold can only remove images, so coverage must never increase along the threshold grid."""
    rng = np.random.default_rng(1)
    # 200 random fake validation predictions and random labels; the values do not matter, only the trend.
    t = threshold_table(softmax(rng.normal(size=(200, 10)) * 3), rng.integers(0, 10, 200))
    # Differences between consecutive coverage values must all be <= 0 (never rising).
    assert (np.diff(t["coverage"]) <= 0).all()

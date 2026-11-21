"""Evaluation tests on fake run directories built from the real test ids (no training)."""

# json/yaml: write the config and metrics files a real run leaves behind; zlib: a stable seed from a run name.
import json
# subprocess/sys: start scripts/evaluate.py as a separate program, like a user would.
import subprocess
import zlib
import sys

# numpy/pandas: arrays of fake predictions and the result tables; pytest: fixtures and approx comparisons.
import numpy as np
import pandas as pd
import pytest
import yaml
# sklearn's f1_score is the trusted reference that our fast macro-F1 must agree with.
from sklearn.metrics import f1_score

# REPO_ROOT: to locate scripts/evaluate.py; split_ids/load_split: the real test-set image ids and labels.
from eurosat_le.constants import REPO_ROOT
from eurosat_le.data.datasets import split_ids
from eurosat_le.data.splits import load_split
# The evaluation functions under test (loading runs, ECE, bootstrap, paired comparison, summary).
from eurosat_le.eval.evaluate import (compare, expected_calibration_error, load_runs, macro_f1_fast,
                                      paired_bootstrap, run_metrics, summarise_comparison)
# classification_metrics: used to write the "logged" metrics of the fake runs.
from eurosat_le.eval.metrics import classification_metrics


# ====================================================================================================
# CONCEPTS USED IN THIS FILE (study notes)
# ====================================================================================================
# - unit test: a check of ONE function on small made-up input: test_macro_f1_fast_matches_sklearn, test_ece,
#     test_paired_bootstrap, test_load_preds_cache and so on.
# - end-to-end (integration) test: runs a whole script like a user would. test_evaluate_script_end_to_end starts
#     scripts/evaluate.py with subprocess and checks its exit code and the files it writes.
# - regression test: pins a known-good result so a later change that alters it is noticed.
#     test_paired_bootstrap_fixed_seed_values pins exact numbers for a fixed seed.
# - assert: a statement that must be true; if false, pytest marks the test as FAILED.
# - fixture (@pytest.fixture): a function that builds shared test input; tests name it as an argument. `runs_dir`
#     builds the 8 fake run folders.
# - tmp_path: built-in pytest fixture: a fresh empty temporary folder (pathlib.Path) per test, so tests never touch
#     results/runs.
# - synthetic / fake data and fake run folder: no model is trained. make_run writes the same files a real run leaves
#     (preds_val.npz, preds_test.npz, config.yaml, metrics.json) but with random fake predictions.
# - logits: the raw output scores of a model, one per class (10 here). The predicted class is the argmax.
# - softmax and argmax: softmax turns logits into probabilities that sum to 1; argmax picks the index of the largest
#     value, i.e. the predicted class. Confidence = the largest probability.
# - accuracy: share of images predicted correctly.
# - macro-F1: F1 per class (harmonic mean of precision and recall), then the plain average over the 10 classes, so
#     rare classes count as much as common ones. macro_f1_fast is a quick numpy version used inside the bootstrap.
# - ECE (expected calibration error): calibration = does 80% confidence mean 80% correct? ECE bins predictions by
#     confidence (15 equal-width bins) and averages |accuracy - confidence| weighted by bin size. 0 = perfect.
# - bootstrap (paired): re-draw the test images many times WITH replacement and recompute the score each time; the
#     spread shows how much a difference could be luck. 'Paired' = the SAME resampled images are used for both runs.
#     n_boot = number of resamples; the 95% interval = 2.5th and 97.5th percentiles (lo, hi).
# - paired comparison by seed: two runs are compared only if everything except the studied factor (and the seed)
#     matches, so they share labelled images, initialisation and test images. delta_f1 = F1(b) - F1(a).
# - align by id: predictions are sorted by image id when loaded, so row i is the same image in every run, even if
#     the data loader ordered the images differently.
# - determinism / seed: a fixed seed (np.random.default_rng(seed), zlib.crc32 of the run name) makes 'random'
#     numbers repeat, so fake runs and bootstrap results are reproducible.
# - float16 vs float32: saved logits are stored in 16-bit floats to save space; they are converted back to float32
#     on loading. Rounding can flip rare ties, hence the tolerance 1e-3 in check_against_logged.
# - cache (load_preds): predictions are read once and kept in memory. The cache key contains path, file modification
#     time (mtime) and size, so a rewritten file is read again. Cached arrays are read-only.
# - subprocess: starts another program (python scripts/evaluate.py ...) and captures its output and exit code
#     (returncode 0 = success).
# - confusion matrix (row-normalised): table of true class (rows) versus predicted class (columns); dividing each
#     row by its total makes every row sum to 1.
# - variant (supervised / pl): 'pl' = pseudo-label student; 'supervised' = plain run with no name suffix.
# - pytest.approx: float comparison with a tolerance.

# ====================================================================================================
# Q&A ABOUT THIS FILE (viva practice)
# ====================================================================================================
# Q: Which part of the project does this file protect?
# A: The evaluation code: eurosat_le/eval/evaluate.py (macro-F1, ECE, bootstrap, pairing of runs, loading and
#    caching of predictions) and the script scripts/evaluate.py that writes the result tables.
# Q: How do you run it?
# A: pytest tests/test_evaluate.py from the repo root. The helper make_run is also imported by
#    tests/test_evaluate_regression.py.
# Q: Why is no training and no image data needed?
# A: Evaluation works only on saved predictions. make_run writes random fake predictions for the REAL val/test image
#    ids, which it reads from the committed splits/split.csv; no pixels and no model are involved.

# Q: What does _fake_logits do?
# A: Builds fake model outputs (n x 10 logits). For each image the 'predicted' class is the true label with
#    probability acc, otherwise a random class; random normal noise is added and the chosen class gets +4 so it
#    becomes (almost always) the argmax.
# Q: Why is it enough?
# A: The evaluation code does not care how the logits were produced, only that accuracy is controllable (about acc)
#    so that better/worse runs can be compared.
def _fake_logits(labels, acc, rng):
    """Helper: make fake model outputs (logits) that are correct for about a fraction `acc` of the images."""
    # For each image, use the true label with probability `acc`, otherwise a random class (may be right by luck).
    pred = np.where(rng.random(len(labels)) < acc, labels, rng.integers(0, 10, len(labels)))
    # Random noise logits for 10 classes ...
    logits = rng.normal(0, 1, (len(labels), 10))
    # ... then raise the chosen class by 4 so that it becomes the arg-max (the fake model's prediction).
    logits[np.arange(len(labels)), pred] += 4
    return logits


# Q: What does this helper do?
# A: Writes one fake finished run folder, e.g. resnet50_rgb_1pct_seed0 (or ..._pl): preds_val.npz and preds_test.npz
#    (float16 logits, labels, ids), config.yaml and metrics.json, with accuracy about acc.
# Q: Why are ids shuffled, and why is the random generator seeded from the run name?
# A: Each run stores its ids in its own order, so the evaluation code must re-align them by id. Seeding from the run
#    name (zlib.crc32) means the same run always gets the same fake predictions, so tests and the regression files
#    are stable.
# Q: Why are the logged metrics computed from float16-rounded logits?
# A: A real training run saves float16 logits and logs metrics from them; copying that makes check_against_logged
#    comparable. Real val/test labels and ids come from load_split()/split_ids().
def make_run(runs_dir, model, modality, fraction, seed, acc, suffix="", rng=None):
    """Helper: write a fake finished run folder (predictions, config, metrics) with roughly accuracy `acc`.

    The folder has the same files a real training run leaves, so the evaluation code can be tested without
    training anything. It is also imported by tests/test_evaluate_regression.py.
    """
    # Seed the random generator from the run's name, so the same run always gets the same fake predictions.
    rng = rng or np.random.default_rng(zlib.crc32(f"{model}{modality}{seed}{suffix}".encode()))
    # Folder name in the same style as real runs, e.g. resnet50_rgb_1pct_seed0 or ..._pl.
    name = f"{model}_{modality}_{fraction * 100:g}pct_seed{seed}" + (f"_{suffix}" if suffix else "")
    d = runs_dir / name
    d.mkdir(parents=True)
    # The real split table, indexed by image id so labels can be looked up by id.
    split = load_split().set_index("id")
    # The metrics.json content starts with the run name; val and test results are added below.
    metrics = {"run_name": name}
    for s in ("val", "test"):
        # The real image ids of this split, shuffled so each run stores them in its own order
        # (the evaluation code has to re-align them by id).
        ids = np.array(split_ids(s))
        ids = ids[rng.permutation(len(ids))]  # each run stores ids in its own order
        # True labels for those ids, and fake logits that match them with accuracy ~acc.
        y = split.loc[ids, "label"].to_numpy()
        logits = _fake_logits(y, acc, rng)
        # Save predictions as float16 (like real runs do) together with labels and ids.
        np.savez(d / f"preds_{s}.npz", logits=logits.astype(np.float16), labels=y, ids=ids)
        # "Logged" metrics are computed from the float16-rounded logits, as a real training run would.
        metrics[s] = classification_metrics(y, logits.astype(np.float16).astype(np.float32).argmax(1))
    # The config.yaml the evaluator reads to know which model/input/fraction/seed/variant this run is.
    cfg = {"model": model, "modality": modality, "fraction": fraction, "seed": seed, "name_suffix": suffix}
    (d / "config.yaml").write_text(yaml.safe_dump(cfg))
    (d / "metrics.json").write_text(json.dumps(metrics))
    return d


# Q: What does the runs_dir fixture build?
# A: A temporary 'runs' folder with 8 fake runs: 2 seeds x 4 settings: ResNet-50 RGB (80%), ResNet-50 13-band (88%),
#    ResNet-50 RGB pseudo-label student 'pl' (84%) and scratch CNN RGB (55%), all at 1% labels.
# Q: Why these settings?
# A: 13-band is clearly better than RGB, pl slightly better than its supervised teacher, scratch worse than ResNet:
#    every comparison has a known winner and two seeds give two pairs. The scratch CNN has no 13-band run, so it
#    must not appear in the RGB-vs-13-band comparison.
@pytest.fixture
def runs_dir(tmp_path):
    """Fixture: a temporary 'runs' folder with 8 fake runs (2 seeds x 4 settings) shared by several tests."""
    rd = tmp_path / "runs"
    # Two seeds per setting, so seed-matched comparisons have two pairs.
    for seed in (0, 1):
        # ResNet-50 on RGB (80% accurate) versus 13 bands (88%): the 13-band run is clearly better.
        make_run(rd, "resnet50", "rgb", 0.01, seed, 0.80)
        make_run(rd, "resnet50", "ms", 0.01, seed, 0.88)
        # A pseudo-label student ("pl") that is slightly better (84%) than its supervised RGB teacher.
        make_run(rd, "resnet50", "rgb", 0.01, seed, 0.84, suffix="pl")
        # A weak scratch CNN (55%), so model comparisons have a clear winner.
        make_run(rd, "scratch_cnn", "rgb", 0.01, seed, 0.55)
    return rd


# Q: What does this test check?
# A: macro_f1_fast (bincount confusion matrix) equals scikit-learn's f1_score(average='macro') on 500 random labels
#    and predictions over 10 classes.
# Q: Why does it matter?
# A: The bootstrap calls macro_f1_fast thousands of times, so it is a faster re-implementation. This test proves the
#    speed-up did not change the number.
def test_macro_f1_fast_matches_sklearn():
    """Our fast macro-F1 (used inside the bootstrap) must give the same number as scikit-learn."""
    # 500 random true and predicted labels over 10 classes.
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 10, 500), rng.integers(0, 10, 500)
    # Compare with sklearn's macro F1 over all 10 classes.
    assert macro_f1_fast(y, p) == pytest.approx(f1_score(y, p, average="macro", labels=range(10)))


# Q: What does this test check?
# A: ECE is 0 for a perfectly confident and always correct model, and about 0.41 for a model that is 91% confident
#    but only 50% correct.
# Q: How are the two cases built?
# A: Perfect: true labels all 0 and probabilities one-hot on class 0 (confidence 1.0, accuracy 1.0 -> gap 0).
#    Overconfident: probability 0.91 on class 0 for every image, but half the true labels are 1, so accuracy 0.50
#    and the gap is 0.91 - 0.50 = 0.41.
# Q: Which function does it exercise?
# A: expected_calibration_error(probs, labels, n_bins=15) in eval/evaluate.py.
def test_ece():
    """Expected calibration error: 0 for a perfectly confident and correct model, large for an overconfident one."""
    # Every true label is class 0; the "perfect" model puts probability 1 on class 0 every time.
    y = np.zeros(1000, int)
    perfect = np.eye(10)[y]
    # Confidence equals accuracy (both 100%), so the calibration error must be 0.
    assert expected_calibration_error(perfect, y) == pytest.approx(0)
    # An overconfident model: 91% probability on class 0, 1% on the others, for every image.
    over = np.full((1000, 10), 0.01)
    over[:, 0] = 0.91
    wrong = np.r_[np.zeros(500, int), np.ones(500, int)]  # 50% correct at 91% confidence
    # Gap between confidence (0.91) and accuracy (0.50) is about 0.41, which is what ECE should report.
    assert expected_calibration_error(over, wrong) == pytest.approx(0.41, abs=0.01)


# Q: What does this test check?
# A: Comparing a run with itself gives exactly (0.0, 0.0, 0.0); a run that fixes half of the mistakes gives a
#    positive difference whose whole 95% interval is above 0 and which lies inside its interval.
# Q: Why does it matter?
# A: It shows the bootstrap reports 'no difference' when there is none and detects a real improvement; the interval
#    is what lets the report say a gain is not just luck.
# Q: How is the fake input built?
# A: 2,000 random true labels; prediction p is right about 70% of the time; 'better' takes the true label for about
#    half of the images and otherwise keeps p's answer. n_boot=200 keeps the test fast.
def test_paired_bootstrap():
    """The paired bootstrap must report no difference for identical runs and a positive one for a better run."""
    # Fake truth y, and predictions p that are right ~70% of the time.
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 2000)
    p = np.where(rng.random(2000) < 0.7, y, rng.integers(0, 10, 2000))
    # Comparing a run with itself: difference and both interval ends must be exactly 0.
    assert paired_bootstrap(y, p, p, n_boot=200) == (0.0, 0.0, 0.0)
    # A "better" run: it takes the true label on half of the images, otherwise keeps p's answer.
    better = np.where(rng.random(2000) < 0.5, y, p)
    d, lo, hi = paired_bootstrap(y, p, better, n_boot=200)
    # Improvement is positive, the whole interval is above 0, and the estimate lies inside its interval.
    assert d > 0 and lo > 0 and lo <= d <= hi


# Q: What does this test check?
# A: With seed 3 and 300 resamples the delta and both interval ends equal three stored numbers (to 1e-9).
# Q: Why does it matter?
# A: paired_bootstrap was rewritten to process resamples in batches for speed. These values came from the original
#    per-resample loop, so passing proves the faster code draws the same resamples and gives the same results.
# Q: What is the weakness of such a test?
# A: If the algorithm is changed on purpose, the numbers must be updated deliberately; the test only guards
#    equivalence, not correctness of the statistics.
def test_paired_bootstrap_fixed_seed_values():
    """Pins the exact output for a fixed seed, so a faster implementation must give the same numbers."""
    # Small fake test set: run a is about 70% correct, run b fixes about half of a's mistakes.
    rng = np.random.default_rng(1)
    y = rng.integers(0, 10, 500)
    a = np.where(rng.random(500) < 0.7, y, rng.integers(0, 10, 500))
    b = np.where(rng.random(500) < 0.5, y, a)
    # Fixed bootstrap seed (3) and 300 resamples, so the output is fully deterministic.
    d, lo, hi = paired_bootstrap(y, a, b, n_boot=300, seed=3)
    # Expected values were produced by the original per-resample loop (before the batching change).
    # If a refactor changes these numbers, the new code is not equivalent to the old one.
    assert d == pytest.approx(0.137514591926854, abs=1e-9)
    assert lo == pytest.approx(0.10746943016001925, abs=1e-9)
    assert hi == pytest.approx(0.16831328287475714, abs=1e-9)


# Q: What does this test check?
# A: compare(runs, 'modality', 'rgb', 'ms') returns 2 pairs (seeds 0 and 1, model resnet50 only) with a positive
#    delta_f1; summarise_comparison reports n_seeds 2 and seeds_b_better 2; compare(runs, 'variant', 'supervised',
#    'pl') gives 2 RGB pairs.
# Q: Why does it matter?
# A: Pairing only runs that match in model, fraction, variant and seed makes the comparison fair. The scratch CNN
#    (no 13-band run) and the 'pl' runs must not be paired with the wrong partner.
# Q: Which functions does it exercise?
# A: load_runs (reads config.yaml of every run), compare (aligns images by sorted id, bootstrap per pair) and
#    summarise_comparison (mean/std over seeds, count of seeds where b is better).
def test_compare_aligns_ids_and_pairs_by_seed(runs_dir):
    """compare() must match runs by seed (and all other settings), align images by id, and fill the table."""
    # Read all fake runs, then compare RGB (a) with 13 bands (b) for every matching seed.
    runs = load_runs(runs_dir)
    pairs = compare(runs, "modality", "rgb", "ms", n_boot=100)
    # Two seeds give two pairs, both ResNet-50 (scratch CNN has no 13-band run, so it is not paired).
    assert len(pairs) == 2 and set(pairs["seed"]) == {0, 1} and (pairs["model"] == "resnet50").all()
    # The 13-band fake runs are more accurate, so the F1 gain must be positive for every pair.
    assert (pairs["delta_f1"] > 0).all()
    # The summary collapses seeds: 2 seeds seen, and 13 bands won in both.
    summ = summarise_comparison(pairs)
    assert summ.loc[0, "n_seeds"] == 2 and summ.loc[0, "seeds_b_better"] == 2
    # Second comparison: plain supervised runs versus their pseudo-label students; only RGB has such pairs.
    pl = compare(runs, "variant", "supervised", "pl", n_boot=100)
    assert len(pl) == 2 and (pl["modality"] == "rgb").all()


# Q: What does this test check?
# A: For every fake run, metrics recomputed from the saved predictions (softmax, argmax) agree with the metrics.json
#    written by the 'training': check_against_logged returns an empty list.
# Q: Why does it matter?
# A: evaluate.py recomputes every number from predictions instead of trusting the log. This cross-check catches
#    mismatched files or a bug in the recomputation (tolerance 1e-3 on accuracy and macro-F1).
def test_recomputed_metrics_match_logged(runs_dir):
    """Metrics recomputed from saved predictions must agree with the metrics.json written by the run."""
    # check_against_logged is imported here because only this test uses it.
    from eurosat_le.eval.evaluate import check_against_logged

    # For each fake run, recompute accuracy/F1 from the predictions; no mismatch messages may be returned.
    for d in sorted(runs_dir.iterdir()):
        assert check_against_logged(d, run_metrics(d)) == []


# Q: What does this test check?
# A: (1) Two calls to load_preds return the very same array objects (cache hit). (2) The arrays are read-only:
#    writing raises ValueError. (3) After the file is rewritten with all-zero logits (and mtime moved +1 s) the next
#    call returns the new data.
# Q: Why does it matter?
# A: A shared cache without invalidation would silently serve stale predictions after a re-run; a writable cache
#    could be corrupted by one caller and change everyone's results.
# Q: Why does it change the modification time by hand?
# A: A fast rewrite can leave the same timestamp, and the cache key uses (path, mtime, size); os.utime guarantees
#    that the key changes. cache_clear() at the start and end keeps other tests unaffected.
def test_load_preds_cache(tmp_path):
    """Predictions are read once, shared read-only, and re-read when the file is rewritten."""
    # os: to change the file's modification time; load_preds: the cached reader under test.
    import os

    from eurosat_le.eval.evaluate import load_preds

    # One fake run, and an empty cache so earlier tests cannot affect this one.
    d = make_run(tmp_path / "runs", "resnet50", "rgb", 0.01, 0, 0.80)
    load_preds.cache_clear()  # start empty, so earlier tests cannot affect this one
    # Read twice: every returned array must be the very same object the second time (cache hit, no re-read).
    first = load_preds(d)
    assert all(a is b for a, b in zip(first, load_preds(d)))  # second call hits the cache
    # The cached arrays are shared, so writing to one must be refused to protect other users of the cache.
    with pytest.raises(ValueError):
        first[0][0, 0] = 1.0  # shared arrays are read-only
    # Rewrite the file with all-zero logits; the next load must return the new contents.
    f = d / "preds_test.npz"
    data = dict(np.load(f))
    data["logits"] = np.zeros_like(data["logits"])
    np.savez(f, **data)
    # Push the modification time one second forward, because a fast rewrite can leave the time unchanged.
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # make sure the mtime changes
    # The reader must notice the change and return the new all-zero logits (any() is False for all zeros).
    assert not load_preds(d)[0].any()
    load_preds.cache_clear()  # leave no cached arrays behind for other tests


# Q: What does this test check?
# A: scripts/evaluate.py, run as a separate process on the 8 fake runs, exits with code 0, prints no WARNING, and
#    writes groups.csv (4 settings, 2 seeds each), per_class.csv (4 x 10 rows), three comparison CSVs and a
#    confusion CSV whose rows sum to 1.
# Q: Why does it matter?
# A: Unit tests can pass while the script's wiring (arguments, file names, grouping) is broken; this catches that.
#    'No WARNING' means the recomputed metrics matched the logged ones.
# Q: Why subprocess?
# A: It runs the script exactly as a user would from the command line (--runs-dir, --out, --n-boot 100), with
#    stdout/stderr captured so a failure message is shown by the assert.
def test_evaluate_script_end_to_end(runs_dir, tmp_path):
    """Run scripts/evaluate.py on the fake runs and check that it succeeds and writes all expected result files."""
    out = tmp_path / "eval"
    # Start the real script as a separate process, pointing it at the fake runs and a temporary output folder.
    res = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "evaluate.py"), "--runs-dir", str(runs_dir),
                          "--out", str(out), "--n-boot", "100"], capture_output=True, text=True)
    # Exit code 0 means success; otherwise show the script's error output in the failure message.
    assert res.returncode == 0, res.stderr
    # The script prints WARNING when recomputed metrics disagree with the logged ones; there must be none.
    assert "WARNING" not in res.stdout
    # groups.csv: one row per setting (4 settings), each averaged over 2 seeds.
    groups = pd.read_csv(out / "groups.csv")
    assert len(groups) == 4 and (groups["n_seeds"] == 2).all()
    # per_class.csv: 10 class rows for each of the 4 settings.
    assert len(pd.read_csv(out / "per_class.csv")) == 4 * 10
    # The three comparison tables (RGB vs 13 bands, supervised vs pseudo-label, scratch vs ResNet) must exist.
    assert (out / "compare_rgb_vs_ms.csv").exists() and (out / "compare_supervised_vs_pl.csv").exists()
    assert (out / "compare_scratch_vs_resnet50.csv").exists()
    # A confusion matrix is row-normalised, so every row (one true class) must add up to 1.
    cm = pd.read_csv(out / "confusion" / "resnet50_ms_1pct_supervised.csv", index_col=0)
    assert np.allclose(cm.sum(axis=1), 1)

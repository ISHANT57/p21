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


def _fake_logits(labels, acc, rng):
    """Helper: make fake model outputs (logits) that are correct for about a fraction `acc` of the images."""
    # For each image, use the true label with probability `acc`, otherwise a random class (may be right by luck).
    pred = np.where(rng.random(len(labels)) < acc, labels, rng.integers(0, 10, len(labels)))
    # Random noise logits for 10 classes ...
    logits = rng.normal(0, 1, (len(labels), 10))
    # ... then raise the chosen class by 4 so that it becomes the arg-max (the fake model's prediction).
    logits[np.arange(len(labels)), pred] += 4
    return logits


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


def test_macro_f1_fast_matches_sklearn():
    """Our fast macro-F1 (used inside the bootstrap) must give the same number as scikit-learn."""
    # 500 random true and predicted labels over 10 classes.
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 10, 500), rng.integers(0, 10, 500)
    # Compare with sklearn's macro F1 over all 10 classes.
    assert macro_f1_fast(y, p) == pytest.approx(f1_score(y, p, average="macro", labels=range(10)))


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


def test_recomputed_metrics_match_logged(runs_dir):
    """Metrics recomputed from saved predictions must agree with the metrics.json written by the run."""
    # check_against_logged is imported here because only this test uses it.
    from eurosat_le.eval.evaluate import check_against_logged

    # For each fake run, recompute accuracy/F1 from the predictions; no mismatch messages may be returned.
    for d in sorted(runs_dir.iterdir()):
        assert check_against_logged(d, run_metrics(d)) == []


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

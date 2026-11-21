"""Evaluation from saved predictions (Phase 6). Nothing here retrains or re-runs a model.

Every finished run directory holds preds_{val,test}.npz (logits, labels, ids). This
module rebuilds all metrics from those files, groups runs over seeds, and builds
paired comparisons (same model, fraction and seed; one factor changed).

It is the library behind `scripts/evaluate.py` (see docs/EVALUATION.md). Main pieces:

- `load_runs` / `load_preds`: discover finished runs and load their saved logits. Predictions are sorted
  by sample id on load, so two runs can be compared image by image no matter how their data loaders
  ordered the test set.
- `run_metrics` / `check_against_logged`: metrics are recomputed from the saved logits (softmax, then
  argmax) and compared with the metrics.json written at training time (tolerance 1e-3 on accuracy and
  macro-F1).
- `expected_calibration_error`: ECE with 15 equal-width confidence bins.
- `compare` / `paired_bootstrap` / `summarise_comparison`: paired comparisons. Two runs are paired only
  when model, modality, fraction, variant (whichever is not the compared factor) AND seed all match, so
  they share the labelled subset, the initialisation seed and the test images. The 95% interval comes
  from resampling test images (1,000 times) and recomputing macro-F1 for both runs on the same resample.
  Results over the 3 seeds are summarised as counts ("b better in k/3 seeds"), not p-values.
- `mean_confusion` / `top_confusions`: seed-averaged, row-normalised confusion matrices and their
  largest off-diagonal entries.

Note: train.engine saves the logits as float16 (np.savez_compressed). They are cast back to float32 here
before softmax. Accuracy and macro-F1 can therefore differ from metrics.json in rare tie cases, which is
what the 1e-3 tolerance in `check_against_logged` allows for.
"""

# Makes type hints such as `int | None` work as plain text on older Python versions.
from __future__ import annotations

# json: read metrics.json; Path: file paths.
import json
from pathlib import Path

# numpy: array maths; pandas: result tables; yaml: read each run's config.yaml.
import numpy as np
import pandas as pd
import yaml

# CLASSES: the 10 class names; classification_metrics: accuracy / F1 / confusion matrix report;
# softmax: turns logits into probabilities.
from eurosat_le.constants import CLASSES
from eurosat_le.eval.metrics import classification_metrics
from eurosat_le.ssl.pseudo_label import softmax

# ======================================== CONCEPTS USED IN THIS FILE ========================================
# saved predictions (preds_val.npz / preds_test.npz): per run, the logits, true labels and image ids saved by
#   train/engine.py; all evaluation is rebuilt from them, nothing is retrained.
# logits -> softmax -> argmax: logits are raw scores; softmax makes probabilities; the predicted class is the argmax.
#   Confidence = the largest probability.
# float16 / float32: logits were saved in half precision; they are cast back to float32 here. This can flip rare ties,
#   hence the 1e-3 tolerance in check_against_logged.
# validation vs test set: val is used for selection during training; test is the final, untouched score. Evaluation here
#   reports both, comparisons use the test set.
# sorting by sample id: predictions are reordered by id when loaded, so two runs can be compared image by image no
#   matter how their loaders ordered the test set.
# cache keyed by (path, mtime, size): load_preds keeps the arrays in memory after the first read; a rewritten file
#   changes the key and is re-read. Cached arrays are read-only (setflags write=False) so one caller cannot corrupt
#   another's results.
# accuracy, macro-F1, confusion matrix: see eval/metrics.py. macro-F1 = mean of the per-class F1; confusion matrix rows
#   = true class, columns = predicted class.
# confusion matrix via bincount: each image is encoded as true*K + predicted (one number per matrix cell) and
#   np.bincount counts them; K = 10 classes.
# ECE (expected calibration error): how far confidence is from actual accuracy. Predictions are binned by confidence (15
#   equal-width bins); ECE = sum over bins of (share of images in bin) x |accuracy in bin - mean confidence in bin|. 0 =
#   perfectly calibrated.
# calibration: a model is calibrated if, among predictions made with e.g. 80% confidence, about 80% are right.
# paired comparison: two runs are paired when everything but the compared factor (model, modality, fraction, or variant)
#   AND the seed match, so they share the labelled subset, the seed and the test images; then differences are not due to
#   other changes.
# variant: "supervised" for a plain run or the name_suffix of a run (e.g. "pl" for a pseudo-label student).
# bootstrap / paired bootstrap / confidence interval: resample the test images with replacement (1,000 times), recompute
#   macro-F1 for both runs on the SAME resample, and take the 2.5th and 97.5th percentiles of the differences as a 95%
#   interval. It shows how much a difference could be luck.
# delta_f1: macro-F1 of run b minus macro-F1 of run a. If the interval excludes 0 the difference is unlikely to be
#   noise.
# seed-level summary: over the 3 seeds results are reported as counts ("b better in k of 3 seeds", "interval excludes 0
#   in k seeds") with mean/std/min/max, not as p-values.
# sample std (ddof=1): standard deviation with n-1 in the denominator (pandas default); NaN if a group has one seed.
# random Generator with fixed seed: np.random.default_rng(seed) makes the bootstrap intervals reproducible.
# vectorised batch of resamples: _BOOT_BATCH = 100 resamples are drawn and scored at once with one rng.integers and one
#   bincount, which is much faster than a Python loop and gives the same numbers.
# seed-averaged, row-normalised confusion matrix: each run's matrix is divided by its row totals (share of each true
#   class) and then averaged over seeds, so every seed counts equally.
# pandas DataFrame / groupby / agg: tables of runs and results; groupby groups rows and agg computes
#   mean/std/min/max/sum per group.
# YAML / JSON: config.yaml (the run's settings) and metrics.json (metrics written at training time) are read here.

# ======================================== Q&A ABOUT THIS FILE ========================================
# Q: What is this file for?
# A: It is the evaluation library (Phase 6): it finds finished runs, reloads their saved predictions, recomputes metrics
#   (including ECE), builds paired comparisons with bootstrap intervals and seed-averaged confusion matrices.
# Q: Where does it fit in the pipeline?
# A: After training. train/engine.py writes preds_*.npz and metrics.json; this module reads them; scripts/evaluate.py is
#   the command-line front end that uses these functions to make the report tables.
# Q: What does it read and write?
# A: It reads each run folder's config.yaml, metrics.json and preds_val/test.npz. It writes no files itself and retrains
#   nothing; it returns DataFrames, dicts and arrays.
# Q: Why recompute metrics from saved logits instead of trusting metrics.json?
# A: It makes the evaluation independent of the training code path and lets us run new analyses (ECE, paired bootstrap,
#   confusion matrices) without retraining; check_against_logged then confirms both agree.

# K = number of classes (10); used as the size of confusion matrices.
K = len(CLASSES)


# Q: What does variant_of do?
# A: It maps a run's name_suffix to a label: no suffix -> "supervised", otherwise the suffix itself (e.g. "pl", "pl2").
# Q: Why is it needed?
# A: Pseudo-label students share model/modality/fraction/seed with their teacher, so "variant" is the key that
#   distinguishes them in comparisons.
def variant_of(suffix: str | None) -> str:
    """Map a run's config `name_suffix` to a variant label; no suffix means the plain supervised run."""
    # An empty or missing suffix becomes "supervised"; any other suffix (e.g. "pl") is its own variant.
    return suffix or "supervised"


# Q: What does load_runs return?
# A: A DataFrame with one row per finished run that has test predictions: run_name, dir, model, modality, fraction,
#   seed, variant and teacher.
# Q: Why search for metrics.json and require preds_test.npz?
# A: metrics.json is written last, so it marks a complete run; tuning runs have no test predictions
#   (evaluate_test=False) and are left out, so they can never enter the evaluation.
# Q: What if nothing qualifies?
# A: It returns an empty DataFrame with no columns.
def load_runs(runs_dir: Path) -> pd.DataFrame:
    """One row per finished run that has test predictions (tuning runs have none).

    Columns: run_name, dir (Path), model, modality, fraction, seed, variant (see `variant_of`), teacher
    (config value, None for runs without a teacher). Rows follow the sorted order of the run folders.
    Returns an empty DataFrame (no columns) when nothing qualifies.
    """
    rows = []
    # Every finished run has a metrics.json (it is written last), so search for those.
    for m in sorted(runs_dir.glob("*/metrics.json")):
        d = m.parent
        # Tuning runs never predicted the test set, so they are left out.
        if not (d / "preds_test.npz").exists():
            continue
        # Read the settings the run was trained with and record the ones that identify its cell.
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        rows.append({"run_name": d.name, "dir": d, "model": cfg["model"], "modality": cfg["modality"],
                     "fraction": cfg["fraction"], "seed": cfg["seed"],
                     "variant": variant_of(cfg.get("name_suffix")), "teacher": cfg.get("teacher")})
    return pd.DataFrame(rows)


# Q: What does load_preds do?
# A: It loads preds_<split>.npz of a run and returns (logits float32 [N, K], labels [N], ids [N] as strings), all sorted
#   by sample id.
# Q: Why sort by id?
# A: Different runs' data loaders may order the test set differently; sorting makes row i the same image in every run,
#   which paired comparisons need.
# Q: Why the cache, and why is the key (path, mtime, size)?
# A: One evaluation reads the same file many times (metrics, confusion matrix, every pairing). Keeping arrays in memory
#   saves time; including mtime and size means a re-written file is read again instead of returning stale data.
# Q: Why are the returned arrays read-only?
# A: They are shared between callers; making them read-only stops one caller's in-place change from silently altering
#   the results of the others.
def load_preds(run_dir: Path, split: str = "test") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load preds_<split>.npz and return (logits [N, K] float32, labels [N], ids [N] as str).

    All three arrays are reordered by sorted string id so that different runs line up row by row.

    Caching: one evaluation reads the same file several times (per-run metrics, confusion matrix, and
    every paired comparison that involves the run), so the sorted arrays are kept in memory after the
    first read. The cache key includes the file's modification time and size, so a file that is
    rewritten (for example a re-run in a notebook session) is read again. The returned arrays are shared
    between callers and therefore read-only; copy them before modifying. `load_preds.cache_clear()`
    empties the cache. Memory: about N * K * 4 bytes per split and run (160 KB for the test set).
    """
    # Locate the file and read its size / modification time.
    path = Path(run_dir) / f"preds_{split}.npz"
    st = path.stat()
    # Cache key: the absolute path, plus mtime and size. If the file is rewritten, the key changes and
    # the new contents are read instead of stale cached arrays.
    key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
    if key not in _PREDS_CACHE:  # first request for this file: read it from disk
        d = np.load(path)
        order = np.argsort(d["ids"].astype(str))  # align runs by id, independent of loader order
        # Reorder logits, labels and ids with the same permutation so they stay matched row by row.
        arrays = (d["logits"].astype(np.float32)[order], d["labels"][order], d["ids"].astype(str)[order])
        # Every caller gets the same array objects, so make them read-only: an in-place change by one
        # caller would otherwise silently change the results of every later caller.
        for a in arrays:
            a.setflags(write=False)
        _PREDS_CACHE[key] = arrays
    return _PREDS_CACHE[key]


# (path, mtime_ns, size) -> (logits, labels, ids), filled by load_preds. A plain dict instead of
# functools.lru_cache, because the key needs the file's stat and not only the function arguments.
_PREDS_CACHE: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
load_preds.cache_clear = _PREDS_CACHE.clear  # same name as lru_cache's method, for tests and notebooks


# Q: What does this function compute?
# A: The expected calibration error: bin images by confidence, and sum over bins (share of images in bin) x |accuracy in
#   bin - average confidence in bin|. 0 means perfectly calibrated.
# Q: Why is calibration worth measuring?
# A: A model can be accurate yet over-confident; ECE shows whether its confidence can be trusted.
# Q: Why np.minimum(..., n_bins - 1)?
# A: A confidence of exactly 1.0 would give bin index n_bins, which does not exist; the minimum puts it in the last bin.
def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error of softmax probabilities `probs` [N, K] against integer `labels` [N].

    Samples are binned by top-class confidence into `n_bins` equal-width bins; ECE is the sum over bins
    of (fraction of samples in the bin) * |accuracy in bin - mean confidence in bin|. Empty bins add 0.

    A model is "well calibrated" when, among predictions made with e.g. 80% confidence, about 80% are
    right; ECE measures how far from that ideal the model is (0 = perfect).
    """
    # Confidence = highest probability of each image; pred = the class that has it.
    conf, pred = probs.max(1), probs.argmax(1)
    # Which of the equal-width confidence bins each image falls into (0 .. n_bins - 1).
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)  # conf == 1.0 would otherwise index bin n_bins
    ece = 0.0
    for b in range(n_bins):
        # Images in this bin.
        m = bins == b
        if m.any():
            # Weight (share of all images) x gap between the bin's actual accuracy and its mean confidence.
            ece += m.mean() * abs((pred[m] == labels[m]).mean() - conf[m].mean())
    return float(ece)


# Q: What does macro_f1_fast do and how is it different from sklearn's f1_score?
# A: It returns the same macro-F1 (zero_division=0, all 10 classes) but computes it from a bincount confusion matrix
#   with F1 = 2TP / (predicted count + true count), so it is much faster.
# Q: Why is speed important here?
# A: The bootstrap calls a macro-F1 computation about 2,000 times per paired comparison; a slower sklearn call each time
#   would add up.
# Q: How is the confusion matrix built?
# A: Each image is encoded as true*K + predicted, np.bincount counts the codes, and reshape(K, K) gives rows = true,
#   columns = predicted.
def macro_f1_fast(y: np.ndarray, p: np.ndarray) -> float:
    """Macro-F1 from integer arrays `y` (true) and `p` (predicted) using only a bincount confusion matrix.

    Same quantity as sklearn's macro F1 with zero_division=0 over all K classes, but much faster, which
    matters inside the bootstrap loop (about 2,000 calls per paired comparison).
    """
    # Encode each image as true*K + predicted (a unique number per matrix cell) and count the occurrences.
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)  # rows = true, cols = predicted
    # True positives per class = the diagonal (correct predictions).
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)  # F1 = 2TP / (2TP + FP + FN) = 2TP / (predicted count + true count)
    # Per-class F1; classes with an empty denominator get 0 instead of a division error.
    f1 = np.divide(2 * tp, denom, out=np.zeros(K), where=denom > 0)
    # Macro = plain average over the classes.
    return float(f1.mean())


# Q: What does run_metrics return?
# A: {"val": m, "test": m} where each m is the classification_metrics dict plus an "ece" entry, all recomputed from the
#   run's saved logits.
# Q: Why softmax before argmax if argmax of logits is the same?
# A: The prediction is identical, but ECE needs the probabilities (confidence), so softmax is applied once and reused.
def run_metrics(run_dir: Path) -> dict:
    """Recompute metrics for one run from its saved predictions.

    Returns {"val": m, "test": m} where each m is the `classification_metrics` dict plus an "ece" entry.
    """
    out = {}
    for split in ("val", "test"):
        # Saved logits and true labels of this split.
        logits, y, _ = load_preds(run_dir, split)
        # Softmax gives probabilities; the predicted class is the one with the highest probability.
        probs = softmax(logits)
        met = classification_metrics(y, probs.argmax(1))
        # Add the calibration error to the usual metrics.
        met["ece"] = expected_calibration_error(probs, y)
        out[split] = met
    return out


# Q: What does check_against_logged do?
# A: It compares the recomputed accuracy and macro-F1 (val and test) with the values in the run's metrics.json and
#   returns a list of mismatch messages; an empty list means they agree.
# Q: Why is there a tolerance of 1e-3?
# A: Logits were saved as float16, so in rare ties the argmax can differ slightly from the training-time float32 result;
#   the tolerance allows for that but still catches real bugs.
# Q: Why is this check useful?
# A: It proves the evaluation pipeline reproduces what training logged, so the reported numbers are trustworthy.
def check_against_logged(run_dir: Path, recomputed: dict, tol: float = 1e-3) -> list[str]:
    """Compare with metrics.json written at training time (fp16 logits can flip rare ties).

    Checks accuracy and macro-F1 on val and test. Returns a list of human-readable mismatch messages
    (differences larger than `tol`, absolute); an empty list means the run agrees with its log.
    """
    # The numbers computed during training.
    logged = json.loads((run_dir / "metrics.json").read_text())
    issues = []
    for split in ("val", "test"):
        for k in ("accuracy", "macro_f1"):
            # Flag the metric if the recomputed value differs from the logged one by more than `tol`.
            if abs(logged[split][k] - recomputed[split][k]) > tol:
                issues.append(f"{run_dir.name} {split} {k}: logged {logged[split][k]:.4f} "
                              f"vs recomputed {recomputed[split][k]:.4f}")
    return issues


# Q: What does paired_bootstrap compute?
# A: The difference in macro-F1 (run b minus run a) on the same test images, and a 95% bootstrap interval for it:
#   returns (delta, ci_low, ci_high).
# Q: What makes it "paired"?
# A: Each resample draws N image indices with replacement and applies the SAME indices to both runs, so the comparison
#   is on identical images and the luck of the draw cancels out.
# Q: Why use a bootstrap rather than a simple difference?
# A: A single difference does not show how reliable it is. Re-drawing the test set many times shows how much the
#   difference varies; if the interval excludes 0, the gain is unlikely to be luck.
# Q: Why a fixed seed and batches of resamples?
# A: The fixed seed makes intervals reproducible. Processing 100 resamples at once (one rng.integers, one bincount) is
#   faster than a plain loop and gives the same numbers, independent of the batch size.
def paired_bootstrap(y, pred_a, pred_b, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Macro-F1(b) - macro-F1(a) on the same test images, with a 95% bootstrap interval.

    `y`, `pred_a`, `pred_b` are aligned integer arrays over the same images. Each bootstrap draw samples
    N image indices with replacement and applies the SAME indices to both runs (that is what makes it a
    paired bootstrap). Returns (delta, ci_low, ci_high); the interval is the 2.5th/97.5th percentile of
    the bootstrap deltas, and `delta` is the point estimate on the original test set. The fixed `seed`
    keeps the intervals reproducible.

    Resamples are processed in batches of `_BOOT_BATCH`: the indices of a whole batch are drawn with one
    `rng.integers` call, the confusion counts come from one `np.bincount`, and macro-F1 is computed for the
    batch at once. NumPy's Generator fills an [B, N] request element by element from the same bit stream
    as B separate [N] requests, so the draws (and the result for a given seed) are the same as in a plain
    per-resample loop and do not depend on the batch size; tests/test_evaluate.py pins the values. Extra
    memory is about 3 * `_BOOT_BATCH` * N int64 values (about 10 MB for the 4,050 test images).

    Bootstrap idea: re-draw the test set many times (with replacement) and see how much the score
    difference changes; the spread of those differences shows how much of it could be luck.
    """
    # Random number generator with a fixed seed, so the same interval comes out every time.
    rng = np.random.default_rng(seed)
    delta = macro_f1_fast(y, pred_b) - macro_f1_fast(y, pred_a)  # point estimate on the real test set
    # Make sure all three inputs are NumPy arrays, and count the test images.
    y, pred_a, pred_b = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    n = len(y)
    # Encode each image as one number: its cell (row = true class, column = predicted) in the K x K
    # confusion matrix. A resample then only needs to gather these codes and count them.
    code_a, code_b = y * K + pred_a, y * K + pred_b  # cell of the confusion matrix for each image
    boots = np.empty(n_boot)  # one macro-F1 difference per resample
    for start in range(0, n_boot, _BOOT_BATCH):
        stop = min(start + _BOOT_BATCH, n_boot)  # the last batch may be smaller
        # Row r holds the n image indices of resample start + r (drawn with replacement). The SAME rows
        # are used for both runs, which is what makes the bootstrap paired.
        idx = rng.integers(0, n, (stop - start, n))
        # Macro-F1 of run b minus run a on every resample in this batch.
        boots[start:stop] = _batched_macro_f1(code_b[idx]) - _batched_macro_f1(code_a[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])  # 95% interval from the middle 95% of the resamples
    return float(delta), float(lo), float(hi)


# Resamples per batch: large enough that the Python loop runs only n_boot / 100 times, small enough that a
# batch of the 4,050 test images stays at a few MB.
_BOOT_BATCH = 100


# Q: What does _batched_macro_f1 do?
# A: It computes macro-F1 for each of B resamples at once, from an array of codes [B, N] where code = true*K +
#   predicted, and returns B values.
# Q: How does one bincount handle all B resamples?
# A: Row r is shifted by r*K*K, so the counts of each resample fall into their own block of K*K bins; the result is
#   reshaped to B confusion matrices.
# Q: What happens to a class that is absent from both truth and prediction?
# A: Its F1 is set to 0 (np.divide with where=denom > 0), matching sklearn's zero_division=0.
def _batched_macro_f1(codes: np.ndarray) -> np.ndarray:
    """Macro-F1 for each row of `codes` [B, N], where a code is `true * K + predicted`.

    Same formula as `macro_f1_fast`, applied to B resamples with a single bincount (row b gets offset
    b * K * K). Returns a float array of length B.
    """
    # Number of resamples in this batch.
    b = len(codes)
    # Shift row r by r * K * K, so the counts of every resample land in their own block of K * K bins.
    offset = (np.arange(b) * K * K)[:, None]
    # One bincount for the whole batch, reshaped to B confusion matrices [B, K, K] (rows = true class).
    cm = np.bincount((codes + offset).ravel(), minlength=b * K * K).reshape(b, K, K)
    tp = np.diagonal(cm, axis1=1, axis2=2).astype(float)  # correct predictions per class, per resample
    denom = cm.sum(1) + cm.sum(2)  # predicted count + true count, per class
    # F1 = 2TP / (predicted + true); a class absent from both gets F1 = 0 (as sklearn's zero_division=0).
    f1 = np.divide(2 * tp, denom, out=np.zeros((b, K)), where=denom > 0)
    return f1.mean(1)  # macro-F1 = unweighted mean over the K classes


# Q: What does compare do?
# A: It finds pairs of runs that differ only in `factor` (a vs b) and share the other keys and the seed, and for each
#   pair computes accuracy, macro-F1 for both, delta_f1 and its bootstrap interval; returns one row per pair.
# Q: Why must everything else, including the seed, match?
# A: Then the pair shares the labelled subset, initialisation seed and test images, so the difference can be blamed on
#   the one factor changed.
# Q: Why the assert on test ids and labels?
# A: As a safety check that both runs were evaluated on exactly the same images; comparing different images would be
#   meaningless.
# Q: Which factors are allowed?
# A: "model", "modality", "fraction" or "variant" (e.g. supervised vs pl).
def compare(runs: pd.DataFrame, factor: str, a, b, n_boot: int = 1000) -> pd.DataFrame:
    """Paired comparisons where only `factor` differs (a -> b); all other keys and the seed match.

    `factor` is one of "model", "modality", "fraction", "variant"; `a` and `b` are its two values. Rows of
    `runs` (from `load_runs`) are matched on the remaining keys plus the seed, so only seeds present in
    both settings are compared. Returns one row per matched pair with accuracy and macro-F1 for each side,
    `delta_f1` (= f1_b - f1_a) and its bootstrap interval; an empty DataFrame if nothing matches.
    Raises AssertionError if a pair does not share identical test ids and labels.
    Assumes (model, modality, fraction, variant, seed) is unique within each side.
    """
    # The columns that must be identical for two runs to count as a pair (everything except the factor).
    keys = [k for k in ("model", "modality", "fraction", "variant") if k != factor]
    # Runs on side A and side B, indexed by those keys plus the seed so they can be matched.
    A = runs[runs[factor] == a].set_index(keys + ["seed"])
    B = runs[runs[factor] == b].set_index(keys + ["seed"])
    rows = []
    # Only cells that exist on both sides can be compared.
    for idx in A.index.intersection(B.index):
        # Load both runs' test predictions (sorted by id, so row i is the same image in both).
        la, ya, ida = load_preds(A.loc[idx, "dir"])
        lb, yb, idb = load_preds(B.loc[idx, "dir"])
        # Safety check: both runs must have been tested on exactly the same images and labels.
        assert np.array_equal(ida, idb) and np.array_equal(ya, yb), "runs are not on the same test set"
        # Predicted class = highest logit.
        pa, pb = la.argmax(1), lb.argmax(1)
        # Score difference b - a with its bootstrap confidence interval.
        d, lo, hi = paired_bootstrap(ya, pa, pb, n_boot)
        # One result row: the cell's keys, the two compared values, both accuracies / F1s and the difference.
        rows.append({**dict(zip(keys + ["seed"], idx)), "a": a, "b": b,
                     "acc_a": float((pa == ya).mean()), "acc_b": float((pb == yb).mean()),
                     "f1_a": macro_f1_fast(ya, pa), "f1_b": macro_f1_fast(yb, pb),
                     "delta_f1": d, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


# Q: What does summarise_comparison do?
# A: It groups the per-seed rows from compare by cell and aggregates them: mean F1 of each side, mean/std/min/max of
#   delta_f1, how many seeds favour b, and how many intervals exclude 0.
# Q: Why report counts such as "b better in k of 3 seeds" instead of p-values?
# A: With only 3 seeds a p-value would be unreliable; simple counts of consistent wins are honest and easy to read.
# Q: Why can the std be NaN?
# A: delta_f1_std uses the sample std (ddof=1), which is undefined for a single seed.
def summarise_comparison(pairs: pd.DataFrame) -> pd.DataFrame:
    """Mean/std/min/max of per-seed deltas; how many seeds favour b; how many CIs exclude 0.

    Groups the per-seed rows from `compare` by the non-seed keys plus (a, b). `delta_f1_std` is the
    pandas sample std (ddof=1), so it is NaN when a group has a single seed. Returns `pairs` unchanged
    when it is empty.
    """
    # Nothing to summarise.
    if pairs.empty:
        return pairs
    # The grouping columns that are present (the factor that was compared is not a column of its own here).
    keys = [c for c in ("model", "modality", "fraction", "variant") if c in pairs.columns]
    # Two helper flags per seed: did b win, and does the 95% interval lie entirely above or below 0
    # (meaning the difference is unlikely to be just noise).
    p = pairs.assign(b_better=pairs["delta_f1"] > 0,
                     ci_excludes_0=(pairs["ci_low"] > 0) | (pairs["ci_high"] < 0))
    # Group the seeds of each cell and aggregate: mean F1 of each side, spread of the difference,
    # and the number of seeds where b wins / where the interval excludes 0.
    return p.groupby(keys + ["a", "b"]).agg(
        n_seeds=("seed", "nunique"), f1_a=("f1_a", "mean"), f1_b=("f1_b", "mean"),
        delta_f1_mean=("delta_f1", "mean"), delta_f1_std=("delta_f1", "std"),
        delta_f1_min=("delta_f1", "min"), delta_f1_max=("delta_f1", "max"),
        seeds_b_better=("b_better", "sum"), seeds_ci_excludes_0=("ci_excludes_0", "sum"),
    ).reset_index()


# Q: What does mean_confusion return?
# A: A K x K matrix averaged over the given runs (usually the 3 seeds), rows = true class, row-normalised by default so
#   each row shows the share of that class predicted as each class.
# Q: Why normalise each run first and then average?
# A: Every seed then counts equally, and the matrix is not dominated by a run with unusual class counts.
def mean_confusion(run_dirs: list[Path], normalise: bool = True) -> np.ndarray:
    """Seed-averaged confusion matrix (rows = true class); row-normalised by default.

    Each run's test confusion matrix is built (and row-normalised if `normalise`) before averaging, so
    every seed counts equally. Returns a [K, K] float array.
    """
    mats = []
    for d in run_dirs:
        # Test predictions of this run; count (true, predicted) pairs into a K x K matrix.
        logits, y, _ = load_preds(d)
        cm = np.bincount(y * K + logits.argmax(1), minlength=K * K).reshape(K, K).astype(float)
        # Row-normalising divides each row by its total, so entries become the share of each true class.
        mats.append(cm / cm.sum(1, keepdims=True) if normalise else cm)
    # Average the matrices of all runs (e.g. the 3 seeds).
    return np.mean(mats, axis=0)


# Q: What does top_confusions return?
# A: The k largest off-diagonal cells of a confusion matrix as (true class, predicted class, rate), largest first; zero
#   cells are dropped.
# Q: Why blank out the diagonal?
# A: The diagonal holds correct predictions; we only want the most common mistakes (which class is mistaken for which).
# Q: How are flat positions turned back into classes?
# A: np.argsort over the flattened array gives positions; i // K is the row (true class) and i % K the column (predicted
#   class).
def top_confusions(cm_norm: np.ndarray, k: int = 5) -> list[tuple[str, str, float]]:
    """The `k` largest off-diagonal cells as (true_class, predicted_class, rate), largest first.

    Zero cells are dropped, so fewer than `k` entries can be returned for a near-perfect matrix.
    """
    # Copy the matrix and blank out the diagonal (correct predictions), leaving only the mistakes.
    off = cm_norm.copy()
    np.fill_diagonal(off, 0)
    # Flat positions of the k biggest mistakes, largest first.
    flat = np.argsort(-off, axis=None)[:k]
    # Convert each flat position back to (row, column) = (true class, predicted class); skip zero entries.
    return [(CLASSES[i // K], CLASSES[i % K], float(off.flat[i])) for i in flat if off.flat[i] > 0]

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

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from eurosat_le.constants import CLASSES
from eurosat_le.eval.metrics import classification_metrics
from eurosat_le.ssl.pseudo_label import softmax

K = len(CLASSES)


def variant_of(suffix: str | None) -> str:
    """Map a run's config `name_suffix` to a variant label; no suffix means the plain supervised run."""
    return suffix or "supervised"


def load_runs(runs_dir: Path) -> pd.DataFrame:
    """One row per finished run that has test predictions (tuning runs have none).

    Columns: run_name, dir (Path), model, modality, fraction, seed, variant (see `variant_of`), teacher
    (config value, None for runs without a teacher). Rows follow the sorted order of the run folders.
    Returns an empty DataFrame (no columns) when nothing qualifies.
    """
    rows = []
    for m in sorted(runs_dir.glob("*/metrics.json")):
        d = m.parent
        if not (d / "preds_test.npz").exists():
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        rows.append({"run_name": d.name, "dir": d, "model": cfg["model"], "modality": cfg["modality"],
                     "fraction": cfg["fraction"], "seed": cfg["seed"],
                     "variant": variant_of(cfg.get("name_suffix")), "teacher": cfg.get("teacher")})
    return pd.DataFrame(rows)


def load_preds(run_dir: Path, split: str = "test") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load preds_<split>.npz and return (logits [N, K] float32, labels [N], ids [N] as str).

    All three arrays are reordered by sorted string id so that different runs line up row by row.
    """
    d = np.load(run_dir / f"preds_{split}.npz")
    order = np.argsort(d["ids"].astype(str))  # align runs by id, independent of loader order
    return d["logits"].astype(np.float32)[order], d["labels"][order], d["ids"].astype(str)[order]


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error of softmax probabilities `probs` [N, K] against integer `labels` [N].

    Samples are binned by top-class confidence into `n_bins` equal-width bins; ECE is the sum over bins
    of (fraction of samples in the bin) * |accuracy in bin - mean confidence in bin|. Empty bins add 0.
    """
    conf, pred = probs.max(1), probs.argmax(1)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)  # conf == 1.0 would otherwise index bin n_bins
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs((pred[m] == labels[m]).mean() - conf[m].mean())
    return float(ece)


def macro_f1_fast(y: np.ndarray, p: np.ndarray) -> float:
    """Macro-F1 from integer arrays `y` (true) and `p` (predicted) using only a bincount confusion matrix.

    Same quantity as sklearn's macro F1 with zero_division=0 over all K classes, but much faster, which
    matters inside the bootstrap loop (about 2,000 calls per paired comparison).
    """
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)  # rows = true, cols = predicted
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)  # F1 = 2TP / (2TP + FP + FN) = 2TP / (predicted count + true count)
    f1 = np.divide(2 * tp, denom, out=np.zeros(K), where=denom > 0)
    return float(f1.mean())


def run_metrics(run_dir: Path) -> dict:
    """Recompute metrics for one run from its saved predictions.

    Returns {"val": m, "test": m} where each m is the `classification_metrics` dict plus an "ece" entry.
    """
    out = {}
    for split in ("val", "test"):
        logits, y, _ = load_preds(run_dir, split)
        probs = softmax(logits)
        met = classification_metrics(y, probs.argmax(1))
        met["ece"] = expected_calibration_error(probs, y)
        out[split] = met
    return out


def check_against_logged(run_dir: Path, recomputed: dict, tol: float = 1e-3) -> list[str]:
    """Compare with metrics.json written at training time (fp16 logits can flip rare ties).

    Checks accuracy and macro-F1 on val and test. Returns a list of human-readable mismatch messages
    (differences larger than `tol`, absolute); an empty list means the run agrees with its log.
    """
    logged = json.loads((run_dir / "metrics.json").read_text())
    issues = []
    for split in ("val", "test"):
        for k in ("accuracy", "macro_f1"):
            if abs(logged[split][k] - recomputed[split][k]) > tol:
                issues.append(f"{run_dir.name} {split} {k}: logged {logged[split][k]:.4f} "
                              f"vs recomputed {recomputed[split][k]:.4f}")
    return issues


def paired_bootstrap(y, pred_a, pred_b, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Macro-F1(b) - macro-F1(a) on the same test images, with a 95% bootstrap interval.

    `y`, `pred_a`, `pred_b` are aligned integer arrays over the same images. Each bootstrap draw samples
    N image indices with replacement and applies the SAME indices to both runs (that is what makes it a
    paired bootstrap). Returns (delta, ci_low, ci_high); the interval is the 2.5th/97.5th percentile of
    the bootstrap deltas, and `delta` is the point estimate on the original test set. The fixed `seed`
    keeps the intervals reproducible.
    """
    rng = np.random.default_rng(seed)
    delta = macro_f1_fast(y, pred_b) - macro_f1_fast(y, pred_a)
    n = len(y)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = macro_f1_fast(y[idx], pred_b[idx]) - macro_f1_fast(y[idx], pred_a[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(delta), float(lo), float(hi)


def compare(runs: pd.DataFrame, factor: str, a, b, n_boot: int = 1000) -> pd.DataFrame:
    """Paired comparisons where only `factor` differs (a -> b); all other keys and the seed match.

    `factor` is one of "model", "modality", "fraction", "variant"; `a` and `b` are its two values. Rows of
    `runs` (from `load_runs`) are matched on the remaining keys plus the seed, so only seeds present in
    both settings are compared. Returns one row per matched pair with accuracy and macro-F1 for each side,
    `delta_f1` (= f1_b - f1_a) and its bootstrap interval; an empty DataFrame if nothing matches.
    Raises AssertionError if a pair does not share identical test ids and labels.
    Assumes (model, modality, fraction, variant, seed) is unique within each side.
    """
    keys = [k for k in ("model", "modality", "fraction", "variant") if k != factor]
    A = runs[runs[factor] == a].set_index(keys + ["seed"])
    B = runs[runs[factor] == b].set_index(keys + ["seed"])
    rows = []
    for idx in A.index.intersection(B.index):
        la, ya, ida = load_preds(A.loc[idx, "dir"])
        lb, yb, idb = load_preds(B.loc[idx, "dir"])
        assert np.array_equal(ida, idb) and np.array_equal(ya, yb), "runs are not on the same test set"
        pa, pb = la.argmax(1), lb.argmax(1)
        d, lo, hi = paired_bootstrap(ya, pa, pb, n_boot)
        rows.append({**dict(zip(keys + ["seed"], idx)), "a": a, "b": b,
                     "acc_a": float((pa == ya).mean()), "acc_b": float((pb == yb).mean()),
                     "f1_a": macro_f1_fast(ya, pa), "f1_b": macro_f1_fast(yb, pb),
                     "delta_f1": d, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def summarise_comparison(pairs: pd.DataFrame) -> pd.DataFrame:
    """Mean/std/min/max of per-seed deltas; how many seeds favour b; how many CIs exclude 0.

    Groups the per-seed rows from `compare` by the non-seed keys plus (a, b). `delta_f1_std` is the
    pandas sample std (ddof=1), so it is NaN when a group has a single seed. Returns `pairs` unchanged
    when it is empty.
    """
    if pairs.empty:
        return pairs
    keys = [c for c in ("model", "modality", "fraction", "variant") if c in pairs.columns]
    p = pairs.assign(b_better=pairs["delta_f1"] > 0,
                     ci_excludes_0=(pairs["ci_low"] > 0) | (pairs["ci_high"] < 0))
    return p.groupby(keys + ["a", "b"]).agg(
        n_seeds=("seed", "nunique"), f1_a=("f1_a", "mean"), f1_b=("f1_b", "mean"),
        delta_f1_mean=("delta_f1", "mean"), delta_f1_std=("delta_f1", "std"),
        delta_f1_min=("delta_f1", "min"), delta_f1_max=("delta_f1", "max"),
        seeds_b_better=("b_better", "sum"), seeds_ci_excludes_0=("ci_excludes_0", "sum"),
    ).reset_index()


def mean_confusion(run_dirs: list[Path], normalise: bool = True) -> np.ndarray:
    """Seed-averaged confusion matrix (rows = true class); row-normalised by default.

    Each run's test confusion matrix is built (and row-normalised if `normalise`) before averaging, so
    every seed counts equally. Returns a [K, K] float array.
    """
    mats = []
    for d in run_dirs:
        logits, y, _ = load_preds(d)
        cm = np.bincount(y * K + logits.argmax(1), minlength=K * K).reshape(K, K).astype(float)
        mats.append(cm / cm.sum(1, keepdims=True) if normalise else cm)
    return np.mean(mats, axis=0)


def top_confusions(cm_norm: np.ndarray, k: int = 5) -> list[tuple[str, str, float]]:
    """The `k` largest off-diagonal cells as (true_class, predicted_class, rate), largest first.

    Zero cells are dropped, so fewer than `k` entries can be returned for a near-perfect matrix.
    """
    off = cm_norm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(-off, axis=None)[:k]
    return [(CLASSES[i // K], CLASSES[i % K], float(off.flat[i])) for i in flat if off.flat[i] > 0]

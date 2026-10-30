"""Evaluation from saved predictions (Phase 6). Nothing here retrains or re-runs a model.

Every finished run directory holds preds_{val,test}.npz (logits, labels, ids). This
module rebuilds all metrics from those files, groups runs over seeds, and builds
paired comparisons (same model, fraction and seed; one factor changed).
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
    return suffix or "supervised"


def load_runs(runs_dir: Path) -> pd.DataFrame:
    """One row per finished run that has test predictions (tuning runs have none)."""
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
    d = np.load(run_dir / f"preds_{split}.npz")
    order = np.argsort(d["ids"].astype(str))  # align runs by id, independent of loader order
    return d["logits"].astype(np.float32)[order], d["labels"][order], d["ids"].astype(str)[order]


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs((pred[m] == labels[m]).mean() - conf[m].mean())
    return float(ece)


def macro_f1_fast(y: np.ndarray, p: np.ndarray) -> float:
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(K), where=denom > 0)
    return float(f1.mean())


def run_metrics(run_dir: Path) -> dict:
    out = {}
    for split in ("val", "test"):
        logits, y, _ = load_preds(run_dir, split)
        probs = softmax(logits)
        met = classification_metrics(y, probs.argmax(1))
        met["ece"] = expected_calibration_error(probs, y)
        out[split] = met
    return out


def check_against_logged(run_dir: Path, recomputed: dict, tol: float = 1e-3) -> list[str]:
    """Compare with metrics.json written at training time (fp16 logits can flip rare ties)."""
    logged = json.loads((run_dir / "metrics.json").read_text())
    issues = []
    for split in ("val", "test"):
        for k in ("accuracy", "macro_f1"):
            if abs(logged[split][k] - recomputed[split][k]) > tol:
                issues.append(f"{run_dir.name} {split} {k}: logged {logged[split][k]:.4f} "
                              f"vs recomputed {recomputed[split][k]:.4f}")
    return issues


def paired_bootstrap(y, pred_a, pred_b, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Macro-F1(b) - macro-F1(a) on the same test images, with a 95% bootstrap interval."""
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
    """Paired comparisons where only `factor` differs (a -> b); all other keys and the seed match."""
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
    """Mean/std/min/max of per-seed deltas; how many seeds favour b; how many CIs exclude 0."""
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
    """Seed-averaged confusion matrix (rows = true class); row-normalised by default."""
    mats = []
    for d in run_dirs:
        logits, y, _ = load_preds(d)
        cm = np.bincount(y * K + logits.argmax(1), minlength=K * K).reshape(K, K).astype(float)
        mats.append(cm / cm.sum(1, keepdims=True) if normalise else cm)
    return np.mean(mats, axis=0)


def top_confusions(cm_norm: np.ndarray, k: int = 5) -> list[tuple[str, str, float]]:
    off = cm_norm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(-off, axis=None)[:k]
    return [(CLASSES[i // K], CLASSES[i % K], float(off.flat[i])) for i in flat if off.flat[i] > 0]

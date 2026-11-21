"""Pseudo-labelling (self-training) from a finished supervised teacher run.

Steps, all without reading the true labels of the unlabelled pool:
1. Softmax the teacher's saved logits on the val set and on the unlabelled pool.
2. Pick the confidence threshold tau on VALIDATION: the smallest tau in the grid at
   which the teacher's val predictions with confidence >= tau are at least
   `target_precision` correct. If no tau reaches it, the most precise tau is used.
3. Keep pool images with confidence >= tau. Optionally balance classes: each
   predicted class keeps at most (its share of the labelled subset) x (number of
   confident images), most confident first. This stops a few easy classes from
   dominating; classes with fewer confident images keep all of them.
4. Write pseudo_labels.csv (id, label, confidence) for the student run.

`quality()` compares pseudo-labels with the hidden true labels. It is analysis for
the report only and is never used to choose tau or anything else.

Where it fits (Phase 5, docs/SSL.md): the teacher is a finished supervised run whose folder holds
preds_val.npz and preds_unlabelled.npz (written by `eurosat_le.train.engine.run`, from the weights
selected on validation macro-F1). `eurosat_le.ssl.self_training.train_student` calls
`make_pseudo_labels` and then trains the student on labelled + pseudo-labelled images.

Leakage safeguards
- Pseudo-labels are made only for the unlabelled pool (train ids outside the labelled subset);
  `make_pseudo_labels` raises if the pool overlaps the labelled subset. Val and test never overlap
  either, by construction of the split.
- tau is chosen from the teacher's VALIDATION predictions only. The test set is not used here.
- The true labels of pool images are read in exactly one place, `quality()`, and that function is
  called only after the student has been trained, writing pseudo_label_quality.json. The student's
  training and the selection never read that file. Computing it later (not before training) means
  the hidden labels cannot, even by accident, influence which images are selected.

Files written into the student run folder: pseudo_labels.csv (the exact training targets),
pseudo_label_thresholds.csv (validation precision and coverage for every tau),
pseudo_label_selection.json (chosen tau, its validation precision / coverage, counts per class).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from eurosat_le.constants import CLASSES
from eurosat_le.data.splits import load_split, load_subset

# Candidate thresholds 0.50, 0.51, ..., 0.99 (rounded to remove floating-point noise from arange).
TAU_GRID = np.round(np.arange(0.50, 0.995, 0.01), 2)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax of (N, C) logits, computed in float64 with the row max subtracted for stability."""
    z = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def threshold_table(val_probs: np.ndarray, val_labels: np.ndarray, grid=TAU_GRID) -> pd.DataFrame:
    """Validation precision and coverage of the teacher's predictions for each threshold tau.

    Args:
        val_probs: (N, C) teacher softmax probabilities on the validation set.
        val_labels: (N,) true validation labels (validation labels are allowed to be used here).
        grid: thresholds to evaluate.
    Returns:
        One row per tau: `coverage` = fraction of validation images with confidence >= tau,
        `precision` = fraction of those predicted correctly (NaN if no image passes).
    Confidence is the maximum softmax probability and the prediction is its argmax.
    """
    conf, pred = val_probs.max(1), val_probs.argmax(1)
    rows = []
    for tau in grid:
        keep = conf >= tau
        rows.append({"tau": float(tau), "coverage": float(keep.mean()),
                     "precision": float((pred[keep] == val_labels[keep]).mean()) if keep.any() else float("nan")})
    return pd.DataFrame(rows)


def choose_threshold(val_probs, val_labels, target_precision: float = 0.95, grid=TAU_GRID) -> tuple[float, pd.DataFrame]:
    """Pick tau on validation data only; return (tau, the full threshold table).

    tau is the smallest grid value whose validation precision is at least `target_precision`: the
    smallest tau keeps the most pseudo-labels while the quality is known to be acceptable. If no
    tau reaches the target, the most precise tau is returned instead (the lowest one among ties).
    """
    table = threshold_table(val_probs, val_labels, grid)
    ok = table[table["precision"] >= target_precision]
    if not ok.empty:
        return float(ok["tau"].min()), table
    return float(table.loc[table["precision"].idxmax(), "tau"]), table


def select(pool_probs: np.ndarray, tau: float, balance: bool = False, class_prior=None) -> np.ndarray:
    """Return indices (into the pool) of the selected items, ordered by decreasing confidence.

    Args:
        pool_probs: (M, C) teacher softmax probabilities on the unlabelled pool.
        tau: keep items whose confidence (max probability) is >= tau.
        balance: if True, cap each predicted class (see below).
        class_prior: per-class counts or weights of the LABELLED subset; required when balance is True.
    Without balancing, all confident items are kept. With balancing, a predicted class keeps at most
    ceil(prior share x number of confident items) items, most confident first; the caps are an upper
    bound only, so a class with fewer confident items keeps all of its items.
    """
    conf, pred = pool_probs.max(1), pool_probs.argmax(1)
    keep = np.flatnonzero(conf >= tau)
    if not balance:
        return keep[np.argsort(-conf[keep], kind="stable")]
    prior = np.asarray(class_prior, dtype=float)
    prior = prior / prior.sum()
    # cap each predicted class at its labelled-set share of the confident selection; classes
    # with fewer confident images keep all of them (a weak class never zeroes the others)
    caps = np.ceil(prior * len(keep)).astype(int)
    chosen = []
    for c in range(len(prior)):
        idx = keep[pred[keep] == c]
        idx = idx[np.argsort(-conf[idx], kind="stable")]
        chosen.append(idx[: caps[c]])
    out = np.concatenate(chosen) if chosen else np.array([], dtype=int)
    return out[np.argsort(-conf[out], kind="stable")]


def _load_npz(path: Path):
    """Load a preds_*.npz file as (float32 logits, labels, string ids); logits were saved as float16."""
    d = np.load(path, allow_pickle=False)
    return d["logits"].astype(np.float32), d["labels"], d["ids"].astype(str)


def make_pseudo_labels(teacher_dir: Path, out_dir: Path, fraction: float, seed: int,
                       target_precision: float = 0.95, balance: bool = False) -> dict:
    """Create out_dir/pseudo_labels.csv from teacher_dir; return the selection summary.

    Args:
        teacher_dir: finished teacher run folder (needs preds_val.npz and preds_unlabelled.npz).
        out_dir: the student's run folder; created if needed. Existing pseudo-label files are overwritten.
        fraction, seed: the labelled-subset settings of the teacher (identify its labelled ids).
        target_precision: validation precision that tau must reach (see `choose_threshold`).
        balance: apply the class-balanced selection in `select`.
    Returns:
        The dict also saved as pseudo_label_selection.json.
    The pseudo-label of an image is the teacher's argmax class. With balance=True the class prior is
    the class distribution of the labelled subset (those labels are known, so this is legitimate).
    """
    val_logits, val_labels, _ = _load_npz(teacher_dir / "preds_val.npz")
    pool_logits, _, pool_ids = _load_npz(teacher_dir / "preds_unlabelled.npz")
    labelled = set(load_subset(fraction, seed))
    # Safeguard: pseudo-labelled images must come from the unlabelled pool, never from the labelled subset.
    if labelled & set(pool_ids):
        raise RuntimeError("teacher's unlabelled pool overlaps the labelled subset")

    tau, table = choose_threshold(softmax(val_logits), val_labels, target_precision)  # validation only
    pool_probs = softmax(pool_logits)
    prior = None
    if balance:
        split = load_split().set_index("id")
        prior = np.bincount(split.loc[sorted(labelled), "label"], minlength=len(CLASSES))
    idx = select(pool_probs, tau, balance, prior)

    out_dir.mkdir(parents=True, exist_ok=True)
    pl = pd.DataFrame({"id": pool_ids[idx], "label": pool_probs[idx].argmax(1),
                       "confidence": pool_probs[idx].max(1).round(5)})
    pl.to_csv(out_dir / "pseudo_labels.csv", index=False)
    table.to_csv(out_dir / "pseudo_label_thresholds.csv", index=False)
    row = table[table["tau"] == tau].iloc[0]
    summary = {
        "teacher": teacher_dir.name, "target_precision": target_precision, "balance": balance,
        "tau": tau, "val_precision_at_tau": float(row["precision"]), "val_coverage_at_tau": float(row["coverage"]),
        "pool_size": int(len(pool_ids)), "n_selected": int(len(pl)),
        "selected_class_counts": dict(zip(CLASSES, np.bincount(pl["label"], minlength=len(CLASSES)).tolist())),
    }
    (out_dir / "pseudo_label_selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def load_pseudo_labels(path: Path) -> tuple[list[str], list[int]]:
    """Read pseudo_labels.csv; return (ids, integer class labels) in file order (most confident first)."""
    pl = pd.read_csv(path)
    return pl["id"].tolist(), pl["label"].astype(int).tolist()


def quality(pseudo_path: Path) -> dict:
    """ANALYSIS ONLY: accuracy of pseudo-labels against the hidden true labels.

    Returns the overall accuracy and, per predicted class, the number of pseudo-labels and the
    fraction that is correct (None for empty classes). This is the only place that reads the
    true labels of unlabelled pool images; the engine calls it after the student has finished
    training, and nothing reads its output during training or selection. It is a diagnostic for
    the report (how large confirmation bias could be), never an input to any decision.
    """
    pl = pd.read_csv(pseudo_path)
    truth = load_split().set_index("id").loc[pl["id"], "label"].to_numpy()
    correct = pl["label"].to_numpy() == truth
    per_class = {}
    for c, name in enumerate(CLASSES):
        m = pl["label"].to_numpy() == c
        per_class[name] = {"n": int(m.sum()), "precision": float(correct[m].mean()) if m.any() else None}
    return {"n": int(len(pl)), "accuracy": float(correct.mean()) if len(pl) else None, "per_class": per_class}

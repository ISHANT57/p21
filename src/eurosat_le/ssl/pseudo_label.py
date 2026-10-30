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
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from eurosat_le.constants import CLASSES
from eurosat_le.data.splits import load_split, load_subset

TAU_GRID = np.round(np.arange(0.50, 0.995, 0.01), 2)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def threshold_table(val_probs: np.ndarray, val_labels: np.ndarray, grid=TAU_GRID) -> pd.DataFrame:
    conf, pred = val_probs.max(1), val_probs.argmax(1)
    rows = []
    for tau in grid:
        keep = conf >= tau
        rows.append({"tau": float(tau), "coverage": float(keep.mean()),
                     "precision": float((pred[keep] == val_labels[keep]).mean()) if keep.any() else float("nan")})
    return pd.DataFrame(rows)


def choose_threshold(val_probs, val_labels, target_precision: float = 0.95, grid=TAU_GRID) -> tuple[float, pd.DataFrame]:
    table = threshold_table(val_probs, val_labels, grid)
    ok = table[table["precision"] >= target_precision]
    if not ok.empty:
        return float(ok["tau"].min()), table
    return float(table.loc[table["precision"].idxmax(), "tau"]), table


def select(pool_probs: np.ndarray, tau: float, balance: bool = False, class_prior=None) -> np.ndarray:
    """Return indices of selected pool items, most confident first within each class."""
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
    d = np.load(path, allow_pickle=False)
    return d["logits"].astype(np.float32), d["labels"], d["ids"].astype(str)


def make_pseudo_labels(teacher_dir: Path, out_dir: Path, fraction: float, seed: int,
                       target_precision: float = 0.95, balance: bool = False) -> dict:
    """Create out_dir/pseudo_labels.csv from teacher_dir; return the selection summary."""
    val_logits, val_labels, _ = _load_npz(teacher_dir / "preds_val.npz")
    pool_logits, _, pool_ids = _load_npz(teacher_dir / "preds_unlabelled.npz")
    labelled = set(load_subset(fraction, seed))
    if labelled & set(pool_ids):
        raise RuntimeError("teacher's unlabelled pool overlaps the labelled subset")

    tau, table = choose_threshold(softmax(val_logits), val_labels, target_precision)
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
    pl = pd.read_csv(path)
    return pl["id"].tolist(), pl["label"].astype(int).tolist()


def quality(pseudo_path: Path) -> dict:
    """ANALYSIS ONLY: accuracy of pseudo-labels against the hidden true labels."""
    pl = pd.read_csv(pseudo_path)
    truth = load_split().set_index("id").loc[pl["id"], "label"].to_numpy()
    correct = pl["label"].to_numpy() == truth
    per_class = {}
    for c, name in enumerate(CLASSES):
        m = pl["label"].to_numpy() == c
        per_class[name] = {"n": int(m.sum()), "precision": float(correct[m].mean()) if m.any() else None}
    return {"n": int(len(pl)), "accuracy": float(correct.mean()) if len(pl) else None, "per_class": per_class}

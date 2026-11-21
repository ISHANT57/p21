"""Classification metrics shared by training and the Phase 6 evaluation.

Two entry points, both over integer class indices 0..len(CLASSES)-1 (the order of
`eurosat_le.constants.CLASSES`):

- `classification_metrics`: full report (accuracy, macro-F1, per-class precision/recall/F1/support,
  confusion matrix). Used when a run finishes (metrics.json) and again by
  `eurosat_le.eval.evaluate.run_metrics`, which recomputes the same numbers from the saved predictions
  and cross-checks them against metrics.json.
- `summary`: accuracy and macro-F1 only, cheap enough to log every epoch.

Design note: `labels=list(range(len(CLASSES)))` is passed everywhere so that all 10 classes are always
present in the output (fixed shape and order), even if a class is missing from `y_true`/`y_pred`;
`zero_division=0` makes an absent class score 0 instead of raising a warning.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from eurosat_le.constants import CLASSES


def classification_metrics(y_true, y_pred) -> dict:
    """Full metrics report for integer labels `y_true` and predictions `y_pred` (same length).

    Returns a JSON-serialisable dict (plain floats/ints/lists) with keys:
    - "accuracy", "macro_f1": floats (macro-F1 = unweighted mean of the per-class F1 scores);
    - "per_class": {class_name: {"precision", "recall", "f1", "support"}};
    - "confusion_matrix": nested list, rows = true class, columns = predicted class, raw counts
      (not normalised).
    """
    labels = list(range(len(CLASSES)))
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def summary(y_true, y_pred) -> dict:
    """Accuracy and macro-F1 only (cheap, for per-epoch logging)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {"accuracy": float((y_true == y_pred).mean()),
            "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(CLASSES))), average="macro",
                                       zero_division=0))}

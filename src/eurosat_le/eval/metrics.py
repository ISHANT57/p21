"""Classification metrics shared by training and the Phase 6 evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from eurosat_le.constants import CLASSES


def classification_metrics(y_true, y_pred) -> dict:
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

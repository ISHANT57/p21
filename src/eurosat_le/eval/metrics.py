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

# Makes type hints such as `int | None` work as plain text on older Python versions.
from __future__ import annotations

# numpy: array handling; sklearn.metrics: ready-made, well-tested scoring functions (accuracy, F1,
# precision/recall, confusion matrix).
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

# CLASSES: the ordered list of the 10 land-use class names.
from eurosat_le.constants import CLASSES


# ======================================== CONCEPTS USED IN THIS FILE ========================================
# accuracy: share of images whose predicted class equals the true class. Simple, but can hide poor results on small
#   classes.
# precision (per class): of the images predicted as a class, the fraction that really are that class. recall: of the
#   images that really are the class, the fraction found.
# F1 score: harmonic mean of precision and recall for one class (high only when both are high).
# macro-F1: the unweighted mean of the 10 per-class F1 scores, so every class counts equally regardless of size. This is
#   the project's main metric.
# support: the number of true images of a class in the evaluated set.
# confusion matrix: a table of counts; row = true class, column = predicted class. The diagonal holds correct
#   predictions, off-diagonal cells are the mistakes.
# labels=list(range(10)) and zero_division=0: always report all 10 classes in a fixed order, and score a class that
#   never appears as 0 instead of raising a warning.
# integer class indices: classes are numbers 0..9 in the order of constants.CLASSES; names are only attached when
#   building the report.
# JSON-serialisable: only plain Python floats / ints / lists, so the result can be saved in metrics.json (NumPy types
#   are converted with float()/int()).
# sklearn.metrics: the scikit-learn library's well-tested scoring functions, used instead of re-implementing the
#   formulas.

# ======================================== Q&A ABOUT THIS FILE ========================================
# Q: What is this file for?
# A: It holds the metric functions shared by training and evaluation: a full report (classification_metrics) and a cheap
#   one (summary).
# Q: Where does it fit in the pipeline?
# A: train/engine.py uses summary every epoch (validation) and classification_metrics when a run ends (metrics.json);
#   eval/evaluate.py recomputes the same numbers from saved predictions and compares them.
# Q: What does it read and write?
# A: Nothing on disk. It takes integer arrays of true and predicted labels and returns dicts.
# Q: Why use one shared implementation?
# A: Training and evaluation then compute metrics identically, so the recomputed numbers can be cross-checked against
#   metrics.json.

# Q: What does classification_metrics return?
# A: A dict with accuracy, macro_f1, per_class (precision, recall, f1, support for each class name) and confusion_matrix
#   (raw counts as a nested list).
# Q: Why pass labels=list(range(len(CLASSES))) everywhere?
# A: So all 10 classes always appear with a fixed shape and order, even if one is missing from y_true / y_pred; without
#   it the arrays could change size and misalign.
# Q: Why macro-F1 in addition to accuracy?
# A: Macro-F1 averages the classes equally, so a weak class is not hidden by strong ones; this matters most with few
#   labels.
def classification_metrics(y_true, y_pred) -> dict:
    """Full metrics report for integer labels `y_true` and predictions `y_pred` (same length).

    Returns a JSON-serialisable dict (plain floats/ints/lists) with keys:
    - "accuracy", "macro_f1": floats (macro-F1 = unweighted mean of the per-class F1 scores);
    - "per_class": {class_name: {"precision", "recall", "f1", "support"}};
    - "confusion_matrix": nested list, rows = true class, columns = predicted class, raw counts
      (not normalised).

    Precision = of the images predicted as a class, how many really are it; recall = of the images
    that really are the class, how many were found; F1 = the harmonic mean of precision and recall;
    support = number of true images of the class.
    """
    # Class indices 0..9; passing them explicitly keeps the output the same shape even if a class is absent.
    labels = list(range(len(CLASSES)))
    # Per-class precision (p), recall (r), F1 (f) and support, computed in one call.
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    # Build the report with plain Python floats / ints / lists so it can be saved as JSON.
    return {
        # Overall fraction of correct predictions.
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # Macro average: compute F1 per class, then take the unweighted mean, so rare classes count equally.
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        # The per-class numbers, labelled with the class names.
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])}
                      for i, c in enumerate(CLASSES)},
        # Counts of (true class, predicted class) pairs; the diagonal holds the correct predictions.
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


# Q: What does summary do and why does it exist separately?
# A: It returns only accuracy and macro-F1. It is cheap, so the training loop can log it after every epoch (it is the
#   validation score used for model selection) without building the full report.
# Q: What are the inputs?
# A: y_true and y_pred, lists or arrays of integer labels of the same length; np.asarray lets either work.
def summary(y_true, y_pred) -> dict:
    """Accuracy and macro-F1 only (cheap, for per-epoch logging).

    Args:
        y_true: true integer labels.
        y_pred: predicted integer labels (same length).
    Returns:
        {"accuracy": float, "macro_f1": float}.
    """
    # Accept lists or arrays alike.
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    # Accuracy = share of positions where prediction equals truth; macro-F1 as in `classification_metrics`.
    return {"accuracy": float((y_true == y_pred).mean()),
            "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(CLASSES))), average="macro",
                                       zero_division=0))}

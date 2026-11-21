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

# Makes type hints such as `int | None` work as plain text on older Python versions.
from __future__ import annotations

# json: write the selection summary; Path: file paths.
import json
from pathlib import Path

# numpy: array maths (softmax, sorting, counting); pandas: tables (threshold table, CSV files).
import numpy as np
import pandas as pd

# CLASSES: the 10 class names. load_split: table of all images with their true labels;
# load_subset: ids of the labelled subset for a given fraction and seed.
from eurosat_le.constants import CLASSES
from eurosat_le.data.splits import load_split, load_subset

# ======================================== CONCEPTS USED IN THIS FILE ========================================
# semi-supervised learning / self-training: using many unlabelled images as well as a few labelled ones. A teacher model
#   labels the unlabelled images itself and a student learns from them too.
# teacher / student: the teacher is a finished supervised run (trained on the labelled subset only); the student is a
#   new run trained on labelled + pseudo-labelled images.
# pseudo-label: a label guessed by the teacher (its argmax class) for an image whose true label is not used.
#   Pseudo-labels can be wrong.
# labelled subset vs unlabelled pool: the labelled subset is the fraction of train images whose labels are known; the
#   pool is every other train image, used without labels.
# logits and softmax: logits are the model's raw scores; softmax makes them probabilities in each row that are positive
#   and add up to 1.
# confidence: the largest softmax probability of an image; the prediction is the class that has it (argmax).
# confidence threshold tau: an image gets a pseudo-label only if confidence >= tau. A high tau gives fewer but more
#   reliable pseudo-labels.
# tau grid (TAU_GRID): the candidate thresholds 0.50, 0.51 ... 0.99 that are tried.
# precision and coverage at tau: coverage = share of images with confidence >= tau; precision = share of those predicted
#   correctly. Raising tau lowers coverage and usually raises precision.
# target_precision (default 0.95): the quality we demand; tau is the smallest grid value whose VALIDATION precision
#   reaches it (keeps the most pseudo-labels). If none reaches it, the most precise tau is used.
# validation vs test set: tau is chosen from the teacher's VALIDATION predictions only; the test set is never used here
#   (no leakage).
# class balancing / class prior: optional cap so each predicted class keeps at most its share (from the labelled
#   subset's class counts) of the confident images; stops easy classes from dominating the pseudo-labels.
# stable sort: argsort(kind="stable") keeps tied items in their original order, so results are reproducible.
# confirmation bias: the risk that the student learns the teacher's mistakes; quality() measures how large it could be
#   but is analysis only.
# leakage safeguards: pool must not overlap the labelled subset (RuntimeError otherwise); true labels of pool images are
#   read only in quality(), after the student is trained.
# .npz / float16: the teacher's saved prediction files; logits were stored as float16 and are converted back to float32
#   here.
# allow_pickle=False: a safety option so np.load only reads plain arrays, never arbitrary Python objects.
# DataFrame / CSV: pandas table and the text file format used for pseudo_labels.csv (id, label, confidence) and the
#   threshold table.

# ======================================== Q&A ABOUT THIS FILE ========================================
# Q: What is this file for?
# A: It turns a finished teacher run's predictions into pseudo-labels: it picks the confidence threshold tau on
#   validation, selects confident unlabelled images and writes pseudo_labels.csv.
# Q: Where does it fit in the pipeline?
# A: Phase 5. train/engine.py saves the teacher's preds_val.npz and preds_unlabelled.npz; ssl/self_training.py
#   train_student calls make_pseudo_labels; engine.run then trains the student and later calls quality().
# Q: What does it read and write?
# A: It reads preds_val.npz and preds_unlabelled.npz from the teacher folder and the split/subset files. It writes
#   pseudo_labels.csv, pseudo_label_thresholds.csv and pseudo_label_selection.json into the student's run folder.
# Q: How does it avoid using the true labels of the unlabelled images?
# A: Selection uses only the teacher's probabilities and validation labels. The pool's true labels are read in exactly
#   one place, quality(), which runs after training and is never used for any decision.

# Candidate thresholds 0.50, 0.51, ..., 0.99 (rounded to remove floating-point noise from arange).
TAU_GRID = np.round(np.arange(0.50, 0.995, 0.01), 2)


# Q: What does softmax do?
# A: It converts each row of raw scores (logits) into probabilities that add up to 1; the biggest score gets the biggest
#   probability.
# Q: Why subtract the row maximum first?
# A: The result is mathematically identical but exp() of a large number cannot overflow; without it big logits could
#   give infinity / NaN.
# Q: Why float64?
# A: Higher precision for the exponentials and sums, so confidences compared with tau are accurate.
def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax of (N, C) logits, computed in float64 with the row max subtracted for stability.

    Softmax turns the model's raw scores (logits) into probabilities that are positive and add up to 1
    in each row; the largest score gets the largest probability.
    """
    # Subtract each row's maximum: the result is mathematically the same but exp() cannot overflow.
    z = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    # Exponentiate, then divide by the row sum so every row adds up to 1.
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# Q: What does threshold_table compute?
# A: For every tau in the grid, the coverage (share of validation images with confidence >= tau) and the precision
#   (share of those that the teacher classifies correctly).
# Q: Why use validation labels here and is that allowed?
# A: Yes: validation labels are meant for decisions like choosing hyper-parameters. Using them to pick tau is
#   legitimate; using the test set would not be.
# Q: What does the table show?
# A: The trade-off: a higher tau gives higher precision but lower coverage. Precision is NaN when no image passes the
#   threshold.
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
    # For each image: confidence = the highest class probability, prediction = the class that has it.
    conf, pred = val_probs.max(1), val_probs.argmax(1)
    rows = []
    for tau in grid:
        # Images the teacher is at least `tau` sure about.
        keep = conf >= tau
        # Coverage = share of images kept; precision = share of the kept images that are right.
        rows.append({"tau": float(tau), "coverage": float(keep.mean()),
                     "precision": float((pred[keep] == val_labels[keep]).mean()) if keep.any() else float("nan")})
    return pd.DataFrame(rows)


# Q: What does choose_threshold do?
# A: It returns the smallest tau whose validation precision is at least target_precision, plus the full table.
# Q: Why the smallest acceptable tau?
# A: A smaller tau keeps more pseudo-labelled images, while the validation precision shows that their quality is still
#   acceptable.
# Q: What if no tau reaches the target?
# A: The most precise tau in the table is used instead (the lowest one among ties), so the function always returns a
#   value.
def choose_threshold(val_probs, val_labels, target_precision: float = 0.95, grid=TAU_GRID) -> tuple[float, pd.DataFrame]:
    """Pick tau on validation data only; return (tau, the full threshold table).

    tau is the smallest grid value whose validation precision is at least `target_precision`: the
    smallest tau keeps the most pseudo-labels while the quality is known to be acceptable. If no
    tau reaches the target, the most precise tau is returned instead (the lowest one among ties).

    Args:
        val_probs: (N, C) teacher probabilities on the validation set.
        val_labels: (N,) true validation labels.
        target_precision: minimum validation precision the chosen tau must reach.
        grid: candidate thresholds.
    """
    # Precision and coverage for every candidate threshold.
    table = threshold_table(val_probs, val_labels, grid)
    # Thresholds that meet the precision target (NaN precision compares as False, so it is excluded).
    ok = table[table["precision"] >= target_precision]
    if not ok.empty:
        # Among acceptable thresholds take the smallest: it keeps the most pseudo-labelled images.
        return float(ok["tau"].min()), table
    # Fallback: no threshold is precise enough, so take the most precise one available.
    return float(table.loc[table["precision"].idxmax(), "tau"]), table


# Q: What does select do?
# A: It returns the positions (indices into the pool) of the images chosen as pseudo-labelled: those with confidence >=
#   tau, ordered from most to least confident.
# Q: What does balance=True change and why?
# A: Each predicted class may keep at most ceil(its share of the labelled subset x number of confident images), most
#   confident first. This stops a few easy classes from dominating; a class with fewer confident images simply keeps all
#   of them.
# Q: Why is class_prior needed only when balancing?
# A: The caps come from the class proportions (counts of the labelled subset); without balancing no proportions are
#   used.
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
    # Confidence and predicted class of every pool image.
    conf, pred = pool_probs.max(1), pool_probs.argmax(1)
    # Positions of the images the teacher is confident enough about.
    keep = np.flatnonzero(conf >= tau)
    if not balance:
        # Sort the kept images from most to least confident (stable sort keeps ties in original order).
        return keep[np.argsort(-conf[keep], kind="stable")]
    # Turn the labelled-set class counts into shares that add up to 1.
    prior = np.asarray(class_prior, dtype=float)
    prior = prior / prior.sum()
    # cap each predicted class at its labelled-set share of the confident selection; classes
    # with fewer confident images keep all of them (a weak class never zeroes the others)
    caps = np.ceil(prior * len(keep)).astype(int)
    chosen = []
    for c in range(len(prior)):
        # Confident images predicted as class c, sorted most confident first.
        idx = keep[pred[keep] == c]
        idx = idx[np.argsort(-conf[idx], kind="stable")]
        # Keep at most caps[c] of them (the most confident ones).
        chosen.append(idx[: caps[c]])
    # Merge the classes' selections, then order the whole selection by confidence again.
    out = np.concatenate(chosen) if chosen else np.array([], dtype=int)
    return out[np.argsort(-conf[out], kind="stable")]


# Q: What does _load_npz return?
# A: A teacher prediction file as (float32 logits, labels, string ids).
# Q: Why astype(np.float32) and allow_pickle=False?
# A: Logits were saved as float16 and need to be converted back for maths; allow_pickle=False means only plain arrays
#   can be loaded, which is safer.
def _load_npz(path: Path):
    """Load a preds_*.npz file as (float32 logits, labels, string ids); logits were saved as float16."""
    # allow_pickle=False is a safety setting: only plain arrays may be loaded, never arbitrary Python objects.
    d = np.load(path, allow_pickle=False)
    # Convert logits back to float32 for the softmax and ids to plain strings.
    return d["logits"].astype(np.float32), d["labels"], d["ids"].astype(str)


# Q: What does make_pseudo_labels do, step by step?
# A: Loads the teacher's validation and pool predictions; checks the pool does not overlap the labelled subset; picks
#   tau on validation; selects confident pool images (optionally class-balanced); writes pseudo_labels.csv (id, label,
#   confidence), the threshold table and a JSON summary; returns the summary.
# Q: What is the pseudo-label of an image?
# A: The teacher's argmax class for that image; its confidence is the max probability.
# Q: Why raise RuntimeError if pool and labelled subset overlap?
# A: A labelled image must never get a pseudo-label; an overlap would mean the split or subset files are wrong and the
#   experiment invalid.
# Q: Why may the class prior come from the labelled subset?
# A: Those labels are known to the teacher anyway, so using their class counts leaks nothing.
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
    # Teacher's validation logits with true labels, and its logits on the unlabelled pool (no labels needed).
    val_logits, val_labels, _ = _load_npz(teacher_dir / "preds_val.npz")
    pool_logits, _, pool_ids = _load_npz(teacher_dir / "preds_unlabelled.npz")
    # Ids of the images whose labels are known to the teacher.
    labelled = set(load_subset(fraction, seed))
    # Safeguard: pseudo-labelled images must come from the unlabelled pool, never from the labelled subset.
    if labelled & set(pool_ids):
        raise RuntimeError("teacher's unlabelled pool overlaps the labelled subset")

    # Choose the confidence threshold from validation predictions only.
    tau, table = choose_threshold(softmax(val_logits), val_labels, target_precision)  # validation only
    # Probabilities of the teacher on the unlabelled pool.
    pool_probs = softmax(pool_logits)
    prior = None
    if balance:
        # Class counts of the labelled subset (their labels are known, so using them is allowed).
        split = load_split().set_index("id")
        prior = np.bincount(split.loc[sorted(labelled), "label"], minlength=len(CLASSES))
    # Positions of the pool images that become pseudo-labelled.
    idx = select(pool_probs, tau, balance, prior)

    # Make sure the student's folder exists, then build the pseudo-label table: id, predicted class, confidence.
    out_dir.mkdir(parents=True, exist_ok=True)
    pl = pd.DataFrame({"id": pool_ids[idx], "label": pool_probs[idx].argmax(1),
                       "confidence": pool_probs[idx].max(1).round(5)})
    # Save the training targets and the full threshold table.
    pl.to_csv(out_dir / "pseudo_labels.csv", index=False)
    table.to_csv(out_dir / "pseudo_label_thresholds.csv", index=False)
    # The table row for the chosen tau (gives its validation precision and coverage).
    row = table[table["tau"] == tau].iloc[0]
    # Summary of what was chosen and selected, saved for the report.
    summary = {
        "teacher": teacher_dir.name, "target_precision": target_precision, "balance": balance,
        "tau": tau, "val_precision_at_tau": float(row["precision"]), "val_coverage_at_tau": float(row["coverage"]),
        "pool_size": int(len(pool_ids)), "n_selected": int(len(pl)),
        "selected_class_counts": dict(zip(CLASSES, np.bincount(pl["label"], minlength=len(CLASSES)).tolist())),
    }
    (out_dir / "pseudo_label_selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# Q: What does load_pseudo_labels return and who uses it?
# A: The image ids and integer class labels from pseudo_labels.csv as Python lists, in file order (most confident
#   first). train/engine.py passes them to the data loader as extra training examples.
def load_pseudo_labels(path: Path) -> tuple[list[str], list[int]]:
    """Read pseudo_labels.csv; return (ids, integer class labels) in file order (most confident first)."""
    # Read the CSV and return its id and label columns as plain Python lists.
    pl = pd.read_csv(path)
    return pl["id"].tolist(), pl["label"].astype(int).tolist()


# Q: What does quality compute?
# A: The accuracy of the pseudo-labels against the images' hidden true labels, overall and per predicted class (count
#   and fraction correct, None for an empty class).
# Q: Why is it "analysis only" and when is it called?
# A: It is the only place that reads the true labels of pool images. The engine calls it after the student is trained
#   and writes pseudo_label_quality.json; nothing reads it during training or selection, so hidden labels cannot
#   influence the student.
# Q: Why compute it at all?
# A: For the report: it shows how many pseudo-labels are wrong, i.e. how large confirmation bias could be.
def quality(pseudo_path: Path) -> dict:
    """ANALYSIS ONLY: accuracy of pseudo-labels against the hidden true labels.

    Returns the overall accuracy and, per predicted class, the number of pseudo-labels and the
    fraction that is correct (None for empty classes). This is the only place that reads the
    true labels of unlabelled pool images; the engine calls it after the student has finished
    training, and nothing reads its output during training or selection. It is a diagnostic for
    the report (how large confirmation bias could be), never an input to any decision.

    Args:
        pseudo_path: the pseudo_labels.csv file to check.
    """
    # Load the pseudo-labels and look up the true label of each of those images.
    pl = pd.read_csv(pseudo_path)
    truth = load_split().set_index("id").loc[pl["id"], "label"].to_numpy()
    # True where the pseudo-label matches the real label.
    correct = pl["label"].to_numpy() == truth
    per_class = {}
    for c, name in enumerate(CLASSES):
        # Pseudo-labels that assign class c.
        m = pl["label"].to_numpy() == c
        # How many there are and what fraction of them is right (None if there are none).
        per_class[name] = {"n": int(m.sum()), "precision": float(correct[m].mean()) if m.any() else None}
    return {"n": int(len(pl)), "accuracy": float(correct.mean()) if len(pl) else None, "per_class": per_class}

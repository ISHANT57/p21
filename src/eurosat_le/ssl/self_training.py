"""Build and train a pseudo-label student from a finished teacher run.

Teacher / student pairing (Phase 5, docs/SSL.md): a student is trained from the teacher's saved
config, so it has the same model, input type, label fraction, seed and hyper-parameters, and it
starts from the same initial weights as the teacher did (fresh pretrained or random weights, not
the teacher's trained weights). Only the training set differs: the student sees the labelled subset
plus the teacher's confident pseudo-labels. Each student can therefore be compared with its own
supervised teacher in the evaluation.

Naming: students live in results/runs/<teacher>_pl; a student used as a teacher gives <teacher>_pl2,
then _pl3, and so on. A custom `name_suffix` override (e.g. "plb" for class-balanced selection)
replaces the automatic suffix.

Flow of `train_student`: check that the teacher finished -> build the student config -> write the
pseudo-labels (tau chosen on validation, see eurosat_le.ssl.pseudo_label) into the student's run
folder -> `eurosat_le.train.engine.run`, which trains on labelled + pseudo-labelled images, selects
and early-stops on the true validation labels exactly as for a supervised run, predicts the test set
once, and finally records pseudo-label accuracy against the hidden labels (analysis only).
"""

# Makes type hints such as `int | None` work as plain text on older Python versions.
from __future__ import annotations

# re: regular expressions, used to read the "pl", "pl2" ... suffix; fields: lists the fields of a dataclass.
import re
from dataclasses import fields

# yaml: reads the teacher's saved config.yaml.
import yaml

# make_pseudo_labels: creates the pseudo-label CSV from the teacher's predictions.
# TrainConfig: the settings of one run. RUNS_DIR / run: where runs live and the function that trains one.
from eurosat_le.ssl.pseudo_label import make_pseudo_labels
from eurosat_le.train.config import TrainConfig
from eurosat_le.train.engine import RUNS_DIR, run


# ======================================== CONCEPTS USED IN THIS FILE ========================================
# self-training / pseudo-labelling: a teacher run labels unlabelled images with its confident predictions; a student is
#   trained on labelled + pseudo-labelled images.
# teacher / student: the teacher is a finished supervised run; the student copies its config (same model, input type,
#   fraction, seed, hyper-parameters) and differs only in its training set.
# fresh initial weights: the student starts from the same kind of initial weights as the teacher did (pretrained
#   ImageNet or random), NOT from the teacher's trained weights, so the comparison is fair.
# pseudo-label CSV (pseudo_labels.csv): ids, labels and confidences of the selected unlabelled images; written into the
#   student's run folder by make_pseudo_labels.
# confidence threshold tau / target_precision: tau is chosen on validation so the teacher's confident predictions reach
#   target_precision (default 0.95); see ssl/pseudo_label.py.
# class-balanced selection (balance): optional cap per predicted class so a few easy classes do not dominate the
#   pseudo-labels.
# self-training rounds / name suffix: the first student is named <teacher>_pl; a student used as a teacher gives _pl2,
#   then _pl3 and so on. name_suffix overrides the automatic suffix.
# regular expression (re.fullmatch): a text pattern; r"pl(\d*)" accepts "pl" optionally followed by digits.
# dataclass fields: fields(TrainConfig) lists the settings TrainConfig knows, used to drop obsolete keys from old
#   config.yaml files.
# skip finished runs: if the student's metrics.json exists (and overwrite is False) it is not retrained.
# validation vs test set: pseudo-label tau, best epoch and early stopping use validation labels only; the test set is
#   predicted once at the end by engine.run.

# ======================================== Q&A ABOUT THIS FILE ========================================
# Q: What is this file for?
# A: It builds a pseudo-label student from a finished teacher run and trains it: it derives the student's config, writes
#   the pseudo-labels, and calls engine.run.
# Q: Where does it fit in the pipeline?
# A: Phase 5. It sits between ssl/pseudo_label.py (creates the pseudo-labels) and train/engine.py (trains);
#   scripts/run_grid.py uses it to build students for the grid.
# Q: What does it read and write?
# A: It reads the teacher's config.yaml, metrics.json, preds_val.npz and preds_unlabelled.npz in results/runs/<teacher>.
#   It writes the student's run folder (pseudo-label files plus the usual run files) via the called functions.
# Q: Why must the student have the same settings as the teacher?
# A: So each student can be compared directly with its own supervised teacher: the only difference is the extra
#   pseudo-labelled training data.


# Q: What does student_suffix do?
# A: It gives the name suffix for the next self-training round: "" -> "pl", "pl" -> "pl2", "pl2" -> "pl3".
# Q: Why raise ValueError for other suffixes?
# A: A teacher with another suffix (e.g. a tuning run "tune_lr...") is not a supervised or pseudo-label run, so naming a
#   student after it would be meaningless.
# Q: Why does an empty digit group count as round 1?
# A: "pl" is the first student, so the next one is round 2: int(m.group(1) or 1) + 1.
def student_suffix(teacher_suffix: str) -> str:
    """'' -> 'pl', 'pl' -> 'pl2', 'pl2' -> 'pl3' (one more self-training round).

    Args:
        teacher_suffix: the `name_suffix` of the teacher run ("" for a plain supervised run).
    Returns:
        The suffix for the student's run name.
    Raises:
        ValueError: if the teacher's suffix is something else (e.g. a tuning run).
    """
    # A plain supervised teacher gives the first-round student "pl".
    if not teacher_suffix:
        return "pl"
    # Accept only "pl" optionally followed by a number.
    m = re.fullmatch(r"pl(\d*)", teacher_suffix)  # rounds continue from default students only
    if not m:
        raise ValueError(f"teacher suffix {teacher_suffix!r} is not a supervised or pseudo-label run")
    # "pl" counts as round 1, so add one to the existing round number (empty digits are treated as 1).
    return f"pl{int(m.group(1) or 1) + 1}"


# Q: What does student_config do?
# A: It builds the student's TrainConfig from a teacher (a run name, read from its config.yaml, or a TrainConfig
#   object): copies all known fields except run_name, sets name_suffix, teacher and pseudo_labels="pseudo_labels.csv",
#   then applies overrides.
# Q: Why can the teacher be a TrainConfig object?
# A: So run_grid.py can list the students of teachers that are not finished yet.
# Q: Why drop run_name and unknown keys?
# A: Dropping run_name makes TrainConfig rebuild a new name from the new suffix; unknown keys from old config.yaml files
#   would otherwise make TrainConfig(...) fail.
# Q: Why does the student keep the teacher's fraction and seed?
# A: To use the same labelled subset and the same initialisation seed, which makes teacher and student a paired
#   comparison.
def student_config(teacher: str | TrainConfig, **overrides) -> TrainConfig:
    """Student config from a teacher run name (reads its config.yaml) or a planned teacher config.

    A planned `TrainConfig` lets run_grid.py list the students of teachers that are not finished yet.
    The student copies every known field of the teacher except `run_name` (so the name is rebuilt from
    the new suffix), and sets `teacher` and `pseudo_labels` ("pseudo_labels.csv", resolved inside the
    student's own run folder). `overrides` are TrainConfig fields applied on top; `name_suffix` in
    them wins over the automatic suffix. Unknown keys in an old config.yaml are dropped.

    Args:
        teacher: the teacher's run name, or a TrainConfig describing a teacher that may not exist yet.
        **overrides: TrainConfig fields to change for the student.
    Returns:
        The student's TrainConfig.
    """
    if isinstance(teacher, str):
        # Teacher given by name: read the settings it was trained with.
        values = yaml.safe_load((RUNS_DIR / teacher / "config.yaml").read_text())
        name = teacher
    else:
        # Teacher given as a config object: use its settings directly.
        values, name = teacher.to_dict(), teacher.run_name
    # Keep only keys that TrainConfig still knows (old files may contain removed settings) and drop the
    # old run name, so the student gets a new one.
    known = {f.name for f in fields(TrainConfig)}
    values = {k: v for k, v in values.items() if k in known and k != "run_name"}
    # Use an explicit suffix override if given, otherwise the next automatic one ("pl", "pl2", ...).
    suffix = overrides.pop("name_suffix", None) or student_suffix(values.get("name_suffix") or "")
    # fraction, seed and model settings stay those of the teacher: same labelled subset, same init seed
    values.update(name_suffix=suffix, teacher=name, pseudo_labels="pseudo_labels.csv", **overrides)
    return TrainConfig(**values)


# Q: What does train_student do, step by step?
# A: Checks that the teacher finished (metrics.json and both prediction files exist); builds the student config; if the
#   student is already finished returns its stored metrics; otherwise writes pseudo-labels (tau chosen on validation),
#   prints tau and the counts, and trains with engine.run.
# Q: Why must the teacher be finished first?
# A: The pseudo-labels are built from the teacher's saved predictions; without them there is nothing to select. A
#   FileNotFoundError says so.
# Q: What are the inputs and the return value?
# A: teacher_name (folder under results/runs), target_precision, balance, overwrite and **overrides for the student
#   config; it returns the student's metrics dict.
# Q: Why return early for a finished student?
# A: run() would skip it anyway; returning before make_pseudo_labels avoids rewriting the pseudo-label files of a
#   finished run.
def train_student(teacher_name: str, target_precision: float = 0.95, balance: bool = False,
                  overwrite: bool = False, **overrides) -> dict:
    """Create pseudo-labels from a finished teacher run and train the student; return its metrics.

    Args:
        teacher_name: run folder name under results/runs/ (a supervised run or an earlier student).
        target_precision: validation precision that tau must reach (default 0.95).
        balance: class-balanced pseudo-label selection (`--balance`).
        overwrite: redo a student that already has metrics.json (otherwise it is skipped).
        **overrides: TrainConfig fields for the student (e.g. name_suffix, lr).
    Raises:
        FileNotFoundError: if the teacher lacks metrics.json, preds_val.npz or preds_unlabelled.npz.
    Side effects: writes pseudo_labels.csv, pseudo_label_*.csv/json and the usual run files into the
    student's run folder. A finished student is skipped before any pseudo-label files are rewritten.
    """
    # The teacher's run folder.
    tdir = RUNS_DIR / teacher_name
    # The teacher must be finished and must have saved the predictions the pseudo-labels are built from.
    for f in ("metrics.json", "preds_val.npz", "preds_unlabelled.npz"):
        if not (tdir / f).exists():
            raise FileNotFoundError(f"teacher {teacher_name} has no {f}; finish the supervised run first")
    # Build the student's settings and locate its run folder.
    cfg = student_config(teacher_name, **overrides)
    sdir = RUNS_DIR / cfg.run_name
    # If the student already finished, `run` just returns its stored metrics; no pseudo-label files are rewritten.
    if (sdir / "metrics.json").exists() and not overwrite:
        return run(cfg)  # skips and returns the stored metrics
    # Choose the confidence threshold on validation data and write the pseudo-label files into the student folder.
    sel = make_pseudo_labels(tdir, sdir, cfg.fraction, cfg.seed, target_precision, balance)
    # Report which threshold was chosen and how many unlabelled images received a pseudo-label.
    print(f"[{cfg.run_name}] teacher {teacher_name}: tau={sel['tau']:.2f} "
          f"(val precision {sel['val_precision_at_tau']:.3f}), {sel['n_selected']}/{sel['pool_size']} pseudo-labels")
    # Train the student on labelled + pseudo-labelled images and return its metrics.
    return run(cfg, overwrite=overwrite)

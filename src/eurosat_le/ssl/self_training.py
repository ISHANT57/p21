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

from __future__ import annotations

import re
from dataclasses import fields

import yaml

from eurosat_le.ssl.pseudo_label import make_pseudo_labels
from eurosat_le.train.config import TrainConfig
from eurosat_le.train.engine import RUNS_DIR, run


def student_suffix(teacher_suffix: str) -> str:
    """'' -> 'pl', 'pl' -> 'pl2', 'pl2' -> 'pl3' (one more self-training round)."""
    if not teacher_suffix:
        return "pl"
    m = re.fullmatch(r"pl(\d*)", teacher_suffix)  # rounds continue from default students only
    if not m:
        raise ValueError(f"teacher suffix {teacher_suffix!r} is not a supervised or pseudo-label run")
    return f"pl{int(m.group(1) or 1) + 1}"


def student_config(teacher: str | TrainConfig, **overrides) -> TrainConfig:
    """Student config from a teacher run name (reads its config.yaml) or a planned teacher config.

    A planned `TrainConfig` lets run_grid.py list the students of teachers that are not finished yet.
    The student copies every known field of the teacher except `run_name` (so the name is rebuilt from
    the new suffix), and sets `teacher` and `pseudo_labels` ("pseudo_labels.csv", resolved inside the
    student's own run folder). `overrides` are TrainConfig fields applied on top; `name_suffix` in
    them wins over the automatic suffix. Unknown keys in an old config.yaml are dropped.
    """
    if isinstance(teacher, str):
        values = yaml.safe_load((RUNS_DIR / teacher / "config.yaml").read_text())
        name = teacher
    else:
        values, name = teacher.to_dict(), teacher.run_name
    known = {f.name for f in fields(TrainConfig)}
    values = {k: v for k, v in values.items() if k in known and k != "run_name"}
    suffix = overrides.pop("name_suffix", None) or student_suffix(values.get("name_suffix") or "")
    # fraction, seed and model settings stay those of the teacher: same labelled subset, same init seed
    values.update(name_suffix=suffix, teacher=name, pseudo_labels="pseudo_labels.csv", **overrides)
    return TrainConfig(**values)


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
    tdir = RUNS_DIR / teacher_name
    for f in ("metrics.json", "preds_val.npz", "preds_unlabelled.npz"):
        if not (tdir / f).exists():
            raise FileNotFoundError(f"teacher {teacher_name} has no {f}; finish the supervised run first")
    cfg = student_config(teacher_name, **overrides)
    sdir = RUNS_DIR / cfg.run_name
    if (sdir / "metrics.json").exists() and not overwrite:
        return run(cfg)  # skips and returns the stored metrics
    sel = make_pseudo_labels(tdir, sdir, cfg.fraction, cfg.seed, target_precision, balance)
    print(f"[{cfg.run_name}] teacher {teacher_name}: tau={sel['tau']:.2f} "
          f"(val precision {sel['val_precision_at_tau']:.3f}), {sel['n_selected']}/{sel['pool_size']} pseudo-labels")
    return run(cfg, overwrite=overwrite)

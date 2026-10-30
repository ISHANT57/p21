"""Build and train a pseudo-label student from a finished teacher run."""

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
    """Student config from a teacher run name (reads its config.yaml) or a planned teacher config."""
    if isinstance(teacher, str):
        values = yaml.safe_load((RUNS_DIR / teacher / "config.yaml").read_text())
        name = teacher
    else:
        values, name = teacher.to_dict(), teacher.run_name
    known = {f.name for f in fields(TrainConfig)}
    values = {k: v for k, v in values.items() if k in known and k != "run_name"}
    suffix = overrides.pop("name_suffix", None) or student_suffix(values.get("name_suffix") or "")
    values.update(name_suffix=suffix, teacher=name, pseudo_labels="pseudo_labels.csv", **overrides)
    return TrainConfig(**values)


def train_student(teacher_name: str, target_precision: float = 0.95, balance: bool = False,
                  overwrite: bool = False, **overrides) -> dict:
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

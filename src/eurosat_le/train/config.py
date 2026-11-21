"""Training configuration: YAML file + command-line overrides -> TrainConfig.

One `TrainConfig` describes one run: the experiment cell (model x input type x label fraction x
seed), the optimisation settings and the data-loading settings. It is built by `load_config` from
a per-model YAML file in configs/ (the frozen hyper-parameters) plus overrides, in this order of
priority: explicit keyword overrides (command-line flags, `--set key=value`) over the YAML file
over the dataclass defaults. Unknown keys raise an error, so a typo in a YAML file or in `--set`
cannot silently be ignored. The resolved config is written to the run folder as config.yaml.

Training budget rule (see docs/MODELS.md, "Training budget"): every run trains for
    max(epochs x one pass over its labelled set, min_steps)
optimiser steps, with the same number of epochs (= validation checks, and the unit of `patience`)
for every label fraction. When one pass is too short (1% / 10% labels), each epoch is stretched to
ceil(min_steps / epochs) steps by sampling with replacement; this is `resolve_steps_per_epoch`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from eurosat_le.data.loaders import DataConfig


@dataclass
class TrainConfig:
    """All settings of one training run; `to_dict()` is what gets saved as config.yaml.

    The field comments below are the authoritative description of each setting. Two fields are
    worth stressing: `seed` drives both the labelled-subset draw (splits/subsets/*_seed<s>.txt) and
    the model initialisation / shuffling / augmentation, and `evaluate_test=False` (tuning runs)
    keeps the test set completely untouched.
    """

    # experiment cell
    model: str = "scratch_cnn"
    modality: str = "rgb"
    fraction: float = 1.0
    seed: int = 0                     # labelled-subset draw AND init/shuffle seed
    # model
    pretrained: bool = True
    init_mode: str = "mean"           # 13-band first-conv init: "mean" or "zero" (see docs/MODELS.md)
    input_size: int | None = None     # None = architecture default (64 or 224)
    # optimisation
    epochs: int = 30
    steps_per_epoch: int | None = None  # None = one pass over the labelled set per epoch
    min_steps: int | None = None      # budget floor: stretch small-label epochs to reach this many steps
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    label_smoothing: float = 0.0
    patience: int = 10                # early stopping on val macro-F1 (epochs without improvement)
    amp: bool = True                  # mixed precision on CUDA
    aug: str = "weak"
    # data loading
    num_workers: int = 2
    eval_batch_size: int = 256
    in_memory: bool = False
    # pseudo-labelling student (Phase 5); set by eurosat_le.ssl.self_training
    pseudo_labels: str | None = None  # CSV (id, label, confidence) inside the run dir
    teacher: str | None = None        # run name of the teacher that produced the pseudo-labels
    # bookkeeping
    run_name: str | None = None       # default: <model>_<modality>_<pct>pct_seed<seed>[_<suffix>]
    name_suffix: str = ""
    evaluate_test: bool = True       # False for tuning runs: the test set stays untouched
    save_checkpoint: bool = False     # best weights to best.pt (large; predictions are always saved)
    max_train_batches: int | None = None  # debug only: cap batches per epoch

    def __post_init__(self):
        # Default run folder name, e.g. "resnet50_ms_1pct_seed0" or "resnet50_ms_1pct_seed0_pl" for a student.
        if self.run_name is None:
            name = f"{self.model}_{self.modality}_{self.fraction * 100:g}pct_seed{self.seed}"
            self.run_name = f"{name}_{self.name_suffix}" if self.name_suffix else name

    def resolve_steps_per_epoch(self, n_labelled: int) -> int | None:
        """Budget rule: total steps = max(epochs x one pass over the labelled set, min_steps).

        Returns None for a plain pass over the data (shuffle, drop_last), or a fixed
        number of steps per epoch (sampling with replacement) when the plain pass
        would fall short of min_steps. The number of epochs, and so the number of
        validation checks and the meaning of `patience`, stays the same.
        """
        if self.steps_per_epoch:
            return self.steps_per_epoch
        # steps of one plain pass: the plain loader uses drop_last, hence floor division
        natural = max(1, n_labelled // self.batch_size)
        if self.min_steps and self.epochs * natural < self.min_steps:
            return math.ceil(self.min_steps / self.epochs)
        return None

    def data_config(self, steps_per_epoch: int | None = None) -> DataConfig:
        """Build the loader settings. `steps_per_epoch` (from `resolve_steps_per_epoch`) wins over the config field.

        `subset_seed` and the loader `seed` are both `self.seed`, tying the labelled subset and the shuffling together.
        """
        return DataConfig(modality=self.modality, fraction=self.fraction, subset_seed=self.seed,
                          batch_size=self.batch_size, eval_batch_size=self.eval_batch_size,
                          num_workers=self.num_workers, aug=self.aug,
                          steps_per_epoch=steps_per_epoch or self.steps_per_epoch,
                          in_memory=self.in_memory, seed=self.seed)

    def to_dict(self) -> dict:
        """Plain dict of all fields (used for config.yaml and to clone a teacher's config for a student)."""
        return asdict(self)


def load_config(path: str | Path | None = None, **overrides) -> TrainConfig:
    """Load a TrainConfig from a YAML file (optional) and keyword overrides.

    Overrides whose value is None are ignored, so unset argparse options (default None) do not
    clobber the YAML value. Raises ValueError for keys that are not TrainConfig fields.
    """
    values = yaml.safe_load(Path(path).read_text()) if path else {}
    values.update({k: v for k, v in overrides.items() if v is not None})
    # `from __future__ import annotations` makes f.type a string such as "float" or "int | None"
    types = {f.name: str(f.type) for f in fields(TrainConfig)}
    unknown = set(values) - set(types)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return TrainConfig(**{k: _coerce(v, types[k]) for k, v in values.items()})


def _coerce(value, type_name: str):
    """YAML 1.1 reads '3e-4' as a string; convert to the field's declared numeric type."""
    if value is None or isinstance(value, bool):
        return value
    if type_name.startswith("float"):
        return float(value)
    if type_name.startswith("int"):
        return int(value)
    return value


def parse_set(items: list[str]) -> dict:
    """Parse ["lr=3e-4", "amp=false"] into typed values using YAML scalars."""
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        out[key.strip()] = yaml.safe_load(value)
    return out

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

# Makes type hints such as `int | None` work as plain text on older Python versions.
from __future__ import annotations

# math: ceil() for rounding the steps-per-epoch up to a whole number.
# dataclasses: the @dataclass decorator, plus helpers to list a dataclass's fields and turn it into a dict.
# pathlib.Path: convenient, OS-independent file paths.
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

# yaml: reads the per-model .yaml config files and writes config.yaml for each run.
import yaml

# DataConfig: the settings object that the data-loading code expects (batch size, workers, augmentation ...).
from eurosat_le.data.loaders import DataConfig


@dataclass
class TrainConfig:
    """All settings of one training run; `to_dict()` is what gets saved as config.yaml.

    The field comments below are the authoritative description of each setting. Two fields are
    worth stressing: `seed` drives both the labelled-subset draw (splits/subsets/*_seed<s>.txt) and
    the model initialisation / shuffling / augmentation, and `evaluate_test=False` (tuning runs)
    keeps the test set completely untouched.

    Every attribute has a default value, so `TrainConfig()` works with no arguments; the YAML file
    and command-line overrides replace only the values they mention.
    """

    # experiment cell
    # Which network to train: "scratch_cnn", "resnet50", "vit_small" or "vit_scratch".
    model: str = "scratch_cnn"
    # Input type: "rgb" (3 bands) or "ms" (all 13 Sentinel-2 bands).
    modality: str = "rgb"
    # Share of the training images whose labels we pretend to know (0.01, 0.1 or 1.0).
    fraction: float = 1.0
    seed: int = 0                     # labelled-subset draw AND init/shuffle seed
    # model
    # Start from ImageNet-pretrained weights (only used for resnet50 / vit_small).
    pretrained: bool = True
    init_mode: str = "mean"           # 13-band first-conv init: "mean" or "zero" (see docs/MODELS.md)
    input_size: int | None = None     # None = architecture default (64 or 224)
    # optimisation
    # Number of epochs; one epoch ends with one validation check.
    epochs: int = 30
    steps_per_epoch: int | None = None  # None = one pass over the labelled set per epoch
    min_steps: int | None = None      # budget floor: stretch small-label epochs to reach this many steps
    # Number of images per optimiser step.
    batch_size: int = 64
    # Peak learning rate: how big a step the optimiser takes along the gradient.
    lr: float = 1e-3
    # AdamW weight decay: a small pull of the weights towards 0 each step, which fights over-fitting.
    weight_decay: float = 0.05
    # Fraction of all steps spent slowly raising the learning rate from ~0 up to `lr`.
    warmup_frac: float = 0.05
    # Softens the one-hot targets slightly (0 = off) so the model is not pushed to be over-confident.
    label_smoothing: float = 0.0
    patience: int = 10                # early stopping on val macro-F1 (epochs without improvement)
    amp: bool = True                  # mixed precision on CUDA
    # Name of the data-augmentation recipe applied to training images.
    aug: str = "weak"
    # data loading
    # Number of background processes that load and augment images.
    num_workers: int = 2
    # Batch size used for validation / test / pool prediction (larger is fine because no gradients are stored).
    eval_batch_size: int = 256
    # Keep all images in RAM instead of reading files from disk each time.
    in_memory: bool = False
    # pseudo-labelling student (Phase 5); set by eurosat_le.ssl.self_training
    pseudo_labels: str | None = None  # CSV (id, label, confidence) inside the run dir
    teacher: str | None = None        # run name of the teacher that produced the pseudo-labels
    # bookkeeping
    run_name: str | None = None       # default: <model>_<modality>_<pct>pct_seed<seed>[_<suffix>]
    # Optional text appended to the automatic run name (e.g. "pl" for pseudo-label students).
    name_suffix: str = ""
    evaluate_test: bool = True       # False for tuning runs: the test set stays untouched
    save_checkpoint: bool = False     # best weights to best.pt (large; predictions are always saved)
    max_train_batches: int | None = None  # debug only: cap batches per epoch

    def __post_init__(self):
        """Runs automatically right after the dataclass is created; fills in a default `run_name`."""
        # Default run folder name, e.g. "resnet50_ms_1pct_seed0" or "resnet50_ms_1pct_seed0_pl" for a student.
        if self.run_name is None:
            # Build the base name from the experiment cell; `:g` prints 0.01*100 as "1" rather than "1.0".
            name = f"{self.model}_{self.modality}_{self.fraction * 100:g}pct_seed{self.seed}"
            # Append the suffix only if there is one.
            self.run_name = f"{name}_{self.name_suffix}" if self.name_suffix else name

    def resolve_steps_per_epoch(self, n_labelled: int) -> int | None:
        """Budget rule: total steps = max(epochs x one pass over the labelled set, min_steps).

        Args:
            n_labelled: number of training images (labelled, plus pseudo-labelled for a student).
        Returns None for a plain pass over the data (shuffle, drop_last), or a fixed
        number of steps per epoch (sampling with replacement) when the plain pass
        would fall short of min_steps. The number of epochs, and so the number of
        validation checks and the meaning of `patience`, stays the same.
        """
        # An explicitly configured value always wins.
        if self.steps_per_epoch:
            return self.steps_per_epoch
        # steps of one plain pass: the plain loader uses drop_last, hence floor division
        natural = max(1, n_labelled // self.batch_size)
        # If the plain passes would give fewer steps than the minimum budget, stretch each epoch
        # to ceil(min_steps / epochs) steps (rounded up so the budget is reached).
        if self.min_steps and self.epochs * natural < self.min_steps:
            return math.ceil(self.min_steps / self.epochs)
        # Otherwise one ordinary pass over the data per epoch is enough.
        return None

    def data_config(self, steps_per_epoch: int | None = None) -> DataConfig:
        """Build the loader settings. `steps_per_epoch` (from `resolve_steps_per_epoch`) wins over the config field.

        `subset_seed` and the loader `seed` are both `self.seed`, tying the labelled subset and the shuffling together.
        Returns a `DataConfig` ready to pass to `build_loaders`.
        """
        # Copy the data-related settings into a DataConfig; the seed is used twice (which labelled
        # images are chosen, and the shuffling order) so one number reproduces the whole run.
        return DataConfig(modality=self.modality, fraction=self.fraction, subset_seed=self.seed,
                          batch_size=self.batch_size, eval_batch_size=self.eval_batch_size,
                          num_workers=self.num_workers, aug=self.aug,
                          steps_per_epoch=steps_per_epoch or self.steps_per_epoch,
                          in_memory=self.in_memory, seed=self.seed)

    def to_dict(self) -> dict:
        """Plain dict of all fields (used for config.yaml and to clone a teacher's config for a student)."""
        # asdict converts the dataclass into an ordinary {field_name: value} dictionary.
        return asdict(self)


def load_config(path: str | Path | None = None, **overrides) -> TrainConfig:
    """Load a TrainConfig from a YAML file (optional) and keyword overrides.

    Args:
        path: YAML file to read, or None to start from the dataclass defaults only.
        **overrides: TrainConfig field values that take priority over the YAML file.
    Returns:
        The resulting TrainConfig.

    Overrides whose value is None are ignored, so unset argparse options (default None) do not
    clobber the YAML value. Raises ValueError for keys that are not TrainConfig fields.
    """
    # Read the YAML file into a dict (or start with an empty dict when no file is given).
    values = yaml.safe_load(Path(path).read_text()) if path else {}
    # Apply the overrides on top, skipping any that were left unset (None).
    values.update({k: v for k, v in overrides.items() if v is not None})
    # `from __future__ import annotations` makes f.type a string such as "float" or "int | None"
    # Map each valid field name to its declared type (as text), used below to coerce numbers.
    types = {f.name: str(f.type) for f in fields(TrainConfig)}
    # Any key that is not a real field is almost certainly a typo, so stop with a clear error.
    unknown = set(values) - set(types)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    # Fix up number types (e.g. "3e-4" read as text) and build the config object.
    return TrainConfig(**{k: _coerce(v, types[k]) for k, v in values.items()})


def _coerce(value, type_name: str):
    """YAML 1.1 reads '3e-4' as a string; convert to the field's declared numeric type.

    Args:
        value: the raw value from YAML or the command line.
        type_name: the declared type of the field as text, e.g. "float", "int | None", "str".
    Returns:
        The value converted to float/int when the field is numeric, otherwise unchanged.
    """
    # None and booleans must pass through untouched (bool is a subclass of int, so check it first).
    if value is None or isinstance(value, bool):
        return value
    # Fields declared float (or "float | None") become real floats.
    if type_name.startswith("float"):
        return float(value)
    # Fields declared int (or "int | None") become real ints.
    if type_name.startswith("int"):
        return int(value)
    # Strings and everything else are kept as they are.
    return value


def parse_set(items: list[str]) -> dict:
    """Parse ["lr=3e-4", "amp=false"] into typed values using YAML scalars.

    Args:
        items: strings of the form "key=value", as given to `--set` on the command line.
    Returns:
        A dict {key: parsed value}; YAML turns "false" into False, "3e-4" into a number, etc.
    """
    out = {}
    for item in items:
        # Split at the first "=": key on the left, value text on the right.
        key, _, value = item.partition("=")
        # Let YAML decide whether the text is a number, boolean, null or string.
        out[key.strip()] = yaml.safe_load(value)
    return out

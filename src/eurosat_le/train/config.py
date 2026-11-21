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


# ======================================== CONCEPTS USED IN THIS FILE ========================================
# dataclass: a Python class whose fields (name, type, default value) are declared as plain annotated lines; Python
#   writes __init__ for it. TrainConfig is one.
# YAML: a human-readable text format for settings (key: value). configs/<model>.yaml holds the frozen
#   hyper-parameters; load_config reads them; config.yaml is also written into every run folder.
# override priority: command-line value (keyword override) beats the YAML file, which beats the default written in the
#   dataclass. See load_config.
# hyper-parameter: a setting chosen by the person (learning rate, batch size, epochs ...), not learned by the model.
# epoch: one round of training followed by one validation check. Here every label fraction gets the same number of
#   epochs.
# batch / batch_size: a group of images processed together; one batch = one optimiser step. batch_size = images per
#   step.
# step (optimiser step): one weight update from one batch. total steps = epochs x steps per epoch.
# training budget rule: total steps = max(epochs x one pass over the labelled set, min_steps); see
#   resolve_steps_per_epoch.
# drop_last / sampling with replacement: a plain pass drops the last incomplete batch (so floor division); small label
#   sets are instead sampled with repeats to reach the step budget.
# fraction (label fraction): share of the training images whose labels we pretend to know (1%, 10%, 100%). The rest is
#   the unlabelled pool.
# seed: number that fixes randomness; here it fixes BOTH which images are labelled and the model init / shuffling /
#   augmentation.
# modality: "rgb" = 3 bands, "ms" = all 13 Sentinel-2 bands (so the model needs 3 or 13 input channels).
# pretrained / init_mode: start from ImageNet weights; init_mode ("mean" or "zero") says how the extra 10 bands of the
#   first layer are initialised (see models/__init__.py).
# learning rate (lr): size of each weight update. AdamW: the optimiser used. weight decay: small pull of weights to 0
#   against over-fitting.
# warm-up (warmup_frac): fraction of all steps used to raise the learning rate slowly from ~0 to lr.
# label smoothing: softens the 0/1 target labels slightly so the model is not pushed to be over-confident (0 = off).
# early stopping / patience: stop if validation macro-F1 has not improved for `patience` epochs in a row.
# amp (automatic mixed precision): do the maths in 16-bit floats where safe (GPU only) for speed and memory.
# validation vs test set: validation is used to pick the best epoch and tune; the test set is only for the final
#   score. evaluate_test=False keeps the test set untouched (tuning runs).
# pseudo-label / teacher / student: a teacher run labels unlabelled images with its own confident predictions; a
#   student is trained on labelled + pseudo-labelled images. Fields pseudo_labels and teacher record this.
# num_workers / in_memory: background processes that load images / keep all images in RAM instead of reading disk each
#   time.
# checkpoint (save_checkpoint): the saved best weights file best.pt; off by default because it is large.

# ======================================== Q&A ABOUT THIS FILE ========================================
# Q: What is this file for?
# A: It defines TrainConfig, the single object that describes one training run, and the helpers that build it from a
#   YAML file plus command-line overrides.
# Q: Where does it fit in the pipeline?
# A: scripts and grid.py build TrainConfig objects; train/engine.py `run` receives one and trains;
#   ssl/self_training.py clones a teacher's config into a student's.
# Q: What does it read and write?
# A: It reads configs/<model>.yaml (and the DataConfig class from data/loaders.py). It writes nothing itself;
#   engine.run saves cfg.to_dict() as config.yaml in the run folder.
# Q: Why use a dataclass and reject unknown keys?
# A: The dataclass gives typed fields with defaults in one place. Rejecting unknown keys means a typo such as
#   "lerning_rate" raises an error instead of being silently ignored.

# Q: What is the TrainConfig class?
# A: A dataclass holding every setting of one run: which model/input/label fraction/seed (the experiment cell),
#   optimisation settings, data-loading settings, pseudo-label fields and bookkeeping.
# Q: Why put all settings in one object instead of passing many arguments around?
# A: One object can be saved as config.yaml, so a run can be reproduced exactly, and it can be copied (to_dict) to
#   make a student config from a teacher.
# Q: Do I have to give every value?
# A: No. Every field has a default, so TrainConfig() works; YAML and command-line overrides replace only the values
#   they mention.
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

    # Q: What does __post_init__ do and when does it run?
    # A: Python calls it automatically right after the dataclass is created. It fills in run_name (e.g.
    #   resnet50_ms_1pct_seed0, plus _<suffix> if given) when no name was supplied.
    # Q: Why build the name from the experiment cell?
    # A: So every (model, modality, fraction, seed, variant) gets a unique, predictable folder name; the name_suffix
    #   (e.g. "pl", "tune_lr0.001") stops tuning runs and students overwriting grid runs.
    def __post_init__(self):
        """Runs automatically right after the dataclass is created; fills in a default `run_name`."""
        # Default run folder name, e.g. "resnet50_ms_1pct_seed0" or "resnet50_ms_1pct_seed0_pl" for a student.
        if self.run_name is None:
            # Build the base name from the experiment cell; `:g` prints 0.01*100 as "1" rather than "1.0".
            name = f"{self.model}_{self.modality}_{self.fraction * 100:g}pct_seed{self.seed}"
            # Append the suffix only if there is one.
            self.run_name = f"{name}_{self.name_suffix}" if self.name_suffix else name

    # Q: What does this function do?
    # A: It applies the training budget rule: it returns None for a normal pass over the data, or a fixed number of
    #   steps per epoch when one pass is too short to reach min_steps.
    # Q: Why is it needed? What would go wrong without it?
    # A: With 1% labels one pass is only a handful of batches, so the model would barely train. Stretching each epoch
    #   (sampling with replacement) gives small label sets enough updates while keeping the same number of epochs
    #   (validation checks, patience) for every fraction.
    # Q: What are the inputs and the output?
    # A: Input n_labelled = number of training images (labelled plus pseudo-labelled for a student). Output: an int
    #   (steps per epoch) or None. An explicitly configured steps_per_epoch always wins.
    # Q: Why floor division and ceil?
    # A: The plain loader drops the last incomplete batch (drop_last), so one pass = n // batch_size steps; the
    #   stretched epoch uses ceil(min_steps / epochs) so the budget is reached, not undershot.
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

    # Q: What does this method do?
    # A: It copies the data-related settings of the run into a DataConfig object that build_loaders needs (modality,
    #   fraction, batch sizes, workers, augmentation, steps per epoch, seeds).
    # Q: Why is the seed passed twice (subset_seed and seed)?
    # A: subset_seed decides WHICH images count as labelled, seed decides shuffling/augmentation order. Using
    #   self.seed for both means one number reproduces the entire run.
    # Q: What is the steps_per_epoch argument?
    # A: The value computed by resolve_steps_per_epoch; if given it overrides the config field, otherwise the field is
    #   used.
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

    # Q: What does to_dict return and why?
    # A: An ordinary {field: value} dict of all settings (dataclasses.asdict). It is saved as config.yaml and read
    #   back to make a student's config from a teacher's.
    def to_dict(self) -> dict:
        """Plain dict of all fields (used for config.yaml and to clone a teacher's config for a student)."""
        # asdict converts the dataclass into an ordinary {field_name: value} dictionary.
        return asdict(self)


# Q: What does this function do?
# A: It builds a TrainConfig from an optional YAML file plus keyword overrides, with priority overrides > YAML >
#   dataclass defaults.
# Q: Why are None overrides ignored?
# A: Unset command-line options arrive as None; if they were applied they would wipe out the value from the YAML file.
# Q: Why check for unknown keys?
# A: A key that is not a TrainConfig field is almost certainly a typo; raising ValueError stops a silently wrong
#   experiment.
# Q: Why call _coerce on each value?
# A: YAML can read a number like 3e-4 as a string; _coerce converts values to the type the field declares.
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


# Q: What does _coerce do?
# A: It converts a raw value to float or int when the field's declared type starts with "float" or "int"; None, bool
#   and other types are returned unchanged.
# Q: Why are booleans checked first?
# A: In Python bool is a subclass of int, so True would otherwise be turned into 1 for an int-typed check.
# Q: Why is it needed at all?
# A: Because of `from __future__ import annotations` the field types are strings (e.g. "int | None"), and YAML 1.1
#   reads "3e-4" as text, not a number.
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


# Q: What does parse_set do?
# A: It turns command-line `--set key=value` strings such as ["lr=3e-4", "amp=false"] into a dict with proper Python
#   types.
# Q: Why use yaml.safe_load for the value?
# A: YAML already knows how to read numbers, true/false, null and strings, so we do not have to write our own parser.
# Q: Why partition at the first "="?
# A: So the key is everything before the first "=" and the value is the rest, even if the value itself contains "=".
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

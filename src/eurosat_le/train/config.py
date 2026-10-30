"""Training configuration: YAML file + command-line overrides -> TrainConfig."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from eurosat_le.data.loaders import DataConfig


@dataclass
class TrainConfig:
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
    # bookkeeping
    run_name: str | None = None       # default: <model>_<modality>_<pct>pct_seed<seed>[_<suffix>]
    name_suffix: str = ""
    save_checkpoint: bool = False     # best weights to best.pt (large; predictions are always saved)
    max_train_batches: int | None = None  # debug only: cap batches per epoch

    def __post_init__(self):
        if self.run_name is None:
            name = f"{self.model}_{self.modality}_{self.fraction * 100:g}pct_seed{self.seed}"
            self.run_name = f"{name}_{self.name_suffix}" if self.name_suffix else name

    def data_config(self) -> DataConfig:
        return DataConfig(modality=self.modality, fraction=self.fraction, subset_seed=self.seed,
                          batch_size=self.batch_size, eval_batch_size=self.eval_batch_size,
                          num_workers=self.num_workers, aug=self.aug, steps_per_epoch=self.steps_per_epoch,
                          in_memory=self.in_memory, seed=self.seed)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path | None = None, **overrides) -> TrainConfig:
    values = yaml.safe_load(Path(path).read_text()) if path else {}
    values.update({k: v for k, v in overrides.items() if v is not None})
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

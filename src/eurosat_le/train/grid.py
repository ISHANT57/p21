"""Experiment grids: the main supervised grid and the learning-rate tuning sweep.

This module only builds lists of `TrainConfig` objects; `scripts/run_grid.py` runs them (and builds
the Phase 5 pseudo-label students from the same grid, see `PL_MODELS` / `PL_FRACTIONS`).

- Main grid (Phase 4): 3 models x 2 input types x 3 label fractions x 3 seeds = 54 runs. Each config
  starts from configs/<model>.yaml, so the learning rate and other hyper-parameters are the same for
  every input type, fraction and seed.
- Tuning sweep: 3 learning rates per model on a single cell (RGB, 10% labels, seed 0), chosen by
  VALIDATION macro-F1 only. Tuning runs set `evaluate_test=False`, so the test set is never used for
  tuning, and their names carry the suffix `tune_lr<lr>` so they never collide with grid runs.
"""

from __future__ import annotations

from itertools import product

from eurosat_le.constants import LABEL_FRACTIONS, REPO_ROOT, SUBSET_SEEDS
from eurosat_le.train.config import TrainConfig, load_config

CONFIG_DIR = REPO_ROOT / "configs"
MAIN_MODELS = ["scratch_cnn", "resnet50", "vit_small"]
MODALITIES = ["rgb", "ms"]
# Phase 5 scope: scratch CNN + one pretrained model (switch to the best pretrained model
# once Phase 4 results are in), 1% and 10% labels.
PL_MODELS = ["scratch_cnn", "resnet50"]
PL_FRACTIONS = [0.01, 0.1]

# Learning-rate candidates for the light tuning sweep (centre = current config value).
TUNE_LRS = {
    "scratch_cnn": [1e-3, 2e-3, 4e-3],
    "resnet50": [2e-4, 5e-4, 1e-3],
    "vit_small": [5e-5, 1e-4, 2e-4],
    "vit_scratch": [5e-4, 1e-3, 2e-3],
}
TUNE_CELL = {"modality": "rgb", "fraction": 0.1, "seed": 0}


def main_grid(models=MAIN_MODELS, modalities=MODALITIES, fractions=LABEL_FRACTIONS, seeds=SUBSET_SEEDS,
              **overrides) -> list[TrainConfig]:
    """All cells, ordered seed-first so that seed 0 of every cell finishes before seed 1 starts.

    Args:
        models, modalities, fractions, seeds: the axes of the grid ("rgb"/"ms" for modalities).
        **overrides: TrainConfig fields applied to every run (e.g. from `--set key=value`).
    Returns:
        One TrainConfig per cell. Seed-first ordering means a session that is cut short still has
        complete coverage of the grid for the early seeds. `--shard k/n` in run_grid.py later takes
        every n-th entry of this list, so the ordering also decides which runs a shard gets.
    """
    return [
        load_config(CONFIG_DIR / f"{m}.yaml", modality=mod, fraction=f, seed=s, **overrides)
        for s, m, mod, f in product(seeds, models, modalities, fractions)
    ]


def tune_grid(models=MAIN_MODELS, **overrides) -> list[TrainConfig]:
    """LR sweep on one cell (RGB, 10% labels, seed 0), selected on validation macro-F1 only.

    Tuning runs never evaluate the test set. Run names end in `_tune_lr<lr>`. Extra `overrides`
    can change the cell (modality / fraction / seed) or any other field.
    """
    return [
        load_config(CONFIG_DIR / f"{m}.yaml", lr=lr, name_suffix=f"tune_lr{lr:g}", evaluate_test=False,
                    **{**TUNE_CELL, **overrides})
        for m in models for lr in TUNE_LRS[m]
    ]

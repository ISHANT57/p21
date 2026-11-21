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

# Makes type hints such as `int | None` work as plain text on older Python versions.
from __future__ import annotations

# product: builds every combination of several lists (the "grid").
from itertools import product

# Project constants: the label fractions and subset seeds of the study, and the repository root folder.
from eurosat_le.constants import LABEL_FRACTIONS, REPO_ROOT, SUBSET_SEEDS
# TrainConfig describes one run; load_config builds one from a YAML file plus overrides.
from eurosat_le.train.config import TrainConfig, load_config

# ======================================== CONCEPTS USED IN THIS FILE ========================================
# experiment grid: the list of every (model x input type x label fraction x seed) combination that is trained; main grid
#   = 3 x 2 x 3 x 3 = 54 runs.
# itertools.product: builds every combination of several lists (a Cartesian product); the LAST list varies fastest, the
#   FIRST is the outer loop.
# seed-first ordering: seed is the outermost loop, so seed 0 of every cell finishes before seed 1 starts; a cut-off
#   session still covers the whole grid.
# label fraction / seed: share of training labels assumed known (1%, 10%, 100%); the seed picks which images those are
#   and the initialisation.
# modality: "rgb" (3 bands) or "ms" (all 13 Sentinel-2 bands).
# learning rate (lr) tuning: trying a few lr values per model and keeping the one with the best VALIDATION macro-F1.
# validation vs test set: tuning may only look at validation; tuning runs set evaluate_test=False so the test set stays
#   untouched (no test leakage).
# name_suffix: text appended to run_name (here "tune_lr<lr>") so tuning runs never overwrite grid runs.
# Phase 5 / pseudo-label students (PL_MODELS, PL_FRACTIONS): the subset of the grid (scratch CNN + ResNet50, 1% and 10%
#   labels) for which self-training students are built.
# frozen hyper-parameters (configs/<model>.yaml): each model's settings are fixed once, so they are identical across
#   input types, fractions and seeds.

# ======================================== Q&A ABOUT THIS FILE ========================================
# Q: What is this file for?
# A: It builds the lists of TrainConfig objects for the main supervised grid and for the learning-rate tuning sweep. It
#   does not train anything.
# Q: Where does it fit in the pipeline?
# A: scripts/run_grid.py imports these lists and runs each config through train.engine.run; it also uses PL_MODELS /
#   PL_FRACTIONS to plan the pseudo-label students.
# Q: What does it read and write?
# A: It reads configs/<model>.yaml (through load_config) and the constants LABEL_FRACTIONS and SUBSET_SEEDS. It writes
#   no files.
# Q: Why is the learning rate tuned on only one cell?
# A: To keep the sweep cheap: RGB, 10% labels, seed 0 is a representative cell; the chosen value is then frozen in the
#   YAML for all other cells.

# Folder holding one YAML file per model with its frozen hyper-parameters.
CONFIG_DIR = REPO_ROOT / "configs"
# The three models compared in the main study.
MAIN_MODELS = ["scratch_cnn", "resnet50", "vit_small"]
# The two input types: 3-band RGB and all-13-band multispectral.
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
# The single experiment cell on which learning rates are tuned.
TUNE_CELL = {"modality": "rgb", "fraction": 0.1, "seed": 0}


# Q: What does main_grid do?
# A: It returns one TrainConfig for every combination of seed, model, modality and label fraction (54 with the
#   defaults), each starting from its model's YAML file.
# Q: Why is the list ordered seed-first?
# A: So that if the compute session is interrupted, every cell already has its seed-0 result; also run_grid.py's --shard
#   k/n takes every n-th entry, so the order decides which runs a shard gets.
# Q: What are the inputs and the return value?
# A: Inputs are the grid axes and optional overrides applied to every run (e.g. from --set); output is a list of
#   TrainConfig objects.
# Q: Why load the YAML for every run instead of setting hyper-parameters here?
# A: It keeps the hyper-parameters frozen in one visible place per model, so changing a cell (fraction, modality, seed)
#   cannot accidentally change the learning rate.
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
    # One config per combination. `product` varies the LAST name fastest, with the seed as the outermost loop
    # (hence seed-first order). Each config starts from the model's YAML file, then sets the cell's input type,
    # label fraction and seed, then any extra overrides.
    return [
        load_config(CONFIG_DIR / f"{m}.yaml", modality=mod, fraction=f, seed=s, **overrides)
        for s, m, mod, f in product(seeds, models, modalities, fractions)
    ]


# Q: What does tune_grid do?
# A: It builds the learning-rate sweep: for each model, one config per candidate lr in TUNE_LRS, all on the RGB / 10% /
#   seed 0 cell.
# Q: Why do tuning runs set evaluate_test=False?
# A: So that the test set is never used to choose a hyper-parameter; choosing on the test set would make the final test
#   score optimistic (leakage).
# Q: Why the suffix tune_lr<lr>?
# A: It makes a unique run name, so tuning runs never collide with, or overwrite, grid runs.
# Q: How can the cell be changed?
# A: Through **overrides; {**TUNE_CELL, **overrides} lets the caller's values beat the default tuning cell.
def tune_grid(models=MAIN_MODELS, **overrides) -> list[TrainConfig]:
    """LR sweep on one cell (RGB, 10% labels, seed 0), selected on validation macro-F1 only.

    Args:
        models: which models to tune.
        **overrides: fields that replace the default tuning cell or any other setting.
    Returns:
        One TrainConfig per (model, candidate learning rate).

    Tuning runs never evaluate the test set. Run names end in `_tune_lr<lr>`. Extra `overrides`
    can change the cell (modality / fraction / seed) or any other field.
    """
    # For every model and each of its candidate learning rates, build a config from the model's YAML with
    # that lr, a unique name suffix (so it never overwrites a grid run) and the test set switched off.
    # `{**TUNE_CELL, **overrides}` merges the dicts so that user overrides beat the default tuning cell.
    return [
        load_config(CONFIG_DIR / f"{m}.yaml", lr=lr, name_suffix=f"tune_lr{lr:g}", evaluate_test=False,
                    **{**TUNE_CELL, **overrides})
        for m in models for lr in TUNE_LRS[m]
    ]

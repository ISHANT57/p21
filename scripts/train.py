"""Train one model on one experiment cell (model x input x label fraction x seed).

Usage:
    python scripts/train.py --config configs/resnet50.yaml --modality rgb --fraction 1.0 --seed 0
    python scripts/train.py --config configs/vit_small.yaml --modality ms --fraction 0.01 --seed 2 \
        --set lr=5e-5 epochs=50
    python scripts/train.py --config configs/scratch_cnn.yaml --set max_train_batches=5 epochs=1  # quick debug

Finished runs (metrics.json present) are skipped unless --overwrite is given.

This is a thin command-line wrapper around eurosat_le.train.engine.run (see docs/MODELS.md). The
hyper-parameters come from the model's YAML file in configs/; --modality, --fraction and --seed pick
the experiment cell, and --set overrides any TrainConfig field (values are parsed as YAML scalars).
Options that are not given do not override the YAML file. The seed fixes both the labelled-subset
draw and the initialisation/shuffling. Output goes to results/runs/<model>_<input>_<pct>pct_seed<s>/.
"""

# argparse reads the command-line options.
import argparse

# load_config builds the training configuration from a YAML file plus overrides; parse_set parses "key=value" strings.
from eurosat_le.train.config import load_config, parse_set
# run performs the actual training of one experiment cell and writes the run folder.
from eurosat_le.train.engine import run

# ==================== CONCEPTS USED IN THIS FILE ====================
# - argparse: standard-library module that reads command-line options (--config, --modality, ...) into an object; used
#   in main().
# - experiment cell: one combination of model x input type (modality) x label fraction x seed; each cell is one training
#   run with its own results folder.
# - YAML config / TrainConfig: the model's hyper-parameters live in a YAML text file in configs/; load_config reads it
#   into a TrainConfig object (a dataclass), then applies command-line overrides on top.
# - modality (rgb / ms): the input type: 3-band RGB pictures or the 13-band multispectral images.
# - label fraction: how much of the training set's labels may be used: 1%, 10% or 100%. The subsets are nested and fixed
#   in splits/.
# - seed: number that fixes the random choices (which labelled subset, weight initialisation, shuffling), so a run can
#   be repeated; different seeds give different draws that are averaged later.
# - hyper-parameter: a setting chosen by the researcher, not learned by training (learning rate lr, number of epochs,
#   ...).
# - epoch: one pass over the training data; the engine checks the validation score after every epoch.
# - --set KEY=VALUE override: replaces any TrainConfig field; parse_set reads the values as YAML scalars, so lr=5e-5
#   becomes a float and amp=false a boolean.
# - skip finished runs: run() returns early when the run folder already has metrics.json (written last), unless
#   --overwrite is given.
# - training engine (what run() does): AdamW optimiser, warm-up then cosine learning-rate schedule, mixed precision on
#   GPU, model selection and early stopping on validation macro-F1, one test prediction at the end.
# - if __name__ == '__main__': runs main() only when executed as a script, not when imported.
# ====================================================================
#
# ---- Q&A: THIS FILE ----
# Q: What is this file for?
# A: It trains one model on one experiment cell. It is a thin command-line wrapper around eurosat_le.train.engine.run,
#    used for single runs and debugging; run_grid.py calls the same engine for many cells.
# Q: What does it read and write?
# A: It reads configs/<model>.yaml plus the fixed splits and the cached images. It writes
#    results/runs/<model>_<input>_<pct>pct_seed<s>/ with config.yaml, env.json, history.csv, preds_*.npz and
#    metrics.json.
# Q: Where does it fit in the pipeline?
# A: After download_data.py, make_splits.py, inspect_dataset.py and build_cache.py, and before evaluate.py /
#    collect_results.py, which read its outputs.


# ---- Q&A: main ----
# Q: What does main() do?
# A: It parses the options, builds a TrainConfig with load_config (YAML file first, then the command-line values on top)
#    and calls run(cfg, overwrite=...).
# Q: Why do --modality, --fraction and --seed have no default?
# A: Their default is None, and load_config ignores None overrides, so an option that was not typed leaves the value
#    from the YAML file unchanged.
# Q: What would go wrong without parse_set?
# A: Values given to --set would stay text; for example lr=5e-5 would be a string instead of a number and training would
#    fail or behave wrongly.
# Q: What are the inputs and what does it return?
# A: Command-line options only (--config is required). It returns None; results are written to results/runs/<run name>/.
def main() -> None:
    """Read the command-line options, build the training configuration and train one experiment cell.

    Takes no arguments (options come from sys.argv) and returns nothing. Results go to results/runs/<run name>/.
    """
    # Build the argument parser; the module docstring doubles as the --help text.
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Path of the YAML file with the model's hyper-parameters (required).
    p.add_argument("--config", required=True)
    # Input type: 3-band RGB images or 13-band multispectral images.
    p.add_argument("--modality", choices=["rgb", "ms"])
    # Share of the training labels that may be used: 1%, 10% or 100%.
    p.add_argument("--fraction", type=float, choices=[0.01, 0.1, 1.0])
    # Seed that fixes the labelled-subset draw and the random initialisation/shuffling.
    p.add_argument("--seed", type=int)
    # Any number of KEY=VALUE overrides for fields of the training configuration.
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override any TrainConfig field.")
    # Flag: retrain even if this run already has a metrics.json.
    p.add_argument("--overwrite", action="store_true")
    # Parse sys.argv into the args object.
    args = p.parse_args()

    # Build the configuration: YAML file first, then the command-line values on top (the --set overrides are
    # converted from "key=value" text to typed values by parse_set).
    # None-valued options (not given on the command line) are ignored by load_config, so the YAML values stay.
    cfg = load_config(args.config, modality=args.modality, fraction=args.fraction, seed=args.seed,
                      **parse_set(args.set))
    # Train the model; a run that already finished is skipped unless --overwrite was given.
    run(cfg, overwrite=args.overwrite)


# Run main() only when the file is executed as a script, not when it is imported.
if __name__ == "__main__":
    main()

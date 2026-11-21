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

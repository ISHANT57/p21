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

import argparse

from eurosat_le.train.config import load_config, parse_set
from eurosat_le.train.engine import run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--modality", choices=["rgb", "ms"])
    p.add_argument("--fraction", type=float, choices=[0.01, 0.1, 1.0])
    p.add_argument("--seed", type=int)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override any TrainConfig field.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    # None-valued options (not given on the command line) are ignored by load_config, so the YAML values stay.
    cfg = load_config(args.config, modality=args.modality, fraction=args.fraction, seed=args.seed,
                      **parse_set(args.set))
    run(cfg, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

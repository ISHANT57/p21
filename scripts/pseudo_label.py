"""Train one pseudo-label student from a finished supervised (or earlier student) run.

Usage:
    python scripts/pseudo_label.py --teacher resnet50_ms_1pct_seed0
    python scripts/pseudo_label.py --teacher resnet50_ms_1pct_seed0_pl            # second round -> _pl2
    python scripts/pseudo_label.py --teacher scratch_cnn_rgb_10pct_seed1 --balance --target-precision 0.98 \
        --set name_suffix=plb98

The student reuses the teacher's config (model, input, fraction, seed, hyper-parameters)
and trains on labelled + pseudo-labelled images. tau is chosen on the validation set.
"""

import argparse

from eurosat_le.ssl.self_training import train_student
from eurosat_le.train.config import parse_set


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", required=True, help="run name under results/runs/")
    p.add_argument("--target-precision", type=float, default=0.95)
    p.add_argument("--balance", action="store_true")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    train_student(args.teacher, args.target_precision, args.balance, overwrite=args.overwrite, **parse_set(args.set))


if __name__ == "__main__":
    main()

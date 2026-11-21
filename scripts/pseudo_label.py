"""Train one pseudo-label student from a finished supervised (or earlier student) run.

Usage:
    python scripts/pseudo_label.py --teacher resnet50_ms_1pct_seed0
    python scripts/pseudo_label.py --teacher resnet50_ms_1pct_seed0_pl            # second round -> _pl2
    python scripts/pseudo_label.py --teacher scratch_cnn_rgb_10pct_seed1 --balance --target-precision 0.98 \
        --set name_suffix=plb98

The student reuses the teacher's config (model, input, fraction, seed, hyper-parameters)
and trains on labelled + pseudo-labelled images. tau is chosen on the validation set.

Command-line wrapper around eurosat_le.ssl.self_training.train_student (see docs/SSL.md).
- tau is the smallest confidence threshold at which the teacher's validation predictions are at
  least --target-precision (default 0.95) correct; the test set plays no part in choosing it.
- --balance caps each predicted class at its labelled-set share of the confident images.
- The student is saved as results/runs/<teacher>_pl (or _pl2, ... when the teacher is itself a
  student); --set name_suffix=... chooses another suffix, e.g. to keep balanced students separate.
- The true labels of the unlabelled images are only read after training, to report pseudo-label
  accuracy in pseudo_label_quality.json; they never influence selection or training.
- A student that already has metrics.json is skipped unless --overwrite is given.
"""

# argparse reads the command-line options.
import argparse

# train_student builds pseudo-labels from a finished teacher run and trains the student model on them.
from eurosat_le.ssl.self_training import train_student
# parse_set turns "key=value" strings from --set into a dictionary of typed values.
from eurosat_le.train.config import parse_set


def main() -> None:
    """Read the command-line options and train one pseudo-label student from the chosen teacher run.

    Takes no arguments (options come from sys.argv) and returns nothing. Results are written into the
    student's own run folder under results/runs/.
    """
    # Build the argument parser; the module docstring doubles as the --help text.
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Name of the finished run (folder in results/runs/) whose predictions provide the pseudo-labels.
    p.add_argument("--teacher", required=True, help="run name under results/runs/")
    # How accurate the teacher's confident predictions must be on the validation set; this decides the threshold tau.
    p.add_argument("--target-precision", type=float, default=0.95)
    # Flag: select pseudo-labels class-balanced instead of with one global threshold.
    p.add_argument("--balance", action="store_true")
    # Any number of KEY=VALUE overrides for the student's training settings.
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    # Flag: retrain even if the student already has a metrics.json.
    p.add_argument("--overwrite", action="store_true")
    # Parse sys.argv into the args object.
    args = p.parse_args()
    # Convert the KEY=VALUE overrides to a dict and hand everything to train_student, which does the real work.
    train_student(args.teacher, args.target_precision, args.balance, overwrite=args.overwrite, **parse_set(args.set))


# Run main() only when the file is executed as a script, not when it is imported.
if __name__ == "__main__":
    main()

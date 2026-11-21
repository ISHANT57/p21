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

# ==================== CONCEPTS USED IN THIS FILE ====================
# - argparse: standard-library module that reads command-line options (--teacher, --balance, ...) into an object; used
#   in main().
# - pseudo-label / self-training: semi-supervised learning: a model trained on the few labelled images predicts labels
#   for the unlabelled images; its confident predictions are treated as if they were true labels and used as extra
#   training data.
# - teacher / student: the teacher is a finished supervised run (or an earlier student) that makes the pseudo-labels;
#   the student is a new model with the same settings, trained on labelled + pseudo-labelled images.
# - confidence and threshold tau: confidence = the highest softmax probability of a prediction. tau is the cut-off: only
#   images with confidence >= tau get a pseudo-label. tau is chosen on the validation set, never on the test set.
# - target precision: the share of the teacher's confident validation predictions that must be correct (default 0.95).
#   tau is the smallest threshold that reaches it.
# - class-balanced selection (--balance): limits how many pseudo-labels each predicted class may contribute (its share
#   of the labelled set), so easy classes cannot dominate.
# - labelled fraction / unlabelled pool: the labelled subset is 1%, 10% or 100% of train; the pool is the rest of train,
#   whose true labels are ignored for training and only used afterwards to measure pseudo-label accuracy.
# - --set KEY=VALUE overrides: nargs='*' collects any number of 'key=value' words; parse_set converts them to typed
#   values (lr=5e-5 becomes a float) and they replace fields of the student's training config.
# - name_suffix: text added to the run name (pl, pl2, plb, ...) so that students never overwrite their teacher or each
#   other.
# - metrics.json / overwrite: metrics.json is the last file a run writes, so its presence means the run finished; a
#   finished student is skipped unless --overwrite.
# - if __name__ == '__main__': runs main() only when executed as a script, not when imported.
# ====================================================================
#
# ---- Q&A: THIS FILE ----
# Q: What is this file for?
# A: It is the command-line wrapper for Phase 5 (semi-supervised learning): it trains one pseudo-label student from one
#    finished teacher run. All the logic is in eurosat_le.ssl.self_training.train_student.
# Q: What does it read and write?
# A: It reads the teacher's folder in results/runs/ (config.yaml, preds_val.npz, preds_unlabelled.npz, metrics.json). It
#    writes a new run folder results/runs/<teacher>_pl (or _pl2, _plb ...) with pseudo_labels.csv, the
#    pseudo_label_*.csv/json files and the usual training outputs.
# Q: How does it relate to run_grid.py?
# A: run_grid.py --plan pl does the same thing for a whole list of teachers by calling train_student directly; this
#    script is for a single teacher.


# ---- Q&A: main ----
# Q: What does main() do?
# A: It parses the options and passes them to train_student(teacher, target_precision, balance, overwrite, **overrides),
#    which chooses tau, writes the pseudo-labels and trains the student.
# Q: Why is tau chosen from the validation set and not the test set?
# A: Choosing a setting with test data would leak the test set into training and make the final test score too
#    optimistic. Validation data exists for exactly this kind of decision.
# Q: What are the inputs and what does it return?
# A: Inputs come from the command line: --teacher (required), --target-precision (default 0.95), --balance, --set
#    KEY=VALUE ..., --overwrite. It returns None; the student's results are saved to disk.
# Q: What does parse_set do here?
# A: It turns the strings given to --set into a dictionary with proper types (numbers, booleans), which is passed on as
#    **overrides to change the student's training settings.
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

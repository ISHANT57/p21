"""Rebuild every evaluation table from saved predictions (Phase 6).

Usage:
    python scripts/evaluate.py                                  # results/runs -> results/summary/eval
    python scripts/evaluate.py --runs-dir /content/drive/MyDrive/p21_runs
    python scripts/evaluate.py --examples resnet50_rgb_1pct_seed0 vit_small_ms_1pct_seed0
    python scripts/evaluate.py --quiet                          # no progress or timing messages

Writes to results/summary/eval/:
    per_run.csv            val/test accuracy, macro-F1, ECE for every run (recomputed)
    groups.csv             mean / std over seeds per model x input x fraction x variant
    per_class.csv          per-class precision / recall / F1, mean and std over seeds
    confusion/*.csv        seed-averaged row-normalised confusion matrix per group
    top_confusions.csv     the most frequent (true -> predicted) mistakes per group
    compare_*.csv          paired comparisons: per seed (pairs) and summarised over seeds
    errors_<run>.png       misclassified test images (with --examples)

Where it fits: Phase 6 (see docs/EVALUATION.md). It reads the saved predictions of every finished run
(preds_val.npz and preds_test.npz in each run folder, plus config.yaml and metrics.json) and never trains
or re-runs a model. scripts/make_figures.py (Phase 7) then turns these tables into figures and tables.

Design decisions:
- Metrics are RECOMPUTED from the saved logits (softmax, then argmax), not copied from metrics.json. As a
  cross-check, accuracy and macro-F1 are compared with the logged values and a WARNING is printed if they
  differ by more than 0.001 (the run is still used; the script does not abort).
- ECE uses 15 confidence bins.
- Every number over seeds is a mean and a sample std (ddof=1) over the labelled-subset seeds (3 in the
  main grid). With a single seed the std is empty (NaN) in the CSV.
- Paired comparisons only match runs that agree on every other factor AND the seed, so both sides use the
  same labelled images, the same initialisation seed and the same test images. The test predictions of
  every run are aligned by sorted sample id. The interval is a 95 percent paired bootstrap over test images
  (--n-boot resamples, default 1000). Results over seeds are reported as counts of seeds, not p-values.
- A comparison with no matching pairs (for example the runs do not exist yet) is silently skipped.
- Progress messages: before the slow part the script prints how many runs and comparison pairs it will
  process, then one line per comparison file ("compare_rgb_vs_ms: 27 pairs, 12.3 s"), and the total time
  next to the final "wrote ..." line. --quiet turns these messages off; tables and warnings still print.
  The messages never change the CSV or PNG outputs.
"""

# argparse reads the options; time measures how long each step takes; Path represents folder paths.
import argparse
import time
from pathlib import Path

# numpy for array operations; pandas for the result tables.
import numpy as np
import pandas as pd

# CLASSES = the class names; RESULTS_DIR = the top-level results folder.
from eurosat_le.constants import CLASSES, RESULTS_DIR
# Evaluation helpers: load runs and predictions, recompute metrics, compare with the logged values, average
# confusion matrices, do paired comparisons and summarise them, and list the most frequent mistakes.
from eurosat_le.eval.evaluate import (check_against_logged, compare, load_preds, load_runs, mean_confusion,
                                      run_metrics, summarise_comparison, top_confusions)
# RUNS_DIR is the default folder holding one sub-folder per run.
from eurosat_le.train.engine import RUNS_DIR

# ==================== CONCEPTS USED IN THIS FILE ====================
# - argparse: standard-library module that reads command-line options (--runs-dir, --out, --n-boot, --examples, --quiet)
#   into an object.
# - saved predictions / logits: each run stored the model's raw class scores (logits) for val and test in preds_val.npz
#   / preds_test.npz; this script reads them and never trains or re-runs a model.
# - softmax: turns logits into probabilities that are positive and sum to 1; argmax of the probabilities is the
#   predicted class, and its value is the confidence.
# - accuracy, precision, recall, F1, macro-F1: accuracy = share correct; precision = of the images predicted as a class,
#   how many are right; recall = of the true images of a class, how many were found; F1 = their harmonic mean; macro-F1
#   = average of the per-class F1 with equal weight.
# - recompute vs logged metrics: numbers are rebuilt from the logits and compared with metrics.json; if they differ by
#   more than 0.001 a WARNING is printed (the run is still used). Logits are stored as float16, which can flip rare
#   ties.
# - ECE (expected calibration error): measures whether confidence is honest: among predictions made with 80% confidence
#   about 80% should be right. Computed with 15 equal-width confidence bins; 0 is perfect.
# - confusion matrix, row-normalised: table of true class (rows) vs predicted class (columns); dividing each row by its
#   total gives the share of each true class sent to each predicted class. Averaged over seeds here.
# - seeds, mean and sample std (ddof=1): each setting has 3 labelled-subset seeds; tables show mean and sample standard
#   deviation over them; with one seed the std is empty (NaN).
# - group / setting / variant: GROUP = model, modality, fraction, variant. 'variant' is 'supervised' for plain runs, or
#   the name_suffix ('pl', 'plb') for pseudo-label students.
# - paired comparison: two runs are compared only if everything except one factor (modality, model or variant) and the
#   seed match, so they share the labelled images, the initialisation seed and the same test images.
# - bootstrap and 95% interval: re-draw the test images with replacement many times (--n-boot, default 1000) and
#   recompute the macro-F1 difference; the middle 95% of the results is the interval. If it excludes 0, the difference
#   is unlikely to be luck. Paired = the same resample is used for both runs.
# - delta_f1, seeds_b_better, seeds_ci_excludes_0: delta_f1 = macro-F1(b) - macro-F1(a); the summary counts in how many
#   seeds b is better and in how many the interval excludes 0 (counts over seeds, not p-values).
# - pandas groupby / agg / DataFrame: tables are grouped by setting and aggregated with mean/std/nunique, then saved as
#   CSV.
# - numpy bincount / argsort / flatnonzero: fast array helpers: count occurrences, sort indices, find the positions of
#   True values (here the misclassified images).
# - memory-mapped cache (load_cache): all RGB images packed in one .npy file read lazily from disk, so it does not fill
#   the RAM; rows follow splits/split.csv.
# - matplotlib / viz style: plotting library; viz.apply_style sets the project's look. Imported inside plot_errors so
#   the rest of the script works without it.
# - lambda / progress function: progress is either print or a function that does nothing (with --quiet), so timing
#   messages can be silenced without touching the results.
# - time.perf_counter: high-resolution clock for measuring how long each step takes.
# - if __name__ == '__main__': runs main() only when executed as a script, not when imported.
# ====================================================================
#
# ---- Q&A: THIS FILE ----
# Q: What is this file for?
# A: It is Phase 6: it rebuilds every evaluation table from the saved predictions of all finished runs - per-run
#    metrics, group means over seeds, per-class scores, confusion matrices, top confusions and paired comparisons with
#    bootstrap intervals.
# Q: What does it read and write?
# A: Reads results/runs/*/ (preds_val.npz, preds_test.npz, config.yaml, metrics.json). Writes results/summary/eval/
#    (per_run.csv, groups.csv, per_class.csv, confusion/*.csv, top_confusions.csv, compare_*.csv, and errors_<run>.png
#    with --examples).
# Q: Where does it fit in the pipeline?
# A: After all training is done; scripts/make_figures.py (Phase 7) then turns these tables into figures. The heavy logic
#    lives in eurosat_le.eval.evaluate; this script is the driver.
# Q: How is it run?
# A: python scripts/evaluate.py [--runs-dir PATH] [--out PATH] [--n-boot N] [--examples RUN ...] [--quiet].

# Keys that define one "setting"; the 3 seeds of a setting are aggregated together. "variant" is "supervised"
# for plain runs and the config name_suffix (e.g. "pl", "plb") for pseudo-label students.
GROUP = ["model", "modality", "fraction", "variant"]
COMPARISONS = [  # (file stem, factor, a, b): b is compared against a
    ("rgb_vs_ms", "modality", "rgb", "ms"),
    ("supervised_vs_pl", "variant", "supervised", "pl"),
    ("supervised_vs_plb", "variant", "supervised", "plb"),
    ("pl_vs_plb", "variant", "pl", "plb"),
    ("scratch_vs_resnet50", "model", "scratch_cnn", "resnet50"),
    ("scratch_vs_vit_small", "model", "scratch_cnn", "vit_small"),
    ("resnet50_vs_vit_small", "model", "resnet50", "vit_small"),
]


# ---- Q&A: group_name ----
# Q: What does group_name() do?
# A: It builds the file-name stem for a group's confusion CSV, for example 'resnet50_rgb_1pct_supervised'.
# Q: Why fraction * 100 with ':g'?
# A: It writes 0.01 as '1' and 1.0 as '100', giving '1pct' and '100pct' without trailing zeros.
# Q: Why must it stay in sync with make_figures.py?
# A: That script rebuilds the same names (CONF_GROUP) to find these files; if the naming differed, the figures would not
#    find their input.
# Q: What are the input and the return value?
# A: Input: a row/dict with the GROUP keys model, modality, fraction, variant. Returns a string.
def group_name(row) -> str:
    """File stem for a group's confusion CSV, e.g. "resnet50_rgb_1pct_supervised".

    `row` needs the GROUP keys. The fraction is written as a percentage with `:g`, so 0.01 -> "1pct" and
    1.0 -> "100pct". scripts/make_figures.py rebuilds these names (CONF_GROUP), so keep both in sync.
    """
    # Join model, input type, label percentage and variant into one name (fraction * 100 turns 0.01 into 1).
    return f"{row['model']}_{row['modality']}_{row['fraction'] * 100:g}pct_{row['variant']}"


# ---- Q&A: count_pairs ----
# Q: What does count_pairs() do?
# A: It counts how many run pairs compare() will match for one comparison, without loading any predictions.
# Q: What is the matching rule?
# A: Every GROUP key except the compared factor, plus the seed, must agree; the code builds an index of those keys for
#    side a and side b and intersects them.
# Q: Why is it needed?
# A: To tell the user up front how much work (pairs x bootstrap resamples) is coming, so a long run does not look
#    frozen.
# Q: What are the inputs and the return value?
# A: runs (table of runs), the factor name and its two values a and b. Returns an int.
def count_pairs(runs: pd.DataFrame, factor: str, a, b) -> int:
    """Number of run pairs `compare` will match for one comparison, without loading any predictions.

    Uses the same rule as `eurosat_le.eval.evaluate.compare`: every GROUP key except `factor`, plus the
    seed, must agree.
    """
    keys = [k for k in GROUP if k != factor] + ["seed"]  # what must match: every other factor and the seed
    A = runs[runs[factor] == a].set_index(keys).index  # settings on side a, e.g. all RGB runs
    B = runs[runs[factor] == b].set_index(keys).index  # settings on side b, e.g. all 13-band runs
    return len(A.intersection(B))  # a pair exists only where both sides have the same key


# ---- Q&A: main ----
# Q: What does main() do?
# A: It loads the finished runs, recomputes val/test metrics for each (warning if they differ from metrics.json), writes
#    per_run.csv, groups.csv and per_class.csv, averages the confusion matrices per group and lists the top mistakes,
#    runs every paired comparison with a bootstrap interval, optionally draws error pictures, and prints the total time.
# Q: Why recompute the metrics instead of copying metrics.json?
# A: The final report numbers should come from one reproducible computation on the saved predictions; the comparison
#    with the logged values is a cross-check that catches mistakes.
# Q: Why are comparisons with no matching pairs skipped silently?
# A: Early in a study some runs do not exist yet; an empty comparison simply produces no file.
# Q: What does --quiet change?
# A: It switches off progress and timing messages only; tables, warnings and results are the same.
def main() -> None:
    """Run the whole Phase 6 evaluation: per-run metrics, group tables, confusions, paired comparisons."""
    # Build the argument parser; the module docstring doubles as the --help text.
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Folder with the run sub-folders to evaluate.
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    # Folder the tables are written into.
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "summary" / "eval")
    # Number of bootstrap resamples used for the confidence intervals of the paired comparisons.
    p.add_argument("--n-boot", type=int, default=1000)
    # Names of runs for which a picture of misclassified test images should be drawn.
    p.add_argument("--examples", nargs="*", default=[], metavar="RUN", help="plot misclassified test images")
    # Flag: suppress progress and timing messages.
    p.add_argument("--quiet", action="store_true",
                   help="no progress/timing messages (results and warnings are still printed)")
    # Parse sys.argv into the args object.
    args = p.parse_args()
    t_start = time.perf_counter()  # wall time of the whole evaluation, reported with the final "wrote" line
    # Progress and timing messages go through `progress`, so --quiet silences them and nothing else.
    progress = (lambda *_: None) if args.quiet else print

    # Table of all finished runs that have saved test predictions.
    runs = load_runs(args.runs_dir)
    # Without such runs there is nothing to evaluate.
    if runs.empty:
        raise SystemExit(f"no finished runs with test predictions in {args.runs_dir}")
    out = args.out
    # Create the output folder and its confusion/ sub-folder before writing into them.
    (out / "confusion").mkdir(parents=True, exist_ok=True)
    # For each comparison, how many matched run pairs exist (used for the progress message below).
    n_pairs = {stem: count_pairs(runs, factor, a, b) for stem, factor, a, b in COMPARISONS}
    # Announce the amount of work before the slow part, so a long run does not look frozen.
    progress(f"{len(runs)} runs found; {sum(n_pairs.values())} comparison pairs in "
             f"{sum(n > 0 for n in n_pairs.values())} comparison files, {args.n_boot} bootstrap resamples each")

    # ---- per run ------------------------------------------------------------------------------
    # rows: one summary row per run; class_rows: one row per run and class; issues: mismatch warnings.
    rows, class_rows, issues = [], [], []
    for _, r in runs.iterrows():
        # Recompute accuracy, macro-F1, ECE and per-class scores from this run's saved predictions.
        met = run_metrics(r["dir"])
        # Compare them with the numbers logged during training; collect a message for every mismatch.
        issues += check_against_logged(r["dir"], met)
        # One row for this run: its identifying fields plus val/test accuracy, macro-F1 and ECE.
        rows.append({**{k: r[k] for k in ["run_name", *GROUP, "seed"]},
                     **{f"{s}_{k}": met[s][k] for s in ("val", "test") for k in ("accuracy", "macro_f1", "ece")}})
        # One row per class for this run: identifying fields, the class name and its precision/recall/F1 on test.
        for c in CLASSES:
            class_rows.append({**{k: r[k] for k in [*GROUP, "seed"]}, "class": c, **met["test"]["per_class"][c]})
    # Save the per-run table.
    per_run = pd.DataFrame(rows)
    per_run.to_csv(out / "per_run.csv", index=False)
    print(f"{len(per_run)} runs evaluated")
    # Warn (but carry on) if any recomputed number differs from the logged one.
    if issues:
        print("WARNING: recomputed metrics differ from metrics.json:\n  " + "\n  ".join(issues))

    # One row per setting; std is NaN when n_seeds == 1. Only the test F1/accuracy/ECE are aggregated here,
    # plus the validation F1 mean for reference.
    groups = per_run.groupby(GROUP).agg(
        n_seeds=("seed", "nunique"),
        test_acc_mean=("test_accuracy", "mean"), test_acc_std=("test_accuracy", "std"),
        test_f1_mean=("test_macro_f1", "mean"), test_f1_std=("test_macro_f1", "std"),
        test_ece_mean=("test_ece", "mean"), val_f1_mean=("val_macro_f1", "mean"),
    ).reset_index()
    groups.to_csv(out / "groups.csv", index=False)

    # Per-class table: test split only; precision/recall are averaged over seeds, F1 also gets a std.
    per_class = pd.DataFrame(class_rows).groupby([*GROUP, "class"]).agg(
        precision=("precision", "mean"), recall=("recall", "mean"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"), n_seeds=("seed", "nunique"),
    ).reset_index()
    per_class.to_csv(out / "per_class.csv", index=False)

    # ---- confusion matrices ---------------------------------------------------------------------
    # Rows for the "most frequent mistakes" table, collected over all groups.
    conf_rows = []
    for key, g in runs.groupby(GROUP):  # key is a tuple in GROUP order (GROUP is a list of 4 columns)
        # Turn the group key tuple into a dict {model:..., modality:..., fraction:..., variant:...}.
        row = dict(zip(GROUP, key))
        # Average the (row-normalised) test confusion matrix over the seeds of this group.
        cm = mean_confusion(list(g["dir"]))
        # Save it as a labelled CSV, one file per group.
        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out / "confusion" / f"{group_name(row)}.csv")
        # Record this group's most frequent mistakes (true class, predicted class, how often).
        for true, pred, rate in top_confusions(cm):
            conf_rows.append({**row, "true": true, "predicted": pred, "rate": rate})
    pd.DataFrame(conf_rows).to_csv(out / "top_confusions.csv", index=False)

    # ---- paired comparisons ---------------------------------------------------------------------
    for stem, factor, a, b in COMPARISONS:
        t0 = time.perf_counter()  # per-file timing: the bootstrap inside compare() is the slow step
        # Match runs that differ only in `factor` (a vs b) and compute the F1 difference with a bootstrap interval.
        pairs = compare(runs, factor, a, b, args.n_boot)
        if pairs.empty:  # nothing to pair yet (one side has no runs)
            continue
        # Save the per-seed pairs, then the summary over seeds.
        pairs.to_csv(out / f"compare_{stem}_pairs.csv", index=False)
        summ = summarise_comparison(pairs)
        summ.to_csv(out / f"compare_{stem}.csv", index=False)
        # The timing line takes the blank separator line; with --quiet the table header keeps it instead.
        progress(f"\ncompare_{stem}: {len(pairs)} pairs, {time.perf_counter() - t0:.1f} s")
        print(("\n" if args.quiet else "") + f"{stem}: macro-F1({b}) - macro-F1({a}), paired by seed")
        # Columns that identify the setting; the compared factor is left out because it is what differs (a vs b).
        cols = [c for c in GROUP if c != factor and c in summ.columns]  # the compared factor is a/b, not a key
        # Print the summary table: setting, number of seeds, mean/std of the F1 difference and per-seed counts.
        print(summ[cols + ["n_seeds", "delta_f1_mean", "delta_f1_std", "seeds_b_better", "seeds_ci_excludes_0"]]
              .to_string(index=False, float_format="%.4f"))

    # Draw the misclassified-images figure for every run name given with --examples.
    for name in args.examples:
        plot_errors(args.runs_dir / name, out / f"errors_{name}.png")
    # The "wrote" line is always printed (as before); only the timing part is a progress message.
    print(f"\nwrote {out}" + ("" if args.quiet else f" ({time.perf_counter() - t_start:.1f} s total)"))


# ---- Q&A: plot_errors ----
# Q: What does plot_errors() do?
# A: It saves a grid of up to n (24) misclassified test images of one run, each titled 'true -> predicted (confidence)'.
# Q: Why are images always taken from the RGB cache, even for 13-band runs?
# A: A human can look at RGB; the cache rows follow split.csv, so an image is found through its position in that table.
# Q: Why a fixed seed when choosing the errors?
# A: So the same grid is drawn every time the script is run.
# Q: What are the inputs and the return value?
# A: run_dir (a finished run with preds_test.npz), path (output PNG), n and seed. Returns None.
def plot_errors(run_dir: Path, path: Path, n: int = 24, seed: int = 0) -> None:
    """Grid of misclassified test images (shown in RGB, also for 13-band runs).

    Args:
        run_dir: a finished run folder with preds_test.npz.
        path: output PNG.
        n: maximum number of images (6 per row); fewer if the run makes fewer errors.
        seed: seed of the random choice among the errors, so the same grid is drawn every time.

    Each title is "true -> predicted (confidence)", the confidence being the top softmax probability.
    Images come from the RGB cache for every run: the cache rows follow splits/split.csv, so an image is
    found through its position in that table.
    """
    # These imports are done here so the rest of the script works without matplotlib until a plot is drawn.
    import matplotlib.pyplot as plt

    # Project plot style, the packed RGB image cache, the split table and the softmax function.
    from eurosat_le import viz
    from eurosat_le.data.cache import load_cache
    from eurosat_le.data.splits import load_split
    from eurosat_le.ssl.pseudo_label import softmax

    viz.apply_style()
    # Saved test predictions: raw class scores (logits), true labels and sample ids (sorted by id).
    logits, y, ids = load_preds(run_dir)
    # Convert the scores to probabilities, so the confidence of each prediction can be shown.
    probs = softmax(logits)
    wrong = np.flatnonzero(probs.argmax(1) != y)  # indices into the id-sorted test arrays
    # Randomly choose up to n of the mistakes (a fixed seed makes the choice repeatable).
    pick = np.random.default_rng(seed).choice(wrong, min(n, len(wrong)), replace=False)
    # All RGB images packed into one array (memory-mapped, so it does not fill the RAM).
    rgb = load_cache("rgb")
    row_of = {sid: i for i, sid in enumerate(load_split()["id"])}  # sample id -> row in split.csv == row in the cache
    # 6 images per row; compute how many rows are needed for the chosen images.
    cols = 6
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.3), squeeze=False)
    # Switch every cell off first, so unused cells in the last row stay empty.
    for ax in axes.flat:
        ax.axis("off")
    # Fill one cell per chosen mistake.
    for ax, i in zip(axes.flat, pick):
        img = rgb[row_of[ids[i]]].transpose(1, 2, 0)  # cache is (C, H, W); imshow wants (H, W, C)
        ax.imshow(img)
        # Title: true class, predicted class and the model's confidence in its (wrong) prediction.
        ax.set_title(f"{CLASSES[y[i]]}\n→ {CLASSES[probs[i].argmax()]}\n({probs[i].max():.2f})", fontsize=6.5,
                     fontweight="normal", color=viz.TEXT_PRIMARY)
    fig.suptitle(f"{run_dir.name}: misclassified test images (true → predicted, confidence)", fontsize=9)
    fig.subplots_adjust(wspace=0.15, hspace=0.6)  # hspace leaves room for the three-line titles
    # Save the figure and free its memory.
    fig.savefig(path)
    plt.close(fig)


# Run main() only when the file is executed as a script, not when it is imported.
if __name__ == "__main__":
    main()

"""Build the evaluation tables that scripts/make_figures.py reads, from the logged metrics of each run.

Where it fits: scripts/evaluate.py recomputes every number from saved predictions, which needs the run
folders. Lanes A and C only kept their notebook logs, so for them the per-run test accuracy and macro-F1
come from the printed result lines (scripts/parse_colab_logs.py -> results/logs/runs_from_logs.csv).
This script turns that table into the same files evaluate.py writes, so the figure and table code is
unchanged:

    groups.csv                       per model x input x fraction x variant: mean / std over seeds
    compare_<stem>_pairs.csv         one row per matched seed (stems as in scripts/evaluate.py)
    compare_<stem>.csv               the same summarised over seeds

Usage:
    python scripts/eval_from_logs.py [--logs results/logs/runs_from_logs.csv] [--out results/summary/eval]
                                     [--runs-dir results/runs]

What is different from evaluate.py (state this wherever the numbers are used):
- There are no bootstrap intervals: ci_low and ci_high are empty and seeds_ci_excludes_0 is empty.
  A paired comparison is the per-seed difference of the two logged macro-F1 values.
- test_ece_mean is empty (calibration is not in the logs).
- Numbers carry the four decimals of the log.
- per_class.csv and the confusion matrices are not made here. They need predictions, which exist only for
  the runs whose folders were kept; build them with scripts/evaluate.py on those folders.

Design decisions:
- Pairs are matched exactly as in evaluate.compare (all keys but the compared factor, plus the seed), and
  the summary reuses evaluate.summarise_comparison, so the columns match.
- With --runs-dir, each logged run that also has a metrics.json is cross-checked against it (the log
  rounds to 4 decimals, so a difference above 0.00006 is reported).
"""

# Standard library: argparse reads the command-line options, json reads metrics.json, Path handles file paths.
import argparse
import json
from pathlib import Path

# numpy supplies NaN (the "empty" value written where a number is not available); pandas holds the tables of runs.
import numpy as np
import pandas as pd

# Project helper that summarises a table of per-seed pairs, so the output columns match evaluate.py exactly.
from eurosat_le.eval.evaluate import summarise_comparison

# ======================================================================
# CONCEPTS USED IN THIS FILE
# ======================================================================
# seed: the number that fixes the random choices of one training run. Every setting is trained several times with
#   different seeds (the run names end in seed0, seed1, ...), so one lucky or unlucky run does not decide the result.
#   Used as the 'seed' column.
# setting (group): one combination of model, modality, fraction and variant (the GROUP list). The seeds of a setting are
#   averaged in groups_table.
# modality: the kind of input image, 'rgb' (3 colour bands) or 'ms' (all 13 satellite bands). Column 'modality'.
# fraction: the share of the labelled training images that was used (0.01, 0.1, 1.0). Column 'fraction'.
# variant: 'supervised' (normal training), 'pl' (pseudo-labelling with one global threshold) or 'plb' (class-balanced
#   pseudo-labelling). Used in the COMPARISONS list.
# pseudo-labelling: a trained teacher model labels unlabelled images itself, and a student model is trained on these
#   extra labels. The student runs are the 'pl' and 'plb' variants compared with 'supervised' here.
# accuracy: the share of test images classified correctly. Column test_acc.
# macro-F1: the F1 score (harmonic mean of precision and recall) computed for each class and then averaged with equal
#   weight per class, so small classes count as much as big ones. Column test_f1.
# validation set vs test set: validation data is used to pick the best epoch; the test set is only for the final score.
#   Columns val_f1 (validation) and test_acc / test_f1 (test).
# mean and sample standard deviation over seeds: the average of the per-seed scores and their spread, with n-1 in the
#   denominator (pandas default, ddof=1). With one seed the std is undefined, so it is NaN. Computed in groups_table
#   with groupby().agg(...).
# paired difference: for two runs that differ in only one factor (and share the seed), the score of b minus the score of
#   a. Pairing removes the seed-to-seed noise. Built in pairs_table as delta_f1 = f1_b - f1_a.
# bootstrap confidence interval: a range for a statistic found by re-sampling the test predictions many times. It needs
#   per-image predictions, which the logs do not hold, so ci_low / ci_high are left empty here.
# ECE (expected calibration error): how far the model's confidence is from its real accuracy. Not in the logs, so
#   test_ece_mean is an empty column here.
# NaN (np.nan): pandas' 'not a number' marker, written as an empty cell in the CSV. Used for every value that cannot be
#   computed from the logs.
# CSV and DataFrame: a CSV is a plain-text table; a pandas DataFrame is the same table in memory. read_csv loads it,
#   to_csv(index=False) saves it without the row numbers.
# groupby / agg: split a table into groups of rows sharing the same key values, then compute summaries (count of
#   distinct seeds, mean, std) for each group. Used in groups_table.
# set_index and Index.intersection: put the matching keys into the row index so the 'a' runs and the 'b' runs can be
#   lined up; the intersection keeps only the keys present on both sides. Used in pairs_table.
# metrics.json cross-check: the training code saved exact test numbers for runs whose folder was kept; the log prints
#   them rounded to 4 decimals, so a difference above 0.00006 (a bit more than the rounding error 0.00005) is reported.
#   Used in cross_check.
# argparse: the standard-library module that turns command-line options (--logs, --out, --runs-dir) into Python values.
#   Used in main.

# ======================================================================
# Q&A ABOUT THIS FILE (file-level viva questions)
# ======================================================================
# Q: What is this file for?
#    A: It builds the evaluation tables (groups.csv, compare_<stem>_pairs.csv, compare_<stem>.csv) from the per-run
#       scores in runs_from_logs.csv, for the runs whose folders and predictions were not kept (Colab lanes A and C).
# Q: Where does it fit in the pipeline?
#    A: parse_colab_logs.py reads the notebook logs and writes runs_from_logs.csv; this script turns that into
#       results/summary/eval/*.csv with the same file names as scripts/evaluate.py; make_figures.py and
#       make_report_tables.py then read those files without any change.
# Q: How is it run and what does it read and write?
#    A: python scripts/eval_from_logs.py [--logs ...] [--out ...] [--runs-dir ...]. It reads runs_from_logs.csv (and,
#       only with --runs-dir, the metrics.json of each kept run folder) and writes groups.csv plus one pairs file and
#       one summary file per comparison that has data.
# Q: What is missing compared with evaluate.py, and why?
#    A: No bootstrap intervals, no calibration (ECE), no per_class.csv and no confusion matrices, because those need the
#       saved predictions and the logs only contain the printed scores.

# The columns that together identify one experimental setting; seeds are averaged inside a setting.
GROUP = ["model", "modality", "fraction", "variant"]
# The pairwise comparisons to make: (output file name stem, column being compared, value "a", value "b").
COMPARISONS = [  # same list as scripts/evaluate.py: (file stem, factor, a, b); b is compared against a
    ("rgb_vs_ms", "modality", "rgb", "ms"),
    ("supervised_vs_pl", "variant", "supervised", "pl"),
    ("supervised_vs_plb", "variant", "supervised", "plb"),
    ("pl_vs_plb", "variant", "pl", "plb"),
    ("scratch_vs_resnet50", "model", "scratch_cnn", "resnet50"),
    ("scratch_vs_vit_small", "model", "scratch_cnn", "vit_small"),
    ("resnet50_vs_vit_small", "model", "resnet50", "vit_small"),
]


# Q: What does this function do?
#    A: For every setting (model, modality, fraction, variant) it counts the seeds and computes the mean and sample std
#       of test accuracy and test macro-F1 and the mean validation F1. One row per setting.
# Q: Why is test_ece_mean inserted as an empty column, and why can a std be NaN?
#    A: The empty column keeps the column order identical to the groups.csv written by evaluate.py, so the later code
#       works on either file. The std is NaN when a setting has only one seed, because the sample std needs at least two
#       values; the tables then print '(n=1)'.
# Q: What are the input and the output?
#    A: Input: a DataFrame with one row per finished run (model, modality, fraction, variant, seed, test_acc, test_f1,
#       val_f1). Output: a DataFrame with one row per setting.
def groups_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample std over seeds per setting (std is NaN for one seed), as evaluate.py writes it.

    Args:
        runs: one row per finished run, with the columns model, modality, fraction, variant, seed,
            test_acc, test_f1 and val_f1.
    Returns:
        One row per setting with the number of seeds and the mean / std of the test scores. The
        calibration column test_ece_mean is present but empty because the logs do not contain it.
    """
    # Group the runs by setting; for each group count the seeds and compute the mean and std of the test
    # accuracy and test F1, plus the mean validation F1. Then turn the group labels back into normal columns.
    g = runs.groupby(GROUP).agg(
        n_seeds=("seed", "nunique"),
        test_acc_mean=("test_acc", "mean"), test_acc_std=("test_acc", "std"),
        test_f1_mean=("test_f1", "mean"), test_f1_std=("test_f1", "std"),
        val_f1_mean=("val_f1", "mean"),
    ).reset_index()
    # Add an empty calibration (ECE) column just before val_f1_mean so the column order matches evaluate.py.
    g.insert(g.columns.get_loc("val_f1_mean"), "test_ece_mean", np.nan)  # same column order as evaluate.py
    # Hand back the finished summary table.
    return g


# Q: What does this function do?
#    A: It pairs every run with factor == a with the run with factor == b that has the same other settings and the same
#       seed, and records both scores and delta_f1 = f1_b - f1_a.
# Q: Why must the other settings and the seed match?
#    A: So that only the compared factor (for example RGB versus 13 bands) differs between the two runs. Then the
#       difference is caused by that factor and not by a different seed or label fraction, and the seed-to-seed noise
#       cancels in the difference.
# Q: Why are ci_low and ci_high empty?
#    A: A bootstrap interval is computed from per-image predictions. The logs contain only summary scores, so there is
#       nothing to re-sample; the columns are kept (as NaN) so the files match evaluate.py.
# Q: What does it return, and what happens when a run has no partner?
#    A: A DataFrame with one row per matched pair (empty if there is none). A run without a partner is simply ignored,
#       because only keys present on both sides are visited.
def pairs_table(runs: pd.DataFrame, factor: str, a: str, b: str) -> pd.DataFrame:
    """Per-seed pairs where only `factor` differs (a -> b); delta_f1 = f1_b - f1_a; no bootstrap interval.

    Args:
        runs: one row per finished run (same table as for groups_table).
        factor: the column being compared, for example "modality" or "variant".
        a: the value of `factor` for the baseline runs.
        b: the value of `factor` for the runs compared against the baseline.
    Returns:
        One row per pair of runs that share every other setting and the seed, with both accuracies and
        F1 scores, the F1 difference delta_f1, and empty (NaN) interval columns ci_low and ci_high.
    """
    # The columns that must be equal for two runs to form a pair: all setting columns except the compared one.
    keys = [k for k in GROUP if k != factor]
    # Split the runs into the "a" side and the "b" side, each indexed by the matching keys plus the seed,
    # so the two sides can be lined up row by row.
    A = runs[runs[factor] == a].set_index(keys + ["seed"])
    B = runs[runs[factor] == b].set_index(keys + ["seed"])
    # One output row will be collected for every matched pair.
    rows = []
    # Visit every key combination (including the seed) that exists on both sides.
    for idx in A.index.intersection(B.index):
        # Fetch the run on each side of this pair.
        ra, rb = A.loc[idx], B.loc[idx]
        # Store the shared keys, the two compared values, both accuracies and F1 scores, and the F1 gain of b
        # over a. There is no bootstrap interval without predictions, so the interval columns stay empty.
        rows.append({**dict(zip(keys + ["seed"], idx)), "a": a, "b": b,
                     "acc_a": ra["test_acc"], "acc_b": rb["test_acc"], "f1_a": ra["test_f1"], "f1_b": rb["test_f1"],
                     "delta_f1": rb["test_f1"] - ra["test_f1"], "ci_low": np.nan, "ci_high": np.nan})
    # Turn the list of row dictionaries into a table (empty if no pair was found).
    return pd.DataFrame(rows)


# Q: What does this function do?
#    A: For every logged run whose folder was kept, it reads metrics.json and compares the saved test accuracy and
#       macro-F1 with the numbers in the log, then prints how many runs were checked and which differ.
# Q: Why is the tolerance 0.00006?
#    A: The log prints 4 decimals, so rounding alone can differ from the exact value by up to 0.00005. Anything larger
#       means the log and the saved metrics really disagree.
# Q: Why are runs without a folder skipped, and what does it return?
#    A: Lanes A and C have no run folders, so there is nothing to compare with. The function returns nothing; its result
#       is the printed summary line and one line per disagreement.
def cross_check(runs: pd.DataFrame, runs_dir: Path) -> None:
    """Compare logged numbers with metrics.json for every run that has a folder; print the outcome.

    Args:
        runs: one row per logged run, with run_name, test_acc and test_f1.
        runs_dir: folder that contains one sub-folder per kept run, each holding a metrics.json.
    Returns:
        Nothing; a summary line (and one line per disagreement) is printed.
    """
    # Count how many runs were compared, and collect a message for every mismatch.
    checked, bad = 0, []
    # Look at every logged run one by one.
    for _, r in runs.iterrows():
        # Where this run's metrics file would be if its folder was kept.
        m = runs_dir / r["run_name"] / "metrics.json"
        # No folder kept (lanes A and C): there is nothing to compare against, so skip this run.
        if not m.exists():
            continue
        # Read the test-set metrics that the training code saved.
        t = json.loads(m.read_text())["test"]
        checked += 1
        # Compare accuracy and macro-F1; the log has 4 decimals, so allow a rounding error of 0.00006.
        for key, logged in (("accuracy", r["test_acc"]), ("macro_f1", r["test_f1"])):
            # A larger difference means the log and the saved metrics disagree, so remember it for the report.
            if abs(t[key] - logged) > 6e-5:
                bad.append(f"{r['run_name']} {key}: log {logged} vs metrics.json {t[key]:.5f}")
    # Print a one-line verdict: how many runs were checked and whether they all agree.
    print(f"cross-check: {checked} runs with a folder compared with metrics.json; "
          f"{'all agree to the 4 printed decimals' if not bad else str(len(bad)) + ' differ'}")
    # List each disagreement on its own line, indented.
    for line in bad:
        print("  " + line)


# Q: What does main do, step by step?
#    A: It reads the options, loads runs_from_logs.csv, optionally runs the cross-check, writes groups.csv, and then for
#       each entry of COMPARISONS builds the per-seed pairs, saves them, summarises them over seeds and saves the
#       summary.
# Q: Why is a comparison skipped when pairs is empty?
#    A: If the logs have no matching runs for it (for example a model that was not run), writing an empty file would
#       only mislead the figure and table code, which treats a missing file as 'no data'.
# Q: Why reuse summarise_comparison and then set seeds_ci_excludes_0 to NaN?
#    A: Reusing evaluate.py's function gives exactly the same columns. Without intervals that function would count 0
#       seeds whose interval excludes zero, which would wrongly look like a real result, so the column is overwritten
#       with empty values.
def main() -> None:
    """Write groups.csv and the compare_* files for every comparison that has at least one pair."""
    # Build the command-line parser; its description is the first line of this file's docstring.
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Option: the table of runs parsed from the notebook logs.
    ap.add_argument("--logs", type=Path, default=Path("results/logs/runs_from_logs.csv"))
    # Option: the folder where the output CSV files are written.
    ap.add_argument("--out", type=Path, default=Path("results/summary/eval"))
    # Option: folder of kept run folders; only used for the optional cross-check.
    ap.add_argument("--runs-dir", type=Path, default=None, help="optional: cross-check against metrics.json here")
    # Read the options the user typed (or the defaults).
    args = ap.parse_args()

    # Load the logged per-run results.
    runs = pd.read_csv(args.logs)
    # Make sure the output folder exists before writing into it.
    args.out.mkdir(parents=True, exist_ok=True)
    # Only when a runs folder was given: verify the logged numbers against the saved metrics.json files.
    if args.runs_dir is not None:
        cross_check(runs, args.runs_dir)
    # Write the per-setting summary (mean / std over seeds), without the row numbers.
    groups_table(runs).to_csv(args.out / "groups.csv", index=False)
    # Tell the user how many runs and how many settings were summarised.
    print(f"groups.csv: {len(runs)} runs in {len(groups_table(runs))} settings")
    # Handle each comparison in turn.
    for stem, factor, a, b in COMPARISONS:
        # Build the per-seed pairs for this comparison.
        pairs = pairs_table(runs, factor, a, b)
        # If the logs contain no matching pairs, there is nothing to write for this comparison.
        if pairs.empty:
            continue
        # Save the per-seed pairs.
        pairs.to_csv(args.out / f"compare_{stem}_pairs.csv", index=False)
        # Summarise the pairs over seeds (mean difference etc.) with the shared evaluate.py function.
        summ = summarise_comparison(pairs)
        summ["seeds_ci_excludes_0"] = np.nan  # no intervals without predictions
        # Save the summary and report how many pairs it was built from.
        summ.to_csv(args.out / f"compare_{stem}.csv", index=False)
        print(f"compare_{stem}: {len(pairs)} pairs")


# Run main() only when the file is executed as a script, not when it is imported.
if __name__ == "__main__":
    main()

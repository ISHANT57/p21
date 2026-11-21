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

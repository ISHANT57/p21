"""Gather every finished run into tables.

Where it fits: a quick summary of whatever has finished, using the numbers each run logged at training
time (metrics.json, config.yaml, env.json, and for pseudo-label runs pseudo_label_quality.json and
pseudo_label_selection.json). It does not read predictions. The Phase 6 script scripts/evaluate.py is the
one that recomputes the final report numbers from the saved predictions; this script is the fast
overview and also writes the LR-sweep (tuning) table.

Usage:
    python scripts/collect_results.py                 # reads results/runs/, writes results/summary/
    python scripts/collect_results.py --runs-dir /content/drive/MyDrive/p21_runs

Writes:
    results/summary/runs.csv         one row per finished run (main grid, tuning, extras)
    results/summary/supervised.csv   main grid: mean / std / n over seeds per model x input x fraction
    results/summary/tuning.csv       LR sweep, validation metrics only
    results/summary/semi_supervised.csv  pseudo-label students vs their supervised teachers (paired by seed)

Design decisions:
- Runs are told apart by the config name_suffix: "" is a main-grid run, a suffix starting with "tune" is an
  LR-sweep run (validation metrics only, no test numbers), and "pl", "plb" (optionally followed by digits)
  are pseudo-label students.
- Student vs teacher is paired through the student's own config "teacher" field (the teacher's run name),
  so each student is compared with the exact teacher run named there, not with a re-matched run.
- Results are mean and sample std (ddof=1) over seeds; with one seed the std is NaN. The printed table
  shows "(n=1)" instead of a std in that case.
- Files are written only when the corresponding group of runs exists.
"""

# argparse reads the options; json reads the files each run saved; Path represents folder paths.
import argparse
import json
from pathlib import Path

# pandas holds the results as tables; yaml reads each run's config.yaml.
import pandas as pd
import yaml

# RESULTS_DIR is the top-level results folder; RUNS_DIR is the default folder holding one sub-folder per run.
from eurosat_le.constants import RESULTS_DIR
from eurosat_le.train.engine import RUNS_DIR

# ==================== CONCEPTS USED IN THIS FILE ====================
# - argparse: standard-library module that reads command-line options (--runs-dir) into an object.
# - pandas DataFrame: a table of rows and columns; load_runs returns one row per run and main() filters, groups and
#   saves it.
# - json / yaml: json.loads reads metrics.json, env.json and the pseudo-label files; yaml.safe_load reads each run's
#   config.yaml (the settings that run was trained with).
# - run folder / metrics.json: every finished run has a folder in results/runs/; metrics.json is written last, so its
#   presence marks a finished run.
# - accuracy and macro-F1: accuracy = share of correct predictions; macro-F1 = the F1 score (balance of precision and
#   recall) computed per class and averaged with equal weight, so rare classes count as much as big ones. Macro-F1 is
#   the main score here.
# - validation vs test scores: val_* come from the validation set (used for tuning and early stopping); test_* come from
#   the test set (final result). Tuning runs have no test scores (NaN).
# - name_suffix: text in the run's config that tells the kind of run: '' = main grid, 'tune...' = learning-rate sweep,
#   'pl' / 'plb' (+ digits) = pseudo-label students.
# - learning-rate sweep (tuning): several learning rates tried on one cell; the best one per model is the row with the
#   highest validation macro-F1 (idxmax).
# - mean and sample std over seeds (ddof=1): each setting is run with 3 seeds; the table shows the mean and the sample
#   standard deviation (divide by n-1). With a single seed the std is NaN, printed as '(n=1)'.
# - groupby / agg / pivot_table: pandas operations: groupby collects the rows of one setting, agg computes mean/std per
#   column, pivot_table lays results out as rows = model/input, columns = label fraction.
# - pseudo-label student vs teacher (paired by seed): each student's config names its teacher; the table compares each
#   student with that exact teacher run and reports delta = student - teacher (positive = pseudo-labelling helped).
# - tau and pseudo-label coverage / accuracy: tau = confidence threshold for accepting pseudo-labels; coverage =
#   selected / pool size (share of unlabelled images that got a pseudo-label); pseudo_acc = share of those pseudo-labels
#   that were correct (measured afterwards).
# - regular expression r'^plb?\d*$': pattern that matches the suffixes pl, pl2, plb, plb2, ... to find the student runs.
# - git commit / device: env.json records which machine (cpu / GPU) and which short git commit produced each run.
# - FileNotFoundError: raised if a run folder lacks config.yaml or env.json.
# - if __name__ == '__main__': runs main() only when executed as a script, not when imported.
# ====================================================================
#
# ---- Q&A: THIS FILE ----
# Q: What is this file for?
# A: It is the quick summary tool: it gathers every finished run into tables (runs.csv, tuning.csv, supervised.csv,
#    semi_supervised.csv) from the numbers each run logged during training.
# Q: How is it different from evaluate.py?
# A: collect_results.py does not read predictions; it copies the logged metrics. evaluate.py recomputes the final report
#    numbers (plus ECE, confusion matrices and paired bootstrap comparisons) from the saved predictions.
# Q: What does it read and write?
# A: Reads results/runs/*/ (metrics.json, config.yaml, env.json and, for students, pseudo_label_quality.json and
#    pseudo_label_selection.json). Writes results/summary/*.csv and prints the tables.
# Q: How is it run?
# A: python scripts/collect_results.py [--runs-dir PATH], for example a Google Drive copy of the runs.

OUT = RESULTS_DIR / "summary"  # same folder that scripts/evaluate.py writes its eval/ subfolder into


# ---- Q&A: load_runs ----
# Q: What does load_runs() do?
# A: It visits every run folder that has metrics.json and builds one table row per run: its config fields (model,
#    modality, fraction, seed, lr, ...), logged metrics (best epoch, train minutes, val and test accuracy/macro-F1),
#    environment (device, short git commit) and, for students, pseudo-label accuracy, tau and coverage.
# Q: Why does it skip folders without metrics.json?
# A: Those runs did not finish (crashed or still running), so their numbers would be incomplete.
# Q: How is it different from eurosat_le.eval.evaluate.load_runs?
# A: That one needs saved test predictions and so ignores tuning runs; this one only needs the logged metrics, so it
#    also includes tuning runs (test columns are NaN).
# Q: What are the input and the return value?
# A: Input: the folder that contains the run folders. Returns a pandas DataFrame (empty if no run is finished).
def load_runs(runs_dir: Path) -> pd.DataFrame:
    """One row per run folder that has metrics.json (sorted by folder name).

    Columns come from config.yaml (model, modality, fraction, seed, name_suffix, lr, init_mode, teacher),
    metrics.json (best_epoch, epochs_run, train_min, n_labelled, val and test accuracy/macro-F1,
    n_pseudo), env.json (device, short git commit) and, for pseudo-label students, the pseudo-label
    quality files (pseudo_acc, tau, pseudo_coverage = selected / pool size). test_acc and test_f1 are
    absent (NaN) for runs without test metrics, such as tuning runs. Unlike
    eurosat_le.eval.evaluate.load_runs this does NOT require saved test predictions. A missing
    config.yaml or env.json raises FileNotFoundError.
    """
    # One dictionary per run; they are turned into a table at the end.
    rows = []
    # Visit every run folder that has a metrics.json (i.e. a finished run), in alphabetical order.
    for mpath in sorted(runs_dir.glob("*/metrics.json")):
        # The metrics the run logged at the end of training.
        m = json.loads(mpath.read_text())
        # The settings the run was trained with (model, input type, label fraction, ...).
        cfg = yaml.safe_load((mpath.parent / "config.yaml").read_text())
        # Information about the machine and code version (device, git commit).
        env = json.loads((mpath.parent / "env.json").read_text())
        # Start the row with the config values we want; .get gives None when a run does not have that field.
        row = {k: cfg.get(k) for k in ("model", "modality", "fraction", "seed", "name_suffix", "lr", "init_mode",
                                       "teacher")}
        row["name_suffix"] = row["name_suffix"] or ""  # None (no suffix) -> "" so .str methods work below
        # Add the logged results: run name, training length and time, label count, device, commit and val scores.
        row.update(run_name=m["run_name"], best_epoch=m["best_epoch"], epochs_run=m["epochs_run"],
                   train_min=m["train_seconds"] / 60, n_labelled=m["n_labelled"], device=env["device"],
                   git_commit=env["git_commit"][:7], val_acc=m["val"]["accuracy"], val_f1=m["val"]["macro_f1"])
        if "test" in m:  # tuning runs skip the test split
            row.update(test_acc=m["test"]["accuracy"], test_f1=m["test"]["macro_f1"])
        # Number of pseudo-labelled images used (0 for ordinary supervised runs).
        row["n_pseudo"] = m.get("n_pseudo", 0)
        # Pseudo-label students also have quality files; read them when present.
        qpath = mpath.parent / "pseudo_label_quality.json"
        if qpath.exists():
            # How many of the pseudo-labels were actually correct (known only after training).
            row["pseudo_acc"] = json.loads(qpath.read_text())["accuracy"]
            # The confidence threshold tau and how many unlabelled images passed it.
            sel = json.loads((mpath.parent / "pseudo_label_selection.json").read_text())
            row.update(tau=sel["tau"], pseudo_coverage=sel["n_selected"] / sel["pool_size"])
        rows.append(row)
    # Combine all rows into one table (an empty table if no run was found).
    return pd.DataFrame(rows)


# ---- Q&A: main ----
# Q: What does main() do?
# A: It loads all runs, saves runs.csv, then prints and saves three summaries: the learning-rate sweep (tuning.csv), the
#    supervised main grid (mean and std over seeds, supervised.csv) and the pseudo-label students against their teachers
#    (semi_supervised.csv). Each file is written only if the matching group of runs exists.
# Q: Why is the student compared with the teacher named in its own config?
# A: So each student is compared with the exact teacher run it was trained from, not with a guessed match; the
#    difference is then a fair paired comparison per seed.
# Q: Why .to_numpy() when copying teacher scores to the student rows?
# A: To line the values up by position; pandas would otherwise try to align them by the (different) row labels.
# Q: What are the inputs and what does it return?
# A: Option --runs-dir (default results/runs). Returns None; stops with a message if no finished run is found.
def main() -> None:
    """Load all runs, write runs.csv, and print and save the tuning, supervised and pseudo-label summaries."""
    # Build the argument parser; the module docstring doubles as the --help text.
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Folder with the run sub-folders (can point to e.g. a Google Drive copy).
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    # Parse sys.argv into the args object.
    args = p.parse_args()

    # Read every finished run into one table.
    df = load_runs(args.runs_dir)
    # Nothing to summarise: stop with a clear message.
    if df.empty:
        raise SystemExit(f"no finished runs in {args.runs_dir}")
    # Make sure the output folder exists before writing into it.
    OUT.mkdir(parents=True, exist_ok=True)
    # Save the full per-run table (one row per run).
    df.to_csv(OUT / "runs.csv", index=False)
    print(f"{len(df)} finished runs -> {OUT / 'runs.csv'}")

    # Learning-rate sweep runs are recognised by a name suffix starting with "tune".
    tune = df[df["name_suffix"].str.startswith("tune")]
    if not tune.empty:
        # Keep only the columns of interest, ordered by model and learning rate.
        tune = tune.sort_values(["model", "lr"])[["model", "lr", "val_acc", "val_f1", "best_epoch", "train_min"]]
        tune.to_csv(OUT / "tuning.csv", index=False)
        print("\nLR sweep (RGB, 10% labels, seed 0; validation only):")
        print(tune.to_string(index=False, float_format="%.4f"))
        # For every model, find the row with the highest validation macro-F1 and print its learning rate.
        best = tune.loc[tune.groupby("model")["val_f1"].idxmax()]
        print("\nbest lr by val macro-F1:", dict(zip(best["model"], best["lr"])))

    # Main grid: plain supervised runs only (no tuning runs, no pseudo-label students).
    main_runs = df[df["name_suffix"] == ""]
    if not main_runs.empty:
        # A "setting" is one model + input type + label fraction; its seeds are averaged together.
        keys = ["model", "modality", "fraction"]
        # Per setting: number of seeds, mean and std of the test accuracy and F1, plus mean val F1 and train time.
        agg = main_runs.groupby(keys).agg(
            n_seeds=("seed", "nunique"),
            test_acc_mean=("test_acc", "mean"), test_acc_std=("test_acc", "std"),
            test_f1_mean=("test_f1", "mean"), test_f1_std=("test_f1", "std"),
            val_f1_mean=("val_f1", "mean"), train_min_mean=("train_min", "mean"),
        ).reset_index()
        agg.to_csv(OUT / "supervised.csv", index=False)
        # Make one printable text per setting, "mean ± std (n=seeds)"; with a single seed there is no std to show.
        cell = agg.assign(v=lambda d: [f"{m:.4f} ± {s:.4f} (n={n})" if n > 1 else f"{m:.4f} (n=1)"
                                       for m, s, n in zip(d.test_f1_mean, d.test_f1_std.fillna(0), d.n_seeds)])
        print("\nTest macro-F1, mean ± std over seeds:")
        # Show it as a grid: one row per model/input type, one column per label fraction.
        print(cell.pivot_table(index=["model", "modality"], columns="fraction", values="v", aggfunc="first")
              .to_string())

    # Pseudo-label students: "pl" (global threshold) or "plb" (class-balanced), optionally with a digit suffix.
    students = df[df["name_suffix"].str.match(r"^plb?\d*$")]
    if not students.empty:
        # All runs indexed by name, so each student's teacher can be looked up directly.
        teach = df.set_index("run_name")
        st = students[students["teacher"].isin(teach.index)].copy()  # drop students whose teacher run is missing
        # .to_numpy() so the teacher values line up by position with the students, not by (different) index
        # The teacher's own test scores, placed next to each student's row.
        st["teacher_test_f1"] = teach.loc[st["teacher"], "test_f1"].to_numpy()
        st["teacher_test_acc"] = teach.loc[st["teacher"], "test_acc"].to_numpy()
        # Improvement of the student over its teacher (positive = pseudo-labelling helped).
        st["delta_f1"] = st["test_f1"] - st["teacher_test_f1"]
        st["delta_acc"] = st["test_acc"] - st["teacher_test_acc"]
        # Group by setting and pseudo-label variant, then average over the seeds.
        keys = ["model", "modality", "fraction", "name_suffix"]
        agg = st.groupby(keys).agg(
            n_seeds=("seed", "nunique"), supervised_f1=("teacher_test_f1", "mean"), pl_f1=("test_f1", "mean"),
            delta_f1_mean=("delta_f1", "mean"), delta_f1_std=("delta_f1", "std"),
            delta_f1_min=("delta_f1", "min"), delta_acc_mean=("delta_acc", "mean"),
            tau_mean=("tau", "mean"), coverage_mean=("pseudo_coverage", "mean"), pseudo_acc_mean=("pseudo_acc", "mean"),
        ).reset_index()
        agg.to_csv(OUT / "semi_supervised.csv", index=False)
        print("\nPseudo-labelling vs supervised teacher (test macro-F1, paired by seed):")
        print(agg.to_string(index=False, float_format="%.4f"))


# Run main() only when the file is executed as a script, not when it is imported.
if __name__ == "__main__":
    main()

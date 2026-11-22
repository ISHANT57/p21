"""Extra report tables from the logged results: every run, pseudo-label details, training cost,
learning-rate sweep and model-against-model comparisons.

Where it fits: scripts/make_figures.py writes the main tables from groups.csv. These tables need the
per-run rows (results/logs/runs_from_logs.csv, results/logs/tuning_from_logs.csv, made by
scripts/parse_colab_logs.py) and the comparison summaries made by scripts/eval_from_logs.py.

Usage:
    python scripts/make_report_tables.py [--logs results/logs] [--eval results/summary/eval]
                                         [--out results/tables]

Writes, each as <name>.md and <name>.tex in --out:
    all_runs          every finished run (test accuracy, macro-F1, validation F1, best epoch, minutes)
    pl_details        every pseudo-label student: threshold, validation precision, pool share, change vs teacher
    training_cost     mean time and best epoch per model and label fraction (supervised runs)
    lr_sweep          the learning-rate sweep (validation scores only)
    model_comparison  scratch CNN -> ResNet-50 -> ViT-Small, matched on seed (supervised runs)
    calibration       mean test ECE per model, input, training type and label fraction (every run with saved
                      predictions, from results/summary/eval/per_run.csv written by scripts/evaluate.py)

Design decisions:
- Scores are shown in percent with one decimal, as in the main tables; a cell with one seed shows
  "(n=1)", never a made-up spread.
- All numbers come straight from the CSV files; nothing is estimated here.
"""

# Standard library: argparse reads command-line options and Path handles file paths.
import argparse
from pathlib import Path

# numpy compares label fractions safely (np.isclose in calibration); pandas builds and reads the tables.
import numpy as np
import pandas as pd

# Project code: tab.write_table saves a table as .md and .tex; the figures module supplies the display names
# (labels for fractions, inputs and models) and the fixed model order so tables match the figures.
from eurosat_le import tables as tab
from eurosat_le.figures import FRACTION_LABEL, MODALITY_LABEL, MODEL_LABEL, MODELS

# ======================================================================
# CONCEPTS USED IN THIS FILE
# ======================================================================
# runs_from_logs.csv: one row per finished run (scores, best epoch, minutes, teacher details), made by
#   parse_colab_logs.py. tuning_from_logs.csv: the learning-rate sweep runs.
# display table: a DataFrame in which every cell is already formatted text (such as '97.8' or '+1.2 ± 0.4'), ready to be
#   written as Markdown or LaTeX.
# mean ± std and '(n=1)': the average over seeds and their spread. With one seed there is no spread, so the cell shows
#   '(n=1)' instead of a made-up number (see _pm).
# percentage points: a difference between two percentages (F1 of 95.0 versus 93.5 is +1.5 points). Scores are multiplied
#   by 100 and shown with one decimal.
# macro-F1 and accuracy: macro-F1 averages the per-class F1 with equal weight; accuracy is the share of correct
#   predictions. Columns test_f1, test_acc, val_f1.
# test vs validation: validation scores choose the best epoch and the learning rate; test scores are the final result.
#   The learning-rate sweep shows validation scores only.
# pseudo-labelling, teacher and student: a teacher model labels the unlabelled pool; the student is trained on those
#   labels. pl_details compares each student with its teacher.
# tau, validation precision, pool share: the confidence threshold for accepting a pseudo-label, the share of accepted
#   labels that were correct on the validation set, and the percentage of pool images that got a label.
# best epoch and training cost: the epoch with the best validation score and the minutes a run took; training_cost
#   averages them per model and fraction.
# learning-rate sweep: short runs with several learning rates, compared on validation scores; the rates that were used
#   afterwards are marked 'yes'.
# paired comparison (matched on seed): two models trained with the same seed are compared, then the differences are
#   summarised. model_comparison reads these from compare_<stem>.csv.
# sort key columns (_v, _m, _o): temporary numeric columns made with assign(...map(...)) so the rows sort in a chosen
#   order (variant, model order, rgb before ms) and not alphabetically.
# groupby / agg: split rows into groups (model and fraction) and compute count and mean per group. Used in
#   training_cost.
# format specs: '{:.1f}' is one decimal, '{:+.1f}' always shows the sign, {:,} adds thousands separators (used for the
#   pseudo-label counts), and 'g' prints 0.0005 compactly.
# NaN and pd.isna: pandas' 'not a number' value; here it marks a missing standard deviation.
# Markdown and LaTeX tables with booktabs: tab.write_table writes <name>.md and <name>.tex; the LaTeX uses toprule /
#   midrule / bottomrule lines; 'wide' means a table* spanning both columns; \scriptsize, \small are LaTeX font sizes.
# argparse: reads the command-line options --logs, --eval and --out. CSV: plain-text table loaded with pd.read_csv.

# ======================================================================
# Q&A ABOUT THIS FILE (file-level viva questions)
# ======================================================================
# Q: What is this file for?
#    A: It writes the extra report tables that make_figures.py does not make: all_runs, pl_details, training_cost,
#       lr_sweep and model_comparison, each as .md and .tex.
# Q: Where does it fit in the pipeline?
#    A: After parse_colab_logs.py (runs_from_logs.csv, tuning_from_logs.csv) and eval_from_logs.py (compare_*.csv).
#       scripts/make_figures.py writes the main tables; this script writes the rest.
# Q: How is it run and what does it read and write?
#    A: python scripts/make_report_tables.py [--logs results/logs] [--eval results/summary/eval] [--out results/tables].
#       It reads the two log CSVs and the compare_<stem>.csv files and writes the tables into --out.
# Q: Where do the numbers come from?
#    A: Straight from the CSV files; nothing is estimated here. A cell with a single seed shows '(n=1)' and no spread.

# Display names for the three training variants (plain supervised, pseudo-labelling, class-balanced pseudo-labelling).
VARIANT = {"supervised": "supervised", "pl": "pseudo-label", "plb": "class-balanced"}


# Q: What does this function do?
#    A: It turns a label fraction such as 0.1 into the text '10%' using FRACTION_LABEL, or a rounded percentage if the
#       fraction has no fixed label.
# Q: Why is the value rounded before the lookup?
#    A: A float like 0.10000000001 would not equal the dictionary key 0.1; rounding to 4 decimals makes the lookup work.
# Q: Why does this file have its own copy?
#    A: The same helper in eurosat_le.tables is private (starts with an underscore), so this script keeps its own copy.
def _frac(f: float) -> str:
    """Label fraction as "1%", "10%", "100%".

    Args:
        f: label fraction between 0 and 1 (for example 0.1).
    Returns:
        The text used in table cells; an unknown fraction is shown as a rounded percentage.
    """
    # Look up the nice label (rounding first so 0.1 and 0.10000001 match); fall back to a computed percentage.
    return FRACTION_LABEL.get(round(f, 4), f"{f:.0%}")


# Q: What does this function do?
#    A: It formats 'mean ± std' for one table cell after multiplying both numbers by scale (100 gives percent). With
#       sign=True it always shows + or -, which is used for differences.
# Q: Why does a missing std give '(n=1)'?
#    A: A NaN std means only one seed. Printing '± 0.0' would wrongly suggest the seeds agreed perfectly, so the cell
#       says '(n=1)'.
# Q: What are the inputs and the output?
#    A: mean, std, scale, digits and sign; it returns a string.
def _pm(mean: float, std: float, scale: float = 100.0, digits: int = 1, sign: bool = False) -> str:
    """"mean ± std" in percent points, or "mean (n=1)" when there is no std (one seed).

    Args:
        mean: the mean value, as a 0-1 score or difference.
        std: the standard deviation over seeds; NaN when only one seed exists.
        scale: factor applied to both numbers (100 turns a 0-1 score into percent).
        digits: number of decimals to show.
        sign: if True, always show a + or - sign in front of the mean (for differences).
    Returns:
        The formatted text for one table cell.
    """
    # Build the number format, for example "{:+.1f}" with a sign or "{:.1f}" without one.
    fmt = f"{{:+.{digits}f}}" if sign else f"{{:.{digits}f}}"
    # No std means one seed: show the mean only and say so, instead of inventing a spread.
    if pd.isna(std):
        return fmt.format(mean * scale) + " (n=1)"
    # Otherwise show the mean and the spread over seeds.
    return fmt.format(mean * scale) + f" ± {std * scale:.{digits}f}"


# Q: What does this function do?
#    A: It builds the table of every finished run with readable names, scores in percent, best epoch and minutes,
#       ordered by variant, model, input, fraction and seed.
# Q: Why are temporary sort-key columns added?
#    A: Alphabetical order would put 'ms' before 'rgb' and the models in the wrong order. Mapping the names to numbers
#       (0, 1, 2 ...) gives the order used everywhere in the report.
# Q: What does it return?
#    A: A DataFrame of display strings; the sort keys are not part of it.
def all_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per finished run, ordered by variant, model, input, fraction, seed.

    Args:
        runs: runs_from_logs.csv as a table (one row per run).
    Returns:
        A table of display strings with the columns Model, Input, Labels, Seed, Variant, Test acc., Test F1,
        Val. F1, Best epoch and Minutes.
    """
    # Add temporary numeric sort keys (_v variant, _m model, _o input) so rows sort in a meaningful order
    # rather than alphabetically, then sort by those keys, the label fraction and the seed.
    r = runs.assign(_v=runs["variant"].map({"supervised": 0, "pl": 1, "plb": 2}),
                    _m=runs["model"].map({m: i for i, m in enumerate(MODELS)}),
                    _o=runs["modality"].map({"rgb": 0, "ms": 1})).sort_values(["_v", "_m", "_o", "fraction", "seed"])
    # Build the display table: readable names for the settings and percentages with two decimals for the scores.
    return pd.DataFrame({
        "Model": r["model"].map(MODEL_LABEL), "Input": r["modality"].map(MODALITY_LABEL),
        "Labels": r["fraction"].map(_frac), "Seed": r["seed"].astype(int), "Variant": r["variant"].map(VARIANT),
        "Test acc.": (r["test_acc"] * 100).map("{:.2f}".format), "Test F1": (r["test_f1"] * 100).map("{:.2f}".format),
        "Val. F1": (r["val_f1"] * 100).map("{:.2f}".format), "Best epoch": r["best_epoch"].astype(int),
        "Minutes": r["minutes"].map("{:.1f}".format)}).reset_index(drop=True)


# Q: What does this function do?
#    A: For every pseudo-label student it shows the threshold tau, the validation precision, how many pool images got a
#       pseudo-label (and their share), the teacher's and the student's test F1 and the change.
# Q: How is the teacher's F1 found?
#    A: Each student row stores its teacher's run name. A Series that maps run_name to test_f1 is used with map() to
#       look the teacher's score up. Change = student F1 minus teacher F1, in points.
# Q: What does it return, and when None?
#    A: A DataFrame of display strings, or None if no run is a student (variant other than 'supervised').
def pl_details(runs: pd.DataFrame) -> pd.DataFrame | None:
    """Pseudo-label students with their threshold, pool share and the change against their teacher.

    Args:
        runs: runs_from_logs.csv as a table; students are the rows whose variant is not "supervised".
    Returns:
        A table of display strings with one row per student, or None if there are no students.
    """
    # Keep only the pseudo-label students (copy so adding columns below does not touch the original table).
    s = runs[runs["variant"] != "supervised"].copy()
    # No students means no table.
    if s.empty:
        return None
    # Map every run name to its test F1, so each student can look up the score of its teacher.
    teacher_f1 = runs.set_index("run_name")["test_f1"]
    s["teacher_f1"] = s["teacher"].map(teacher_f1)
    # Add temporary sort keys (model order, input order) and sort the students for display.
    s = s.assign(_m=s["model"].map({m: i for i, m in enumerate(MODELS)}),
                 _o=s["modality"].map({"rgb": 0, "ms": 1})).sort_values(["_m", "_o", "fraction", "seed"])
    # Build the display table: threshold tau, validation precision, how many pool images got a pseudo-label
    # (and their share), the teacher's and student's test F1, and the change (student minus teacher) in points.
    return pd.DataFrame({
        "Model": s["model"].map(MODEL_LABEL), "Input": s["modality"].map(MODALITY_LABEL),
        "Labels": s["fraction"].map(_frac), "Seed": s["seed"].astype(int),
        "$\\tau$": s["tau"].map("{:.2f}".format), "Val. precision": s["val_precision"].map("{:.3f}".format),
        "Pseudo-labelled": [f"{int(a):,} / {int(b):,}" for a, b in zip(s["n_pseudo"], s["n_pool"])],
        "Share (%)": (s["n_pseudo"] / s["n_pool"] * 100).map("{:.1f}".format),
        "Teacher F1": (s["teacher_f1"] * 100).map("{:.1f}".format), "Student F1": (s["test_f1"] * 100).map("{:.1f}".format),
        "$\\Delta$": ((s["test_f1"] - s["teacher_f1"]) * 100).map("{:+.1f}".format)}).reset_index(drop=True)


# Q: What does this function do?
#    A: For the supervised runs it counts the runs and averages the minutes and the best epoch for each model and label
#       fraction.
# Q: Why only the supervised runs?
#    A: They are the baseline of each model; filtering on variant == 'supervised' keeps the pseudo-label students out of
#       the averages.
# Q: What does it return?
#    A: A DataFrame of display strings, ordered by the fixed model order and then by fraction.
def training_cost(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean time and best epoch per model and label fraction, for supervised runs.

    Args:
        runs: runs_from_logs.csv as a table (one row per run).
    Returns:
        A table of display strings with one row per model and label fraction.
    """
    # Only the plain supervised runs are compared for cost.
    s = runs[runs["variant"] == "supervised"]
    # For each model and fraction: number of runs, mean minutes and mean best epoch.
    g = s.groupby(["model", "fraction"]).agg(n=("run_name", "count"), minutes=("minutes", "mean"),
                                           epoch=("best_epoch", "mean")).reset_index()
    # Sort by the fixed model order, then by fraction.
    g = g.assign(_m=g["model"].map({m: i for i, m in enumerate(MODELS)})).sort_values(["_m", "fraction"])
    # Build the display table with readable names and one decimal for the means.
    return pd.DataFrame({"Model": g["model"].map(MODEL_LABEL), "Labels": g["fraction"].map(_frac), "Runs": g["n"].astype(int),
                         "Mean time (min)": g["minutes"].map("{:.1f}".format),
                         "Mean best epoch": g["epoch"].map("{:.1f}".format)}).reset_index(drop=True)


# Q: What does this function do?
#    A: It formats the learning-rate sweep (validation accuracy and F1 per model and learning rate) and marks the
#       learning rate that was used for the main experiments.
# Q: Why are only validation scores shown, and where does the 'Used' mark come from?
#    A: The rate is chosen on validation data so the test set stays untouched. The chosen pairs are written in the set
#       'used' in the code (scratch CNN 0.002, ResNet-50 0.0005); if the choice changed, this set must be edited.
def lr_sweep(tune: pd.DataFrame) -> pd.DataFrame:
    """Learning-rate sweep rows; the rate that was used for all later runs is marked.

    Args:
        tune: tuning_from_logs.csv as a table (one row per sweep run, validation scores only).
    Returns:
        A table of display strings with a "Used" column that says "yes" for the chosen learning rates.
    """
    # The (model, learning rate) pairs that were picked and used for the main experiments.
    used = {("scratch_cnn", 0.002), ("resnet50", 0.0005)}
    # Sort the sweep rows by the fixed model order and then by learning rate.
    t = tune.assign(_m=tune["model"].map({m: i for i, m in enumerate(MODELS)})).sort_values(["_m", "lr"])
    # Build the display table; the learning rate is printed compactly (0.0005 -> "0.0005", 1e-05 stays short).
    return pd.DataFrame({
        "Model": t["model"].map(MODEL_LABEL), "Learning rate": t["lr"].map(lambda v: f"{v:g}"),
        "Val. acc.": (t["val_acc"] * 100).map("{:.2f}".format), "Val. F1": (t["val_f1"] * 100).map("{:.2f}".format),
        "Best epoch": t["best_epoch"].astype(int), "Minutes": t["minutes"].map("{:.1f}".format),
        "Used": ["yes" if (m, lr) in used else "" for m, lr in zip(t["model"], t["lr"])]}).reset_index(drop=True)


# Q: What does this function do?
#    A: It reads the compare_<stem>.csv files for scratch CNN vs ResNet-50, scratch CNN vs ViT-Small and ResNet-50 vs
#       ViT-Small and shows, for supervised runs, the mean ± std difference in macro-F1 points (second model minus
#       first) and how many seeds the second model won.
# Q: Why is the data 'matched on seed'?
#    A: Two models trained with the same seed are compared, so the difference is not caused by different random starts.
#       'Seeds better' is seeds_b_better out of n_seeds.
# Q: What does it return, and what is the sort key lambda for?
#    A: A DataFrame, or None if none of the files exist. The lambda maps the modality names to 0 / 1 so RGB comes before
#       13-band; other columns sort normally.
def model_comparison(eval_dir: Path) -> pd.DataFrame | None:
    """Supervised model-against-model differences (b minus a, in macro-F1 points), matched on seed.

    Args:
        eval_dir: folder with the compare_<stem>.csv files written by scripts/eval_from_logs.py.
    Returns:
        A table of display strings, or None if none of the comparison files exist.
    """
    # Display rows collected from all comparison files.
    rows = []
    # The three model comparisons: (file stem, row label shown in the table).
    for stem, label in (("scratch_vs_resnet50", "ResNet-50 $-$ scratch CNN"),
                        ("scratch_vs_vit_small", "ViT-Small $-$ scratch CNN"),
                        ("resnet50_vs_vit_small", "ViT-Small $-$ ResNet-50")):
        # Where this comparison's summary file should be.
        f = eval_dir / f"compare_{stem}.csv"
        # A missing file just means that comparison is skipped.
        if not f.exists():
            continue
        c = pd.read_csv(f)
        # Keep supervised rows only, ordered by input (RGB before 13-band) and then by fraction. The sort key
        # function converts the modality names to numbers 0 / 1 and leaves other columns unchanged.
        c = c[c["variant"] == "supervised"].sort_values(["modality", "fraction"], key=lambda col: col.map(
            {"rgb": 0, "ms": 1}) if col.name == "modality" else col)
        # Turn each summary row into one display row: mean difference ± std in points, and how many seeds
        # (out of all seeds) the second model won.
        for _, r in c.iterrows():
            rows.append({"Comparison": label, "Input": MODALITY_LABEL[r["modality"]], "Labels": _frac(r["fraction"]),
                         "$\\Delta$ F1 (points)": _pm(r["delta_f1_mean"], r["delta_f1_std"], sign=True),
                         "Seeds better": f"{int(r['seeds_b_better'])}/{int(r['n_seeds'])}"})
    # Return a table if anything was collected, otherwise None so the caller can skip it.
    return pd.DataFrame(rows) if rows else None


# Q: What does this function do?
#    A: It builds the calibration table: for every model, input and training type (supervised or pseudo-label student)
#       the expected calibration error (ECE) of the test predictions, as mean ± std over seeds, one column per label
#       fraction. It reads results/summary/eval/per_run.csv (column test_ece, written by scripts/evaluate.py).
# Q: What is the ECE?
#    A: Expected calibration error. The test images are sorted into 15 bins by the model's confidence (the probability
#       it gives to its predicted class). In each bin the average confidence is compared with the share of correct
#       predictions; the ECE is the bin-size-weighted average of these gaps. A model that says '90%' and is right 90% of
#       the time has an ECE near 0; a small value means the confidence can be trusted.
# Q: Why are accuracy and the mean confidence not columns of this table?
#    A: The accuracy is already in the main tables, and per_run.csv stores only the ECE, not the mean confidence (the
#       sign of the gap, over- or under-confidence, was not saved either), so the table shows the ECE alone.
# Q: Why does the table only cover some runs?
#    A: The ECE needs the predicted probabilities, so it exists only for runs whose predictions were saved. Cells
#       without a run show '-' (for example pseudo-label students were not trained with 100% of the labels).
def calibration(eval_dir: Path) -> pd.DataFrame | None:
    """Mean test ECE (percent) per model, input and training type, one column per label fraction.

    Args:
        eval_dir: folder with per_run.csv, written by scripts/evaluate.py (columns model, modality, fraction,
            variant, seed and test_ece, where test_ece is a 0-1 value).
    Returns:
        A table of display strings, or None if per_run.csv does not exist or holds no rows.
    """
    # Where the per-run evaluation table should be; without it there is no calibration to show.
    f = eval_dir / "per_run.csv"
    if not f.exists():
        return None
    runs = pd.read_csv(f)
    # One column per label fraction that occurs in the data, from the smallest to the largest.
    fractions = sorted(runs["fraction"].unique())
    # Display rows, one per (training type, model, input) combination that has at least one run.
    rows = []
    # Supervised runs first, then the pseudo-label students, so students never mix into a supervised cell.
    for variant in ("supervised", "pl"):
        # Models in the fixed order used everywhere else (scratch CNN, ResNet-50, ViT-Small ...).
        for model in MODELS:
            # RGB before 13 bands, as in the other tables.
            for modality in ("rgb", "ms"):
                # All runs of this combination, over every seed and fraction.
                combo = runs[(runs["variant"] == variant) & (runs["model"] == model) & (runs["modality"] == modality)]
                # A combination that was never run gets no row.
                if combo.empty:
                    continue
                # Row label columns: readable model name, input and training type.
                row = {"Model": MODEL_LABEL[model], "Input": MODALITY_LABEL[modality], "Training": VARIANT[variant]}
                # One cell per label fraction: mean ± std of the test ECE over the seeds, '(n=1)' for one seed.
                for fraction in fractions:
                    # np.isclose compares the floats safely (0.1 might be stored as 0.10000000001).
                    cell = combo[np.isclose(combo["fraction"], fraction)]
                    row[_frac(fraction)] = _pm(cell["test_ece"].mean(), cell["test_ece"].std(ddof=1)) if len(cell) else "-"
                rows.append(row)
    # Return a table if anything was collected, otherwise None so the caller can skip it.
    return pd.DataFrame(rows) if rows else None


# Q: What does main do?
#    A: It loads runs_from_logs.csv, builds the five tables with their LaTeX captions and sizes, and writes each one as
#       .md and .tex, printing how many rows were written.
# Q: Why are some tables skipped?
#    A: A table that is None or empty (for example model_comparison when no compare_*.csv exists) is reported as
#       'skipped' instead of crashing the script. The sweep table is only attempted if tuning_from_logs.csv exists.
# Q: What should you remember about the captions?
#    A: They are written into the code (for example 'All 51 finished runs'), so they must be updated if the number of
#       runs changes.
def main() -> None:
    """Write the five tables."""
    # Build the command-line parser; its description is the first line of this file's docstring.
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Option: folder with runs_from_logs.csv and tuning_from_logs.csv.
    ap.add_argument("--logs", type=Path, default=Path("results/logs"))
    # Option: folder with the compare_*.csv summaries.
    ap.add_argument("--eval", type=Path, default=Path("results/summary/eval"))
    # Option: folder where the .md and .tex tables are written.
    ap.add_argument("--out", type=Path, default=Path("results/tables"))
    # Read what the user typed (or the defaults).
    args = ap.parse_args()
    # Load the table of all finished runs.
    runs = pd.read_csv(args.logs / "runs_from_logs.csv")
    # The tables to write, each as (file name, table, LaTeX caption, spans two columns?, LaTeX font size).
    made = [
        ("all_runs", all_runs(runs), "All 52 finished runs. Scores are test accuracy and macro-F1 (\\%) as printed at the end of each run; "
         "Val. F1 is the validation macro-F1 of the selected weights.", True, r"\scriptsize"),
        ("pl_details", pl_details(runs), "Pseudo-label students: the threshold $\\tau$ chosen on the validation set, the validation "
         "precision at $\\tau$, the number of pool images that received a pseudo-label, and the test macro-F1 change against "
         "the teacher of the same seed.", True, r"\scriptsize"),
        ("training_cost", training_cost(runs), "Training cost of the supervised runs on a free-tier Tesla T4.", False, r"\small"),
    ]
    # The learning-rate sweep table is only made if the sweep log exists.
    tune_csv = args.logs / "tuning_from_logs.csv"
    if tune_csv.exists():
        made.append(("lr_sweep", lr_sweep(pd.read_csv(tune_csv)), "Learning-rate sweep (RGB, 10\\% labels, seed 0). "
                     "Validation scores only; the test set was not evaluated.", False, r"\small"))
    # The model-against-model table is built from the comparison summaries.
    made.append(("model_comparison", model_comparison(args.eval), "Differences in test macro-F1 between models, supervised runs, "
                 "matched on seed (second model minus first).", True, r"\small"))
    # Write each table, skipping the ones without data.
    for name, df, caption, wide, font in made:
        # A None or empty table means its input was missing, so report it and move on.
        if df is None or df.empty:
            print(f"skipped {name}: no data")
            continue
        # Save the table as <name>.md and <name>.tex in the output folder.
        tab.write_table(df, args.out, name, caption, wide=wide, font=font)
        print(f"wrote {name} ({len(df)} rows)")


# Run main() only when the file is executed as a script, not when it is imported.
if __name__ == "__main__":
    main()

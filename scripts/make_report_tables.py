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

Design decisions:
- Scores are shown in percent with one decimal, as in the main tables; a cell with one seed shows
  "(n=1)", never a made-up spread.
- All numbers come straight from the CSV files; nothing is estimated here.
"""

# Standard library: argparse reads command-line options and Path handles file paths.
import argparse
from pathlib import Path

# numpy is imported here but not used directly in this file; pandas builds and reads the tables.
import numpy as np
import pandas as pd

# Project code: tab.write_table saves a table as .md and .tex; the figures module supplies the display names
# (labels for fractions, inputs and models) and the fixed model order so tables match the figures.
from eurosat_le import tables as tab
from eurosat_le.figures import FRACTION_LABEL, MODALITY_LABEL, MODEL_LABEL, MODELS

# Display names for the three training variants (plain supervised, pseudo-labelling, class-balanced pseudo-labelling).
VARIANT = {"supervised": "supervised", "pl": "pseudo-label", "plb": "class-balanced"}


def _frac(f: float) -> str:
    """Label fraction as "1%", "10%", "100%".

    Args:
        f: label fraction between 0 and 1 (for example 0.1).
    Returns:
        The text used in table cells; an unknown fraction is shown as a rounded percentage.
    """
    # Look up the nice label (rounding first so 0.1 and 0.10000001 match); fall back to a computed percentage.
    return FRACTION_LABEL.get(round(f, 4), f"{f:.0%}")


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
        ("all_runs", all_runs(runs), "All 51 finished runs. Scores are test accuracy and macro-F1 (\\%) as printed at the end of each run; "
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

"""Build the report figures and tables from the Phase 6 evaluation tables (Phase 7).

Usage:
    python scripts/evaluate.py            # first: writes results/summary/eval/
    python scripts/make_figures.py        # then: results/figures/*.png, results/tables/*.{md,tex}
    python scripts/make_figures.py --eval-dir path/to/eval --fraction 0.01

A figure or table whose inputs are missing is skipped with a message, so the script
can run on a partial set of results.

Where it fits: Phase 7 (report figures and tables). It does no evaluation itself; it only reads the CSV
files from results/summary/eval/ (groups.csv, per_class.csv, compare_*.csv, compare_*_pairs.csv and
confusion/*.csv) and calls eurosat_le.figures and eurosat_le.tables.

Writes (names fixed in main):
    figures/label_efficiency_f1.png, label_efficiency_acc.png, rgb_vs_ms_gain.png, pseudo_label_gain.png,
    per_class_f1_<fraction>pct.png, confusion_<fraction>pct.png
    tables/supervised_f1, supervised_acc, semi_supervised, per_class_f1_<fraction>pct  (each as .md and .tex)

Design decisions:
- Every output is one row of the `outputs` list in main: its name for the "skipped" message, whether its
  inputs exist, and a function that builds it. One loop runs the rows in order. A row whose inputs do not
  exist is left out silently; a row whose function returns None is reported as "skipped <name>: no data yet".
  To add a figure or table, add a row.
- groups.csv is mandatory; the script exits with an error if it is missing. Everything else is optional.
- --fraction (default 1.0) selects the label fraction for the per-class heatmap, per-class table and
  confusion matrices; confusion matrices are also drawn for 1 percent labels when that fraction exists.
- The confusion matrices and per-class views use the single "best model": the one with the highest mean
  test macro-F1 (over RGB and 13-band) at --fraction, as chosen by eurosat_le.tables.best_model.
- Colours, markers and labels are fixed in eurosat_le.figures so a model looks the same in every figure.
"""

# Standard library: argparse reads command-line options and Path handles file paths.
import argparse
from pathlib import Path

# pandas reads the evaluation CSV files into tables.
import pandas as pd

# Project code: figures draws the plots, tables writes the .md/.tex tables, RESULTS_DIR is the results folder.
from eurosat_le import figures as fig
from eurosat_le import tables as tab
from eurosat_le.constants import RESULTS_DIR

# ======================================================================
# CONCEPTS USED IN THIS FILE
# ======================================================================
# evaluation CSV files: groups.csv (mean / std over seeds per setting), per_class.csv (per-class F1), compare_*.csv
#   (summaries of paired comparisons), compare_*_pairs.csv (one row per seed) and confusion/*.csv. All are written
#   earlier by scripts/evaluate.py (or eval_from_logs.py).
# label fraction: the share of the labelled training images used (0.01, 0.1, 1.0). --fraction picks the one shown in the
#   per-class views and the confusion matrices.
# macro-F1 and accuracy: macro-F1 averages the per-class F1 with equal weight; accuracy is the share of correct
#   predictions. The label-efficiency figure and the supervised tables show both.
# mean ± std over seeds: the average over the repeated runs (seeds) and their sample standard deviation. '(n=1)' marks a
#   single seed. The captions say 'over 3 seeds'.
# paired comparison (pairs): two runs with the same seed that differ in one factor; the per-seed difference in macro-F1
#   is plotted by rgb_vs_ms_gain and pseudo_label_gain.
# pseudo-labelling (pl, plb): a teacher labels unlabelled images and a student trains on them; 'plb' is the
#   class-balanced version. The files compare_supervised_vs_pl / _plb compare student with teacher.
# confusion matrix and row normalisation: a table of true class (rows) against predicted class (columns); dividing each
#   row by its total makes each row sum to 100 percent, so classes of different size compare fairly.
# per-class F1 heatmap: a coloured grid with one row per land-cover class and one column per model / input.
# best model: the model with the highest mean test macro-F1 (averaged over RGB and 13-band) at --fraction; found by
#   tables.best_model and used for the confusion matrices and the per-class views.
# DataFrame and CSV: a pandas DataFrame is a table in memory; pd.read_csv loads a CSV (plain-text table) into one.
# table-driven outputs list: 'outputs' holds one tuple per figure or table (name, inputs exist?, function that builds
#   it) and a single loop runs them, instead of repeating the same if/print code for each one.
# lambda with a default argument (lambda f=f: ...): a small unnamed function; 'f=f' freezes the current loop value,
#   otherwise every lambda would see the last f. Used for the confusion-matrix rows.
# Markdown (.md) and LaTeX (.tex) tables: write_table saves each table in both formats; the .tex file uses booktabs
#   rules and is included in the report. In captions '\%' is a LaTeX-escaped percent sign and $\pm$ is a math
#   plus-minus.
# argparse and RawDescriptionHelpFormatter: argparse reads command-line options; the formatter keeps the docstring's
#   line breaks when --help prints it.
# SystemExit: raising it stops the script with a message and a non-zero exit code. Used when groups.csv is missing.
# format spec ':g' in CONF_GROUP: prints 1.0 as '1' and 100.0 as '100', so the name matches the file names written by
#   evaluate.py (for example resnet50_rgb_1pct_supervised).

# ======================================================================
# Q&A ABOUT THIS FILE (file-level viva questions)
# ======================================================================
# Q: What is this file for?
#    A: It is the last step (Phase 7) that turns the evaluation CSV files into the report figures (PNG) and tables (.md
#       and .tex). It computes no scores itself.
# Q: Where does it fit in the pipeline, and how is it run?
#    A: After scripts/evaluate.py (or eval_from_logs.py) has written results/summary/eval/. Run python
#       scripts/make_figures.py [--eval-dir ...] [--figures-dir ...] [--tables-dir ...] [--fraction 0.01].
# Q: What does it read and write?
#    A: It reads groups.csv (required) and per_class.csv, compare_*.csv, compare_*_pairs.csv and confusion/*.csv
#       (optional). It writes PNG files into results/figures and .md/.tex tables into results/tables.
# Q: What happens when some inputs are missing?
#    A: A figure or table whose input file is missing is left out or reported as 'skipped ... no data yet', so the
#       script works on partial results. Only a missing groups.csv stops it.

# Must match scripts/evaluate.py:group_name for a supervised group (file stem of confusion/<stem>.csv).
CONF_GROUP = "{model}_{modality}_{pct:g}pct_supervised"


# Q: What does this function do?
#    A: It reads a CSV file into a DataFrame, or returns None if the file does not exist.
# Q: Why return None instead of raising an error?
#    A: Most inputs are optional. The caller turns None into 'skip this output', so the script can run on a partial set
#       of results.
# Q: What are the input and output?
#    A: Input: a Path. Output: a DataFrame or None.
def read(path: Path) -> pd.DataFrame | None:
    """Read a CSV, or return None if the file does not exist (the callers then skip that output).

    Args:
        path: location of the CSV file.
    Returns:
        The table in the file, or None when the file is missing.
    """
    # Only read the file if it exists, so a missing optional input does not crash the script.
    return pd.read_csv(path) if path.exists() else None


# Q: What does main do?
#    A: It reads the options and the CSV files, finds the best model, builds the list of figures and tables to make,
#       runs each builder and finally prints the paths of the written files.
# Q: Why is groups.csv mandatory but the other files are not?
#    A: Every figure and table needs groups.csv (it also decides the best model), so without it there is nothing useful
#       to draw. The other files only feed individual outputs.
# Q: Why is the work organised as a list of rows instead of a long series of if statements?
#    A: Each row has the same form, one loop handles skipping, building and reporting, and adding a new figure or table
#       needs only one new row.
# Q: What is the 'best model' used for?
#    A: The confusion matrices, the per-class heatmap and the per-class table show one model only, chosen as the one
#       with the highest mean test macro-F1 at --fraction.
def main() -> None:
    """Build all figures and tables that the available evaluation CSVs allow, then list what was written."""
    # Build the command-line parser; the module docstring is shown as the help text, with its layout kept.
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Option: folder holding the evaluation CSV files (the inputs).
    p.add_argument("--eval-dir", type=Path, default=RESULTS_DIR / "summary" / "eval")
    # Option: folder where the PNG figures are written.
    p.add_argument("--figures-dir", type=Path, default=RESULTS_DIR / "figures")
    # Option: folder where the .md and .tex tables are written.
    p.add_argument("--tables-dir", type=Path, default=RESULTS_DIR / "tables")
    # Option: which label fraction the per-class figure, per-class table and confusion matrices show.
    p.add_argument("--fraction", type=float, default=1.0, help="label fraction for the per-class views")
    # Read what the user typed (or the defaults).
    args = p.parse_args()

    # Short names for the three folders used many times below.
    ev, fd, td = args.eval_dir, args.figures_dir, args.tables_dir
    # Load the two main input tables; per_class.csv is optional and becomes None if absent.
    groups, per_class = read(ev / "groups.csv"), read(ev / "per_class.csv")
    # Without groups.csv there is nothing to plot, so stop with a helpful message.
    if groups is None:
        raise SystemExit(f"{ev / 'groups.csv'} not found: run scripts/evaluate.py first")
    # Pick the single best model at the chosen fraction (used by the confusion matrices and per-class views).
    best = tab.best_model(groups, args.fraction)
    # Are there any supervised runs at all? The supervised tables are skipped if not.
    has_supervised = (groups["variant"] == "supervised").any()
    # Per-seed teacher-vs-student pairs for plain (pl) and class-balanced (plb) pseudo-labelling.
    pl_pairs = {v: read(ev / f"compare_supervised_vs_{v}_pairs.csv") for v in ("pl", "plb")}
    # Summarised (over seeds) teacher-vs-student tables, not the per-seed _pairs files used by the figure.
    compare = {v: read(ev / f"compare_supervised_vs_{v}.csv") for v in ("pl", "plb")}
    # rgb_vs_ms_gain wants a DataFrame, so a missing CSV becomes an empty one (-> the figure returns None).
    rgb_vs_ms = read(ev / "compare_rgb_vs_ms_pairs.csv")
    rgb_vs_ms = pd.DataFrame() if rgb_vs_ms is None else rgb_vs_ms

    # Q: What does this function do?
    #    A: For the best model at label fraction f it loads the RGB and the 13-band confusion matrix CSV (when they
    #       exist) and returns a dictionary from panel title to the 10 x 10 matrix.
    # Q: Why is cm.set_index(cm.columns[0]).to_numpy() needed?
    #    A: The first CSV column holds the class names. Making it the index leaves only the 10 x 10 numbers, which is
    #       what the figure function expects.
    # Q: Why is this a nested function?
    #    A: It uses variables of main (ev, best) and is called once per fraction, so defining it inside main is simpler
    #       than passing them in.
    def confusion_mats(f: float) -> dict:
        """Panels for the best model at fraction `f`: title -> [10, 10] matrix, for the inputs that have a CSV.

        Args:
            f: label fraction, for example 0.01 for one percent of the labels.
        Returns:
            A dictionary that maps a panel title (model and input type) to its 10 x 10 confusion matrix.
        """
        # Collected panels, one per input type that has a confusion-matrix file.
        mats = {}
        # Look for the RGB version and the 13-band version of the best model.
        for mo in ("rgb", "ms"):
            # Read the confusion matrix saved by evaluate.py for this model, input type and fraction.
            cm = read(ev / "confusion" / (CONF_GROUP.format(model=best, modality=mo, pct=f * 100) + ".csv"))
            # Skip input types for which no file exists.
            if cm is not None:
                # The CSV has the class names as first column (index); keep only the [10, 10] numbers.
                mats[f"{fig.MODEL_LABEL[best]}, {fig.MODALITY_LABEL[mo]}"] = cm.set_index(cm.columns[0]).to_numpy()
        return mats

    # Q: What does this function do?
    #    A: It writes one table as .md and .tex through tab.write_table, or returns None if the table could not be built
    #       because its data is missing.
    # Q: Why are caption, wide and font arguments needed?
    #    A: The caption goes into the LaTeX file; wide=True makes the table span both columns of the report page; font
    #       sets a smaller LaTeX size for tables with many columns.
    # Q: What does it return?
    #    A: The list of written file paths, or None, which main reports as 'skipped ... no data yet'.
    def table(df, name: str, caption: str, wide: bool = False, font: str = r"\small"):
        """Write one table (.md and .tex), or return None if its data is missing. `wide`: a two-column float.

        Args:
            df: the table to write, or None if its input was missing.
            name: file name (without extension) for the table.
            caption: caption text used in the LaTeX version.
            wide: True if the table should span both columns of the report page.
            font: LaTeX font-size command for the table body (a smaller size for tables with many columns).
        Returns:
            The list of written files, or None when df is None.
        """
        # No data means nothing to write (None, which the caller reports); otherwise write the .md and .tex files.
        return None if df is None else tab.write_table(df, td, name, caption, wide, font)

    # One row per output: (name used in the "skipped" message, whether its inputs exist, what builds it).
    # A row whose inputs do not exist is left out silently; a row that returns None is reported as skipped.
    # The rows run in this order, which is the order of the final list of files.
    outputs = [
        # Figure: macro-F1 against the amount of labels, one curve per model.
        ("label-efficiency curves", True,
         lambda: fig.label_efficiency(groups, fd / "label_efficiency_f1.png")),
        # The same figure for accuracy instead of macro-F1.
        ("label-efficiency (accuracy)", True,
         lambda: fig.label_efficiency(groups, fd / "label_efficiency_acc.png", metric="acc")),
        # Figure: how much the 13-band input gains over RGB.
        ("RGB vs 13-band gain", True,
         lambda: fig.rgb_vs_ms_gain(rgb_vs_ms, fd / "rgb_vs_ms_gain.png")),
        # Figure: how much pseudo-labelling changes the score relative to the teacher.
        ("pseudo-label gains", True,
         lambda: fig.pseudo_label_gain(pl_pairs, fd / "pseudo_label_gain.png")),
        # Figure: per-class F1 heatmap; only possible when per_class.csv exists.
        ("per-class F1 heatmap", per_class is not None,
         lambda: fig.per_class_heatmap(per_class, args.fraction, fd / f"per_class_f1_{args.fraction * 100:g}pct.png")),
    ]
    # Confusion matrices need a best model; without one this block is skipped.
    if best:
        # The chosen fraction plus 1 percent (the label-scarce case), limited to fractions that were run.
        for f in sorted({args.fraction, 0.01} & set(groups["fraction"])):
            # Add one confusion-matrix figure per fraction. "f=f" fixes the current value inside the lambda.
            outputs.append((f"confusion matrices at {f:.0%}", True,
                            lambda f=f: fig.confusion_grid(
                                confusion_mats(f), fd / f"confusion_{f * 100:g}pct.png",
                                f"Confusion matrices (rows sum to 100), {fig.FRACTION_LABEL.get(round(f, 4))} labels, "
                                "mean of 3 seeds")))
    # Now the tables; they are appended after the figures so the final list keeps this order.
    outputs += [
        # Table: supervised test macro-F1 per model, input type and label fraction.
        ("supervised F1", has_supervised,
         lambda: table(tab.supervised_table(groups), "supervised_f1",
                       "Test macro-F1 (\\%), mean $\\pm$ std over 3 seeds; (n=1) is a single seed.")),
        # The same table for accuracy.
        ("supervised accuracy", has_supervised,
         lambda: table(tab.supervised_table(groups, metric="acc"), "supervised_acc",
                       "Test accuracy (\\%), mean $\\pm$ std over 3 seeds; (n=1) is a single seed.")),
        # Table: pseudo-labelling results compared with the supervised teacher.
        ("semi-supervised table", True,
         lambda: table(tab.semi_supervised_table(groups, compare), "semi_supervised",
                       "Pseudo-labelling against the supervised teacher, matched on seed.", wide=True, font=r"\footnotesize")),
        # Table: per-class F1 of the best model; needs both a best model and per_class.csv.
        ("per-class table", bool(best) and per_class is not None,
         lambda: table(tab.per_class_table(per_class, best, args.fraction),
                       f"per_class_f1_{args.fraction * 100:g}pct",  # one table per fraction, so runs do not overwrite
                       f"Per-class test F1 (\\%) of {fig.MODEL_LABEL[best]} at "
                       # chr(92) is a backslash: a bare % would start a LaTeX comment and swallow the closing brace
                       f"{fig.FRACTION_LABEL.get(round(args.fraction, 4), f'{args.fraction:.0%}').replace('%', chr(92) + '%')}"
                       " of the labels, "
                       "mean $\\pm$ std over 3 seeds.")),
    ]

    # Paths of every file actually written.
    made: list[Path] = []
    # Run each output builder in order.
    for what, inputs_exist, build in outputs:
        # Leave out outputs whose input files do not exist.
        if not inputs_exist:
            continue
        # Build the figure or table; the result is a path, a list of paths, or None.
        result = build()
        # None means the builder found no data, so say so instead of failing.
        if result is None:
            print(f"skipped {what}: no data yet")
        else:
            # Normalise to a list so single paths and lists are handled the same way.
            made.extend(result if isinstance(result, list) else [result])

    # Finish with a summary: how many files were written and their paths.
    print(f"wrote {len(made)} files:")
    for m in made:
        print("  ", m)


# Run main() only when the file is executed as a script, not when it is imported.
if __name__ == "__main__":
    main()

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

# Must match scripts/evaluate.py:group_name for a supervised group (file stem of confusion/<stem>.csv).
CONF_GROUP = "{model}_{modality}_{pct:g}pct_supervised"


def read(path: Path) -> pd.DataFrame | None:
    """Read a CSV, or return None if the file does not exist (the callers then skip that output).

    Args:
        path: location of the CSV file.
    Returns:
        The table in the file, or None when the file is missing.
    """
    # Only read the file if it exists, so a missing optional input does not crash the script.
    return pd.read_csv(path) if path.exists() else None


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

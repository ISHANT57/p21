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

import argparse
from pathlib import Path

import pandas as pd

from eurosat_le import figures as fig
from eurosat_le import tables as tab
from eurosat_le.constants import RESULTS_DIR

# Must match scripts/evaluate.py:group_name for a supervised group (file stem of confusion/<stem>.csv).
CONF_GROUP = "{model}_{modality}_{pct:g}pct_supervised"


def read(path: Path) -> pd.DataFrame | None:
    """Read a CSV, or return None if the file does not exist (the callers then skip that output)."""
    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    """Build all figures and tables that the available evaluation CSVs allow, then list what was written."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", type=Path, default=RESULTS_DIR / "summary" / "eval")
    p.add_argument("--figures-dir", type=Path, default=RESULTS_DIR / "figures")
    p.add_argument("--tables-dir", type=Path, default=RESULTS_DIR / "tables")
    p.add_argument("--fraction", type=float, default=1.0, help="label fraction for the per-class views")
    args = p.parse_args()

    ev, fd, td = args.eval_dir, args.figures_dir, args.tables_dir
    groups, per_class = read(ev / "groups.csv"), read(ev / "per_class.csv")
    if groups is None:
        raise SystemExit(f"{ev / 'groups.csv'} not found: run scripts/evaluate.py first")
    best = tab.best_model(groups, args.fraction)
    has_supervised = (groups["variant"] == "supervised").any()
    pl_pairs = {v: read(ev / f"compare_supervised_vs_{v}_pairs.csv") for v in ("pl", "plb")}
    # Summarised (over seeds) teacher-vs-student tables, not the per-seed _pairs files used by the figure.
    compare = {v: read(ev / f"compare_supervised_vs_{v}.csv") for v in ("pl", "plb")}
    # rgb_vs_ms_gain wants a DataFrame, so a missing CSV becomes an empty one (-> the figure returns None).
    rgb_vs_ms = read(ev / "compare_rgb_vs_ms_pairs.csv")
    rgb_vs_ms = pd.DataFrame() if rgb_vs_ms is None else rgb_vs_ms

    def confusion_mats(f: float) -> dict:
        """Panels for the best model at fraction `f`: title -> [10, 10] matrix, for the inputs that have a CSV."""
        mats = {}
        for mo in ("rgb", "ms"):
            cm = read(ev / "confusion" / (CONF_GROUP.format(model=best, modality=mo, pct=f * 100) + ".csv"))
            if cm is not None:
                # The CSV has the class names as first column (index); keep only the [10, 10] numbers.
                mats[f"{fig.MODEL_LABEL[best]}, {fig.MODALITY_LABEL[mo]}"] = cm.set_index(cm.columns[0]).to_numpy()
        return mats

    def table(df, name: str, caption: str, wide: bool = False):
        """Write one table (.md and .tex), or return None if its data is missing. `wide`: a two-column float."""
        return None if df is None else tab.write_table(df, td, name, caption, wide)

    # One row per output: (name used in the "skipped" message, whether its inputs exist, what builds it).
    # A row whose inputs do not exist is left out silently; a row that returns None is reported as skipped.
    # The rows run in this order, which is the order of the final list of files.
    outputs = [
        ("label-efficiency curves", True,
         lambda: fig.label_efficiency(groups, fd / "label_efficiency_f1.png")),
        ("label-efficiency (accuracy)", True,
         lambda: fig.label_efficiency(groups, fd / "label_efficiency_acc.png", metric="acc")),
        ("RGB vs 13-band gain", True,
         lambda: fig.rgb_vs_ms_gain(rgb_vs_ms, fd / "rgb_vs_ms_gain.png")),
        ("pseudo-label gains", True,
         lambda: fig.pseudo_label_gain(pl_pairs, fd / "pseudo_label_gain.png")),
        ("per-class F1 heatmap", per_class is not None,
         lambda: fig.per_class_heatmap(per_class, args.fraction, fd / f"per_class_f1_{args.fraction * 100:g}pct.png")),
    ]
    if best:
        # The chosen fraction plus 1 percent (the label-scarce case), limited to fractions that were run.
        for f in sorted({args.fraction, 0.01} & set(groups["fraction"])):
            outputs.append((f"confusion matrices at {f:.0%}", True,
                            lambda f=f: fig.confusion_grid(
                                confusion_mats(f), fd / f"confusion_{f * 100:g}pct.png",
                                f"Confusion matrices (rows sum to 100), {fig.FRACTION_LABEL.get(round(f, 4))} labels, "
                                "mean of 3 seeds")))
    outputs += [
        ("supervised F1", has_supervised,
         lambda: table(tab.supervised_table(groups), "supervised_f1",
                       "Test macro-F1 (\\%), mean $\\pm$ std over 3 seeds; (n=1) is a single seed.")),
        ("supervised accuracy", has_supervised,
         lambda: table(tab.supervised_table(groups, metric="acc"), "supervised_acc",
                       "Test accuracy (\\%), mean $\\pm$ std over 3 seeds; (n=1) is a single seed.")),
        ("semi-supervised table", True,
         lambda: table(tab.semi_supervised_table(groups, compare), "semi_supervised",
                       "Pseudo-labelling against the supervised teacher, matched on seed.", wide=True)),
        ("per-class table", bool(best) and per_class is not None,
         lambda: table(tab.per_class_table(per_class, best, args.fraction),
                       f"per_class_f1_{args.fraction * 100:g}pct",  # one table per fraction, so runs do not overwrite
                       f"Per-class test F1 (\\%) of {fig.MODEL_LABEL[best]} at "
                       f"{fig.FRACTION_LABEL.get(round(args.fraction, 4), f'{args.fraction:.0%}')} of the labels, "
                       "mean $\\pm$ std over 3 seeds.")),
    ]

    made: list[Path] = []
    for what, inputs_exist, build in outputs:
        if not inputs_exist:
            continue
        result = build()
        if result is None:
            print(f"skipped {what}: no data yet")
        else:
            made.extend(result if isinstance(result, list) else [result])

    print(f"wrote {len(made)} files:")
    for m in made:
        print("  ", m)


if __name__ == "__main__":
    main()

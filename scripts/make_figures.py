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
    tables/supervised_f1, supervised_acc, semi_supervised, per_class_f1  (each as .md and .tex)

Design decisions:
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
    made: list[Path] = []

    def keep(result, what: str) -> None:
        """Collect the written path(s) of one output; report a skipped output (None result) by name."""
        if result is None:
            print(f"skipped {what}: no data yet")
        else:
            made.extend(result if isinstance(result, list) else [result])

    # ---- figures ---------------------------------------------------------------------------------
    keep(fig.label_efficiency(groups, fd / "label_efficiency_f1.png"), "label-efficiency curves")
    keep(fig.label_efficiency(groups, fd / "label_efficiency_acc.png", metric="acc"), "label-efficiency (accuracy)")
    # rgb_vs_ms_gain wants a DataFrame, so a missing CSV becomes an empty one (-> the figure returns None).
    keep(fig.rgb_vs_ms_gain(read(ev / "compare_rgb_vs_ms_pairs.csv") if (ev / "compare_rgb_vs_ms_pairs.csv").exists()
                            else pd.DataFrame(), fd / "rgb_vs_ms_gain.png"), "RGB vs 13-band gain")
    pl_pairs = {v: read(ev / f"compare_supervised_vs_{v}_pairs.csv") for v in ("pl", "plb")}
    keep(fig.pseudo_label_gain(pl_pairs, fd / "pseudo_label_gain.png"), "pseudo-label gains")
    if per_class is not None:
        keep(fig.per_class_heatmap(per_class, args.fraction, fd / f"per_class_f1_{args.fraction * 100:g}pct.png"),
             "per-class F1 heatmap")

    best = tab.best_model(groups, args.fraction)
    if best:
        # The chosen fraction plus 1 percent (the label-scarce case), limited to fractions that were run.
        for f in sorted({args.fraction, 0.01} & set(groups["fraction"])):
            mats = {}
            for mo in ("rgb", "ms"):
                cm = read(ev / "confusion" / (CONF_GROUP.format(model=best, modality=mo, pct=f * 100) + ".csv"))
                if cm is not None:
                    # The CSV has the class names as first column (index); keep only the [10, 10] numbers.
                    mats[f"{fig.MODEL_LABEL[best]}, {fig.MODALITY_LABEL[mo]}"] = cm.set_index(cm.columns[0]).to_numpy()
            keep(fig.confusion_grid(mats, fd / f"confusion_{f * 100:g}pct.png",
                                    f"Confusion matrices (rows sum to 100), {fig.FRACTION_LABEL.get(round(f, 4))} labels, "
                                    "mean of 3 seeds"), f"confusion matrices at {f:.0%}")

    # ---- tables ----------------------------------------------------------------------------------
    sup = tab.supervised_table(groups)
    if sup is not None:
        keep(tab.write_table(sup, td, "supervised_f1", "Test macro-F1 (\\%), mean $\\pm$ std over 3 seeds."), "supervised F1")
    sup_acc = tab.supervised_table(groups, metric="acc")
    if sup_acc is not None:
        keep(tab.write_table(sup_acc, td, "supervised_acc", "Test accuracy (\\%), mean $\\pm$ std over 3 seeds."),
             "supervised accuracy")
    # Summarised (over seeds) teacher-vs-student tables, not the per-seed _pairs files used by the figure.
    compare = {v: read(ev / f"compare_supervised_vs_{v}.csv") for v in ("pl", "plb")}
    semi = tab.semi_supervised_table(groups, compare)
    keep(tab.write_table(semi, td, "semi_supervised", "Pseudo-labelling against the supervised teacher, matched on seed.")
         if semi is not None else None, "semi-supervised table")
    if best and per_class is not None:
        pc = tab.per_class_table(per_class, best, args.fraction)
        keep(tab.write_table(pc, td, "per_class_f1", f"Per-class test F1 (\\%) for {fig.MODEL_LABEL[best]}.")
             if pc is not None else None, "per-class table")

    print(f"wrote {len(made)} files:")
    for m in made:
        print("  ", m)


if __name__ == "__main__":
    main()

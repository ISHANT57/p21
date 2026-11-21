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

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eurosat_le.eval.evaluate import summarise_comparison

GROUP = ["model", "modality", "fraction", "variant"]
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
    """Mean and sample std over seeds per setting (std is NaN for one seed), as evaluate.py writes it."""
    g = runs.groupby(GROUP).agg(
        n_seeds=("seed", "nunique"),
        test_acc_mean=("test_acc", "mean"), test_acc_std=("test_acc", "std"),
        test_f1_mean=("test_f1", "mean"), test_f1_std=("test_f1", "std"),
        val_f1_mean=("val_f1", "mean"),
    ).reset_index()
    g.insert(g.columns.get_loc("val_f1_mean"), "test_ece_mean", np.nan)  # same column order as evaluate.py
    return g


def pairs_table(runs: pd.DataFrame, factor: str, a: str, b: str) -> pd.DataFrame:
    """Per-seed pairs where only `factor` differs (a -> b); delta_f1 = f1_b - f1_a; no bootstrap interval."""
    keys = [k for k in GROUP if k != factor]
    A = runs[runs[factor] == a].set_index(keys + ["seed"])
    B = runs[runs[factor] == b].set_index(keys + ["seed"])
    rows = []
    for idx in A.index.intersection(B.index):
        ra, rb = A.loc[idx], B.loc[idx]
        rows.append({**dict(zip(keys + ["seed"], idx)), "a": a, "b": b,
                     "acc_a": ra["test_acc"], "acc_b": rb["test_acc"], "f1_a": ra["test_f1"], "f1_b": rb["test_f1"],
                     "delta_f1": rb["test_f1"] - ra["test_f1"], "ci_low": np.nan, "ci_high": np.nan})
    return pd.DataFrame(rows)


def cross_check(runs: pd.DataFrame, runs_dir: Path) -> None:
    """Compare logged numbers with metrics.json for every run that has a folder; print the outcome."""
    checked, bad = 0, []
    for _, r in runs.iterrows():
        m = runs_dir / r["run_name"] / "metrics.json"
        if not m.exists():
            continue
        t = json.loads(m.read_text())["test"]
        checked += 1
        for key, logged in (("accuracy", r["test_acc"]), ("macro_f1", r["test_f1"])):
            if abs(t[key] - logged) > 6e-5:
                bad.append(f"{r['run_name']} {key}: log {logged} vs metrics.json {t[key]:.5f}")
    print(f"cross-check: {checked} runs with a folder compared with metrics.json; "
          f"{'all agree to the 4 printed decimals' if not bad else str(len(bad)) + ' differ'}")
    for line in bad:
        print("  " + line)


def main() -> None:
    """Write groups.csv and the compare_* files for every comparison that has at least one pair."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--logs", type=Path, default=Path("results/logs/runs_from_logs.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/summary/eval"))
    ap.add_argument("--runs-dir", type=Path, default=None, help="optional: cross-check against metrics.json here")
    args = ap.parse_args()

    runs = pd.read_csv(args.logs)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.runs_dir is not None:
        cross_check(runs, args.runs_dir)
    groups_table(runs).to_csv(args.out / "groups.csv", index=False)
    print(f"groups.csv: {len(runs)} runs in {len(groups_table(runs))} settings")
    for stem, factor, a, b in COMPARISONS:
        pairs = pairs_table(runs, factor, a, b)
        if pairs.empty:
            continue
        pairs.to_csv(args.out / f"compare_{stem}_pairs.csv", index=False)
        summ = summarise_comparison(pairs)
        summ["seeds_ci_excludes_0"] = np.nan  # no intervals without predictions
        summ.to_csv(args.out / f"compare_{stem}.csv", index=False)
        print(f"compare_{stem}: {len(pairs)} pairs")


if __name__ == "__main__":
    main()

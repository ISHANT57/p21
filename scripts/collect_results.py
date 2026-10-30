"""Gather every finished run into tables.

Usage:
    python scripts/collect_results.py                 # reads results/runs/, writes results/summary/
    python scripts/collect_results.py --runs-dir /content/drive/MyDrive/p21_runs

Writes:
    results/summary/runs.csv         one row per finished run (main grid, tuning, extras)
    results/summary/supervised.csv   main grid: mean / std / n over seeds per model x input x fraction
    results/summary/tuning.csv       LR sweep, validation metrics only
    results/summary/semi_supervised.csv  pseudo-label students vs their supervised teachers (paired by seed)
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from eurosat_le.constants import RESULTS_DIR
from eurosat_le.train.engine import RUNS_DIR

OUT = RESULTS_DIR / "summary"


def load_runs(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for mpath in sorted(runs_dir.glob("*/metrics.json")):
        m = json.loads(mpath.read_text())
        cfg = yaml.safe_load((mpath.parent / "config.yaml").read_text())
        env = json.loads((mpath.parent / "env.json").read_text())
        row = {k: cfg.get(k) for k in ("model", "modality", "fraction", "seed", "name_suffix", "lr", "init_mode",
                                       "teacher")}
        row["name_suffix"] = row["name_suffix"] or ""
        row.update(run_name=m["run_name"], best_epoch=m["best_epoch"], epochs_run=m["epochs_run"],
                   train_min=m["train_seconds"] / 60, n_labelled=m["n_labelled"], device=env["device"],
                   git_commit=env["git_commit"][:7], val_acc=m["val"]["accuracy"], val_f1=m["val"]["macro_f1"])
        if "test" in m:
            row.update(test_acc=m["test"]["accuracy"], test_f1=m["test"]["macro_f1"])
        row["n_pseudo"] = m.get("n_pseudo", 0)
        qpath = mpath.parent / "pseudo_label_quality.json"
        if qpath.exists():
            row["pseudo_acc"] = json.loads(qpath.read_text())["accuracy"]
            sel = json.loads((mpath.parent / "pseudo_label_selection.json").read_text())
            row.update(tau=sel["tau"], pseudo_coverage=sel["n_selected"] / sel["pool_size"])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = p.parse_args()

    df = load_runs(args.runs_dir)
    if df.empty:
        raise SystemExit(f"no finished runs in {args.runs_dir}")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "runs.csv", index=False)
    print(f"{len(df)} finished runs -> {OUT / 'runs.csv'}")

    tune = df[df["name_suffix"].str.startswith("tune")]
    if not tune.empty:
        tune = tune.sort_values(["model", "lr"])[["model", "lr", "val_acc", "val_f1", "best_epoch", "train_min"]]
        tune.to_csv(OUT / "tuning.csv", index=False)
        print("\nLR sweep (RGB, 10% labels, seed 0; validation only):")
        print(tune.to_string(index=False, float_format="%.4f"))
        best = tune.loc[tune.groupby("model")["val_f1"].idxmax()]
        print("\nbest lr by val macro-F1:", dict(zip(best["model"], best["lr"])))

    main_runs = df[df["name_suffix"] == ""]
    if not main_runs.empty:
        keys = ["model", "modality", "fraction"]
        agg = main_runs.groupby(keys).agg(
            n_seeds=("seed", "nunique"),
            test_acc_mean=("test_acc", "mean"), test_acc_std=("test_acc", "std"),
            test_f1_mean=("test_f1", "mean"), test_f1_std=("test_f1", "std"),
            val_f1_mean=("val_f1", "mean"), train_min_mean=("train_min", "mean"),
        ).reset_index()
        agg.to_csv(OUT / "supervised.csv", index=False)
        cell = agg.assign(v=lambda d: [f"{m:.4f} ± {s:.4f} (n={n})" if n > 1 else f"{m:.4f} (n=1)"
                                       for m, s, n in zip(d.test_f1_mean, d.test_f1_std.fillna(0), d.n_seeds)])
        print("\nTest macro-F1, mean ± std over seeds:")
        print(cell.pivot_table(index=["model", "modality"], columns="fraction", values="v", aggfunc="first")
              .to_string())

    students = df[df["name_suffix"].str.match(r"^plb?\d*$")]
    if not students.empty:
        teach = df.set_index("run_name")
        st = students[students["teacher"].isin(teach.index)].copy()
        st["teacher_test_f1"] = teach.loc[st["teacher"], "test_f1"].to_numpy()
        st["teacher_test_acc"] = teach.loc[st["teacher"], "test_acc"].to_numpy()
        st["delta_f1"] = st["test_f1"] - st["teacher_test_f1"]
        st["delta_acc"] = st["test_acc"] - st["teacher_test_acc"]
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


if __name__ == "__main__":
    main()

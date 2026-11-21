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

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eurosat_le import tables as tab
from eurosat_le.figures import FRACTION_LABEL, MODALITY_LABEL, MODEL_LABEL, MODELS

VARIANT = {"supervised": "supervised", "pl": "pseudo-label", "plb": "class-balanced"}


def _frac(f: float) -> str:
    """Label fraction as "1%", "10%", "100%"."""
    return FRACTION_LABEL.get(round(f, 4), f"{f:.0%}")


def _pm(mean: float, std: float, scale: float = 100.0, digits: int = 1, sign: bool = False) -> str:
    """"mean ± std" in percent points, or "mean (n=1)" when there is no std (one seed)."""
    fmt = f"{{:+.{digits}f}}" if sign else f"{{:.{digits}f}}"
    if pd.isna(std):
        return fmt.format(mean * scale) + " (n=1)"
    return fmt.format(mean * scale) + f" ± {std * scale:.{digits}f}"


def all_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per finished run, ordered by variant, model, input, fraction, seed."""
    r = runs.assign(_v=runs["variant"].map({"supervised": 0, "pl": 1, "plb": 2}),
                    _m=runs["model"].map({m: i for i, m in enumerate(MODELS)}),
                    _o=runs["modality"].map({"rgb": 0, "ms": 1})).sort_values(["_v", "_m", "_o", "fraction", "seed"])
    return pd.DataFrame({
        "Model": r["model"].map(MODEL_LABEL), "Input": r["modality"].map(MODALITY_LABEL),
        "Labels": r["fraction"].map(_frac), "Seed": r["seed"].astype(int), "Variant": r["variant"].map(VARIANT),
        "Test acc.": (r["test_acc"] * 100).map("{:.2f}".format), "Test F1": (r["test_f1"] * 100).map("{:.2f}".format),
        "Val. F1": (r["val_f1"] * 100).map("{:.2f}".format), "Best epoch": r["best_epoch"].astype(int),
        "Minutes": r["minutes"].map("{:.1f}".format)}).reset_index(drop=True)


def pl_details(runs: pd.DataFrame) -> pd.DataFrame | None:
    """Pseudo-label students with their threshold, pool share and the change against their teacher."""
    s = runs[runs["variant"] != "supervised"].copy()
    if s.empty:
        return None
    teacher_f1 = runs.set_index("run_name")["test_f1"]
    s["teacher_f1"] = s["teacher"].map(teacher_f1)
    s = s.assign(_m=s["model"].map({m: i for i, m in enumerate(MODELS)}),
                 _o=s["modality"].map({"rgb": 0, "ms": 1})).sort_values(["_m", "_o", "fraction", "seed"])
    return pd.DataFrame({
        "Model": s["model"].map(MODEL_LABEL), "Input": s["modality"].map(MODALITY_LABEL),
        "Labels": s["fraction"].map(_frac), "Seed": s["seed"].astype(int),
        "$\\tau$": s["tau"].map("{:.2f}".format), "Val. precision": s["val_precision"].map("{:.3f}".format),
        "Pseudo-labelled": [f"{int(a):,} / {int(b):,}" for a, b in zip(s["n_pseudo"], s["n_pool"])],
        "Share (%)": (s["n_pseudo"] / s["n_pool"] * 100).map("{:.1f}".format),
        "Teacher F1": (s["teacher_f1"] * 100).map("{:.1f}".format), "Student F1": (s["test_f1"] * 100).map("{:.1f}".format),
        "$\\Delta$": ((s["test_f1"] - s["teacher_f1"]) * 100).map("{:+.1f}".format)}).reset_index(drop=True)


def training_cost(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean time and best epoch per model and label fraction, for supervised runs."""
    s = runs[runs["variant"] == "supervised"]
    g = s.groupby(["model", "fraction"]).agg(n=("run_name", "count"), minutes=("minutes", "mean"),
                                           epoch=("best_epoch", "mean")).reset_index()
    g = g.assign(_m=g["model"].map({m: i for i, m in enumerate(MODELS)})).sort_values(["_m", "fraction"])
    return pd.DataFrame({"Model": g["model"].map(MODEL_LABEL), "Labels": g["fraction"].map(_frac), "Runs": g["n"].astype(int),
                         "Mean time (min)": g["minutes"].map("{:.1f}".format),
                         "Mean best epoch": g["epoch"].map("{:.1f}".format)}).reset_index(drop=True)


def lr_sweep(tune: pd.DataFrame) -> pd.DataFrame:
    """Learning-rate sweep rows; the rate that was used for all later runs is marked."""
    used = {("scratch_cnn", 0.002), ("resnet50", 0.0005)}
    t = tune.assign(_m=tune["model"].map({m: i for i, m in enumerate(MODELS)})).sort_values(["_m", "lr"])
    return pd.DataFrame({
        "Model": t["model"].map(MODEL_LABEL), "Learning rate": t["lr"].map(lambda v: f"{v:g}"),
        "Val. acc.": (t["val_acc"] * 100).map("{:.2f}".format), "Val. F1": (t["val_f1"] * 100).map("{:.2f}".format),
        "Best epoch": t["best_epoch"].astype(int), "Minutes": t["minutes"].map("{:.1f}".format),
        "Used": ["yes" if (m, lr) in used else "" for m, lr in zip(t["model"], t["lr"])]}).reset_index(drop=True)


def model_comparison(eval_dir: Path) -> pd.DataFrame | None:
    """Supervised model-against-model differences (b minus a, in macro-F1 points), matched on seed."""
    rows = []
    for stem, label in (("scratch_vs_resnet50", "ResNet-50 $-$ scratch CNN"),
                        ("scratch_vs_vit_small", "ViT-Small $-$ scratch CNN"),
                        ("resnet50_vs_vit_small", "ViT-Small $-$ ResNet-50")):
        f = eval_dir / f"compare_{stem}.csv"
        if not f.exists():
            continue
        c = pd.read_csv(f)
        c = c[c["variant"] == "supervised"].sort_values(["modality", "fraction"], key=lambda col: col.map(
            {"rgb": 0, "ms": 1}) if col.name == "modality" else col)
        for _, r in c.iterrows():
            rows.append({"Comparison": label, "Input": MODALITY_LABEL[r["modality"]], "Labels": _frac(r["fraction"]),
                         "$\\Delta$ F1 (points)": _pm(r["delta_f1_mean"], r["delta_f1_std"], sign=True),
                         "Seeds better": f"{int(r['seeds_b_better'])}/{int(r['n_seeds'])}"})
    return pd.DataFrame(rows) if rows else None


def main() -> None:
    """Write the five tables."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--logs", type=Path, default=Path("results/logs"))
    ap.add_argument("--eval", type=Path, default=Path("results/summary/eval"))
    ap.add_argument("--out", type=Path, default=Path("results/tables"))
    args = ap.parse_args()
    runs = pd.read_csv(args.logs / "runs_from_logs.csv")
    made = [
        ("all_runs", all_runs(runs), "All 51 finished runs. Scores are test accuracy and macro-F1 (\\%) as printed at the end of each run; "
         "Val. F1 is the validation macro-F1 of the selected weights.", True, r"\scriptsize"),
        ("pl_details", pl_details(runs), "Pseudo-label students: the threshold $\\tau$ chosen on the validation set, the validation "
         "precision at $\\tau$, the number of pool images that received a pseudo-label, and the test macro-F1 change against "
         "the teacher of the same seed.", True, r"\small"),
        ("training_cost", training_cost(runs), "Training cost of the supervised runs on a free-tier Tesla T4.", False, r"\small"),
    ]
    tune_csv = args.logs / "tuning_from_logs.csv"
    if tune_csv.exists():
        made.append(("lr_sweep", lr_sweep(pd.read_csv(tune_csv)), "Learning-rate sweep (RGB, 10\\% labels, seed 0). "
                     "Validation scores only; the test set was not evaluated.", False, r"\small"))
    made.append(("model_comparison", model_comparison(args.eval), "Differences in test macro-F1 between models, supervised runs, "
                 "matched on seed (second model minus first).", True, r"\small"))
    for name, df, caption, wide, font in made:
        if df is None or df.empty:
            print(f"skipped {name}: no data")
            continue
        tab.write_table(df, args.out, name, caption, wide=wide, font=font)
        print(f"wrote {name} ({len(df)} rows)")


if __name__ == "__main__":
    main()

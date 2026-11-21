"""Rebuild every evaluation table from saved predictions (Phase 6).

Usage:
    python scripts/evaluate.py                                  # results/runs -> results/summary/eval
    python scripts/evaluate.py --runs-dir /content/drive/MyDrive/p21_runs
    python scripts/evaluate.py --examples resnet50_rgb_1pct_seed0 vit_small_ms_1pct_seed0

Writes to results/summary/eval/:
    per_run.csv            val/test accuracy, macro-F1, ECE for every run (recomputed)
    groups.csv             mean / std over seeds per model x input x fraction x variant
    per_class.csv          per-class precision / recall / F1, mean and std over seeds
    confusion/*.csv        seed-averaged row-normalised confusion matrix per group
    top_confusions.csv     the most frequent (true -> predicted) mistakes per group
    compare_*.csv          paired comparisons: per seed (pairs) and summarised over seeds
    errors_<run>.png       misclassified test images (with --examples)

Where it fits: Phase 6 (see docs/EVALUATION.md). It reads the saved predictions of every finished run
(preds_val.npz and preds_test.npz in each run folder, plus config.yaml and metrics.json) and never trains
or re-runs a model. scripts/make_figures.py (Phase 7) then turns these tables into figures and tables.

Design decisions:
- Metrics are RECOMPUTED from the saved logits (softmax, then argmax), not copied from metrics.json. As a
  cross-check, accuracy and macro-F1 are compared with the logged values and a WARNING is printed if they
  differ by more than 0.001 (the run is still used; the script does not abort).
- ECE uses 15 confidence bins.
- Every number over seeds is a mean and a sample std (ddof=1) over the labelled-subset seeds (3 in the
  main grid). With a single seed the std is empty (NaN) in the CSV.
- Paired comparisons only match runs that agree on every other factor AND the seed, so both sides use the
  same labelled images, the same initialisation seed and the same test images. The test predictions of
  every run are aligned by sorted sample id. The interval is a 95 percent paired bootstrap over test images
  (--n-boot resamples, default 1000). Results over seeds are reported as counts of seeds, not p-values.
- A comparison with no matching pairs (for example the runs do not exist yet) is silently skipped.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from eurosat_le.constants import CLASSES, RESULTS_DIR
from eurosat_le.eval.evaluate import (check_against_logged, compare, load_preds, load_runs, mean_confusion,
                                      run_metrics, summarise_comparison, top_confusions)
from eurosat_le.train.engine import RUNS_DIR

# Keys that define one "setting"; the 3 seeds of a setting are aggregated together. "variant" is "supervised"
# for plain runs and the config name_suffix (e.g. "pl", "plb") for pseudo-label students.
GROUP = ["model", "modality", "fraction", "variant"]
COMPARISONS = [  # (file stem, factor, a, b): b is compared against a
    ("rgb_vs_ms", "modality", "rgb", "ms"),
    ("supervised_vs_pl", "variant", "supervised", "pl"),
    ("supervised_vs_plb", "variant", "supervised", "plb"),
    ("pl_vs_plb", "variant", "pl", "plb"),
    ("scratch_vs_resnet50", "model", "scratch_cnn", "resnet50"),
    ("scratch_vs_vit_small", "model", "scratch_cnn", "vit_small"),
    ("resnet50_vs_vit_small", "model", "resnet50", "vit_small"),
]


def group_name(row) -> str:
    """File stem for a group's confusion CSV, e.g. "resnet50_rgb_1pct_supervised".

    `row` needs the GROUP keys. The fraction is written as a percentage with `:g`, so 0.01 -> "1pct" and
    1.0 -> "100pct". scripts/make_figures.py rebuilds these names (CONF_GROUP), so keep both in sync.
    """
    return f"{row['model']}_{row['modality']}_{row['fraction'] * 100:g}pct_{row['variant']}"


def count_pairs(runs: pd.DataFrame, factor: str, a, b) -> int:
    """Number of run pairs `compare` will match for one comparison, without loading any predictions.

    Uses the same rule as `eurosat_le.eval.evaluate.compare`: every GROUP key except `factor`, plus the
    seed, must agree.
    """
    keys = [k for k in GROUP if k != factor] + ["seed"]
    A = runs[runs[factor] == a].set_index(keys).index
    B = runs[runs[factor] == b].set_index(keys).index
    return len(A.intersection(B))


def main() -> None:
    """Run the whole Phase 6 evaluation: per-run metrics, group tables, confusions, paired comparisons."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "summary" / "eval")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--examples", nargs="*", default=[], metavar="RUN", help="plot misclassified test images")
    args = p.parse_args()
    t_start = time.perf_counter()  # wall time of the whole evaluation, reported with the final "wrote" line

    runs = load_runs(args.runs_dir)
    if runs.empty:
        raise SystemExit(f"no finished runs with test predictions in {args.runs_dir}")
    out = args.out
    (out / "confusion").mkdir(parents=True, exist_ok=True)
    n_pairs = {stem: count_pairs(runs, factor, a, b) for stem, factor, a, b in COMPARISONS}
    # Announce the amount of work before the slow part, so a long run does not look frozen.
    print(f"{len(runs)} runs found; {sum(n_pairs.values())} comparison pairs in "
          f"{sum(n > 0 for n in n_pairs.values())} comparison files, {args.n_boot} bootstrap resamples each")

    # ---- per run ------------------------------------------------------------------------------
    rows, class_rows, issues = [], [], []
    for _, r in runs.iterrows():
        met = run_metrics(r["dir"])
        issues += check_against_logged(r["dir"], met)
        rows.append({**{k: r[k] for k in ["run_name", *GROUP, "seed"]},
                     **{f"{s}_{k}": met[s][k] for s in ("val", "test") for k in ("accuracy", "macro_f1", "ece")}})
        for c in CLASSES:
            class_rows.append({**{k: r[k] for k in [*GROUP, "seed"]}, "class": c, **met["test"]["per_class"][c]})
    per_run = pd.DataFrame(rows)
    per_run.to_csv(out / "per_run.csv", index=False)
    print(f"{len(per_run)} runs evaluated")
    if issues:
        print("WARNING: recomputed metrics differ from metrics.json:\n  " + "\n  ".join(issues))

    # One row per setting; std is NaN when n_seeds == 1. Only the test F1/accuracy/ECE are aggregated here,
    # plus the validation F1 mean for reference.
    groups = per_run.groupby(GROUP).agg(
        n_seeds=("seed", "nunique"),
        test_acc_mean=("test_accuracy", "mean"), test_acc_std=("test_accuracy", "std"),
        test_f1_mean=("test_macro_f1", "mean"), test_f1_std=("test_macro_f1", "std"),
        test_ece_mean=("test_ece", "mean"), val_f1_mean=("val_macro_f1", "mean"),
    ).reset_index()
    groups.to_csv(out / "groups.csv", index=False)

    # Per-class table: test split only; precision/recall are averaged over seeds, F1 also gets a std.
    per_class = pd.DataFrame(class_rows).groupby([*GROUP, "class"]).agg(
        precision=("precision", "mean"), recall=("recall", "mean"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"), n_seeds=("seed", "nunique"),
    ).reset_index()
    per_class.to_csv(out / "per_class.csv", index=False)

    # ---- confusion matrices ---------------------------------------------------------------------
    conf_rows = []
    for key, g in runs.groupby(GROUP):  # key is a tuple in GROUP order (GROUP is a list of 4 columns)
        row = dict(zip(GROUP, key))
        cm = mean_confusion(list(g["dir"]))
        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out / "confusion" / f"{group_name(row)}.csv")
        for true, pred, rate in top_confusions(cm):
            conf_rows.append({**row, "true": true, "predicted": pred, "rate": rate})
    pd.DataFrame(conf_rows).to_csv(out / "top_confusions.csv", index=False)

    # ---- paired comparisons ---------------------------------------------------------------------
    for stem, factor, a, b in COMPARISONS:
        t0 = time.perf_counter()  # per-file timing: the bootstrap inside compare() is the slow step
        pairs = compare(runs, factor, a, b, args.n_boot)
        if pairs.empty:  # nothing to pair yet (one side has no runs)
            continue
        pairs.to_csv(out / f"compare_{stem}_pairs.csv", index=False)
        summ = summarise_comparison(pairs)
        summ.to_csv(out / f"compare_{stem}.csv", index=False)
        print(f"\ncompare_{stem}: {len(pairs)} pairs, {time.perf_counter() - t0:.1f} s")
        print(f"{stem}: macro-F1({b}) - macro-F1({a}), paired by seed")
        cols = [c for c in GROUP if c != factor and c in summ.columns]  # the compared factor is a/b, not a key
        print(summ[cols + ["n_seeds", "delta_f1_mean", "delta_f1_std", "seeds_b_better", "seeds_ci_excludes_0"]]
              .to_string(index=False, float_format="%.4f"))

    for name in args.examples:
        plot_errors(args.runs_dir / name, out / f"errors_{name}.png")
    print(f"\nwrote {out} ({time.perf_counter() - t_start:.1f} s total)")


def plot_errors(run_dir: Path, path: Path, n: int = 24, seed: int = 0) -> None:
    """Grid of misclassified test images (shown in RGB, also for 13-band runs).

    Args:
        run_dir: a finished run folder with preds_test.npz.
        path: output PNG.
        n: maximum number of images (6 per row); fewer if the run makes fewer errors.
        seed: seed of the random choice among the errors, so the same grid is drawn every time.

    Each title is "true -> predicted (confidence)", the confidence being the top softmax probability.
    Images come from the RGB cache for every run: the cache rows follow splits/split.csv, so an image is
    found through its position in that table.
    """
    import matplotlib.pyplot as plt

    from eurosat_le import viz
    from eurosat_le.data.cache import load_cache
    from eurosat_le.data.splits import load_split
    from eurosat_le.ssl.pseudo_label import softmax

    viz.apply_style()
    logits, y, ids = load_preds(run_dir)
    probs = softmax(logits)
    wrong = np.flatnonzero(probs.argmax(1) != y)  # indices into the id-sorted test arrays
    pick = np.random.default_rng(seed).choice(wrong, min(n, len(wrong)), replace=False)
    rgb = load_cache("rgb")
    row_of = {sid: i for i, sid in enumerate(load_split()["id"])}  # sample id -> row in split.csv == row in the cache
    cols = 6
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.3), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, pick):
        img = rgb[row_of[ids[i]]].transpose(1, 2, 0)  # cache is (C, H, W); imshow wants (H, W, C)
        ax.imshow(img)
        ax.set_title(f"{CLASSES[y[i]]}\n→ {CLASSES[probs[i].argmax()]}\n({probs[i].max():.2f})", fontsize=6.5,
                     fontweight="normal", color=viz.TEXT_PRIMARY)
    fig.suptitle(f"{run_dir.name}: misclassified test images (true → predicted, confidence)", fontsize=9)
    fig.subplots_adjust(wspace=0.15, hspace=0.6)  # hspace leaves room for the three-line titles
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()

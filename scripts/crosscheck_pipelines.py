"""Cross-check our pipeline against the independent pipeline of a teammate (Ayush Kumar, a separate repository).

Two team members built two independent pipelines for the same task (EuroSAT label efficiency). This script
compares them where they overlap, so that the report can say how far the two agree.

Usage:
    # first time: summarise the other repository (a local clone) and write the comparison tables
    python scripts/crosscheck_pipelines.py --other-repo <path to clone> --commit <git commit of the clone>
    # afterwards (summaries already saved in results/crosscheck/): only rebuild the tables
    python scripts/crosscheck_pipelines.py

Reads:   results/logs/runs_from_logs.csv (our runs), and either the other repository's results/runs_*.csv or
         the saved summaries results/crosscheck/other_pipeline_*.csv
Writes:  results/crosscheck/other_pipeline_rgb_summary.csv, other_pipeline_band_control.csv and SOURCE.md
         (only with --other-repo), results/tables/crosscheck.{md,tex} and band_control.{md,tex}

What is compared:
- RGB (JPEG) test accuracy of the scratch CNN, ResNet-50 (ImageNet) and ViT-Small (ImageNet) at 1%, 10% and 100% of
  the labels. The two pipelines use different splits (80/10/10 and 70/15/15), different recipes and different
  seeds, so a small difference is expected; the point is whether the conclusions agree.
- A control for the 13-band comparison (ResNet-50, ImageNet weights). Our RGB baseline is the 8-bit JPEG version
  of EuroSAT, while the 13-band input comes from the 16-bit GeoTIFF files. The other pipeline also trained on
  the three RGB bands cut from the GeoTIFF files, which separates the effect of the input source from the effect
  of the extra bands.

Design decisions:
- Only accuracy is compared, because the other pipeline did not save macro-F1 for every run.
- The two protocols are never merged into one table: each number stays tied to its pipeline.
"""

import argparse
from pathlib import Path

import pandas as pd

from eurosat_le import tables as tab

# (model, init) names in the other pipeline <-> (model name, label) in ours
MODEL_PAIRS = [("cnn", "scratch", "scratch_cnn", "Scratch CNN"), ("resnet50", "imagenet", "resnet50", "ResNet-50 (ImageNet)"),
               ("vit_s", "imagenet", "vit_small", "ViT-Small (ImageNet)")]
FRACTIONS = (0.01, 0.1, 1.0)
LABEL = {0.01: "1\\%", 0.1: "10\\%", 1.0: "100\\%"}


def _diff(x: float) -> str:
    """A signed difference with one decimal; a value that rounds to zero is written +0.0, never -0.0."""
    return f"{round(float(x), 1) + 0.0:+.1f}"


def summarise_other(repo: Path, out: Path, commit: str) -> None:
    """Condense the other repository's result tables into two small CSV files (and a note on where they come from)."""
    rgb = pd.read_csv(repo / "results/runs_rgb.csv")
    rgb = rgb[rgb["frac"].isin(FRACTIONS)].assign(acc=lambda d: d.test_acc * 100)
    summary = rgb.groupby(["model", "init", "frac"]).acc.agg(mean_acc="mean", std_acc="std", n="count").reset_index()
    ms = pd.concat([pd.read_csv(repo / f"results/runs_ms_{k}.csv") for k in "ab"])
    ms = ms[ms["frac"].isin(FRACTIONS) & ms["input"].isin(["rgb_ms", "all13"])].assign(acc=lambda d: d.test_acc * 100)
    control = ms.groupby(["input", "frac"]).acc.agg(mean_acc="mean", std_acc="std", n="count").reset_index()
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "other_pipeline_rgb_summary.csv", index=False)
    control.to_csv(out / "other_pipeline_band_control.csv", index=False)
    (out / "SOURCE.md").write_text(
        "# Source of the other pipeline's numbers\n\n"
        "Summaries of results/runs_rgb.csv and results/runs_ms_a.csv / runs_ms_b.csv of the repository\n"
        "https://github.com/kumarAyush24029/Eurosat-label-efficiency (author: Ayush Kumar, a member of this team),\n"
        f"at commit {commit}. His protocol: stratified 80/10/10 split (2,700 test images), 3 seeds, test accuracy.\n"
        "Mean and sample std over the seeds. Not merged with our own results: the protocols differ.\n", encoding="utf-8")


def crosscheck_table(summary: pd.DataFrame, ours: pd.DataFrame) -> pd.DataFrame:
    """Rows: model x label fraction. Columns: other pipeline, ours, difference (ours minus other), in accuracy points."""
    mine = ours[(ours["variant"] == "supervised") & (ours["modality"] == "rgb")].assign(acc=lambda d: d.test_acc * 100)
    rows = []
    for om, oi, mm, label in MODEL_PAIRS:
        for f in FRACTIONS:
            a = summary[(summary["model"] == om) & (summary["init"] == oi) & (summary["frac"] == f)]["mean_acc"].iloc[0]
            b = mine[(mine["model"] == mm) & (mine["fraction"] == f)]["acc"].mean()
            rows.append({"Model": label, "Labels": LABEL[f], "Other pipeline": f"{a:.1f}", "Ours": f"{b:.1f}",
                         "Difference": _diff(b - a)})
    return pd.DataFrame(rows)


def band_control_table(control: pd.DataFrame, summary: pd.DataFrame, ours: pd.DataFrame) -> pd.DataFrame:
    """ResNet-50 (ImageNet): effect of the input source versus the effect of the extra bands, in accuracy points."""
    jpeg = summary[(summary["model"] == "resnet50") & (summary["init"] == "imagenet")].set_index("frac")["mean_acc"]
    tif = control[control["input"] == "rgb_ms"].set_index("frac")["mean_acc"]
    all13 = control[control["input"] == "all13"].set_index("frac")["mean_acc"]
    mine = ours[(ours["variant"] == "supervised") & (ours["model"] == "resnet50")]
    rows = []
    for f in FRACTIONS:
        gain = (mine[(mine["modality"] == "ms") & (mine["fraction"] == f)].test_acc.mean()
                - mine[(mine["modality"] == "rgb") & (mine["fraction"] == f)].test_acc.mean()) * 100
        rows.append({"Labels": LABEL[f], "GeoTIFF RGB $-$ JPEG RGB": _diff(tif[f] - jpeg[f]),
                     "13 bands $-$ GeoTIFF RGB": _diff(all13[f] - tif[f]),
                     "13 bands $-$ JPEG RGB (other)": _diff(all13[f] - jpeg[f]),
                     "13 bands $-$ JPEG RGB (ours)": _diff(gain)})
    return pd.DataFrame(rows)


def main() -> None:
    """Optionally summarise the other repository, then write the two comparison tables."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--other-repo", type=Path, default=None, help="local clone of the other repository")
    ap.add_argument("--commit", default="unknown", help="git commit of that clone (written to SOURCE.md)")
    ap.add_argument("--logs", type=Path, default=Path("results/logs/runs_from_logs.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/crosscheck"))
    ap.add_argument("--tables", type=Path, default=Path("results/tables"))
    args = ap.parse_args()
    if args.other_repo is not None:
        summarise_other(args.other_repo, args.out, args.commit)
    summary = pd.read_csv(args.out / "other_pipeline_rgb_summary.csv")
    control = pd.read_csv(args.out / "other_pipeline_band_control.csv")
    ours = pd.read_csv(args.logs)
    t1 = crosscheck_table(summary, ours)
    tab.write_table(t1, args.tables, "crosscheck",
                    "Test accuracy (\\%) of the same three models on RGB (JPEG) in two independent pipelines, mean over "
                    "seeds. The pipelines differ in split (80/10/10 and 70/15/15), recipe and seeds.")
    t2 = band_control_table(control, summary, ours)
    tab.write_table(t2, args.tables, "band_control",
                    "ResNet-50 (ImageNet): the effect of the input source and of the extra bands, in accuracy points. "
                    "``Other'' is the teammate's pipeline, which also trained on RGB cut from the 13-band files.")
    print(t1.to_string(index=False))
    print(t2.to_string(index=False))


if __name__ == "__main__":
    main()

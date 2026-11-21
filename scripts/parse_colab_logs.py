"""Turn the result lines of Colab notebook outputs into a table of finished runs.

Where it fits: the normal route is scripts/collect_results.py and scripts/evaluate.py on the run folders
(metrics.json and predictions). For two of the three Colab lanes (A and C) only the notebooks, with their
printed training logs, were kept, not the run folders. Every finished run prints one line like

    [resnet50_ms_10pct_seed1] best epoch 29: val acc 0.9810 f1 0.9804 | test acc 0.9783 f1 0.9771 | 10.1 min

and every pseudo-label student also prints its teacher, threshold and pseudo-label count:

    [resnet50_rgb_1pct_seed0_pl] teacher resnet50_rgb_1pct_seed0: tau=0.89 (val precision 0.950), 15935/18708 pseudo-labels

This script reads those lines, so the numbers in the report come from the logs as printed (4 decimals).

Usage:
    python scripts/parse_colab_logs.py NOTEBOOK.ipynb [NOTEBOOK.ipynb ...] [--out results/logs]

Writes (into --out):
    runs_from_logs.csv     one row per finished run: run_name, model, modality, fraction, seed, variant,
                           val_acc, val_f1, test_acc, test_f1, best_epoch, minutes, source, and for
                           students teacher, tau, val_precision, n_pseudo, n_pool
    <notebook stem>.txt    only the result lines found in that notebook (the evidence for the CSV)

Design decisions:
- Learning-rate sweep runs (names containing "_tune_") are skipped: they have validation numbers only.
- If a run appears more than once (an interrupted session was run again), the LAST occurrence in the LAST
  notebook given wins, and every disagreement is printed so it can be checked.
- Only runs whose result line was printed count. A run that was interrupted has no such line.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

NAME = re.compile(r"^(?P<model>scratch_cnn|resnet50|vit_small|vit_scratch)_(?P<modality>rgb|ms)_(?P<pct>\d+)pct_"
                  r"seed(?P<seed>\d+)(?:_(?P<variant>pl|plb)\d*)?$")
RESULT = re.compile(r"^\[(?P<name>[^\]]+)\] best epoch (?P<best_epoch>\d+): val acc (?P<val_acc>[\d.]+) f1 (?P<val_f1>[\d.]+)"
                    r" \| test acc (?P<test_acc>[\d.]+) f1 (?P<test_f1>[\d.]+) \| (?P<minutes>[\d.]+) min")
TEACHER = re.compile(r"^\[(?P<name>[^\]]+)\] teacher (?P<teacher>\S+): tau=(?P<tau>[\d.]+) \(val precision (?P<prec>[\d.]+)\), "
                     r"(?P<n>\d+)/(?P<pool>\d+) pseudo-labels")


def notebook_text(path: Path) -> str:
    """All stream outputs of every code cell, joined (error tracebacks are ignored)."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for cell in nb["cells"]:
        for o in cell.get("outputs", []):
            if o.get("output_type") == "stream":
                out.append("".join(o["text"]))
    return "\n".join(out)


def parse(path: Path) -> tuple[list[dict], list[str]]:
    """Rows of finished runs in one notebook, plus the evidence lines (result and teacher lines)."""
    teachers: dict[str, dict] = {}
    rows: dict[str, dict] = {}
    evidence = []
    for line in notebook_text(path).splitlines():
        line = line.strip()
        m = TEACHER.match(line)
        if m:
            teachers[m["name"]] = dict(teacher=m["teacher"], tau=float(m["tau"]), val_precision=float(m["prec"]),
                                       n_pseudo=int(m["n"]), n_pool=int(m["pool"]))
            evidence.append(line)
            continue
        m = RESULT.match(line)
        if not m or "_tune_" in m["name"]:
            continue
        n = NAME.match(m["name"])
        if not n:
            print(f"  skipped (unknown run name): {m['name']}")
            continue
        evidence.append(line)
        rows[m["name"]] = dict(
            run_name=m["name"], model=n["model"], modality=n["modality"], fraction=int(n["pct"]) / 100,
            seed=int(n["seed"]), variant=n["variant"] or "supervised",
            val_acc=float(m["val_acc"]), val_f1=float(m["val_f1"]), test_acc=float(m["test_acc"]),
            test_f1=float(m["test_f1"]), best_epoch=int(m["best_epoch"]), minutes=float(m["minutes"]),
            source=path.name)
    for name, row in rows.items():
        row.update(teachers.get(name, {}))
    return list(rows.values()), evidence


def main() -> None:
    """Parse the given notebooks in order and write runs_from_logs.csv and one evidence file per notebook."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("notebooks", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/logs"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    merged: dict[str, dict] = {}
    for nb in args.notebooks:
        rows, evidence = parse(nb)
        slug = re.sub(r"\W+", "_", nb.stem).strip("_")  # a file name without spaces or brackets
        (args.out / f"{slug}.txt").write_text("\n".join(evidence) + "\n", encoding="utf-8")
        print(f"{nb.name}: {len(rows)} finished runs")
        for r in rows:
            old = merged.get(r["run_name"])
            if old and abs(old["test_f1"] - r["test_f1"]) > 1e-9:
                print(f"  differs: {r['run_name']} test F1 {old['test_f1']} ({old['source']}) vs {r['test_f1']} ({r['source']}); the last one is kept")
            merged[r["run_name"]] = r
    df = pd.DataFrame(merged.values()).sort_values(["variant", "model", "modality", "fraction", "seed"])
    df.to_csv(args.out / "runs_from_logs.csv", index=False)
    print(f"{len(df)} runs -> {args.out / 'runs_from_logs.csv'}")
    print(df.groupby(["variant", "model"]).size().to_string())


if __name__ == "__main__":
    main()

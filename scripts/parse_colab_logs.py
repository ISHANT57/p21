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

# Standard library: argparse reads command-line options, json opens the notebook files, re matches the log
# lines with regular expressions, and Path handles file paths.
import argparse
import json
import re
from pathlib import Path

# pandas builds, sorts and saves the final table of runs.
import pandas as pd

# Pattern for a run name such as "resnet50_ms_10pct_seed1_pl": it splits out the model, the input type
# (rgb or ms), the percentage of labels, the seed and the optional pseudo-label variant (pl or plb).
NAME = re.compile(r"^(?P<model>scratch_cnn|resnet50|vit_small|vit_scratch)_(?P<modality>rgb|ms)_(?P<pct>\d+)pct_"
                  r"seed(?P<seed>\d+)(?:_(?P<variant>pl|plb)\d*)?$")
# Pattern for the line printed when a run finishes: name, best epoch, validation and test accuracy / F1, minutes.
RESULT = re.compile(r"^\[(?P<name>[^\]]+)\] best epoch (?P<best_epoch>\d+): val acc (?P<val_acc>[\d.]+) f1 (?P<val_f1>[\d.]+)"
                    r" \| test acc (?P<test_acc>[\d.]+) f1 (?P<test_f1>[\d.]+) \| (?P<minutes>[\d.]+) min")
# Pattern for the line a pseudo-label student prints: its teacher, the confidence threshold tau, the teacher's
# validation precision, and how many of the unlabeled pool images received a pseudo-label.
TEACHER = re.compile(r"^\[(?P<name>[^\]]+)\] teacher (?P<teacher>\S+): tau=(?P<tau>[\d.]+) \(val precision (?P<prec>[\d.]+)\), "
                     r"(?P<n>\d+)/(?P<pool>\d+) pseudo-labels")


def notebook_text(path: Path) -> str:
    """All stream outputs of every code cell, joined (error tracebacks are ignored).

    Args:
        path: the .ipynb notebook file.
    Returns:
        One string with the printed text of the whole notebook, one output chunk after another.
    """
    # A notebook is a JSON file; load it into Python lists and dictionaries.
    nb = json.loads(path.read_text(encoding="utf-8"))
    # Collect the printed text pieces here.
    out = []
    # Go through the cells in order.
    for cell in nb["cells"]:
        # Each code cell can have several outputs (printed text, images, errors).
        for o in cell.get("outputs", []):
            # Keep only the printed text (stream) outputs; the pieces of one output are joined together.
            if o.get("output_type") == "stream":
                out.append("".join(o["text"]))
    # Join all outputs into one big text with a newline between outputs.
    return "\n".join(out)


def parse(path: Path) -> tuple[list[dict], list[str]]:
    """Rows of finished runs in one notebook, plus the evidence lines (result and teacher lines).

    Args:
        path: the .ipynb notebook file.
    Returns:
        A pair (rows, evidence). rows is a list with one dictionary per finished run (names, settings,
        scores and, for students, the teacher details). evidence is the list of log lines the rows came from.
    """
    # Teacher information of the pseudo-label students, found by run name (it may be printed before the result).
    teachers: dict[str, dict] = {}
    # Finished runs found so far, keyed by run name so that a later repeat of a run replaces the earlier one.
    rows: dict[str, dict] = {}
    # The raw log lines that were used, saved as proof for the CSV.
    evidence = []
    # Read the notebook output line by line.
    for line in notebook_text(path).splitlines():
        # Remove spaces around the line so the patterns (which start with "[") can match.
        line = line.strip()
        # Is this a "teacher ... pseudo-labels" line?
        m = TEACHER.match(line)
        if m:
            # Remember the teacher, threshold, precision and pseudo-label counts for this student run.
            teachers[m["name"]] = dict(teacher=m["teacher"], tau=float(m["tau"]), val_precision=float(m["prec"]),
                                       n_pseudo=int(m["n"]), n_pool=int(m["pool"]))
            evidence.append(line)
            # Nothing more to do with this line.
            continue
        # Otherwise check whether it is a "run finished" result line.
        m = RESULT.match(line)
        # Ignore other lines, and skip learning-rate sweep runs because they have no test numbers.
        if not m or "_tune_" in m["name"]:
            continue
        # Split the run name into model, modality, fraction, seed and variant.
        n = NAME.match(m["name"])
        # A name that does not follow the naming scheme cannot be placed in the table, so report and skip it.
        if not n:
            print(f"  skipped (unknown run name): {m['name']}")
            continue
        evidence.append(line)
        # Build the table row: settings come from the name, scores from the result line. The percentage is
        # turned into a fraction (10pct -> 0.1) and a name without a suffix is the plain supervised variant.
        rows[m["name"]] = dict(
            run_name=m["name"], model=n["model"], modality=n["modality"], fraction=int(n["pct"]) / 100,
            seed=int(n["seed"]), variant=n["variant"] or "supervised",
            val_acc=float(m["val_acc"]), val_f1=float(m["val_f1"]), test_acc=float(m["test_acc"]),
            test_f1=float(m["test_f1"]), best_epoch=int(m["best_epoch"]), minutes=float(m["minutes"]),
            source=path.name)
    # Attach the teacher details to each student run (supervised runs have none, so nothing is added).
    for name, row in rows.items():
        row.update(teachers.get(name, {}))
    return list(rows.values()), evidence


# Pattern for the line printed when a learning-rate sweep run finishes: it includes the learning rate and has
# validation scores only (no test scores).
TUNE = re.compile(r"^\[(?P<name>(?P<model>scratch_cnn|resnet50|vit_small)_(?P<modality>rgb|ms)_(?P<pct>\d+)pct_seed(?P<seed>\d+)"
                  r"_tune_lr(?P<lr>[\d.e+-]+))\] best epoch (?P<best_epoch>\d+): val acc (?P<val_acc>[\d.]+) "
                  r"f1 (?P<val_f1>[\d.]+) \| (?P<minutes>[\d.]+) min")


def parse_tuning(path: Path) -> list[dict]:
    """Finished learning-rate sweep runs of one notebook (validation numbers only; the test set is not used).

    Args:
        path: the .ipynb notebook file.
    Returns:
        A list with one dictionary per finished sweep run (settings, learning rate, validation scores).
    """
    # Sweep runs found so far, keyed by name so that a repeated run replaces the earlier one.
    rows: dict[str, dict] = {}
    # Read the notebook output line by line.
    for line in notebook_text(path).splitlines():
        # Does this line report a finished sweep run?
        m = TUNE.match(line.strip())
        if m:
            # Store the settings, the learning rate and the validation scores of this run.
            rows[m["name"]] = dict(run_name=m["name"], model=m["model"], modality=m["modality"],
                                   fraction=int(m["pct"]) / 100, seed=int(m["seed"]), lr=float(m["lr"]),
                                   val_acc=float(m["val_acc"]), val_f1=float(m["val_f1"]),
                                   best_epoch=int(m["best_epoch"]), minutes=float(m["minutes"]), source=path.name)
    return list(rows.values())


def main() -> None:
    """Parse the given notebooks in order and write runs_from_logs.csv and one evidence file per notebook."""
    # Build the command-line parser; its description is the first line of this file's docstring.
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # One or more notebook files to read (later notebooks win when a run appears twice).
    ap.add_argument("notebooks", nargs="+", type=Path)
    # Option: the folder where the outputs are written.
    ap.add_argument("--out", type=Path, default=Path("results/logs"))
    # Read what the user typed.
    args = ap.parse_args()
    # Make sure the output folder exists before writing into it.
    args.out.mkdir(parents=True, exist_ok=True)

    # All finished runs from all notebooks, keyed by run name.
    merged: dict[str, dict] = {}
    # Process the notebooks in the order given.
    for nb in args.notebooks:
        # Extract the finished runs and the evidence lines from this notebook.
        rows, evidence = parse(nb)
        slug = re.sub(r"\W+", "_", nb.stem).strip("_")  # a file name without spaces or brackets
        # Save the evidence lines next to the CSV so the numbers can be traced back to the log.
        (args.out / f"{slug}.txt").write_text("\n".join(evidence) + "\n", encoding="utf-8")
        print(f"{nb.name}: {len(rows)} finished runs")
        # Merge this notebook's runs into the overall table.
        for r in rows:
            # Was the same run already found in an earlier notebook?
            old = merged.get(r["run_name"])
            # If the two copies disagree on the test F1, tell the user; the later copy replaces the earlier one.
            if old and abs(old["test_f1"] - r["test_f1"]) > 1e-9:
                print(f"  differs: {r['run_name']} test F1 {old['test_f1']} ({old['source']}) vs {r['test_f1']} ({r['source']}); the last one is kept")
            merged[r["run_name"]] = r
    # Build the final table, sorted so that the rows come in a stable, readable order, and save it.
    df = pd.DataFrame(merged.values()).sort_values(["variant", "model", "modality", "fraction", "seed"])
    df.to_csv(args.out / "runs_from_logs.csv", index=False)
    print(f"{len(df)} runs -> {args.out / 'runs_from_logs.csv'}")
    # Print how many runs each variant / model combination has, as a quick completeness check.
    print(df.groupby(["variant", "model"]).size().to_string())
    # Collect the learning-rate sweep runs from all notebooks (a later notebook overrides an earlier one).
    tune = {r["run_name"]: r for nb in args.notebooks for r in parse_tuning(nb)}
    # Only write the sweep table if any sweep runs were found.
    if tune:
        # Sort by model and learning rate so the sweep reads naturally, then save it.
        t = pd.DataFrame(tune.values()).sort_values(["model", "lr"])
        t.to_csv(args.out / "tuning_from_logs.csv", index=False)
        print(f"{len(t)} learning-rate sweep runs -> {args.out / 'tuning_from_logs.csv'}")


# Run main() only when the file is executed as a script, not when it is imported.
if __name__ == "__main__":
    main()

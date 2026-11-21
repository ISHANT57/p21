"""Tests for scripts/parse_colab_logs.py and scripts/eval_from_logs.py (results taken from notebook logs).

The project's real runs were done in Colab notebooks, whose printed output is the only record of the
results. These tests build a small fake notebook containing realistic log lines and check that the
parser reads the right numbers from it and that the summary tables are computed correctly.
"""

# importlib.util: load a script file as a module; json: write the fake notebook; sys: register the module.
import importlib.util
import json
import sys
# Path: file paths for the scripts folder and the fake notebook.
from pathlib import Path

# numpy/pandas: compare numbers with tolerance and inspect the result tables.
import numpy as np
import pandas as pd

# The repository's top folder: this file is in tests/, so go one level up from its folder.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    """Import a script from scripts/ as a module (scripts are not a package)."""
    # Describe where the file is, create an empty module from that description, ...
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    # ... register it under its name (some code expects that) and then run the file to fill the module.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load the two scripts under test once, when this test file is imported.
parse_logs = _load("parse_colab_logs")
from_logs = _load("eval_from_logs")  # noqa: E402

# A fake notebook output with the log lines the training notebooks print. It deliberately contains:
# finished supervised runs (two seeds, RGB and 13-band), a learning-rate sweep run ("tune", which must be
# ignored), a pseudo-label student with its teacher line, and an epoch line (which is not a result).
LOG = """\
[resnet50_rgb_1pct_seed0] best epoch 7: val acc 0.8966 f1 0.8916 | test acc 0.8904 f1 0.8853 | 5.4 min
[resnet50_rgb_1pct_seed0_tune_lr0.001] best epoch 23: val acc 0.9696 f1 0.9691 | 9.5 min
[resnet50_rgb_1pct_seed0_pl] teacher resnet50_rgb_1pct_seed0: tau=0.89 (val precision 0.950), 15935/18708 pseudo-labels
  epoch   1  loss 0.8069  train_acc 0.7735  val_acc 0.9558  val_f1 0.9548   90.4s  *
[resnet50_rgb_1pct_seed0_pl] best epoch 2: val acc 0.9195 f1 0.9161 | test acc 0.9123 f1 0.9095 | 12.4 min
[resnet50_ms_1pct_seed0] best epoch 29: val acc 0.9134 f1 0.9115 | test acc 0.9015 f1 0.8988 | 10.1 min
[resnet50_rgb_1pct_seed1] best epoch 29: val acc 0.8909 f1 0.8858 | test acc 0.8894 f1 0.8845 | 9.4 min
[resnet50_ms_1pct_seed1] best epoch 28: val acc 0.8988 f1 0.8948 | test acc 0.8998 f1 0.8961 | 10.0 min
"""


def _notebook(tmp_path: Path) -> Path:
    """Helper: write a minimal .ipynb file whose only printed output is LOG, and return its path."""
    # A notebook is JSON: one code cell with two outputs. The first is the printed log text (a "stream");
    # the second is an error output (as after a KeyboardInterrupt), which the parser must ignore.
    nb = {"cells": [{"cell_type": "code", "source": [], "outputs": [
        {"output_type": "stream", "name": "stdout", "text": LOG.splitlines(keepends=True)},
        {"output_type": "error", "traceback": ["KeyboardInterrupt"]}]}]}
    # The file name has a space and brackets, like a real downloaded Colab file ("lane (1).ipynb").
    p = tmp_path / "lane (1).ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    return p


def test_parse_finished_runs_skips_tuning_and_reads_teacher(tmp_path):
    """The parser must keep finished real runs, skip the LR sweep, and read the teacher details of a student."""
    # Parse the fake notebook; rows = one dict per finished run, evidence = the log lines they came from.
    rows, evidence = parse_logs.parse(_notebook(tmp_path))
    # Put the rows in a table indexed by run name for easy lookup.
    df = pd.DataFrame(rows).set_index("run_name")
    # 5 finished runs are expected (4 supervised + 1 student); the "tune" run must not be among them.
    assert len(df) == 5 and not any("tune" in n for n in df.index)  # the LR-sweep run is not a result
    # A supervised run: its settings must be read from the run name ...
    r = df.loc["resnet50_ms_1pct_seed0"]
    assert (r.model, r.modality, r.fraction, r.seed, r.variant) == ("resnet50", "ms", 0.01, 0, "supervised")
    # ... and its scores and training time from the result line.
    assert r.test_f1 == 0.8988 and r.test_acc == 0.9015 and r.minutes == 10.1
    # A pseudo-label student: variant "pl", plus threshold tau and pool counts from the teacher line.
    s = df.loc["resnet50_rgb_1pct_seed0_pl"]
    assert s.variant == "pl" and s.tau == 0.89 and s.n_pseudo == 15935 and s.n_pool == 18708
    # Every evidence line is a raw log line starting with "[run name]", so it can be checked by hand later.
    assert all(e.startswith("[") for e in evidence)


def test_groups_and_pairs_from_logs(tmp_path):
    """The summary tables built from the logs must give correct means, std handling, pairs and differences."""
    # Parse the fake notebook into a table of runs.
    rows, _ = parse_logs.parse(_notebook(tmp_path))
    runs = pd.DataFrame(rows)
    # groups_table: one row per setting (model, input, fraction, variant); index it by those for lookup.
    g = from_logs.groups_table(runs).set_index(["model", "modality", "fraction", "variant"])
    # ResNet-50 on 13 bands has two seeds, so its mean test F1 must be the average of the two logged values.
    ms = g.loc[("resnet50", "ms", 0.01, "supervised")]
    assert ms.n_seeds == 2 and np.isclose(ms.test_f1_mean, (0.8988 + 0.8961) / 2)
    # The pseudo-label student has only one seed, so its std is undefined (NaN) rather than 0.
    assert pd.isna(g.loc[("resnet50", "rgb", 0.01, "pl")].test_f1_std)  # one seed: no std
    # RGB-versus-13-band pairs: seeds 0 and 1 both have both runs, giving 2 pairs.
    p = from_logs.pairs_table(runs, "modality", "rgb", "ms")
    assert len(p) == 2 and p["ci_low"].isna().all()  # no bootstrap interval without predictions
    # For seed 0 the F1 gain is the 13-band F1 minus the RGB F1 of the same seed.
    assert np.isclose(p.sort_values("seed")["delta_f1"].iloc[0], 0.8988 - 0.8853)
    # Supervised-versus-pseudo-label: only seed 0 has a student, so one pair, compared with its own teacher.
    q = from_logs.pairs_table(runs, "variant", "supervised", "pl")
    assert len(q) == 1 and np.isclose(q["delta_f1"].iloc[0], 0.9095 - 0.8853)  # student vs its own teacher

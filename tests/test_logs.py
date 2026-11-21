"""Tests for scripts/parse_colab_logs.py and scripts/eval_from_logs.py (results taken from notebook logs)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    """Import a script from scripts/ as a module (scripts are not a package)."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


parse_logs = _load("parse_colab_logs")
from_logs = _load("eval_from_logs")  # noqa: E402

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
    nb = {"cells": [{"cell_type": "code", "source": [], "outputs": [
        {"output_type": "stream", "name": "stdout", "text": LOG.splitlines(keepends=True)},
        {"output_type": "error", "traceback": ["KeyboardInterrupt"]}]}]}
    p = tmp_path / "lane (1).ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    return p


def test_parse_finished_runs_skips_tuning_and_reads_teacher(tmp_path):
    rows, evidence = parse_logs.parse(_notebook(tmp_path))
    df = pd.DataFrame(rows).set_index("run_name")
    assert len(df) == 5 and not any("tune" in n for n in df.index)  # the LR-sweep run is not a result
    r = df.loc["resnet50_ms_1pct_seed0"]
    assert (r.model, r.modality, r.fraction, r.seed, r.variant) == ("resnet50", "ms", 0.01, 0, "supervised")
    assert r.test_f1 == 0.8988 and r.test_acc == 0.9015 and r.minutes == 10.1
    s = df.loc["resnet50_rgb_1pct_seed0_pl"]
    assert s.variant == "pl" and s.tau == 0.89 and s.n_pseudo == 15935 and s.n_pool == 18708
    assert all(e.startswith("[") for e in evidence)


def test_groups_and_pairs_from_logs(tmp_path):
    rows, _ = parse_logs.parse(_notebook(tmp_path))
    runs = pd.DataFrame(rows)
    g = from_logs.groups_table(runs).set_index(["model", "modality", "fraction", "variant"])
    ms = g.loc[("resnet50", "ms", 0.01, "supervised")]
    assert ms.n_seeds == 2 and np.isclose(ms.test_f1_mean, (0.8988 + 0.8961) / 2)
    assert pd.isna(g.loc[("resnet50", "rgb", 0.01, "pl")].test_f1_std)  # one seed: no std
    p = from_logs.pairs_table(runs, "modality", "rgb", "ms")
    assert len(p) == 2 and p["ci_low"].isna().all()  # no bootstrap interval without predictions
    assert np.isclose(p.sort_values("seed")["delta_f1"].iloc[0], 0.8988 - 0.8853)
    q = from_logs.pairs_table(runs, "variant", "supervised", "pl")
    assert len(q) == 1 and np.isclose(q["delta_f1"].iloc[0], 0.9095 - 0.8853)  # student vs its own teacher

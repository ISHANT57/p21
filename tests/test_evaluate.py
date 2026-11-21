"""Evaluation tests on fake run directories built from the real test ids (no training)."""

import json
import subprocess
import zlib
import sys

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import f1_score

from eurosat_le.constants import REPO_ROOT
from eurosat_le.data.datasets import split_ids
from eurosat_le.data.splits import load_split
from eurosat_le.eval.evaluate import (compare, expected_calibration_error, load_runs, macro_f1_fast,
                                      paired_bootstrap, run_metrics, summarise_comparison)
from eurosat_le.eval.metrics import classification_metrics


def _fake_logits(labels, acc, rng):
    pred = np.where(rng.random(len(labels)) < acc, labels, rng.integers(0, 10, len(labels)))
    logits = rng.normal(0, 1, (len(labels), 10))
    logits[np.arange(len(labels)), pred] += 4
    return logits


def make_run(runs_dir, model, modality, fraction, seed, acc, suffix="", rng=None):
    rng = rng or np.random.default_rng(zlib.crc32(f"{model}{modality}{seed}{suffix}".encode()))
    name = f"{model}_{modality}_{fraction * 100:g}pct_seed{seed}" + (f"_{suffix}" if suffix else "")
    d = runs_dir / name
    d.mkdir(parents=True)
    split = load_split().set_index("id")
    metrics = {"run_name": name}
    for s in ("val", "test"):
        ids = np.array(split_ids(s))
        ids = ids[rng.permutation(len(ids))]  # each run stores ids in its own order
        y = split.loc[ids, "label"].to_numpy()
        logits = _fake_logits(y, acc, rng)
        np.savez(d / f"preds_{s}.npz", logits=logits.astype(np.float16), labels=y, ids=ids)
        metrics[s] = classification_metrics(y, logits.astype(np.float16).astype(np.float32).argmax(1))
    cfg = {"model": model, "modality": modality, "fraction": fraction, "seed": seed, "name_suffix": suffix}
    (d / "config.yaml").write_text(yaml.safe_dump(cfg))
    (d / "metrics.json").write_text(json.dumps(metrics))
    return d


@pytest.fixture
def runs_dir(tmp_path):
    rd = tmp_path / "runs"
    for seed in (0, 1):
        make_run(rd, "resnet50", "rgb", 0.01, seed, 0.80)
        make_run(rd, "resnet50", "ms", 0.01, seed, 0.88)
        make_run(rd, "resnet50", "rgb", 0.01, seed, 0.84, suffix="pl")
        make_run(rd, "scratch_cnn", "rgb", 0.01, seed, 0.55)
    return rd


def test_macro_f1_fast_matches_sklearn():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 10, 500), rng.integers(0, 10, 500)
    assert macro_f1_fast(y, p) == pytest.approx(f1_score(y, p, average="macro", labels=range(10)))


def test_ece():
    y = np.zeros(1000, int)
    perfect = np.eye(10)[y]
    assert expected_calibration_error(perfect, y) == pytest.approx(0)
    over = np.full((1000, 10), 0.01)
    over[:, 0] = 0.91
    wrong = np.r_[np.zeros(500, int), np.ones(500, int)]  # 50% correct at 91% confidence
    assert expected_calibration_error(over, wrong) == pytest.approx(0.41, abs=0.01)


def test_paired_bootstrap():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 2000)
    p = np.where(rng.random(2000) < 0.7, y, rng.integers(0, 10, 2000))
    assert paired_bootstrap(y, p, p, n_boot=200) == (0.0, 0.0, 0.0)
    better = np.where(rng.random(2000) < 0.5, y, p)
    d, lo, hi = paired_bootstrap(y, p, better, n_boot=200)
    assert d > 0 and lo > 0 and lo <= d <= hi


def test_compare_aligns_ids_and_pairs_by_seed(runs_dir):
    runs = load_runs(runs_dir)
    pairs = compare(runs, "modality", "rgb", "ms", n_boot=100)
    assert len(pairs) == 2 and set(pairs["seed"]) == {0, 1} and (pairs["model"] == "resnet50").all()
    assert (pairs["delta_f1"] > 0).all()
    summ = summarise_comparison(pairs)
    assert summ.loc[0, "n_seeds"] == 2 and summ.loc[0, "seeds_b_better"] == 2
    pl = compare(runs, "variant", "supervised", "pl", n_boot=100)
    assert len(pl) == 2 and (pl["modality"] == "rgb").all()


def test_recomputed_metrics_match_logged(runs_dir):
    from eurosat_le.eval.evaluate import check_against_logged

    for d in sorted(runs_dir.iterdir()):
        assert check_against_logged(d, run_metrics(d)) == []


def test_load_preds_cache(tmp_path):
    """Predictions are read once, shared read-only, and re-read when the file is rewritten."""
    import os

    from eurosat_le.eval.evaluate import load_preds

    d = make_run(tmp_path / "runs", "resnet50", "rgb", 0.01, 0, 0.80)
    load_preds.cache_clear()  # start empty, so earlier tests cannot affect this one
    first = load_preds(d)
    assert all(a is b for a, b in zip(first, load_preds(d)))  # second call hits the cache
    with pytest.raises(ValueError):
        first[0][0, 0] = 1.0  # shared arrays are read-only
    # Rewrite the file with all-zero logits; the next load must return the new contents.
    f = d / "preds_test.npz"
    data = dict(np.load(f))
    data["logits"] = np.zeros_like(data["logits"])
    np.savez(f, **data)
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # make sure the mtime changes
    assert not load_preds(d)[0].any()
    load_preds.cache_clear()  # leave no cached arrays behind for other tests


def test_evaluate_script_end_to_end(runs_dir, tmp_path):
    out = tmp_path / "eval"
    res = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "evaluate.py"), "--runs-dir", str(runs_dir),
                          "--out", str(out), "--n-boot", "100"], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "WARNING" not in res.stdout
    groups = pd.read_csv(out / "groups.csv")
    assert len(groups) == 4 and (groups["n_seeds"] == 2).all()
    assert len(pd.read_csv(out / "per_class.csv")) == 4 * 10
    assert (out / "compare_rgb_vs_ms.csv").exists() and (out / "compare_supervised_vs_pl.csv").exists()
    assert (out / "compare_scratch_vs_resnet50.csv").exists()
    cm = pd.read_csv(out / "confusion" / "resnet50_ms_1pct_supervised.csv", index_col=0)
    assert np.allclose(cm.sum(axis=1), 1)

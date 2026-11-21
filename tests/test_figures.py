"""Phase 7: figure and table builders, tested on synthetic evaluation tables (no real runs needed)."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from eurosat_le import figures, tables
from eurosat_le.constants import CLASSES, REPO_ROOT

MODELS = ["scratch_cnn", "resnet50", "vit_small"]
FRACS = [0.01, 0.1, 1.0]


def _groups() -> pd.DataFrame:
    rows = []
    for i, m in enumerate(MODELS):
        for mo, bonus in (("rgb", 0.0), ("ms", 0.01)):
            for j, f in enumerate(FRACS):
                f1 = 0.5 + 0.1 * i + 0.12 * j + bonus
                rows.append(dict(model=m, modality=mo, fraction=f, variant="supervised", n_seeds=3,
                                 test_acc_mean=f1 + 0.01, test_acc_std=0.01, test_f1_mean=f1, test_f1_std=0.01,
                                 test_ece_mean=0.05, val_f1_mean=f1))
    return pd.DataFrame(rows)


def _per_class(groups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in groups.iterrows():
        for k, c in enumerate(CLASSES):
            rows.append(dict(model=g["model"], modality=g["modality"], fraction=g["fraction"], variant="supervised",
                             **{"class": c}, precision=0.9, recall=0.9, f1_mean=g["test_f1_mean"] - 0.01 * k,
                             f1_std=0.01, n_seeds=3))
    return pd.DataFrame(rows)


def _pairs(factor_a: str, factor_b: str, **fixed) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        for f in FRACS:
            for s in range(3):
                rows.append(dict(model=m, fraction=f, variant="supervised", seed=s, a=factor_a, b=factor_b,
                                 f1_a=0.8, f1_b=0.82 + 0.005 * s, delta_f1=0.02 + 0.005 * s, ci_low=0.0, ci_high=0.04,
                                 **fixed))
    return pd.DataFrame(rows)


def _summary(pairs: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("model", "modality", "fraction") if c in pairs.columns]
    return pairs.groupby(keys).agg(f1_a=("f1_a", "mean"), f1_b=("f1_b", "mean"), delta_f1_mean=("delta_f1", "mean"),
                                   delta_f1_std=("delta_f1", "std"), n_seeds=("seed", "nunique"),
                                   seeds_b_better=("delta_f1", lambda d: int((d > 0).sum()))).reset_index()


def test_label_efficiency_and_heatmap(tmp_path):
    g = _groups()
    assert figures.label_efficiency(g, tmp_path / "a.png").stat().st_size > 1000
    assert figures.label_efficiency(g, tmp_path / "b.png", metric="acc").exists()
    assert figures.per_class_heatmap(_per_class(g), 1.0, tmp_path / "c.png").exists()


def test_missing_data_returns_none(tmp_path):
    empty = _groups().iloc[0:0]
    assert figures.label_efficiency(empty, tmp_path / "a.png") is None
    assert figures.rgb_vs_ms_gain(pd.DataFrame(), tmp_path / "b.png") is None
    assert figures.pseudo_label_gain({"pl": None, "plb": pd.DataFrame()}, tmp_path / "c.png") is None
    assert figures.confusion_grid({}, tmp_path / "d.png", "t") is None
    assert not list(tmp_path.iterdir())


def test_gain_and_pseudo_label_figures(tmp_path):
    assert figures.rgb_vs_ms_gain(_pairs("rgb", "ms"), tmp_path / "gain.png").exists()
    pl = _pairs("supervised", "pl", modality="rgb")
    plb = _pairs("supervised", "plb", modality="ms")
    assert figures.pseudo_label_gain({"pl": pl, "plb": plb}, tmp_path / "pl.png").exists()


def test_confusion_grid(tmp_path):
    cm = np.eye(10) * 0.9 + 0.01
    assert figures.confusion_grid({"A": cm, "B": cm}, tmp_path / "cm.png", "title").exists()


def test_label_spread_keeps_order_and_gap():
    out = figures._spread([0.9, 0.9, 0.91, 0.5], gap=0.03)
    assert out[3] == 0.5
    s = sorted(out)
    assert all(b - a >= 0.03 - 1e-9 for a, b in zip(s[:-1], s[1:]))


def test_supervised_table_bolds_best_per_column():
    t = tables.supervised_table(_groups())
    assert t.shape == (6, 3)
    for c in t.columns:
        assert t[c].str.startswith("**").sum() == 1
    assert "ViT-Small, 13 bands" in t.index and "±" in t.iloc[0, 0]


def test_semi_supervised_and_per_class_tables():
    comp = {"pl": _summary(_pairs("supervised", "pl", modality="rgb")), "plb": None}
    t = tables.semi_supervised_table(_groups(), comp)
    assert len(t) == 9 and t["Seeds better"].eq("3/3").all()
    assert tables.semi_supervised_table(_groups(), {"pl": None}) is None
    g = _groups()
    assert tables.best_model(g) == "vit_small"
    pc = tables.per_class_table(_per_class(g), "vit_small", 1.0)
    assert list(pc.index) == CLASSES and list(pc.columns) == ["RGB (3 bands)", "13 bands"]


def test_markdown_and_latex_escaping(tmp_path):
    t = tables.supervised_table(_groups())
    paths = tables.write_table(t, tmp_path, "sup", "Test macro-F1 (\\%).")
    md, tex = (p.read_text(encoding="utf-8") for p in paths)
    assert md.startswith("|  |") and "**" in md
    assert r"$\pm$" in tex and r"\textbf{" in tex and "**" not in tex and "±" not in tex
    assert r"\begin{tabular}{lccc}" in tex and r"\label{tab:sup}" in tex


@pytest.mark.parametrize("partial", [False, True])
def test_cli_end_to_end(tmp_path, partial):
    ev = tmp_path / "eval"
    (ev / "confusion").mkdir(parents=True)
    g = _groups()
    g.to_csv(ev / "groups.csv", index=False)
    if not partial:
        _per_class(g).to_csv(ev / "per_class.csv", index=False)
        pr = _pairs("rgb", "ms")
        pr.to_csv(ev / "compare_rgb_vs_ms_pairs.csv", index=False)
        for f in (1.0, 0.01):
            for mo in ("rgb", "ms"):
                pd.DataFrame(np.eye(10), index=CLASSES, columns=CLASSES).to_csv(
                    ev / "confusion" / f"vit_small_{mo}_{f * 100:g}pct_supervised.csv")
    res = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "make_figures.py"), "--eval-dir", str(ev),
                          "--figures-dir", str(tmp_path / "f"), "--tables-dir", str(tmp_path / "t")],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "f" / "label_efficiency_f1.png").exists()
    assert (tmp_path / "t" / "supervised_f1.tex").exists()
    assert (tmp_path / "f" / "confusion_100pct.png").exists() == (not partial)
    assert "skipped pseudo-label gains" in res.stdout


def test_cli_needs_groups(tmp_path):
    res = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "make_figures.py"), "--eval-dir",
                          str(tmp_path)], capture_output=True, text=True)
    assert res.returncode != 0 and "run scripts/evaluate.py first" in res.stderr


def test_single_seed_cell_shows_zero_spread_and_figure_draws(tmp_path):
    # Protects the 100% runs, which use one seed: groups.csv then has a NaN std. The table must print
    # 0.0 for the spread (never "nan") and the label-efficiency figure must still be drawn.
    g = _groups()
    one = (g["model"] == "resnet50") & (g["modality"] == "rgb") & (g["fraction"] == 1.0)
    g.loc[one, ["n_seeds", "test_f1_std", "test_acc_std"]] = [1, np.nan, np.nan]
    for metric in ("f1", "acc"):
        cell = tables.supervised_table(g, metric=metric).at["ResNet-50, RGB (3 bands)", "100%"]
        assert cell.rstrip("*").endswith("± 0.0") and "nan" not in cell.lower()
    assert figures.label_efficiency(g, tmp_path / "f1.png").stat().st_size > 1000
    assert figures.label_efficiency(g, tmp_path / "acc.png", metric="acc").stat().st_size > 1000


def test_missing_model_or_cell_in_supervised_table(tmp_path):
    # Protects partial results: a model with no 13-band runs gets no row, and a single missing cell
    # shows an en dash. Neither may crash the table, and the figure must still be drawn.
    g = _groups()
    no_ms = g[~((g["model"] == "vit_small") & (g["modality"] == "ms"))]
    t = tables.supervised_table(no_ms)
    assert t.shape == (5, 3) and "ViT-Small, 13 bands" not in t.index and "ViT-Small, RGB (3 bands)" in t.index
    assert figures.label_efficiency(no_ms, tmp_path / "a.png").exists()
    no_cell = g[~((g["model"] == "resnet50") & (g["modality"] == "ms") & (g["fraction"] == 0.1))]
    t = tables.supervised_table(no_cell)
    assert t.shape == (6, 3) and t.at["ResNet-50, 13 bands", "10%"] == "–"
    assert t.at["ResNet-50, 13 bands", "1%"] != "–"


def test_empty_groups_return_none(tmp_path):
    # Protects a run with no results yet: with an empty groups table (or no supervised rows) the
    # figure functions return None and write no file.
    empty = _groups().iloc[0:0]
    assert figures.label_efficiency(empty, tmp_path / "a.png") is None
    assert figures.label_efficiency(empty, tmp_path / "b.png", metric="acc") is None
    assert figures.per_class_heatmap(_per_class(_groups()).iloc[0:0], 1.0, tmp_path / "c.png") is None
    pl_only = _groups().assign(variant="pl")  # rows exist, but none are supervised
    assert figures.label_efficiency(pl_only, tmp_path / "d.png") is None
    assert not list(tmp_path.iterdir())

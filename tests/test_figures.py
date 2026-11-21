"""Phase 7: figure and table builders, tested on synthetic evaluation tables (no real runs needed)."""

# subprocess/sys: run scripts/make_figures.py as a separate program in the command-line tests.
import subprocess
import sys

# numpy/pandas: build fake evaluation tables and confusion matrices; pytest: parametrise tests.
import numpy as np
import pandas as pd
import pytest

# The two modules under test: figure drawing and table building.
from eurosat_le import figures, tables
# CLASSES: the 10 class names; REPO_ROOT: to find the make_figures script.
from eurosat_le.constants import CLASSES, REPO_ROOT

# The models and label fractions used by the fake result tables below.
MODELS = ["scratch_cnn", "resnet50", "vit_small"]
FRACS = [0.01, 0.1, 1.0]


def _groups() -> pd.DataFrame:
    """Helper: fake groups.csv (one row per model x input x label fraction) with made-up, rising scores."""
    rows = []
    for i, m in enumerate(MODELS):
        # 13-band input ("ms") gets a small bonus of +0.01 F1 over RGB.
        for mo, bonus in (("rgb", 0.0), ("ms", 0.01)):
            for j, f in enumerate(FRACS):
                # Score grows with model index i and with the label fraction index j, so the best model
                # (the last one, vit_small) and the best fraction (100%) are known in advance.
                f1 = 0.5 + 0.1 * i + 0.12 * j + bonus
                # All the columns that the real groups.csv has (3 seeds, small standard deviations).
                rows.append(dict(model=m, modality=mo, fraction=f, variant="supervised", n_seeds=3,
                                 test_acc_mean=f1 + 0.01, test_acc_std=0.01, test_f1_mean=f1, test_f1_std=0.01,
                                 test_ece_mean=0.05, val_f1_mean=f1))
    return pd.DataFrame(rows)


def _per_class(groups: pd.DataFrame) -> pd.DataFrame:
    """Helper: fake per_class.csv with one row per group and class; F1 falls slightly with the class index."""
    rows = []
    for _, g in groups.iterrows():
        for k, c in enumerate(CLASSES):
            # "class" is a reserved Python word, so the column is passed through a dict instead of class=...
            rows.append(dict(model=g["model"], modality=g["modality"], fraction=g["fraction"], variant="supervised",
                             **{"class": c}, precision=0.9, recall=0.9, f1_mean=g["test_f1_mean"] - 0.01 * k,
                             f1_std=0.01, n_seeds=3))
    return pd.DataFrame(rows)


def _pairs(factor_a: str, factor_b: str, **fixed) -> pd.DataFrame:
    """Helper: fake per-seed comparison table (like compare_*_pairs.csv) where setting b beats setting a."""
    rows = []
    for m in MODELS:
        for f in FRACS:
            for s in range(3):
                # b is always better than a: delta grows slightly with the seed number s.
                # `fixed` adds extra constant columns (for example modality="rgb").
                rows.append(dict(model=m, fraction=f, variant="supervised", seed=s, a=factor_a, b=factor_b,
                                 f1_a=0.8, f1_b=0.82 + 0.005 * s, delta_f1=0.02 + 0.005 * s, ci_low=0.0, ci_high=0.04,
                                 **fixed))
    return pd.DataFrame(rows)


def _summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """Helper: average the per-seed pairs into one row per setting (a small copy of summarise_comparison)."""
    # Group by whichever of these identifying columns the pairs table has.
    keys = [c for c in ("model", "modality", "fraction") if c in pairs.columns]
    # Mean scores and delta, std of the delta, number of seeds, and how many seeds favour b.
    return pairs.groupby(keys).agg(f1_a=("f1_a", "mean"), f1_b=("f1_b", "mean"), delta_f1_mean=("delta_f1", "mean"),
                                   delta_f1_std=("delta_f1", "std"), n_seeds=("seed", "nunique"),
                                   seeds_b_better=("delta_f1", lambda d: int((d > 0).sum()))).reset_index()


def test_label_efficiency_and_heatmap(tmp_path):
    """The main line chart (F1 and accuracy versions) and the per-class heat map must be drawn to files."""
    g = _groups()
    # The F1 chart must exist and not be an empty image (more than 1000 bytes).
    assert figures.label_efficiency(g, tmp_path / "a.png").stat().st_size > 1000
    # The accuracy version of the chart is also produced.
    assert figures.label_efficiency(g, tmp_path / "b.png", metric="acc").exists()
    # The per-class heat map at 100% labels is produced.
    assert figures.per_class_heatmap(_per_class(g), 1.0, tmp_path / "c.png").exists()


def test_missing_data_returns_none(tmp_path):
    """With no data every figure function must return None and write no file (instead of crashing)."""
    # A groups table with the right columns but zero rows.
    empty = _groups().iloc[0:0]
    # Each of the four figure functions gets "nothing to plot" input and must answer None.
    assert figures.label_efficiency(empty, tmp_path / "a.png") is None
    assert figures.rgb_vs_ms_gain(pd.DataFrame(), tmp_path / "b.png") is None
    assert figures.pseudo_label_gain({"pl": None, "plb": pd.DataFrame()}, tmp_path / "c.png") is None
    assert figures.confusion_grid({}, tmp_path / "d.png", "t") is None
    # The folder must still be empty: no file was written.
    assert not list(tmp_path.iterdir())


def test_gain_and_pseudo_label_figures(tmp_path):
    """The RGB-versus-13-band gain figure and the pseudo-label gain figure must be created."""
    # Gain of 13 bands over RGB, from fake pairs.
    assert figures.rgb_vs_ms_gain(_pairs("rgb", "ms"), tmp_path / "gain.png").exists()
    # Two pseudo-label variants: plain ("pl", on RGB) and class-balanced ("plb", on 13 bands).
    pl = _pairs("supervised", "pl", modality="rgb")
    plb = _pairs("supervised", "plb", modality="ms")
    assert figures.pseudo_label_gain({"pl": pl, "plb": plb}, tmp_path / "pl.png").exists()


def test_confusion_grid(tmp_path):
    """A grid of confusion matrices (here two identical ones) must be drawn to a file."""
    # A strongly diagonal 10x10 matrix: mostly correct, a little confusion everywhere else.
    cm = np.eye(10) * 0.9 + 0.01
    assert figures.confusion_grid({"A": cm, "B": cm}, tmp_path / "cm.png", "title").exists()


def test_label_spread_keeps_order_and_gap():
    """_spread moves overlapping curve labels apart: the minimum gap holds and far-away labels stay put."""
    # Three labels crowd around 0.9; one is far away at 0.5.
    out = figures._spread([0.9, 0.9, 0.91, 0.5], gap=0.03)
    # The isolated label (the 4th) must not move.
    assert out[3] == 0.5
    # After sorting, each neighbouring pair must be at least `gap` apart (1e-9 allows rounding error).
    s = sorted(out)
    assert all(b - a >= 0.03 - 1e-9 for a, b in zip(s[:-1], s[1:]))


def test_label_spread_stays_below_top():
    """When labels bunch up near the top of the plot, _spread must push them down, never above `top`."""
    # Real scores end close together near the top of the axes (0.975 to 0.99): labels must be pushed
    # down inside the axes, not up above them.
    out = figures._spread([0.975, 0.9863, 0.9876], gap=0.016, top=0.99)
    # The highest label must not exceed the top limit.
    assert max(out) <= 0.99 + 1e-9
    # And the minimum spacing must still be respected between all neighbours.
    s = sorted(out)
    assert all(b - a >= 0.016 - 1e-9 for a, b in zip(s[:-1], s[1:]))
    # Labels that are already far enough apart must be returned unchanged.
    assert figures._spread([0.5, 0.9], gap=0.03, top=1.0) == [0.5, 0.9]  # nothing to move


def test_supervised_table_bolds_best_per_column():
    """The main results table must have 6 rows x 3 columns, with exactly one bold (best) cell per column."""
    t = tables.supervised_table(_groups())
    # 3 models x 2 input types = 6 rows; 3 label fractions = 3 columns.
    assert t.shape == (6, 3)
    # In each column exactly one cell starts with "**" (the best score).
    for c in t.columns:
        assert t[c].str.startswith("**").sum() == 1
    # Row names use readable labels, and cells show "mean ± std".
    assert "ViT-Small, 13 bands" in t.index and "±" in t.iloc[0, 0]


def test_semi_supervised_and_per_class_tables():
    """Check the pseudo-label table, the best-model picker and the per-class table on fake data."""
    # Only the plain pseudo-label comparison is given; the class-balanced one is missing (None).
    comp = {"pl": _summary(_pairs("supervised", "pl", modality="rgb")), "plb": None}
    t = tables.semi_supervised_table(_groups(), comp)
    # 3 models x 3 fractions = 9 rows, and the student beat its teacher in all 3 seeds in every row.
    assert len(t) == 9 and t["Seeds better"].eq("3/3").all()
    # If there is no usable comparison at all, the function returns None instead of an empty table.
    assert tables.semi_supervised_table(_groups(), {"pl": None}) is None
    g = _groups()
    # In the fake data vit_small has the highest scores, so it must be chosen as the best model.
    assert tables.best_model(g) == "vit_small"
    # Per-class table: one row per class (in the standard order) and one column per input type.
    pc = tables.per_class_table(_per_class(g), "vit_small", 1.0)
    assert list(pc.index) == CLASSES and list(pc.columns) == ["RGB (3 bands)", "13 bands"]


def test_markdown_and_latex_escaping(tmp_path):
    """write_table must produce both a Markdown and a LaTeX file, with bold and plus-minus converted for LaTeX."""
    t = tables.supervised_table(_groups())
    # Writes sup.md and sup.tex into tmp_path and returns the two paths.
    paths = tables.write_table(t, tmp_path, "sup", "Test macro-F1 (\\%).")
    # Read both files back as text (md first, then tex).
    md, tex = (p.read_text(encoding="utf-8") for p in paths)
    # Markdown starts with an empty header cell and keeps the ** bold markers.
    assert md.startswith("|  |") and "**" in md
    # LaTeX must use $\pm$ and \textbf{...} instead of the raw "±" and "**", which LaTeX cannot handle.
    assert r"$\pm$" in tex and r"\textbf{" in tex and "**" not in tex and "±" not in tex
    # The table environment has 1 label column + 3 centred columns, and the label for \ref is set.
    assert r"\begin{tabular}{lccc}" in tex and r"\label{tab:sup}" in tex


# Run once with every input file present, once with only groups.csv ("partial" results).
@pytest.mark.parametrize("partial", [False, True])
def test_cli_end_to_end(tmp_path, partial):
    """Run scripts/make_figures.py on fake evaluation files; it must work with full and with partial results."""
    # Build the folder layout that evaluate.py would have written: eval/ and eval/confusion/.
    ev = tmp_path / "eval"
    (ev / "confusion").mkdir(parents=True)
    # groups.csv is the one file that is always present.
    g = _groups()
    g.to_csv(ev / "groups.csv", index=False)
    if not partial:
        # Full case: add the per-class table and the RGB-vs-13-band pairs.
        _per_class(g).to_csv(ev / "per_class.csv", index=False)
        pr = _pairs("rgb", "ms")
        pr.to_csv(ev / "compare_rgb_vs_ms_pairs.csv", index=False)
        # Also add fake confusion matrices for the 100% and 1% runs of the best model, for both inputs.
        for f in (1.0, 0.01):
            for mo in ("rgb", "ms"):
                pd.DataFrame(np.eye(10), index=CLASSES, columns=CLASSES).to_csv(
                    ev / "confusion" / f"vit_small_{mo}_{f * 100:g}pct_supervised.csv")
    # Start the real script as a separate process with temporary input and output folders.
    res = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "make_figures.py"), "--eval-dir", str(ev),
                          "--figures-dir", str(tmp_path / "f"), "--tables-dir", str(tmp_path / "t")],
                         capture_output=True, text=True)
    # Success exit code, and the error text is shown if it crashed.
    assert res.returncode == 0, res.stderr
    # Outputs that only need groups.csv must exist in both cases.
    assert (tmp_path / "f" / "label_efficiency_f1.png").exists()
    assert (tmp_path / "t" / "supervised_f1.tex").exists()
    # The confusion figure needs the confusion files: present only in the full case.
    assert (tmp_path / "f" / "confusion_100pct.png").exists() == (not partial)
    # The script must say it skipped the pseudo-label figure (no pseudo-label results were supplied).
    assert "skipped pseudo-label gains" in res.stdout


def test_cli_needs_groups(tmp_path):
    """Without groups.csv the script must fail with a clear message that tells the user what to run first."""
    # tmp_path is an empty folder, so groups.csv is missing.
    res = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "make_figures.py"), "--eval-dir",
                          str(tmp_path)], capture_output=True, text=True)
    # Non-zero exit code and a helpful hint on stderr.
    assert res.returncode != 0 and "run scripts/evaluate.py first" in res.stderr


def test_single_seed_cell_says_n1_and_figure_draws(tmp_path):
    """One-seed results (the 100% runs) must show "(n=1)" in the table, not NaN or a fake spread."""
    # Protects the 100% runs, which use one seed: groups.csv then has a NaN std. The table must say
    # "(n=1)", never "nan" and never "± 0.0" (no spread was measured), and the label-efficiency figure
    # must still be drawn.
    g = _groups()
    # Select the ResNet-50 / RGB / 100% row and turn it into a one-seed result with undefined (NaN) std.
    one = (g["model"] == "resnet50") & (g["modality"] == "rgb") & (g["fraction"] == 1.0)
    g.loc[one, ["n_seeds", "test_f1_std", "test_acc_std"]] = [1, np.nan, np.nan]
    # Check both the F1 and the accuracy version of the table.
    for metric in ("f1", "acc"):
        cell = tables.supervised_table(g, metric=metric).at["ResNet-50, RGB (3 bands)", "100%"]
        # The cell may end with "**" if it is the bold best one, so strip those before checking "(n=1)".
        assert cell.rstrip("*").endswith("(n=1)") and "nan" not in cell.lower() and "±" not in cell
    # The line charts must still be drawn despite the NaN std.
    assert figures.label_efficiency(g, tmp_path / "f1.png").stat().st_size > 1000
    assert figures.label_efficiency(g, tmp_path / "acc.png", metric="acc").stat().st_size > 1000


def test_missing_model_or_cell_in_supervised_table(tmp_path):
    """Partial results must not crash the table: missing rows are left out, single missing cells show a dash."""
    # Protects partial results: a model with no 13-band runs gets no row, and a single missing cell
    # shows an en dash. Neither may crash the table, and the figure must still be drawn.
    g = _groups()
    # Remove all 13-band runs of vit_small.
    no_ms = g[~((g["model"] == "vit_small") & (g["modality"] == "ms"))]
    t = tables.supervised_table(no_ms)
    # 5 rows instead of 6: only the missing 13-band row disappears, the RGB row of vit_small stays.
    assert t.shape == (5, 3) and "ViT-Small, 13 bands" not in t.index and "ViT-Small, RGB (3 bands)" in t.index
    # The figure must still be produced from the incomplete data.
    assert figures.label_efficiency(no_ms, tmp_path / "a.png").exists()
    # Now remove only one single cell: ResNet-50, 13 bands, 10% labels.
    no_cell = g[~((g["model"] == "resnet50") & (g["modality"] == "ms") & (g["fraction"] == 0.1))]
    t = tables.supervised_table(no_cell)
    # All rows remain; the missing cell shows an en dash while its neighbour in the 1% column is filled.
    assert t.shape == (6, 3) and t.at["ResNet-50, 13 bands", "10%"] == "–"
    assert t.at["ResNet-50, 13 bands", "1%"] != "–"


def test_empty_groups_return_none(tmp_path):
    """Empty inputs (no results yet, or only non-supervised rows) must give None and create no files."""
    # Protects a run with no results yet: with an empty groups table (or no supervised rows) the
    # figure functions return None and write no file.
    empty = _groups().iloc[0:0]
    # F1 and accuracy line charts, and the heat map from an empty per-class table.
    assert figures.label_efficiency(empty, tmp_path / "a.png") is None
    assert figures.label_efficiency(empty, tmp_path / "b.png", metric="acc") is None
    assert figures.per_class_heatmap(_per_class(_groups()).iloc[0:0], 1.0, tmp_path / "c.png") is None
    pl_only = _groups().assign(variant="pl")  # rows exist, but none are supervised
    assert figures.label_efficiency(pl_only, tmp_path / "d.png") is None
    # No file may have been created by any of the calls above.
    assert not list(tmp_path.iterdir())


def test_wide_table_spans_both_columns(tmp_path):
    """write_table(wide=True) must produce a two-column-wide LaTeX float (table*), the default a normal one."""
    # The pseudo-label table has 8 columns and would run off one column of a two-column page, so
    # write_table(wide=True) must produce a table* float; the default stays a one-column table.
    # A tiny 2-column table is enough: only the LaTeX wrapper is being checked.
    df = pd.DataFrame({"a": ["1"], "b": ["2"]})
    tables.write_table(df, tmp_path, "narrow", "c")
    tables.write_table(df, tmp_path, "wide", "c", wide=True)
    # Default file begins with the one-column "table" environment.
    assert (tmp_path / "narrow.tex").read_text().startswith(r"\begin{table}[t]")
    # Wide file begins and ends with the two-column "table*" environment.
    wide = (tmp_path / "wide.tex").read_text()
    assert wide.startswith(r"\begin{table*}[t]") and wide.rstrip().endswith(r"\end{table*}")


def test_generated_latex_has_no_bare_percent_sign(tmp_path):
    """A bare % in a generated .tex file starts a LaTeX comment and swallows the closing brace (compile error)."""
    # Fake evaluation folder with groups, per-class scores and confusion files for the 1% fraction.
    ev = tmp_path / "eval"
    (ev / "confusion").mkdir(parents=True)
    g = _groups()
    g.to_csv(ev / "groups.csv", index=False)
    _per_class(g).to_csv(ev / "per_class.csv", index=False)
    # Run the real script for the 1% fraction: its per-class table caption contains the fraction label "1%".
    res = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "make_figures.py"), "--eval-dir", str(ev),
                          "--figures-dir", str(tmp_path / "f"), "--tables-dir", str(tmp_path / "t"),
                          "--fraction", "0.01"], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    # The per-class table for 1% must exist, and every percent sign in every .tex must be escaped as \%.
    tex_files = list((tmp_path / "t").glob("*.tex"))
    assert (tmp_path / "t" / "per_class_f1_1pct.tex").exists()
    for f in tex_files:
        text = f.read_text(encoding="utf-8")
        # Any % not preceded by a backslash is a bare percent sign.
        bare = [i for i, ch in enumerate(text) if ch == "%" and (i == 0 or text[i - 1] != "\\")]
        assert not bare, f"{f.name} contains a bare % (LaTeX comment character)"

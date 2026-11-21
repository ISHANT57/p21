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

# ====================================================================================================
# CONCEPTS USED IN THIS FILE (study notes)
# ====================================================================================================
# - unit test: a check of ONE function on small made-up input (e.g. _spread, supervised_table, write_table).
# - end-to-end (CLI) test: test_cli_end_to_end, test_cli_needs_groups and test_generated_latex_... start
#     scripts/make_figures.py as a separate program with subprocess and check its exit code, messages and files.
# - regression test: several tests are named after a bug or a promise they protect: one-seed cells say (n=1),
#     partial results do not crash, empty input returns None, no bare % in LaTeX.
# - assert: a statement that must be true; if false, pytest marks the test as FAILED.
# - tmp_path (pytest fixture): a fresh empty temporary folder per test: figures, tables and fake CSV files are
#     written there, never into results/.
# - pytest.mark.parametrize: test_cli_end_to_end runs twice: partial=False (all input files present) and
#     partial=True (only groups.csv).
# - synthetic (fake) evaluation tables: _groups, _per_class, _pairs and _summary build DataFrames shaped like
#     groups.csv, per_class.csv and compare_*_pairs.csv, with made-up scores. No training or real results needed.
# - groups.csv / per_class.csv / pairs: tables written by scripts/evaluate.py: groups = mean and std of the test
#     scores per setting (model, input, fraction, variant); per_class = F1 per class; pairs = one row per seed of a
#     paired comparison with delta_f1.
# - label fraction (1%, 10%, 100%): the share of train images treated as labelled; the x axis of the label-
#     efficiency figure.
# - macro-F1 and accuracy: macro-F1 = mean of the per-class F1; accuracy = share correct. Figures and tables can
#     show either (metric='f1' or 'acc').
# - mean +- std over seeds, (n=1), NaN: a cell is 'mean ± std' over the 3 seeds. With one seed the std is NaN; the
#     table then writes '(n=1)' (never 'nan' or '± 0.0') and the figure draws a zero-length error bar.
# - matplotlib figure / PNG file: figures.* draw with matplotlib and write a PNG. The tests check that the file
#     exists and is larger than 1000 bytes (an empty or broken image would be tiny).
# - label-efficiency figure: line chart of test score against label fraction, one line per model, one panel per
#     input type (RGB, 13 bands).
# - heat map: per-class F1 as a colour grid (classes down, model x input across), numbers printed in the cells.
# - confusion matrix grid: row-normalised confusion matrices (true class vs predicted class) drawn side by side.
# - gain figures: bars/dots of the F1 difference: 13 bands minus RGB, and pseudo-label student minus its teacher;
#     one dot per seed.
# - pseudo-label variants pl / plb: 'pl' = pseudo-labels with one global threshold tau; 'plb' = class-balanced
#     selection.
# - returns None on missing data: each figure/table function returns None (and writes nothing) when its input is
#     empty, so the script can run on partial results.
# - _spread (label placement): moves overlapping curve labels apart so each pair is at least `gap` apart; with `top`
#     it pushes labels DOWN instead of above the axes.
# - bold best per column: supervised_table wraps the highest mean of every column in '**...**'.
# - Markdown and LaTeX output (write_table): the same table is written as .md and .tex. LaTeX needs escaping: ± ->
#     $\pm$, **x** -> \textbf{x}, % -> \%. 'lccc' means one left-aligned and three centred columns;
#     \label{tab:<name>} lets the report refer to the table.
# - table vs table* (wide=True): table* is a LaTeX float that spans both columns of a two-column page.
# - bare percent sign: in LaTeX an unescaped % starts a comment and swallows the rest of the line, including a
#     closing brace, so compilation fails.
# - en dash for a missing cell: a setting that was not run shows '–' instead of crashing the table.
# - best_model: the model with the highest mean test F1 (over RGB and 13 bands) at one fraction; its confusion
#     matrices and per-class views are shown in the report.

# ====================================================================================================
# Q&A ABOUT THIS FILE (viva practice)
# ====================================================================================================
# Q: Which part of the project does this file protect?
# A: The report output stage (Phase 7): eurosat_le/figures.py, eurosat_le/tables.py and scripts/make_figures.py.
#    They turn the evaluation CSVs into the figures and LaTeX/Markdown tables of the report.
# Q: How do you run it?
# A: pytest tests/test_figures.py from the repo root (needs matplotlib and pandas). The CLI tests start
#    scripts/make_figures.py in a subprocess.
# Q: Why are no real results needed?
# A: The builders only read tables. _groups() makes 18 rows (3 models x 2 inputs x 3 fractions) with rising scores,
#    so the best model (ViT-Small) and the best cell are known in advance.

# The models and label fractions used by the fake result tables below.
MODELS = ["scratch_cnn", "resnet50", "vit_small"]
FRACS = [0.01, 0.1, 1.0]


# Q: What does _groups build?
# A: A fake groups.csv with 18 supervised rows (3 models x rgb/ms x 3 fractions), 3 seeds each. The score is 0.5 +
#    0.1 x model index + 0.12 x fraction index, plus 0.01 for 13 bands.
# Q: Why is it enough?
# A: Scores rise with the model and with the label fraction, so ViT-Small with 13 bands at 100% is the known best
#    cell. Tests can predict which cell is bold and which model is best.
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


# Q: What does _per_class build?
# A: A fake per_class.csv: for every row of groups and every class, one row with f1_mean = the group's F1 minus 0.01
#    x class index (precision and recall 0.9, std 0.01, 3 seeds).
# Q: Why is the column 'class' passed through a dict?
# A: class is a reserved word in Python, so class=... cannot be written as a keyword argument; **{'class': c} works.
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


# Q: What does _pairs build?
# A: A fake compare_*_pairs.csv: 3 models x 3 fractions x 3 seeds, where setting b always beats a (delta_f1 = 0.02 +
#    0.005 x seed, interval 0.0 to 0.04). **fixed adds constant columns such as modality='rgb'.
# Q: Why is it enough?
# A: The gain figures and the semi-supervised table only need per-seed rows with f1_a, f1_b and delta_f1.
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


# Q: What does _summary do?
# A: A small copy of summarise_comparison: groups the pairs by model/modality/fraction and gives mean F1 of each
#    side, mean and std of delta_f1, number of seeds and how many seeds favour b.
# Q: Why a copy and not the real function?
# A: The test then does not depend on eval/evaluate.py; it only provides input in the format
#    tables.semi_supervised_table expects.
def _summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """Helper: average the per-seed pairs into one row per setting (a small copy of summarise_comparison)."""
    # Group by whichever of these identifying columns the pairs table has.
    keys = [c for c in ("model", "modality", "fraction") if c in pairs.columns]
    # Mean scores and delta, std of the delta, number of seeds, and how many seeds favour b.
    return pairs.groupby(keys).agg(f1_a=("f1_a", "mean"), f1_b=("f1_b", "mean"), delta_f1_mean=("delta_f1", "mean"),
                                   delta_f1_std=("delta_f1", "std"), n_seeds=("seed", "nunique"),
                                   seeds_b_better=("delta_f1", lambda d: int((d > 0).sum()))).reset_index()


# Q: What does this test check?
# A: label_efficiency writes a PNG larger than 1000 bytes for the F1 version and a file for metric='acc';
#    per_class_heatmap writes the heat map for 100% labels.
# Q: Why check the file size?
# A: A figure that failed to draw would produce an empty or tiny file; more than 1000 bytes shows a real image.
# Q: Which functions does it exercise?
# A: figures.label_efficiency and figures.per_class_heatmap (and viz styling, _save).
def test_label_efficiency_and_heatmap(tmp_path):
    """The main line chart (F1 and accuracy versions) and the per-class heat map must be drawn to files."""
    g = _groups()
    # The F1 chart must exist and not be an empty image (more than 1000 bytes).
    assert figures.label_efficiency(g, tmp_path / "a.png").stat().st_size > 1000
    # The accuracy version of the chart is also produced.
    assert figures.label_efficiency(g, tmp_path / "b.png", metric="acc").exists()
    # The per-class heat map at 100% labels is produced.
    assert figures.per_class_heatmap(_per_class(g), 1.0, tmp_path / "c.png").exists()


# Q: What does this test check?
# A: With empty or missing input, label_efficiency, rgb_vs_ms_gain, pseudo_label_gain and confusion_grid all return
#    None, and no file is created.
# Q: Why does it matter?
# A: make_figures.py must work on partial results (e.g. before pseudo-label runs exist). Returning None lets it
#    print 'skipped ...' instead of crashing.
# Q: How are the empty inputs built?
# A: _groups().iloc[0:0] (right columns, zero rows), an empty DataFrame, a dict whose values are None or empty, and
#    an empty dict.
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


# Q: What does this test check?
# A: rgb_vs_ms_gain draws from fake pairs (rgb vs ms), and pseudo_label_gain draws from two variants: pl on RGB and
#    plb (class-balanced) on 13 bands.
# Q: Why does it matter?
# A: These are the report's key gain figures (what the extra bands add; what pseudo-labelling adds). The test makes
#    sure they can be produced for both variants.
def test_gain_and_pseudo_label_figures(tmp_path):
    """The RGB-versus-13-band gain figure and the pseudo-label gain figure must be created."""
    # Gain of 13 bands over RGB, from fake pairs.
    assert figures.rgb_vs_ms_gain(_pairs("rgb", "ms"), tmp_path / "gain.png").exists()
    # Two pseudo-label variants: plain ("pl", on RGB) and class-balanced ("plb", on 13 bands).
    pl = _pairs("supervised", "pl", modality="rgb")
    plb = _pairs("supervised", "plb", modality="ms")
    assert figures.pseudo_label_gain({"pl": pl, "plb": plb}, tmp_path / "pl.png").exists()


# Q: What does this test check?
# A: confusion_grid draws two identical 10 x 10 matrices side by side into a PNG file.
# Q: How is the matrix built?
# A: np.eye(10) * 0.9 + 0.01: 0.91 on the diagonal and 0.01 elsewhere; every row sums to 1.0, like a row-normalised
#    confusion matrix.
def test_confusion_grid(tmp_path):
    """A grid of confusion matrices (here two identical ones) must be drawn to a file."""
    # A strongly diagonal 10x10 matrix: mostly correct, a little confusion everywhere else.
    cm = np.eye(10) * 0.9 + 0.01
    assert figures.confusion_grid({"A": cm, "B": cm}, tmp_path / "cm.png", "title").exists()


# Q: What does this test check?
# A: _spread([0.9, 0.9, 0.91, 0.5], gap=0.03) leaves the isolated label (0.5) unchanged and separates the three
#    crowded labels by at least 0.03.
# Q: Why does it matter?
# A: Direct labels at the end of the curves must not overlap on the figure; labels that are already apart must not
#    move, otherwise a label could end up next to the wrong line.
def test_label_spread_keeps_order_and_gap():
    """_spread moves overlapping curve labels apart: the minimum gap holds and far-away labels stay put."""
    # Three labels crowd around 0.9; one is far away at 0.5.
    out = figures._spread([0.9, 0.9, 0.91, 0.5], gap=0.03)
    # The isolated label (the 4th) must not move.
    assert out[3] == 0.5
    # After sorting, each neighbouring pair must be at least `gap` apart (1e-9 allows rounding error).
    s = sorted(out)
    assert all(b - a >= 0.03 - 1e-9 for a, b in zip(s[:-1], s[1:]))


# Q: What does this test check?
# A: With bunched labels near the top, _spread keeps every label <= top (0.99) while still keeping the minimum gap;
#    labels already far apart are returned unchanged.
# Q: Why does it matter?
# A: Real scores can end close together near the top (about 0.975 to 0.99), so pushing labels UP would put them
#    above the axes. The function pins the highest label at `top` and pushes the others down.
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


# Q: What does this test check?
# A: supervised_table has 6 rows (3 models x 2 inputs) and 3 columns (1%, 10%, 100%), exactly one bold '**' cell per
#    column, the row label 'ViT-Small, 13 bands', and '±' in the cells.
# Q: Why does it matter?
# A: This is the main results table of the report; bolding the best value per column and using readable row names is
#    part of its format.
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


# Q: What does this test check?
# A: semi_supervised_table with only the 'pl' comparison has 9 rows (3 models x 3 fractions) with 'Seeds better' =
#    '3/3' everywhere, and returns None when there is no usable comparison; best_model picks 'vit_small';
#    per_class_table has the 10 classes as rows and the columns 'RGB (3 bands)' and '13 bands'.
# Q: Why does it matter?
# A: Missing variants (here plb is None) must be skipped without errors; the best-model picker decides which model's
#    per-class views appear in the report.
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


# Q: What does this test check?
# A: write_table writes sup.md and sup.tex. The Markdown starts with an empty header cell and keeps '**'. The LaTeX
#    contains $\pm$ and \textbf{...}, no raw '**' or '±', the 'lccc' tabular and \label{tab:sup}.
# Q: Why does it matter?
# A: LaTeX cannot print '±' or Markdown bold markers directly; one display table feeds both outputs, so the
#    conversion must be right or the report will not compile or will look wrong.
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


# Q: What does test_cli_end_to_end check?
# A: scripts/make_figures.py, run with fake eval files, exits with code 0 in both cases and writes
#    label_efficiency_f1.png and supervised_f1.tex; confusion_100pct.png exists only in the full case; stdout says
#    'skipped pseudo-label gains'.
# Q: What are 'full' and 'partial'?
# A: Full: groups.csv, per_class.csv, compare_rgb_vs_ms_pairs.csv and confusion CSVs (identity matrices) for the
#    best model ViT-Small at 100% and 1%. Partial: only groups.csv.
# Q: Why does it matter?
# A: The script must never crash because an optional input is missing; it skips that output and reports it.
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


# Q: What does this test check?
# A: In an empty folder (no groups.csv) the script exits with a non-zero code and prints 'run scripts/evaluate.py
#    first' on stderr.
# Q: Why does it matter?
# A: groups.csv is the one mandatory input. A clear message tells the user which step to run first, instead of a
#    confusing traceback.
def test_cli_needs_groups(tmp_path):
    """Without groups.csv the script must fail with a clear message that tells the user what to run first."""
    # tmp_path is an empty folder, so groups.csv is missing.
    res = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "make_figures.py"), "--eval-dir",
                          str(tmp_path)], capture_output=True, text=True)
    # Non-zero exit code and a helpful hint on stderr.
    assert res.returncode != 0 and "run scripts/evaluate.py first" in res.stderr


# Q: What does this test check?
# A: A one-seed cell (ResNet-50, RGB, 100%, std = NaN) is shown as '... (n=1)' in the F1 and accuracy tables, never
#    as 'nan' or with '±', and label_efficiency still draws the figure.
# Q: Why does it matter?
# A: The 100% runs use one seed. '± 0.0' would wrongly suggest the seeds agreed exactly; honest reporting says no
#    spread was measured.
# Q: How is the cell found?
# A: A boolean mask selects the row; its n_seeds becomes 1 and both std columns become NaN. cell.rstrip('*') removes
#    the bold markers in case it is the best cell.
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


# Q: What does this test check?
# A: Without ViT-Small 13-band rows the table has 5 rows (RGB row kept) and the figure is still drawn; with one cell
#    removed (ResNet-50, 13 bands, 10%) the table keeps 6 rows, that cell shows '–' and the 1% cell is filled.
# Q: Why does it matter?
# A: Results arrive in stages (Colab lanes). The report code must produce tables and figures from incomplete data
#    instead of failing.
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


# Q: What does this test check?
# A: label_efficiency returns None for an empty table (F1 and accuracy) and for a table with only pseudo-label rows;
#    per_class_heatmap returns None for an empty per-class table; no file is created.
# Q: Why does it matter?
# A: A run with no results yet must not crash or leave empty image files behind.
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


# Q: What does this test check?
# A: write_table(..., wide=True) produces a LaTeX file that starts with \begin{table*}[t] and ends with
#    \end{table*}; the default starts with \begin{table}[t].
# Q: Why does it matter?
# A: The report uses a two-column layout. A table with many columns (the pseudo-label table) would run off a single
#    column, so it must be a table* float spanning both.
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


# Q: What does this test check?
# A: After running make_figures.py with --fraction 0.01, per_class_f1_1pct.tex exists and every % in every generated
#    .tex file is escaped as \%.
# Q: What bug would make it fail?
# A: A caption such as 'at 1% of the labels' with an unescaped %. In LaTeX it starts a comment that swallows the
#    closing brace of the caption, so the report would not compile.
# Q: How is a bare % found?
# A: For each file, find every '%' whose previous character is not a backslash; the list must be empty.
def test_generated_latex_has_no_bare_percent_sign(tmp_path):
    """A bare % in a generated .tex file starts a LaTeX comment and swallows the closing brace (compile error)."""
    # Fake evaluation folder with groups.csv, per_class.csv and an (empty) confusion/ sub-folder.
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

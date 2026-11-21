"""Report tables (Markdown and LaTeX) built from the Phase 6 tables. Phase 7.

Each function returns a DataFrame of display strings; `write_table` saves it as
<name>.md and <name>.tex. A cell is "mean ± std" over the seeds. The best value in
each column is bold in the LaTeX/Markdown output.

Pipeline position: Phase 7. Reads the DataFrames loaded from results/summary/eval/*.csv by
`scripts/make_figures.py` (groups.csv, per_class.csv, compare_supervised_vs_<variant>.csv) and
writes results/tables/<name>.md and <name>.tex for the report.

Design decisions:
- Scores are shown in percent with one decimal. When only one seed exists the std is NaN and is shown
  as 0.0 (never as "nan").
- Bolding is done with Markdown "**x**" markers in the cell strings; `to_latex` converts them to
  LaTeX bold commands and escapes the Unicode symbols, so the same DataFrame feeds both outputs.
- The LaTeX uses booktabs rules (toprule, midrule, bottomrule), so the report preamble must load
  the booktabs package. The first column is left-aligned, all others are centred.
- Model, input and fraction names come from `eurosat_le.figures` so tables and figures use the same labels.
- Each table builder returns None when its data is missing, so the script can skip it.
"""

# Lets type hints such as "pd.DataFrame | None" be written the modern way on all supported Python versions.
from __future__ import annotations

# Path is used for the output folder of write_table.
from pathlib import Path

# pandas holds every table; it is also used to detect NaN values.
import pandas as pd

# CLASSES is the ordered list of the ten land-cover class names (row order of the per-class table).
from eurosat_le.constants import CLASSES
# Shared display names and the fixed model order, so tables and figures use identical labels.
from eurosat_le.figures import FRACTION_LABEL, MODEL_LABEL, MODELS, MODALITY_LABEL, metric_columns


def _pm(mean: float, std: float, scale: float = 100.0, digits: int = 1) -> str:
    """Format "mean ± std" after multiplying both by `scale` (100 turns 0-1 scores into percent).

    A NaN std means a single seed. No spread was measured then, so the cell reads "mean (n=1)" and not
    "± 0.0", which would suggest the seeds agreed exactly.

    Args:
        mean: the mean score (0-1 by default).
        std: the standard deviation over seeds, or NaN for a single seed.
        scale: factor applied to both numbers.
        digits: number of decimals to show.
    Returns:
        The text for one table cell.
    """
    # Single seed: show the mean only and mark it, because no spread exists.
    if pd.isna(std):
        return f"{mean * scale:.{digits}f} (n=1)"
    # Several seeds: show mean and spread.
    return f"{mean * scale:.{digits}f} ± {std * scale:.{digits}f}"


def _frac(f: float) -> str:
    """Label fraction as a column header such as "1%", "10%", "100%" (rounded to match the dict keys).

    Args:
        f: label fraction between 0 and 1.
    Returns:
        The header text; a fraction without a fixed label is shown as a rounded percentage.
    """
    # Round first so values like 0.10000000001 still find the key 0.1.
    return FRACTION_LABEL.get(round(f, 4), f"{f:.0%}")


def _bold_best(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Bold the highest value per column (raw has the numbers behind the strings).

    `df` holds display strings and `raw` the same-shaped floats; the cell where `raw` is largest in each
    column gets "**...**" around its string. Columns that are missing from `raw` or all NaN are left
    alone. Ties are resolved by `idxmax`, which picks the first row. Returns a modified copy.

    Args:
        df: the table of display strings.
        raw: a table of the real numbers with the same rows and columns.
    Returns:
        A copy of `df` in which the best cell of each column is wrapped in "**" bold markers.
    """
    # Work on a copy so the caller's table is not changed.
    out = df.copy()
    # Handle each column separately: the best value is chosen within a column.
    for c in out.columns:
        # Only columns with real numbers can have a best value.
        if c in raw and raw[c].notna().any():
            # Find the row with the highest raw number and wrap its text in Markdown bold markers.
            out.loc[raw[c].idxmax(), c] = f"**{out.loc[raw[c].idxmax(), c]}**"
    return out


def supervised_table(groups: pd.DataFrame, metric: str = "f1") -> pd.DataFrame | None:
    """Rows: model x input. Columns: label fractions. Cell: test score in %, mean ± std over seeds.

    Args:
        groups: groups.csv as a DataFrame; only variant == "supervised" rows are used.
        metric: "f1" for test macro-F1, anything else for test accuracy.

    Returns None if there are no supervised rows. Row order follows MODELS, with RGB before 13-band;
    combinations with no data are skipped, and individual missing cells show an en dash. The best
    mean in each fraction column (across all rows) is bold.
    """
    # Keep only the supervised rows; pseudo-label students are compared separately.
    g = groups[groups["variant"] == "supervised"]
    # Nothing supervised means no table.
    if g.empty:
        return None
    # Names of the mean and std columns for the chosen metric (F1 or accuracy).
    mean, std = metric_columns(metric)
    # The label fractions that exist, smallest first; they become the table columns.
    fr = sorted(g["fraction"].unique())
    # The (model, input) combinations that have data, in the fixed model order with RGB before 13-band.
    idx = [(m, mo) for m in MODELS for mo in ("rgb", "ms") if ((g["model"] == m) & (g["modality"] == mo)).any()]
    # Empty table of display strings: one row per combination, one column per fraction.
    disp = pd.DataFrame(index=[f"{MODEL_LABEL[m]}, {MODALITY_LABEL[mo]}" for m, mo in idx],
                        columns=[_frac(f) for f in fr], dtype=object)
    # A second table of the same shape that keeps the real numbers, needed to find the best cell per column.
    raw = pd.DataFrame(index=disp.index, columns=disp.columns, dtype=float)
    # Fill both tables cell by cell.
    for (m, mo), name in zip(idx, disp.index):
        for f in fr:
            # The single groups.csv row for this model, input and fraction (empty if it was not run).
            r = g[(g["model"] == m) & (g["modality"] == mo) & (g["fraction"] == f)]
            if not r.empty:
                # Display text "mean ± std" and the plain mean for the bold comparison.
                disp.at[name, _frac(f)] = _pm(r[mean].iloc[0], r[std].iloc[0])
                raw.at[name, _frac(f)] = r[mean].iloc[0]
    # Show an en dash in cells without data, and bold the best value in each column.
    return _bold_best(disp.fillna("–"), raw)


def semi_supervised_table(groups: pd.DataFrame, compare: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """Teacher vs student, matched on seed: macro-F1 (%) and the change in points.

    `compare` maps a variant ('pl', 'plb') to its summarised compare_supervised_vs_<variant> table
    (one row per model/modality/fraction, already averaged over seeds). `groups` is accepted but not used.
    Delta is student minus supervised teacher in macro-F1 points: mean +- std over seeds, std shown as
    0.0 if NaN. "Seeds better" is "<seeds where the student beat its teacher>/<seeds>".
    Returns None if every table is missing or empty; the row order is variant, then model, modality, fraction.
    """
    # One display row per (variant, model, input, fraction) combination.
    rows = []
    # Go through the pseudo-label variants (plain "pl" and class-balanced "plb").
    for variant, c in compare.items():
        # Skip a variant whose summary table is missing or empty.
        if c is None or c.empty:
            continue
        # Visit the summary rows in a stable order: model, input, fraction.
        for _, r in c.sort_values(["model", "modality", "fraction"]).iterrows():
            # Turn the summary row into display text (percent with one decimal).
            rows.append({
                "Model": MODEL_LABEL.get(r["model"], r["model"]),
                "Input": MODALITY_LABEL[r["modality"]],
                "Labels": _frac(r["fraction"]),
                "Variant": "balanced" if variant == "plb" else "global τ",
                # f1_a is the supervised teacher's F1 and f1_b the student's F1.
                "Supervised F1": f"{r['f1_a'] * 100:.1f}",
                "Student F1": f"{r['f1_b'] * 100:.1f}",
                # The change in points; with one seed there is no std, so mark it as (n=1).
                "Δ (points)": (f"{r['delta_f1_mean'] * 100:+.1f} (n=1)" if pd.isna(r["delta_f1_std"])
                              else f"{r['delta_f1_mean'] * 100:+.1f} ± {r['delta_f1_std'] * 100:.1f}"),
                # How many seeds the student beat its teacher, out of all seeds.
                "Seeds better": f"{int(r['seeds_b_better'])}/{int(r['n_seeds'])}",
            })
    # Return the table, or None when no variant had any data.
    return pd.DataFrame(rows) if rows else None


def per_class_table(per_class: pd.DataFrame, model: str, fraction: float) -> pd.DataFrame | None:
    """Per-class test F1 (%) for one model, RGB and 13-band, at one label fraction.

    Rows are the classes in CLASSES order (index named "Class"); one column per input type that has
    data, cell = mean +- std over seeds, "–" for a class missing in that column. Supervised rows only.
    Returns None if nothing matches. `fraction` is an exact float match.
    """
    # Keep the supervised rows of the chosen model at the chosen label fraction.
    d = per_class[(per_class["variant"] == "supervised") & (per_class["model"] == model)
                  & (per_class["fraction"] == fraction)]
    # Nothing matches: no table.
    if d.empty:
        return None
    # Start with one empty row per class, in the fixed class order.
    out = pd.DataFrame(index=CLASSES)
    # One column for RGB and one for the 13-band input.
    for mo in ("rgb", "ms"):
        # The rows for this input, looked up by class name.
        x = d[d["modality"] == mo].set_index("class")
        # Only add the column if this input has data.
        if not x.empty:
            # For every class write "mean ± std" of its F1, or an en dash if the class is missing.
            out[MODALITY_LABEL[mo]] = [_pm(x.at[c, "f1_mean"], x.at[c, "f1_std"]) if c in x.index else "–"
                                       for c in CLASSES]
    # Name the row labels "Class" so the first column gets that header in the output.
    out.index.name = "Class"
    return out


def best_model(groups: pd.DataFrame, fraction: float = 1.0) -> str | None:
    """Model with the highest mean test macro-F1 over both inputs, supervised, at one fraction.

    The score is the average of the model's test_f1_mean over its RGB and 13-band rows. Returns the model
    name, or None if there is no supervised row at that fraction. Used to pick which model's confusion
    matrices and per-class views are shown in the report.
    """
    # Keep the supervised rows at the requested fraction.
    g = groups[(groups["variant"] == "supervised") & (groups["fraction"] == fraction)]
    # No such rows: there is no best model.
    if g.empty:
        return None
    # Average each model's F1 over its inputs and return the name of the highest one.
    return g.groupby("model")["test_f1_mean"].mean().idxmax()


def to_markdown(df: pd.DataFrame) -> str:
    """Render a display DataFrame as a pipe-delimited Markdown table (string ends with a newline).

    A named or non-default index becomes the first column (an unnamed one gets an empty header).
    Cell strings are written as they are, so existing "**bold**" markers render as bold.
    """
    # If the row labels carry information (a name or non-numbered index), make them the first column.
    d = df.reset_index() if df.index.name or not isinstance(df.index, pd.RangeIndex) else df
    # An unnamed index column is called "index" by pandas; give it an empty header instead.
    d = d.rename(columns={"index": ""})
    # The header row, the separator row under it, and then one line per table row.
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    # Join everything with newlines and end with a final newline.
    return "\n".join([head, sep, *body]) + "\n"


def to_latex(df: pd.DataFrame, caption: str, label: str, wide: bool = False, font: str = r"\small") -> str:
    """Render a display DataFrame as a booktabs `table` float (string ends with a newline).

    With `wide=True` the float is a `table*`, which spans both columns of a two-column page; use it for
    tables with many columns (the pseudo-label table), which would run off a single column.

    `caption` is inserted as given (the caller writes LaTeX, e.g. an escaped percent sign and a math
    plus-minus); `label` is the LaTeX label key. The index becomes the first column (header "Setting" if
    it has no name).

    Args:
        df: the table of display strings.
        caption: LaTeX caption text.
        label: LaTeX label used to refer to the table from the report text.
        wide: True for a table that spans both page columns.
        font: LaTeX font-size command applied to the table.
    Returns:
        The LaTeX source of the table.
    """
    # Make a meaningful index the first column, as in to_markdown.
    d = df.reset_index() if df.index.name or not isinstance(df.index, pd.RangeIndex) else df
    # An unnamed index column gets the header "Setting".
    d = d.rename(columns={"index": "Setting"})

    def esc(s) -> str:
        """Escape one cell/header: Unicode symbols to math or ASCII, % and _ escaped, **x** to bold.

        Args:
            s: the cell or header value (anything that can be turned into text).
        Returns:
            Text that is safe to put into LaTeX.
        """
        # Replace the Unicode symbols by their LaTeX math forms and escape the percent sign, which starts a
        # comment in LaTeX.
        s = str(s).replace("±", r"$\pm$").replace("Δ", r"$\Delta$").replace("τ", r"$\tau$").replace("%", r"\%")
        # Use plain ASCII dashes for the minus sign and en dash, and escape underscores (special in LaTeX).
        s = s.replace("−", "-").replace("–", "--").replace("_", r"\_")
        # Turn each pair of "**" markers into a \textbf{...} command, one pair at a time.
        while "**" in s:  # **bold** -> \textbf{bold}
            a = s.index("**")
            b = s.index("**", a + 2)
            s = s[:a] + r"\textbf{" + s[a + 2:b] + "}" + s[b + 2:]
        return s

    # Column layout: the first column is left-aligned, all the other columns are centred.
    cols = "l" + "c" * (len(d.columns) - 1)
    # A two-column page needs the starred float so the table can span both columns.
    env = "table*" if wide else "table"
    # The opening lines: float, caption, label, font size, table header with rules and the column names.
    lines = [rf"\begin{{{env}}}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             font, rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(esc(c) for c in d.columns) + r" \\", r"\midrule"]
    # One line per table row, cells separated by "&" and each line ended with "\\".
    lines += [" & ".join(esc(v) for v in row) + r" \\" for row in d.itertuples(index=False)]
    # The closing lines: bottom rule and the end of the table and float.
    lines += [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"]
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, out_dir: Path, name: str, caption: str, wide: bool = False,
                font: str = r"\small") -> list[Path]:
    """Write `df` to <out_dir>/<name>.md and <name>.tex (creating out_dir) and return both paths.

    The LaTeX label is "tab:<name>". The caption is used in the .tex file only. `wide` makes the LaTeX
    float span both columns (see `to_latex`).
    """
    # Make sure the output folder exists before writing into it.
    out_dir.mkdir(parents=True, exist_ok=True)
    # The two output file paths.
    md, tex = out_dir / f"{name}.md", out_dir / f"{name}.tex"
    # Write the Markdown version (for quick reading) and the LaTeX version (for the report).
    md.write_text(to_markdown(df), encoding="utf-8")
    tex.write_text(to_latex(df, caption, f"tab:{name}", wide, font), encoding="utf-8")
    # Return both paths so the caller can list the written files.
    return [md, tex]

"""Report tables (Markdown and LaTeX) built from the Phase 6 tables. Phase 7.

Each function returns a DataFrame of display strings; `write_table` saves it as
<name>.md and <name>.tex. A cell is "mean ± std" over the seeds. The best value in
each column is bold in the LaTeX/Markdown output.

Pipeline position: Phase 7. Reads the DataFrames loaded from results/summary/eval/*.csv by
`scripts/make_figures.py` (groups.csv, per_class.csv, compare_supervised_vs_<variant>.csv) and
writes results/tables/<name>.md and <name>.tex for the report.

Design decisions:
- Scores are shown in percent with one decimal. When only one seed exists the std is NaN and the cell
  reads "x (n=1)" (never "nan", and never "± 0.0", which would suggest the seeds agreed exactly).
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


# ======================================================================
# CONCEPTS USED IN THIS FILE
# ======================================================================
# display table: a DataFrame in which every cell is already formatted text (such as '97.8 ± 0.3'), so the same table can
#   be written as Markdown and as LaTeX. Built by supervised_table, semi_supervised_table and per_class_table.
# mean ± std over seeds: the average over the repeated runs (seeds) and their sample standard deviation. A NaN std (one
#   seed) is shown as '(n=1)' in _pm. The numbers come from groups.csv.
# macro-F1 and accuracy: macro-F1 averages the per-class F1 (harmonic mean of precision and recall) with equal weight
#   per class; accuracy is the share of correct predictions. metric_columns picks the columns.
# percent with one decimal: scores are multiplied by 100 (the scale argument) and formatted with .1f.
# label fraction: the share of the labelled training images used (0.01, 0.1, 1.0), shown as 1% / 10% / 100% by _frac. It
#   is rounded before the dictionary lookup because floats such as 0.1 can differ in the last digits.
# groups.csv, per_class.csv, compare_supervised_vs_<variant>.csv: the evaluation tables written by scripts/evaluate.py.
#   They hold mean / std per setting, per-class F1, and teacher-vs-student summaries.
# teacher and student (pseudo-labelling): a teacher model labels unlabelled images and a student trains on those labels.
#   In semi_supervised_table f1_a is the teacher's F1 and f1_b the student's.
# matched on seed / paired difference: teacher and student with the same seed are compared and the per-seed difference
#   is summarised: delta_f1_mean, delta_f1_std, seeds_b_better out of n_seeds.
# percentage points: the difference between two percentages (95.0 vs 93.5 = +1.5 points). Column 'Δ (points)'.
# best model: the model with the highest test macro-F1 averaged over its RGB and 13-band rows
#   (groupby('model').mean().idxmax()), at one label fraction. Used by make_figures.py.
# groupby, idxmax, notna, fillna: groupby splits rows into groups; idxmax gives the row label of the largest value
#   (first one if tied); notna tests for real numbers; fillna replaces missing cells (here with an en dash).
# NaN and pd.isna: pandas' 'not a number' marker for a missing value.
# Markdown table: plain text with | separators and a |---| line under the header; to_markdown writes it.
# LaTeX table and booktabs: the report is typeset with LaTeX. booktabs gives the clean horizontal rules \toprule,
#   \midrule, \bottomrule. 'table' is a one-column float, 'table*' spans both columns (wide=True). \caption and \label
#   give the title and the reference key.
# LaTeX escaping: % starts a comment, _ is special and Unicode symbols may not compile, so to_latex.esc rewrites them
#   (\%, \_, $\pm$, $\Delta$, $\tau$, '--').
# bold best value: _bold_best wraps the best cell of each column in **...** (Markdown bold), which esc later turns into
#   \textbf{...} for LaTeX.
# index and reset_index / RangeIndex: a DataFrame's row labels. A named or text index (such as the model names) is moved
#   into a normal first column before writing.
# itertuples: loops over the rows of a DataFrame as tuples; used to write the table body.
# 'from __future__ import annotations': lets type hints like 'pd.DataFrame | None' work on older Python 3 versions.

# ======================================================================
# Q&A ABOUT THIS FILE (file-level viva questions)
# ======================================================================
# Q: What is this file for?
#    A: It turns the evaluation tables (groups.csv and friends) into report tables: it builds tables of display strings
#       and writes each one as Markdown (.md) and LaTeX (.tex).
# Q: Where does it fit in the pipeline?
#    A: Phase 7. scripts/make_figures.py and scripts/make_report_tables.py load the CSVs, call the table builders here
#       and call write_table to save the files into results/tables for the report.
# Q: Why is each table a DataFrame of strings rather than numbers?
#    A: The rounding, the '± std', the '(n=1)' marker and the bold markers are decided once, so the Markdown and the
#       LaTeX output always show identical text.
# Q: Why do the builders sometimes return None?
#    A: A table whose data is missing returns None so the calling script can skip it with a message and still produce
#       the other outputs.

# Q: What does this function do?
#    A: It formats one table cell as 'mean ± std' after multiplying both numbers by scale (100 turns a 0-1 score into
#       percent).
# Q: Why does a NaN std give '(n=1)'?
#    A: NaN means only one seed, so no spread was measured. '± 0.0' would wrongly suggest the seeds agreed exactly.
# Q: What are the inputs and the output?
#    A: mean, std, scale and digits; it returns a string.
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


# Q: What does this function do?
#    A: It turns a label fraction (0.01, 0.1, 1.0) into the header text '1%', '10%', '100%' using FRACTION_LABEL.
# Q: Why is f rounded first?
#    A: A float such as 0.10000000001 would not be equal to the dictionary key 0.1. Rounding to 4 decimals makes the
#       lookup reliable; unknown fractions fall back to a computed percentage.
def _frac(f: float) -> str:
    """Label fraction as a column header such as "1%", "10%", "100%" (rounded to match the dict keys).

    Args:
        f: label fraction between 0 and 1.
    Returns:
        The header text; a fraction without a fixed label is shown as a rounded percentage.
    """
    # Round first so values like 0.10000000001 still find the key 0.1.
    return FRACTION_LABEL.get(round(f, 4), f"{f:.0%}")


# Q: What does this function do?
#    A: For every column it wraps the string of the best (largest) value in ** markers, so the best result per column
#       appears in bold.
# Q: Why is a second table 'raw' needed?
#    A: The display cells are strings like '97.8 ± 0.3'. Comparing strings is wrong ('9.5' sorts above '10.1'), so the
#       largest value is found in the table of real numbers and only its text is marked.
# Q: What about ties and empty columns? What does it return?
#    A: idxmax picks the first row on a tie; a column that is missing from raw or all NaN is left alone. It returns a
#       modified copy, the input table is not changed.
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


# Q: What does this function do?
#    A: It builds the table of supervised test scores: one row per model and input type (RGB, 13 bands), one column per
#       label fraction, each cell 'mean ± std' in percent, with the best mean per column in bold.
# Q: Why are there two tables, disp and raw?
#    A: disp holds the text that will be shown; raw holds the plain means so the best cell of each column can be found
#       and bolded.
# Q: What are the inputs and what is returned?
#    A: groups (groups.csv as a DataFrame) and metric ('f1' for macro-F1, anything else for accuracy). It returns the
#       table, or None if there are no supervised rows; a missing cell shows an en dash.
# Q: Why only variant == 'supervised'?
#    A: Pseudo-label students are compared with their teachers in a separate table.
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


# Q: What does this function do?
#    A: It lists, for each pseudo-label variant, model, input and label fraction, the teacher's F1, the student's F1,
#       the change in points (mean ± std over seeds) and how many seeds the student beat the teacher.
# Q: Why is the comparison 'matched on seed'?
#    A: The student of seed s is compared with the teacher of the same seed, so the change measures the effect of
#       pseudo-labelling and not the difference between random starts.
# Q: What is 'groups' used for?
#    A: Nothing: it is accepted only so the call looks like the other table builders. All numbers come from the compare
#       tables.
# Q: What does it return?
#    A: A DataFrame of strings, or None if every compare table is missing or empty. A single seed is shown as '(n=1)'.
def semi_supervised_table(groups: pd.DataFrame, compare: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """Teacher vs student, matched on seed: macro-F1 (%) and the change in points.

    `compare` maps a variant ('pl', 'plb') to its summarised compare_supervised_vs_<variant> table
    (one row per model/modality/fraction, already averaged over seeds). `groups` is accepted but not used.
    Delta is student minus supervised teacher in macro-F1 points: mean +- std over seeds, shown as
    "(n=1)" with no std if the std is NaN. "Seeds better" is "<seeds where the student beat its teacher>/<seeds>".
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


# Q: What does this function do?
#    A: It builds a table of per-class test F1 (mean ± std over seeds, in percent) for one model at one label fraction:
#       one row per class, one column for RGB and one for 13 bands.
# Q: Why start from CLASSES and use set_index('class')?
#    A: CLASSES fixes the row order for every table. A class missing in a column shows an en dash instead of crashing.
# Q: What does it return?
#    A: The table, or None if no supervised rows match the model and fraction (an exact float match).
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


# Q: What does this function do?
#    A: It finds the supervised model with the highest test macro-F1 at one label fraction, averaging each model's RGB
#       and 13-band means.
# Q: Why is it needed?
#    A: The confusion matrices and per-class views show only one model; this picks which one.
# Q: What does it return?
#    A: The model name, or None if there is no supervised row at that fraction.
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


# Q: What does this function do?
#    A: It writes a display DataFrame as a Markdown pipe table: header row, a '|---|' separator row and one line per
#       table row.
# Q: Why is the index sometimes reset?
#    A: If the row labels carry information (for example the model names), they must become the first column; an unnamed
#       index column gets an empty header.
# Q: Are the ** markers changed?
#    A: No. Cells are written as they are, so '**bold**' is rendered as bold by Markdown viewers.
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


# Q: What does this function do?
#    A: It writes a display DataFrame as a LaTeX booktabs table inside a table (or table*) float, with caption, label,
#       font size, header, rows and closing lines.
# Q: Why is escaping needed?
#    A: LaTeX treats % as a comment and _ as special, and the Unicode symbols (±, Δ, τ, en dash) may not compile.
#       Without escaping the report would fail or lose text.
# Q: Why booktabs, and what does wide do?
#    A: booktabs rules look cleaner than full grid lines (the report preamble must load the package). wide=True uses
#       table*, which spans both columns of a two-column page.
# Q: How are columns aligned?
#    A: The first column is left-aligned ('l'), all others centred ('c').
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

    # Q: What does this function do?
    #    A: It makes one cell or header safe for LaTeX: symbols to math or ASCII, % and _ escaped, and each pair of **
    #       markers turned into \textbf{...}.
    # Q: Why is a while loop used for the bold markers?
    #    A: Each pass finds the first '**' and the next one and replaces that pair; the loop repeats until no pair is
    #       left.
    # Q: Why is it nested inside to_latex?
    #    A: It is used only there, so it is kept local.
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


# Q: What does this function do?
#    A: It saves a table as <name>.md and <name>.tex in out_dir, creating the folder first.
# Q: Why return the two paths?
#    A: The calling scripts collect them to print the list of written files.
# Q: What is the LaTeX label?
#    A: 'tab:<name>', so the report can refer to the table with \ref. The caption is used only in the .tex file.
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

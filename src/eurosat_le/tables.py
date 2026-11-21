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

from __future__ import annotations

from pathlib import Path

import pandas as pd

from eurosat_le.constants import CLASSES
from eurosat_le.figures import FRACTION_LABEL, MODEL_LABEL, MODELS, MODALITY_LABEL, metric_columns


def _pm(mean: float, std: float, scale: float = 100.0, digits: int = 1) -> str:
    """Format "mean ± std" after multiplying both by `scale` (100 turns 0-1 scores into percent).

    A NaN std means a single seed. No spread was measured then, so the cell reads "mean (n=1)" and not
    "± 0.0", which would suggest the seeds agreed exactly.
    """
    if pd.isna(std):
        return f"{mean * scale:.{digits}f} (n=1)"
    return f"{mean * scale:.{digits}f} ± {std * scale:.{digits}f}"


def _frac(f: float) -> str:
    """Label fraction as a column header such as "1%", "10%", "100%" (rounded to match the dict keys)."""
    return FRACTION_LABEL.get(round(f, 4), f"{f:.0%}")


def _bold_best(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Bold the highest value per column (raw has the numbers behind the strings).

    `df` holds display strings and `raw` the same-shaped floats; the cell where `raw` is largest in each
    column gets "**...**" around its string. Columns that are missing from `raw` or all NaN are left
    alone. Ties are resolved by `idxmax`, which picks the first row. Returns a modified copy.
    """
    out = df.copy()
    for c in out.columns:
        if c in raw and raw[c].notna().any():
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
    g = groups[groups["variant"] == "supervised"]
    if g.empty:
        return None
    mean, std = metric_columns(metric)
    fr = sorted(g["fraction"].unique())
    idx = [(m, mo) for m in MODELS for mo in ("rgb", "ms") if ((g["model"] == m) & (g["modality"] == mo)).any()]
    disp = pd.DataFrame(index=[f"{MODEL_LABEL[m]}, {MODALITY_LABEL[mo]}" for m, mo in idx],
                        columns=[_frac(f) for f in fr], dtype=object)
    raw = pd.DataFrame(index=disp.index, columns=disp.columns, dtype=float)
    for (m, mo), name in zip(idx, disp.index):
        for f in fr:
            r = g[(g["model"] == m) & (g["modality"] == mo) & (g["fraction"] == f)]
            if not r.empty:
                disp.at[name, _frac(f)] = _pm(r[mean].iloc[0], r[std].iloc[0])
                raw.at[name, _frac(f)] = r[mean].iloc[0]
    return _bold_best(disp.fillna("–"), raw)


def semi_supervised_table(groups: pd.DataFrame, compare: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """Teacher vs student, matched on seed: macro-F1 (%) and the change in points.

    `compare` maps a variant ('pl', 'plb') to its summarised compare_supervised_vs_<variant> table
    (one row per model/modality/fraction, already averaged over seeds). `groups` is accepted but not used.
    Delta is student minus supervised teacher in macro-F1 points: mean +- std over seeds, std shown as
    0.0 if NaN. "Seeds better" is "<seeds where the student beat its teacher>/<seeds>".
    Returns None if every table is missing or empty; the row order is variant, then model, modality, fraction.
    """
    rows = []
    for variant, c in compare.items():
        if c is None or c.empty:
            continue
        for _, r in c.sort_values(["model", "modality", "fraction"]).iterrows():
            rows.append({
                "Model": MODEL_LABEL.get(r["model"], r["model"]),
                "Input": MODALITY_LABEL[r["modality"]],
                "Labels": _frac(r["fraction"]),
                "Variant": "balanced" if variant == "plb" else "global τ",
                "Supervised F1": f"{r['f1_a'] * 100:.1f}",
                "Student F1": f"{r['f1_b'] * 100:.1f}",
                "Δ (points)": (f"{r['delta_f1_mean'] * 100:+.1f} (n=1)" if pd.isna(r["delta_f1_std"])
                              else f"{r['delta_f1_mean'] * 100:+.1f} ± {r['delta_f1_std'] * 100:.1f}"),
                "Seeds better": f"{int(r['seeds_b_better'])}/{int(r['n_seeds'])}",
            })
    return pd.DataFrame(rows) if rows else None


def per_class_table(per_class: pd.DataFrame, model: str, fraction: float) -> pd.DataFrame | None:
    """Per-class test F1 (%) for one model, RGB and 13-band, at one label fraction.

    Rows are the classes in CLASSES order (index named "Class"); one column per input type that has
    data, cell = mean +- std over seeds, "–" for a class missing in that column. Supervised rows only.
    Returns None if nothing matches. `fraction` is an exact float match.
    """
    d = per_class[(per_class["variant"] == "supervised") & (per_class["model"] == model)
                  & (per_class["fraction"] == fraction)]
    if d.empty:
        return None
    out = pd.DataFrame(index=CLASSES)
    for mo in ("rgb", "ms"):
        x = d[d["modality"] == mo].set_index("class")
        if not x.empty:
            out[MODALITY_LABEL[mo]] = [_pm(x.at[c, "f1_mean"], x.at[c, "f1_std"]) if c in x.index else "–"
                                       for c in CLASSES]
    out.index.name = "Class"
    return out


def best_model(groups: pd.DataFrame, fraction: float = 1.0) -> str | None:
    """Model with the highest mean test macro-F1 over both inputs, supervised, at one fraction.

    The score is the average of the model's test_f1_mean over its RGB and 13-band rows. Returns the model
    name, or None if there is no supervised row at that fraction. Used to pick which model's confusion
    matrices and per-class views are shown in the report.
    """
    g = groups[(groups["variant"] == "supervised") & (groups["fraction"] == fraction)]
    if g.empty:
        return None
    return g.groupby("model")["test_f1_mean"].mean().idxmax()


def to_markdown(df: pd.DataFrame) -> str:
    """Render a display DataFrame as a pipe-delimited Markdown table (string ends with a newline).

    A named or non-default index becomes the first column (an unnamed one gets an empty header).
    Cell strings are written as they are, so existing "**bold**" markers render as bold.
    """
    d = df.reset_index() if df.index.name or not isinstance(df.index, pd.RangeIndex) else df
    d = d.rename(columns={"index": ""})
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, sep, *body]) + "\n"


def to_latex(df: pd.DataFrame, caption: str, label: str, wide: bool = False, font: str = r"\small") -> str:
    """Render a display DataFrame as a booktabs `table` float (string ends with a newline).

    With `wide=True` the float is a `table*`, which spans both columns of a two-column page; use it for
    tables with many columns (the pseudo-label table), which would run off a single column.

    `caption` is inserted as given (the caller writes LaTeX, e.g. an escaped percent sign and a math
    plus-minus); `label` is the LaTeX label key. The index becomes the first column (header "Setting" if
    it has no name).
    """
    d = df.reset_index() if df.index.name or not isinstance(df.index, pd.RangeIndex) else df
    d = d.rename(columns={"index": "Setting"})

    def esc(s) -> str:
        """Escape one cell/header: Unicode symbols to math or ASCII, % and _ escaped, **x** to bold."""
        s = str(s).replace("±", r"$\pm$").replace("Δ", r"$\Delta$").replace("τ", r"$\tau$").replace("%", r"\%")
        s = s.replace("−", "-").replace("–", "--").replace("_", r"\_")
        while "**" in s:  # **bold** -> \textbf{bold}
            a = s.index("**")
            b = s.index("**", a + 2)
            s = s[:a] + r"\textbf{" + s[a + 2:b] + "}" + s[b + 2:]
        return s

    cols = "l" + "c" * (len(d.columns) - 1)
    env = "table*" if wide else "table"
    lines = [rf"\begin{{{env}}}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             font, rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(esc(c) for c in d.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(v) for v in row) + r" \\" for row in d.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"]
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, out_dir: Path, name: str, caption: str, wide: bool = False,
                font: str = r"\small") -> list[Path]:
    """Write `df` to <out_dir>/<name>.md and <name>.tex (creating out_dir) and return both paths.

    The LaTeX label is "tab:<name>". The caption is used in the .tex file only. `wide` makes the LaTeX
    float span both columns (see `to_latex`).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    md, tex = out_dir / f"{name}.md", out_dir / f"{name}.tex"
    md.write_text(to_markdown(df), encoding="utf-8")
    tex.write_text(to_latex(df, caption, f"tab:{name}", wide, font), encoding="utf-8")
    return [md, tex]

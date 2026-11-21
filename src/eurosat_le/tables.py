"""Report tables (Markdown and LaTeX) built from the Phase 6 tables. Phase 7.

Each function returns a DataFrame of display strings; `write_table` saves it as
<name>.md and <name>.tex. A cell is "mean ± std" over the seeds. The best value in
each column is bold in the LaTeX/Markdown output.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from eurosat_le.constants import CLASSES
from eurosat_le.figures import FRACTION_LABEL, MODEL_LABEL, MODELS, MODALITY_LABEL


def _pm(mean: float, std: float, scale: float = 100.0, digits: int = 1) -> str:
    return f"{mean * scale:.{digits}f} ± {0.0 if pd.isna(std) else std * scale:.{digits}f}"


def _frac(f: float) -> str:
    return FRACTION_LABEL.get(round(f, 4), f"{f:.0%}")


def _bold_best(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Bold the highest value per column (raw has the numbers behind the strings)."""
    out = df.copy()
    for c in out.columns:
        if c in raw and raw[c].notna().any():
            out.loc[raw[c].idxmax(), c] = f"**{out.loc[raw[c].idxmax(), c]}**"
    return out


def supervised_table(groups: pd.DataFrame, metric: str = "f1") -> pd.DataFrame | None:
    """Rows: model x input. Columns: label fractions. Cell: test score in %, mean ± std over seeds."""
    g = groups[groups["variant"] == "supervised"]
    if g.empty:
        return None
    mean, std = f"test_{metric if metric == 'f1' else 'acc'}_mean", f"test_{metric if metric == 'f1' else 'acc'}_std"
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

    `compare` maps a variant ('pl', 'plb') to its summarised compare_supervised_vs_<variant> table.
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
                "Δ (points)": f"{r['delta_f1_mean'] * 100:+.1f} ± {0.0 if pd.isna(r['delta_f1_std']) else r['delta_f1_std'] * 100:.1f}",
                "Seeds better": f"{int(r['seeds_b_better'])}/{int(r['n_seeds'])}",
            })
    return pd.DataFrame(rows) if rows else None


def per_class_table(per_class: pd.DataFrame, model: str, fraction: float) -> pd.DataFrame | None:
    """Per-class test F1 (%) for one model, RGB and 13-band, at one label fraction."""
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
    """Model with the highest mean test macro-F1 over both inputs, supervised, at one fraction."""
    g = groups[(groups["variant"] == "supervised") & (groups["fraction"] == fraction)]
    if g.empty:
        return None
    return g.groupby("model")["test_f1_mean"].mean().idxmax()


def to_markdown(df: pd.DataFrame) -> str:
    d = df.reset_index() if df.index.name or not isinstance(df.index, pd.RangeIndex) else df
    d = d.rename(columns={"index": ""})
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, sep, *body]) + "\n"


def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    d = df.reset_index() if df.index.name or not isinstance(df.index, pd.RangeIndex) else df
    d = d.rename(columns={"index": "Setting"})

    def esc(s) -> str:
        s = str(s).replace("±", r"$\pm$").replace("Δ", r"$\Delta$").replace("τ", r"$\tau$").replace("%", r"\%")
        s = s.replace("−", "-").replace("–", "--").replace("_", r"\_")
        while "**" in s:  # **bold** -> \textbf{bold}
            a = s.index("**")
            b = s.index("**", a + 2)
            s = s[:a] + r"\textbf{" + s[a + 2:b] + "}" + s[b + 2:]
        return s

    cols = "l" + "c" * (len(d.columns) - 1)
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\small", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(esc(c) for c in d.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(v) for v in row) + r" \\" for row in d.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, out_dir: Path, name: str, caption: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md, tex = out_dir / f"{name}.md", out_dir / f"{name}.tex"
    md.write_text(to_markdown(df), encoding="utf-8")
    tex.write_text(to_latex(df, caption, f"tab:{name}"), encoding="utf-8")
    return [md, tex]

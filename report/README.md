# Report

Two versions of the same report:

- `main.tex` is the single-column report in the layout of the ELCVIA journal template (A4, 11 pt Times, 17 x 25 cm
  text area, 2 cm left margin, bold title / section / subsection sizes of 16 / 14 / 12 pt, running head, numbered
  references). The layout is rebuilt with standard packages, because the journal's `elcvia.cls` is not in this
  repository; to use the journal's own class, put `elcvia.cls` and `elcvia.sty` next to `main.tex` and change the
  preamble as the template shows. To stay near the recommended 8 to 12 pages, this version leaves out a few tables
  and the two long appendices (all runs, design decisions); they are in `results/tables/all_runs.md` and
  `docs/DESIGN_DECISIONS.md`.
- `main_ieee.tex` is the full two-column IEEE version (every table and figure, with the appendices).

Both read the figures and tables from `../results/figures` and `../results/tables`, so build those first:

```bash
python scripts/evaluate.py             # results/summary/eval/ (needs run folders)
# or, when only the run logs exist:
python scripts/parse_colab_logs.py <notebooks...>
python scripts/eval_from_logs.py
python scripts/make_figures.py --fraction 0.01      # results/figures/*.png, results/tables/*.tex
python scripts/make_report_tables.py                # the extra tables
cd report && latexmk -pdf main.tex
```

`references.bib` holds the references. The numbers in the text were checked against `results/summary` and
`results/tables`; if the results change, recheck them by hand.

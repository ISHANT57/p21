# Report

Two versions of the same report:

- `main.tex` is the single-column report in the style of a NeurIPS paper (title between two rules, numbered
  sections, "Figure 1:" and "Table 1:" captions, numbered references). It is the version to submit. To stay
  near 12 pages it leaves out a few tables and one figure that the full version has.
- `main_ieee.tex` is the full two-column IEEE version (every table and figure).

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

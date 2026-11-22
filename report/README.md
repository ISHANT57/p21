# Report

`team21report.pdf` is a compiled copy of the report (12 pages, compiled on Overleaf from `main.tex`).

Three versions of the same report:

- `Elcvia2e.tex` is the report in the structure of the instructor's ELCVIA template (`elcvia2e.tex`): it loads the
  template's own class files (`\usepackage{elcvia}`, `psfig.sty`), uses its title/author/abstract layout and a
  hand-written reference list. **The class files are not in this repository.** To compile it, open a copy of the
  instructor's template project on Overleaf, upload `Elcvia2e.tex` (it replaces the template's file) together with the
  folders `figures/` and `tables/` (the figures and tables of `../results`), and recompile. Afterwards set `\finpag` in
  the file to the real last page.
- `main.tex` is the same text as a stand-alone article that imitates the template layout with standard packages
  (A4, 11 pt Times, 17 x 25 cm text area, 2 cm left margin). It compiles anywhere, without the class files.
- `main_ieee.tex` is the full two-column IEEE version (every table and figure, with the appendices).

To stay near the recommended 8 to 12 pages, `Elcvia2e.tex` and `main.tex` leave out a few tables and the two long
appendices (all runs, design decisions); they are in `results/tables/all_runs.md` and `docs/DESIGN_DECISIONS.md`.

The versions read the figures and tables from `../results/figures` and `../results/tables` (or the `figures/` and
`tables/` folders on Overleaf), so build those first:

```bash
python scripts/evaluate.py             # results/summary/eval/ (needs run folders)
# or, when only the run logs exist:
python scripts/parse_colab_logs.py <notebooks...>
python scripts/eval_from_logs.py
python scripts/make_figures.py --fraction 0.01      # results/figures/*.png, results/tables/*.tex
python scripts/make_report_tables.py                # the extra tables
cd report && latexmk -pdf main.tex
```

`references.bib` holds the references (`Elcvia2e.tex` has them converted to the template's `thebibliography` list).
The numbers in the text were checked against `results/summary` and `results/tables`; if the results change, recheck
them by hand.

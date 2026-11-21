# Report

IEEE two-column skeleton. Every `\TODO{...}` is text still to be written.

```bash
python scripts/evaluate.py        # results/summary/eval/
python scripts/make_figures.py    # results/figures/*.png, results/tables/*.tex
cd report && latexmk -pdf main.tex
```

`main.tex` reads the figures and tables from `../results/figures` and `../results/tables`, so build those first.
The `Dosovitskiy` and `Sohn` author lists are shortened with "others"; complete them before submission.

# Figures and tables for the report

`scripts/make_figures.py` builds the report figures (`results/figures/`) and tables (`results/tables/`) from the CSV files that `scripts/evaluate.py` writes to `results/summary/eval/`. Code: `src/eurosat_le/figures.py`, `src/eurosat_le/tables.py`.

## Figures

| File | Source CSV | Columns used | What it shows |
|---|---|---|---|
| `label_efficiency_f1.png` | `groups.csv` (rows with `variant == supervised`) | `model`, `modality`, `fraction`, `test_f1_mean`, `test_f1_std` | Test macro-F1 against the share of labelled training images, one line per model with error bars (std over seeds), RGB and 13 bands in separate panels. |
| `label_efficiency_acc.png` | `groups.csv` (supervised rows) | `model`, `modality`, `fraction`, `test_acc_mean`, `test_acc_std` | The same figure for test accuracy. |
| `rgb_vs_ms_gain.png` | `compare_rgb_vs_ms_pairs.csv` (supervised rows) | `model`, `fraction`, `delta_f1` | The change in test macro-F1 when 13-band input replaces RGB, per label fraction and model; bar = mean over seeds, dot = one seed. |
| `pseudo_label_gain.png` | `compare_supervised_vs_pl_pairs.csv`, `compare_supervised_vs_plb_pairs.csv` | `modality`, `model`, `fraction`, `delta_f1` | The change in test macro-F1 of each pseudo-label student against its own supervised teacher (same seed); class-balanced students are hatched. |
| `per_class_f1_<pct>pct.png` | `per_class.csv` (supervised rows at `--fraction`, default 1.0) | `model`, `modality`, `class`, `f1_mean` | Test F1 for each class for every model and input type, as a heatmap with the values printed. |
| `confusion_<pct>pct.png` | `confusion/<model>_<input>_<pct>pct_supervised.csv` | the 10 × 10 matrix | Row-normalised confusion matrices of the best model (highest mean test macro-F1 at `--fraction`), RGB and 13 bands side by side, averaged over seeds. Also drawn for 1% labels when that fraction exists. |

## Tables

Each table is written as `<name>.md` and `<name>.tex` in `results/tables/`.

| Name | Source CSV | Columns used | What it shows |
|---|---|---|---|
| `supervised_f1` | `groups.csv` (supervised rows) | `model`, `modality`, `fraction`, `test_f1_mean`, `test_f1_std` | Test macro-F1 (%) as mean ± std over seeds; rows are model and input, columns are label fractions; the best value per column is bold. |
| `supervised_acc` | `groups.csv` (supervised rows) | `model`, `modality`, `fraction`, `test_acc_mean`, `test_acc_std` | The same table for test accuracy. |
| `semi_supervised` | `compare_supervised_vs_pl.csv`, `compare_supervised_vs_plb.csv` | `model`, `modality`, `fraction`, `f1_a`, `f1_b`, `delta_f1_mean`, `delta_f1_std`, `seeds_b_better`, `n_seeds` | Supervised teacher F1, student F1, the change in points (mean ± std) and the number of seeds where the student was better, for each model, input and fraction. |
| `per_class_f1_<pct>pct` | `per_class.csv` (supervised rows) | `class`, `f1_mean`, `f1_std` | Test F1 (%) per class, RGB and 13 bands, for the best model at `--fraction`. |

## Colour rules

The shared style is in [`src/eurosat_le/viz.py`](../src/eurosat_le/viz.py).

- A model keeps the same colour and marker in every figure (taken in order from `viz.CATEGORICAL` and never cycled): Scratch CNN blue with a circle, ResNet-50 orange with a square, ViT-Small green with a triangle, ViT-Tiny from scratch amber with a diamond.
- Colour is not the only cue: lines also have their own marker and a direct label, because the green has low contrast on the background.
- Text uses grey text colours, never a series colour.
- Confusion matrices and the per-class heatmap use one blue ramp (`viz.SEQUENTIAL`), not model colours.

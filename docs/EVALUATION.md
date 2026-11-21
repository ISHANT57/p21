# Evaluation (Phase 6)

`python scripts/evaluate.py` rebuilds every number in the report from the predictions each run saved (`preds_val.npz`, `preds_test.npz`). Nothing is retrained.

As a check, it recomputes accuracy and macro-F1 and compares them with the `metrics.json` written at training time. It prints a warning if they differ by more than 0.001.

## Metrics

| Metric | Why |
|---|---|
| Overall accuracy | The headline metric used in the EuroSAT paper |
| Macro-F1 | The mean of the per-class F1 scores. Classes are mildly imbalanced (2,000–3,000 images each), so this is the main metric for comparing models; it is also used for model selection |
| Per-class precision / recall / F1 | Shows which land-use classes need labels or extra bands |
| ECE (15 bins) | Expected calibration error: whether softmax confidence matches accuracy. This matters for choosing pseudo-label thresholds |
| Confusion matrix | Row-normalised (rows = true class) and averaged over seeds |

Every result is reported as **mean ± std over the 3 labelled-subset seeds**. The test set (4,050 images) is the same for every run.

## Paired comparisons

Two settings are compared only when everything else matches: model, label fraction, variant **and seed**. That means the same labelled images, the same initialisation seed and the same test images. For each matched pair:

- **Per-seed difference** in test macro-F1 (b − a).
- **95% paired bootstrap interval** on the test set: test images are resampled 1,000 times, and macro-F1 is computed for both runs on the same resample. This shows how much of a difference could be test-set noise.
- **Over seeds:** the mean, std, minimum and maximum of the differences; how many seeds favour b; and how many seeds have an interval that excludes 0.

With only 3 seeds, the report states, for example, "13-band better in 3/3 seeds, intervals exclude 0 in 2/3", rather than a p-value.

| File | Comparison |
|---|---|
| `compare_rgb_vs_ms` | RGB → 13-band, the brief's "value of non-RGB bands" |
| `compare_supervised_vs_pl`, `compare_supervised_vs_plb` | teacher → pseudo-label student (Phase 5) |
| `compare_pl_vs_plb` | global-τ students → class-balanced students |
| `compare_scratch_vs_resnet50`, `compare_scratch_vs_vit_small`, `compare_resnet50_vs_vit_small` | between models |

## Outputs (`results/summary/eval/`)

`per_run.csv`, `groups.csv` (the main table), `per_class.csv`, `confusion/<model>_<input>_<pct>pct_<variant>.csv`, `top_confusions.csv`, `compare_*.csv` (summary over seeds) and `compare_*_pairs.csv` (per seed). With `--examples <run> ...`, it also writes `errors_<run>.png`, a grid of misclassified test images shown in RGB (also for 13-band runs), labelled true → predicted (confidence).

## Progress messages

The script prints how many runs and comparison pairs it will process before the slow part (the paired bootstrap), then one line per comparison file with its pair count and time (for example `compare_rgb_vs_ms: 27 pairs, 12.3 s`), and the total time next to the final `wrote ...` line. Pass `--quiet` to turn these messages off; the result tables and any metric warnings are still printed. The messages do not change any output file.

## Results taken from logs (final results of this project)

Only the run folders of one lane (ViT-Small at 1% and 10%, and ResNet-50 on 13 bands at 100%; 13 runs) were kept. For the other two lanes only the Colab notebooks, with their printed training logs, were kept, so their predictions do not exist and the full evaluation above cannot be run on them. The final tables and figures are therefore built from the numbers each run printed when it finished:

1. `python scripts/parse_colab_logs.py <notebooks...>` reads the result line of every finished run (`best epoch ... | test acc ... f1 ...`) and, for students, the teacher, threshold and pseudo-label count. It writes `results/logs/runs_from_logs.csv` and, for each notebook, the lines it used (`results/logs/<notebook>.txt`).
2. `python scripts/eval_from_logs.py --runs-dir results/runs` writes `groups.csv` and the `compare_*` files in the same layout as `evaluate.py`. It cross-checks the 13 runs that have a folder against their `metrics.json`: all agree to the four printed decimals.
3. `python scripts/evaluate.py --out results/summary/eval_laneB` evaluates those 13 runs fully. Their `per_class.csv` and `confusion/` are copied into `results/summary/eval/`, so `make_figures.py --fraction 0.01` can draw the per-class heatmap and confusion matrices of ViT-Small at 1% labels.

What differs from a full evaluation, and has to be said wherever these numbers are used:

- **No bootstrap intervals.** A paired comparison is the per-seed difference of two logged macro-F1 values, and `seeds_ci_excludes_0` is empty. Only the ViT-Small RGB versus 13-band comparison has intervals (`results/summary/eval_laneB/`).
- **No calibration (ECE)** and numbers with four decimals, as printed.
- **Per-class scores and confusion matrices exist only for the 13 runs with predictions** (ViT-Small at 1% and 10%, ResNet-50 on 13 bands at 100%).
- **Single-seed cells** (the 100% runs, and the scratch CNN student) are shown as `(n=1)`, with no spread.
- **Incomplete pseudo-labelling.** Students that had not finished when the sessions ended are not counted: 8 ResNet-50 students (seeds 0 and 1) and 1 scratch CNN student (RGB, 1%, seed 0).

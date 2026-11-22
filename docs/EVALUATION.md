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

## Final results (all 52 runs evaluated from saved predictions)

The training ran in several Colab and Kaggle sessions, and the run folders of all 52 finished runs (42 supervised runs and 10 pseudo-label students) were collected into `results/runs/` (not in git: the prediction files are large). The final tables and figures are built like this:

1. `python scripts/evaluate.py` evaluates every run folder (accuracy, macro-F1, ECE, per-class scores, confusion matrices, paired bootstrap intervals) into `results/summary/eval/`.
2. `python scripts/make_figures.py --fraction 0.01` (and `--fraction 0.1`) writes the figures and the main tables; `python scripts/make_report_tables.py` writes the extra tables (all runs, pseudo-label details, training cost, learning-rate sweep, model-against-model differences).
3. The notebooks' printed result lines are kept in `results/logs/` (`scripts/parse_colab_logs.py` -> `runs_from_logs.csv`). They agree with the scores recomputed from the predictions for all 52 runs (four-decimal rounding only). One finished student (scratch CNN, RGB, 10% labels) is missing from the lane C log, which ends after its pseudo-labelling line, so its row in `runs_from_logs.csv` was added from its run folder (`metrics.json`, `pseudo_label_selection.json`).
4. `results/summary/pseudo_label_quality.csv` lists the accuracy of each student's pseudo-labels against the hidden pool labels (from `pseudo_label_quality.json` of each student's run folder) next to the threshold and validation precision.

What remains limited, and has to be said wherever the numbers are used:

- **Single-seed cells** (the 100% runs, the scratch CNN students) are shown as `(n=1)`, with no spread.
- **Incomplete pseudo-labelling.** Only 10 of the planned 24 students finished: 8 ResNet-50 students (seeds 0 and 1) and 2 scratch CNN students (RGB, seed 0, at 1% and 10%). A scratch CNN student for 13 bands (1%, seed 0) was interrupted and has no result.
- **ECE only.** The sign of the calibration gap (over- or under-confidence) was not saved.

`results/summary/eval_laneB/` is the earlier evaluation of the first 13 runs with predictions; it is kept for reference and is superseded by `results/summary/eval/`.

## Cross-check with an independent pipeline

A fourth team member, Ayush Kumar, built an independent pipeline for the same task (folder
[`ayush_pipeline/`](../ayush_pipeline/): 80/10/10 split, six label fractions, three seeds everywhere). `scripts/crosscheck_pipelines.py` compares the two where they overlap:

- RGB (JPEG) test accuracy of the same three models at 1%, 10% and 100% of the labels (`results/tables/crosscheck.md`): the pretrained models agree within 0.6 points and the scratch CNN within 1.0 point.
- A control for the 13-band comparison (`results/tables/band_control.md`): his pipeline also trained ResNet-50 on the three RGB bands cut from the 13-band GeoTIFF files. At 1% of the labels this input scores 1.0 point above the JPEG RGB, and all 13 bands add -0.2 points on top of it, so our ResNet-50 gain over JPEG RGB (+0.8) is explained by the input source and not by the extra bands.

His summarised numbers are saved in `results/crosscheck/` (`SOURCE.md` says which files in `ayush_pipeline/results/` they come from); `python scripts/crosscheck_pipelines.py --other-repo ayush_pipeline` rebuilds them. The two protocols are never merged into one table.

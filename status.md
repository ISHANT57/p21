# Project status (P21)

**Last updated:** 2026-10-30
**Current phase:** Phase 6 code done. All code that does not need a GPU is in place. Waiting on GPU runs: the baseline, tuning, the 54-run grid, and 30 pseudo-label students. Phase 7 figure/table code and the report skeleton are in place; they need the real results to produce the final figures.

The plan and its task lists are in [phases.md](phases.md). Update this file whenever a phase task finishes or a decision is made.

## Overview

| Phase | Title | State | Done |
|---|---|---|---|
| 1 | Repository, download, inspection, splits | ✅ Done | 2026-10-30 |
| 2 | Preprocessing, augmentation, loaders | ✅ Done; Colab test passed 2026-11-21, Kaggle test pending | 2026-10-30 |
| 3 | Models and working baseline | ✅ Done: baseline reproduced on Colab (98.67% test accuracy) | 2026-11-21 |
| 4 | Supervised label-efficiency runs (54) | ⏳ Tooling done; GPU runs pending | |
| 5 | Pseudo-labelling (24 runs) | ⏳ Code done; GPU runs pending (they need the Phase 4 teachers) | |
| 6 | Evaluation | ⏳ Code done; waiting for real runs | |
| 7 | Figures, tables, README, report | ⏳ Figure/table code and report skeleton done; waiting for real results and the report text | |

States: ⬜ Not started · ⏳ Next / in progress · ✅ Done · ⚠️ Blocked

## Run tracker

| Grid | Planned | Done |
|---|---|---|
| Supervised, RGB | 27 | 0 |
| Supervised, 13-band | 27 | 0 |
| Pseudo-labelling | 24 | 0 |
| Pseudo-labelling, class-balanced | 6 | 0 |
| *(optional)* ViT from scratch | 18 | 0 |
| *(optional)* Band ablation | n/a | 0 |

## Change log

### 2026-11-21: Baseline reproduced on Colab
- ResNet-50, RGB, 100% labels, seed 0, Tesla T4: **test accuracy 0.9867, macro-F1 0.9863**, validation accuracy 0.9889, best epoch 29 of 30, 43.1 min. Target was ≥ 97%; the EuroSAT paper reports about 98.6%.
- The best epoch was the last one, so early stopping never triggered. A 100% pretrained run takes about 43 min (86 s per epoch of 295 steps, about 0.29 s per step).
- On the same Colab session: 69 tests passed (62 s) and the smoke test (it runs on CPU) passed 8/8 model × input combinations.
- Parallel lanes (three Colab accounts, `notebooks/lane_*.ipynb`): A = ResNet-50 and its students, B = ViT-Small 1%/10% plus ResNet-50 13-band and ViT-Small RGB at 100%, C = scratch CNN, ViT-Small 13-band at 100%, scratch CNN students. Students run with `--set epochs=10` (about the same compute as their teachers); state this in the report.
- Time plan (deadline 2026-11-22 night): no tuning sweep (default `lr` from the configs); 100% runs with seed 0 only; pseudo-labelling for ResNet-50 at 1% and 10%; optional runs dropped. State these cuts in the report.

### 2026-11-21: Phase 7 code and report skeleton (branch `feat/report`)
- `src/eurosat_le/figures.py` and `scripts/make_figures.py` build, from `results/summary/eval/`: label-efficiency curves (macro-F1 and accuracy), RGB-vs-13-band gain per seed, pseudo-label gain per seed, confusion matrices, a per-class F1 heatmap. A figure whose inputs are missing is skipped with a message.
- `src/eurosat_le/tables.py` writes the supervised grid, the semi-supervised grid and the per-class table as Markdown and LaTeX, best value per column in bold.
- Model colours and markers are fixed across figures. The palette passes `validate_palette.js`; the green is below 3:1 contrast on the surface, so every line also has a marker and a direct label.
- `report/`: IEEE two-column skeleton (`main.tex`, `references.bib`) that reads `results/figures` and `results/tables`. Every `\TODO{}` is text still to write.
- Tested on synthetic tables only (`tests/test_figures.py`, 11 tests); nothing has been rendered from real runs yet.
- Fixed a stray line at the top of `phases.md` and the outdated GitHub note below.

### 2026-10-30: Phase 6 code finished (branch `feat/evaluation`)
- `scripts/evaluate.py` rebuilds every metric from the saved predictions:
  - accuracy, macro-F1, per-class metrics, ECE, with a cross-check against `metrics.json`
  - confusion matrices averaged over seeds, and the top confused class pairs
  - paired comparisons matched on seed, with 95% bootstrap intervals on the test set
  - a misclassified-image grid
- Method: `docs/EVALUATION.md`. Tests use fake run folders built from the real test ids. 58 tests pass.

### 2026-10-30: Phase 5 code finished (branch `feat/pseudo-label`)
- `src/eurosat_le/ssl/`:
  - τ is chosen on validation as the smallest τ with confident-prediction precision ≥ 95%.
  - Pool images above τ get the teacher's class as their pseudo-label.
  - A student with the same configuration trains on labelled + pseudo-labelled images.
  - Optional per-class cap (`--balance`) and extra rounds (`_pl2`).
- Pseudo-label accuracy against the hidden labels is computed only after training, for analysis (`pseudo_label_quality.json`).
- `run_grid.py --plan pl` plans 24 students, each waiting for its teacher. `scripts/pseudo_label.py` runs a single student. `collect_results.py` writes `semi_supervised.csv`, with each student paired with its teacher by seed.
- Method and safeguards: `docs/SSL.md`. 52 tests pass.
- CPU check with a deliberately weak teacher (scratch CNN, 1%, 3 epochs, 61% test accuracy): τ = 0.98, validation precision 96.6%, actual pool pseudo-label accuracy **95.6%**. Validation estimated pseudo-label quality well.
- Finding: the weak teacher's confident predictions fell in only 5 of 10 classes (Industrial 973, Residential 368, SeaLake 243, Pasture 69, HerbaceousVegetation 10). The first version of `--balance` would then have selected nothing. It now caps each class at its labelled-set share instead. Global-τ selection can skew the class mix a lot, so the class-balanced variant is worth running for at least one model.

### 2026-10-30: Phase 4 tooling finished (branch `feat/grid-runner`)
- Budget rule (confirmed by the user): every run trains for `max(epochs × one pass over its labelled set, min_steps)` steps. Small-label epochs are stretched, so the number of validation checks is the same for every fraction. See `docs/MODELS.md`.
- `scripts/run_grid.py`:
  - `--plan tune` runs 9 learning-rate runs and never evaluates the test set.
  - `--plan main` runs the 54-run grid, seed 0 first.
  - It resumes after a disconnect, splits work with `--shard k/n`, stops starting runs after `--max-hours`, and logs failures and carries on.
- `scripts/collect_results.py` builds `runs.csv`, `tuning.csv` and `supervised.csv` (mean ± std over seeds).
- Notebook section 6 runs the tuning sweep and the grid on Colab or Kaggle.
- Tested on CPU with throwaway runs: skip and resume, tuning without test, teacher predictions for 1% runs, the tables. The test runs were then deleted. 42 tests pass.

### 2026-10-30: Phase 3 code finished (branch `feat/models`)
- Models: `scratch_cnn` (2.8M parameters), `resnet50` (23.5M), `vit_small` (21.7M) and the optional `vit_scratch` (5.4M), each for 3 or 13 input channels. Models that need 224 px upsample on the GPU.
- 13-band first layer: RGB filters go to B04/B03/B02, the other bands get the mean filter, and everything is scaled by 3/13. A `zero` init is available as an ablation. Details in `docs/MODELS.md`.
- Training engine and `scripts/train.py`:
  - AdamW, warm-up + cosine schedule, mixed precision, early stopping on validation macro-F1
  - full run logging, including predictions on the unlabelled pool (the Phase 5 teacher)
  - finished runs are skipped when re-launched
- 38 tests pass. Smoke test with pretrained weights: 8/8 model × input combinations memorise a batch.
- Local CPU run (scratch CNN, RGB, 10% labels): validation accuracy 42% → 71% in 5 epochs, so the full loop learns.
- Code zip option for Colab without GitHub: `scripts/make_code_zip.sh` creates `dist/p21_code.zip` and records the commit, so `env.json` still names it.
- Bug fixed: YAML read `3e-4` as a string; config values are now converted to the field's type.
- Note: the laptop is short on memory (2 GB of swap in use). The first epoch of a 13-band run on CPU took about 18 minutes; later epochs took about 45 s. The laptop is for checks only.

### 2026-10-30: Phase 2 code finished (branch `feat/data-loaders`)
- Packed image caches: `data/cache/rgb.npy` (27000×3×64×64 `uint8`, 0.33 GB) and `ms.npy` (27000×13×64×64 `uint16`, 2.88 GB). Rows follow the order of `split.csv`. A cache built for a different split is rejected.
- `make_dataset` / `build_loaders`: RGB or 13-band input, train-only normalisation, weak/strong augmentation, unlabelled pool with weak and strong views, seeded workers.
- 24 tests pass. On the full train split, normalised data has mean 0 and std 1 to within 0.001 for every band. The cache matches the raw files exactly.
- Loader speed with 2 CPU workers: RGB 5,700 img/s, 13-band 1,700 img/s; with weak+strong views, 1,900 and 570 img/s. Training on a free-tier GPU should be slower than this, so loading won't be the bottleneck.
- Two choices changed from the plan:
  - Resizing to 224 px moves to the GPU, inside the model.
  - The strong augmentation is identical for RGB and 13-band input.

### 2026-10-30: Phase 1 finished
- Repository structure, package `eurosat_le`, requirements and lock file, CI workflow.
- Downloaded RGB (27,000 JPEGs) and 13-band (27,000 GeoTIFFs, 2.8 GB) data. Both passed the hash check. The Zenodo RGB archive failed its MD5 check (cut-off download), so the Hugging Face mirror was used instead.
- Split 18,899 / 4,051 / 4,050 (train/val/test), stratified, seed 42. Labelled subsets of 191 (1%), 1,890 (10%) and 18,899 (100%) images, with 3 seeds; each smaller subset is contained in the larger one.
- `make_splits.py --check` passes. 9 unit tests pass.
- Dataset figures are in `results/dataset_inspection/`. Normalisation statistics (train only) are in `splits/norm_stats.json`.

## Findings to carry into the report

1. **The 13-band files store B8A last** (`B01 … B12, B8A`), not after B08. The order was confirmed from band statistics and from the correlation with the RGB images. See docs/DATA.md.
2. **No exact duplicate images** anywhere in the dataset, so none can cross between splits.
3. **B10 (cirrus) carries almost no information:** mean 12 DN, std 4.8. Worth an ablation that leaves it out.
4. **Class imbalance is mild:** 2,000 to 3,000 images per class; Pasture is the smallest.
5. A random split can overestimate accuracy, because nearby patches look alike. Mention this in the limitations section.

## Open decisions (need the user / group)

- [x] First local git commit for Phase 1 (`c19ab73` on `main`)
- [x] GitHub repository: https://github.com/ISHANT57/p21 (private), pushed on 2026-11-21
- [ ] Give the group members access to the GitHub repository
- [ ] Group members and who owns which area (see the CONTRIBUTING.md table)
- [ ] Report template, page limit and deadline from the course
- [x] Phase 5 scope: class-balanced variant added for the pretrained model at 1% (+6 runs, `*_plb`); decided on 2026-10-30
- [x] Phase 4 budget rule confirmed by the user on 2026-10-30: `max(epochs × one pass, min_steps)` (see docs/MODELS.md)

## Risks

| Risk | Mitigation |
|---|---|
| Free Colab/Kaggle sessions disconnect and have usage quotas | Run script skips finished runs and saves to Drive after each run |
| Local disk has 13 GB free | Delete download zips after extraction; build the packed cache only where training runs |
| Zenodo downloads are unreliable | Hugging Face mirror with hash check is already used as a fallback |
| 54 + 24 runs may not fit the GPU quota | Do the main grid first; optional runs only afterwards. Budget per pretrained model: about 71k optimiser steps over 18 runs (8,850 steps for each 100% run, 1,500 for each 1%/10% run). Measure throughput in the baseline run, then estimate GPU hours. |
| Kaggle has no Google Drive | Save results as notebook output (*Save Version*) or download `results/runs/` at the end of each session |

## Next actions

1. **Baseline (user, on Colab; no GitHub needed):** upload `dist/p21_code.zip` to Colab and run `notebooks/colab_setup.ipynb`; it covers setup, the smoke test and the ResNet-50 baseline. Then copy `results/runs/resnet50_rgb_100pct_seed0/` back here, or report the numbers.
2. Group access to the GitHub repo, and run the Colab setup notebook from it (Phase 2/3 Colab items).
3. After the baseline: the tuning sweep (notebook section 6, 9 runs), then freeze `lr` in `configs/*.yaml` and commit.
4. Main grid: 54 runs, which can be split across Colab, Kaggle and group members with `--shard`.
5. Pseudo-label students: `run_grid.py --plan pl` (24), plus `--plan pl --balance --models resnet50 --fractions 0.01` (6); notebook section 7.
6. Meanwhile, offline: write the parts of `report/main.tex` that do not need results (introduction, related work, data, method). After the runs: `evaluate.py`, then `make_figures.py`, then fill in the results text.

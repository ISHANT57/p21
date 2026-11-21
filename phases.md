# Project phases (P21)

This file is the plan: what each phase delivers and how we know it is finished.
Progress lives in [status.md](status.md). Tick a box here when a task is done, then add a line to status.md.

Legend: `[x]` done · `[ ]` to do · *(optional)* = only if time and compute allow.

---

## Phase 1: Repository, data download, inspection, splits

**Goal:** everyone works from the same verified data and the same fixed splits.

- [x] Repository structure, `pyproject.toml`, installable package `eurosat_le`
- [x] `requirements.txt` (minimum versions, works on Colab/Kaggle) and `requirements-lock.txt` (exact local versions)
- [x] Download script: two mirrors (Zenodo, Hugging Face), size and hash check, safe extraction
- [x] Check that the RGB and 13-band folders contain the same 27,000 sample ids
- [x] Stratified 70/15/15 split (seed 42) → `splits/split.csv`
- [x] Nested 1% / 10% / 100% labelled subsets from train only, 3 seeds → `splits/subsets/`
- [x] Leakage checks and a SHA-256 manifest (`splits/manifest.json`); `make_splits.py --check`
- [x] Dataset inspection: class counts, sample grid, spectral heatmap, duplicate check
- [x] Per-channel normalisation statistics from the train split only → `splits/norm_stats.json`
- [x] Unit tests (splits, download helpers); CI workflow
- [x] README, CONTRIBUTING, docs/DATA.md

**Done when:** `pytest` passes, `make_splits.py --check` passes, and the dataset figures exist.

---

## Phase 2: Preprocessing, augmentation, data loaders

**Goal:** one dataset class serves RGB and 13-band input, for both labelled and unlabelled data.

- [x] Pack the images into one array file per input type for fast loading (`data/cache/`): RGB 0.33 GB, 13-band 2.88 GB `uint16` → `scripts/build_cache.py`
- [x] `EuroSATDataset` reading from the packed file (memory-mapped or loaded into RAM), with a fallback to the raw files → `make_dataset(modality, ids, ...)`
- [x] Normalisation using `splits/norm_stats.json` (per band, train only)
- [x] Augmentation that works the same for 3 or 13 channels: random flips and 90° rotations (the 8 symmetries of a square), random resized crop
- [x] ~~Resize option in the data pipeline~~ → the data pipeline always works at 64 px. Models that need 224 px upsample on the GPU (Phase 3), which keeps the 2-core Colab CPU from slowing training.
- [x] Weak and strong augmentation pairs for semi-supervised learning. **Changed from the plan:** the strong policy uses the same operations for RGB and 13-band input (one brightness/contrast change shared by all bands, noise, blur, cutout), not RGB-only colour jitter, so both input types get identical augmentation.
- [x] Unlabelled-pool dataset (train ids not in the labelled subset; labels hidden)
- [x] DataLoader factory (`DataConfig`, `build_loaders`): seeded workers, optional fixed number of steps per epoch, 2 workers by default
- [x] Tests: shapes, normalised mean/std exactly 0/1 on the full train split, labelled and unlabelled sets do not overlap, cache identical to the raw files
- [x] Colab/Kaggle setup notebook (`notebooks/colab_setup.ipynb`): clone, install, Drive link, download, check, GPU batch
- [ ] Run the notebook once on Colab and once on Kaggle. **Needs the GitHub repo URL.**

**Done when:** a batch of each input type loads on CPU and in Colab, and the tests pass.

---

## Phase 3: Models and a working baseline

**Goal:** three model families run end to end and give one sanity-check result.

- [x] CNN from scratch (`scratch_cnn`: small ResNet-style network, 2.8M params, 3 or 13 channels, 64 px input)
- [x] ImageNet CNN (`resnet50`, timm `resnet50.a1_in1k`), first conv widened to 13 channels:
  - copy the pretrained RGB filters into the B04/B03/B02 positions
  - set each other band's filters to the mean of the RGB filters
  - scale all filters by 3/13 so activations keep about the same size
  - documented in `docs/MODELS.md`, including its limitations; `init_mode: zero` is available as an ablation
- [x] ViT-Small (`vit_small`, timm `vit_small_patch16_224.augreg_in21k_ft_in1k`, 224 px), patch embedding widened to 13 channels the same way
- [x] *(optional)* Small ViT from scratch (`vit_scratch`: ViT-Tiny, patch 8, 64 px). The code exists; whether to run it is decided in Phase 4.
- [x] Training loop: AdamW, cosine schedule with warm-up, mixed precision (AMP), early stopping on validation macro-F1. The best weights are kept in memory; `best.pt` is saved only on request.
- [x] Run logging: `config.yaml`, `env.json` (versions, GPU, git commit), `history.csv`, `metrics.json`, plus predictions for val, test and the unlabelled pool (`preds_*.npz`)
- [x] YAML configs in `configs/`, one per model; `scripts/train.py` takes `--modality/--fraction/--seed` and `--set key=value` overrides; finished runs are skipped
- [x] Smoke test (`scripts/smoke_test.py`): every model × input type memorises one batch on CPU with pretrained weights (8/8 pass)
- [x] Local CPU check: scratch CNN, RGB, 10% labels; validation accuracy 42% → 71% in 5 epochs
- [ ] **Baseline:** ResNet-50, RGB, 100% labels on Colab. Target ≥ 97% test accuracy (the paper reports about 98.6%). **Needs the GitHub repo** (or a manual upload) to run on Colab.

**Done when:** the baseline is reproduced and every model passes the smoke test with both input types.

---

## Phase 4: Supervised label-efficiency runs

**Goal:** the main result grid, run under the same budget for every condition.

Grid: 3 models × 2 input types × 3 label fractions × 3 seeds = **54 runs** (+18 if the optional ViT from scratch is included).

- [x] Training budget rule: every run trains for `max(epochs × one pass over its labelled set, min_steps)` optimiser steps, with the same number of epochs (validation checks) for every fraction. `min_steps` is 1,500 for the pretrained models and 3,000 for the scratch CNN. Confirmed by the user on 2026-10-30. Documented in `docs/MODELS.md`.
- [x] Tuning sweep code: 3 learning rates per model on RGB / 10% labels / seed 0, picked by validation macro-F1; test set never evaluated (`run_grid.py --plan tune`)
- [ ] Run the tuning sweep on GPU (9 runs), update `lr` in `configs/*.yaml`, commit ("freeze hyper-parameters")
- [x] Grid runner `scripts/run_grid.py`: skips finished runs, `--shard k/n` to split the grid across machines, `--max-hours` for session limits, logs failures and continues. Results go to Drive through the notebook's `results/runs` link.
- [x] Results collector `scripts/collect_results.py` → `results/summary/runs.csv`, `supervised.csv` (mean ± std over seeds), `tuning.csv`
- [ ] Run the RGB grid (27 runs)
- [ ] Run the 13-band grid (27 runs)
- [ ] *(optional)* Band ablation: 13 bands without B10, and RGB+NIR (4 bands)
- [ ] *(optional)* `init_mode: zero` vs `mean` for the 13-band pretrained models
- [ ] Gather all runs → `results/summary/supervised.csv`

**Done when:** all 54 runs have `metrics.json` and the summary table is filled.

---

## Phase 5: Semi-supervised learning (pseudo-labelling)

**Goal:** measure how much the unlabelled training images help at 1% and 10% labels. Reported **separately** from the supervised results.

- [x] Pseudo-labelling code (`src/eurosat_le/ssl/`). The Phase 4 run is the teacher; its saved predictions on the unlabelled pool are filtered by τ; a student (same model, same starting weights, same budget rule) trains on labelled + pseudo-labelled images.
- [x] τ chosen on the validation set only: the smallest τ at which the teacher's confident val predictions are ≥ 95% correct. *(optional)* class-balanced selection (`--balance`) and more rounds (`--teacher <run>_pl` gives `_pl2`) are supported.
- [x] Analysis only: pseudo-label accuracy (overall and per class) and coverage, measured against the hidden train labels after training (`pseudo_label_quality.json`)
- [x] Runner: `run_grid.py --plan pl` (24 students; each waits until its teacher has finished) and `scripts/pseudo_label.py --teacher <run>` for a single student
- [x] Collector: `results/summary/semi_supervised.csv`, with each student paired with its own supervised teacher (same seed) and the change in test macro-F1
- [ ] Choose the pretrained model for the scope (default `resnet50`; switch in `PL_MODELS` in `src/eurosat_le/train/grid.py` if ViT-Small wins in Phase 4)
- [ ] Run the 24 default students on GPU: scratch CNN + best pretrained model × RGB/13-band × 1%/10% × 3 seeds
- [ ] Class-balanced variant (decided by the user on 2026-10-30): pretrained model × RGB/13-band × 1% × 3 seeds = 6 runs (`--plan pl --balance`, students named `*_plb`)
- [ ] *(optional)* second self-training round for one model
- [ ] *(optional)* FixMatch for one model, only if Phase 5 stays on schedule (the weak/strong unlabelled loader already exists)

**Done when:** a separate table and figure show supervised vs pseudo-label results with matched settings.

---

## Phase 6: Evaluation

**Goal:** the same complete evaluation for every run, on the fixed test set.

- [x] Metrics recomputed from saved predictions: overall accuracy, macro-F1, per-class precision / recall / F1, and a cross-check against `metrics.json`
- [x] Confusion matrices: row-normalised and averaged over seeds, per group; top confused class pairs
- [x] Mean ± std over the 3 seeds for every setting (`groups.csv`, `per_class.csv`)
- [x] Paired comparisons (RGB vs 13-band, supervised vs pseudo-label, global vs balanced pseudo-label, between models), matched on seed: per-seed differences plus 95% paired bootstrap intervals on the test set
- [x] Error analysis: misclassified-image grid (`--examples <run>`); confused pairs in `top_confusions.csv`
- [x] Calibration (ECE, 15 bins) for every run
- [x] Unit tests on fake run folders built from the real test ids; method in `docs/EVALUATION.md`
- [ ] Run on the real results once Phases 4–5 finish; interpret the results in the report

**Done when:** `scripts/evaluate.py` rebuilds every metric from the saved predictions with one command.

---

## Phase 7: Figures, tables, README, report

**Goal:** submission-ready repository and IEEE report.

- [x] Figure code (`src/eurosat_le/figures.py`, `scripts/make_figures.py`): label-efficiency curves (macro-F1 and accuracy, RGB and 13-band side by side), RGB-vs-13-band gain per seed, pseudo-label gain per seed, confusion matrices, per-class F1 heatmap. Tested on synthetic tables.
- [ ] Render the figures from the real results (needs Phases 4-5)
- [x] Table code (`src/eurosat_le/tables.py`): supervised grid (macro-F1 and accuracy), semi-supervised grid, per-class F1 for the best model; Markdown and LaTeX output
- [ ] Fill the tables from the real results (needs Phases 4-5)
- [x] LaTeX report skeleton (IEEE two-column, `report/`): all sections, figure and table includes, bibliography
- [ ] Write the report text: abstract, introduction, related work, method, setup, results, discussion, limitations, conclusion (every `\TODO{}` in `report/main.tex`)
- [ ] README: final results summary, steps to reproduce every number
- [ ] Repository clean-up: remove dead code, tests pass, CI green, licence, citation
- [ ] Final check: a fresh clone + the README steps reproduce one baseline number

**Done when:** the report PDF is built and the repository is pushed.

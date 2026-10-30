okay so what next we do # Project phases (P21)

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

- [ ] Fix the training budget rule, for example the same number of optimiser steps for every fraction, with a minimum number of epochs; document it
- [ ] Tune hyper-parameters lightly on the **validation set only**, for one setting per model, then freeze them
- [ ] Run script that skips finished runs (so it can resume after a Colab disconnect) and syncs `results/runs/` to Drive
- [ ] Run the RGB grid
- [ ] Run the 13-band grid
- [ ] *(optional)* Band ablation: 13 bands without B10, and RGB+NIR (4 bands)
- [ ] Gather all runs → `results/summary/supervised.csv`

**Done when:** all 54 runs have `metrics.json` and the summary table is filled.

---

## Phase 5: Semi-supervised learning (pseudo-labelling)

**Goal:** measure how much the unlabelled training images help at 1% and 10% labels. Reported **separately** from the supervised results.

- [ ] Pseudo-labelling: the Phase 4 model (teacher) labels the unlabelled pool; keep predictions above a confidence threshold τ; retrain a student on labelled + pseudo-labelled images
- [ ] Choose τ on the validation set only; *(optional)* class-balanced thresholds, 2 rounds
- [ ] Analysis only: pseudo-label accuracy and coverage, measured against the hidden train labels
- [ ] Scope: best pretrained model and the scratch CNN × RGB/13-band × 1%/10% × 3 seeds = 24 runs (extend if compute allows)
- [ ] *(optional)* FixMatch for one model, only if Phase 5 stays on schedule
- [ ] Gather → `results/summary/semi_supervised.csv`

**Done when:** a separate table and figure show supervised vs pseudo-label results with matched settings.

---

## Phase 6: Evaluation

**Goal:** the same complete evaluation for every run, on the fixed test set.

- [ ] Metrics: overall accuracy, macro-F1, per-class precision / recall / F1
- [ ] Confusion matrices (counts and row-normalised) per run; averaged over seeds
- [ ] Mean ± std over the 3 seeds for every setting
- [ ] Paired comparisons (RGB vs 13-band, supervised vs pseudo-labelling) using the same seeds; report the per-seed differences
- [ ] Error analysis: which classes get confused (e.g. Highway / River, PermanentCrop / AnnualCrop); example images of misclassifications
- [ ] *(optional)* Calibration (ECE), since it matters for pseudo-label thresholds

**Done when:** `scripts/evaluate.py` rebuilds every metric from the saved predictions with one command.

---

## Phase 7: Figures, tables, README, report

**Goal:** submission-ready repository and IEEE report.

- [ ] Figures: label-efficiency curves (accuracy and macro-F1 against label fraction, log x-axis, one line per model, RGB and 13-band side by side), RGB-vs-13-band gain by fraction, confusion matrices, per-class F1 heatmap, pseudo-label gains
- [ ] Tables: main supervised grid, semi-supervised grid, per-class F1 for the best model
- [ ] LaTeX report (IEEE two-column, `report/`): abstract, introduction, related work, method, experimental setup, results, discussion, limitations, conclusion, references
- [ ] README: final results summary, steps to reproduce every number
- [ ] Repository clean-up: remove dead code, tests pass, CI green, licence, citation
- [ ] Final check: a fresh clone + the README steps reproduce one baseline number

**Done when:** the report PDF is built and the repository is pushed (after the account and repo are confirmed).

# Project status (P21)

**Last updated:** 2026-10-30
**Current phase:** Phase 3 code done. The ResNet-50 baseline on Colab is blocked until the code is on GitHub.

The plan and its task lists are in [phases.md](phases.md). Update this file whenever a phase task finishes or a decision is made.

## Overview

| Phase | Title | State | Done |
|---|---|---|---|
| 1 | Repository, download, inspection, splits | ✅ Done | 2026-10-30 |
| 2 | Preprocessing, augmentation, loaders | ✅ Done (local), Colab test pending | 2026-10-30 |
| 3 | Models and working baseline | ⚠️ Code done; Colab baseline blocked (no repo yet) | |
| 4 | Supervised label-efficiency runs (54) | ⬜ Not started | |
| 5 | Pseudo-labelling (24 runs) | ⬜ Not started | |
| 6 | Evaluation | ⬜ Not started | |
| 7 | Figures, tables, README, report | ⬜ Not started | |

States: ⬜ Not started · ⏳ Next / in progress · ✅ Done · ⚠️ Blocked

## Run tracker

| Grid | Planned | Done |
|---|---|---|
| Supervised, RGB | 27 | 0 |
| Supervised, 13-band | 27 | 0 |
| Pseudo-labelling | 24 | 0 |
| *(optional)* ViT from scratch | 18 | 0 |
| *(optional)* Band ablation | n/a | 0 |

## Change log

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
- [ ] GitHub account, repository name and member access. **Nothing will be pushed until these are confirmed.**
- [ ] Group members and who owns which area (see the CONTRIBUTING.md table)
- [ ] Report template, page limit and deadline from the course
- [ ] Training budget rule for Phase 4 (fixed number of steps vs. epochs)

## Risks

| Risk | Mitigation |
|---|---|
| Free Colab/Kaggle sessions disconnect and have usage quotas | Run script skips finished runs and saves to Drive after each run |
| Local disk has 13 GB free | Delete download zips after extraction; build the packed cache only where training runs |
| Zenodo downloads are unreliable | Hugging Face mirror with hash check is already used as a fallback |
| 54 + 24 runs may not fit the GPU quota | Do the main grid first; optional runs only afterwards |

## Next actions

1. **Baseline now (no GitHub needed):** upload `dist/p21_code.zip` to Colab and run `notebooks/colab_setup.ipynb`; it covers setup, the smoke test and the ResNet-50 baseline. Then copy `results/runs/resnet50_rgb_100pct_seed0/` back here, or report the numbers.
2. GitHub repo (account and repo name from the user) for the group workflow.
3. Phase 4 prep: choose the training budget rule, write a grid runner script that resumes after a disconnect, do light tuning on the validation set.

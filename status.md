# Project status (P21)

**Last updated:** 2026-10-30
**Current phase:** Phase 2, Preprocessing and data loaders (not started)

The plan and its task lists are in [phases.md](phases.md). Update this file whenever a phase task finishes or a decision is made.

## Overview

| Phase | Title | State | Done |
|---|---|---|---|
| 1 | Repository, download, inspection, splits | ✅ Done | 2026-10-30 |
| 2 | Preprocessing, augmentation, loaders | ⏳ Next | |
| 3 | Models and working baseline | ⬜ Not started | |
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

- [ ] Make the first local git commit for Phase 1?
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

1. Phase 2: packed data cache, `EuroSATDataset`, augmentation, unlabelled pool, tests.
2. Colab/Kaggle bootstrap notebook and a test run of loading data on a GPU runtime.

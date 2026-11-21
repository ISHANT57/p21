# Design decisions

This lists the main choices made in the project, why each was made, what the alternative was, and the known weakness. It is meant for review: read the linked code or document, then agree or disagree and say why. Disagreement is useful, especially where a weakness is listed.

| # | Decision | Why | Alternative | Known weakness | Where |
|---|---|---|---|---|---|
| 1 | One fixed random, class-stratified 70/15/15 split (seed 42) | Matches the original EuroSAT protocol; every run shares the same validation and test sets | Spatially disjoint split by region | Neighbouring patches look alike, so scores can be optimistic for new regions | `docs/DATA.md`, `scripts/make_splits.py` |
| 2 | Labelled subsets are nested (1% in 10% in 100%) with 3 seeds | Adding labels never removes images already in use, so curves are easier to compare | Independent random subsets per fraction | Only 3 seeds, so uncertainty is large at 1% | `docs/DATA.md`, `src/eurosat_le/data/splits.py` |
| 3 | Normalisation statistics from the training split only | Nothing from validation or test leaks into preprocessing | Statistics from all data | The statistics include images that count as unlabelled at 1% and 10% (no labels are used, so not label leakage) | `splits/norm_stats.json` |
| 4 | Models keep the best weights by validation macro-F1, with early stopping; the test set is predicted once | The test set never influences any choice | Fixed number of epochs | Validation scores wobble at 1% labels, so the selected epoch is noisy | `src/eurosat_le/train/engine.py` |
| 5 | Equal-budget rule: max(epochs x one pass, min_steps) steps | Small label sets get enough steps to converge, with the same number of validation checks for every fraction | Same number of epochs for every fraction | The 1% and 10% runs get more passes over the data than the 100% runs | `docs/MODELS.md` |
| 6 | 13-band first layer built from pretrained RGB filters (mean filter for other bands, scaled by 3/13) | Lets ImageNet weights be used with 13 bands and keeps the first-layer output size | Zero-initialise the other bands; or pretrain on multispectral data | ImageNet knows nothing about infrared; the RGB filters are weakened by the scaling | `src/eurosat_le/models/__init__.py`, `docs/MODELS.md` |
| 7 | Macro-F1 is the main metric (accuracy also reported) | Classes are mildly imbalanced and each class should count equally | Accuracy only | Macro-F1 on 1% runs is noisy | `src/eurosat_le/eval/metrics.py` |
| 8 | Pseudo-label threshold tau chosen on validation (smallest tau with at least 95% precision) | No test data is used, and as many pseudo-labels are kept as possible at a known quality | A fixed tau such as 0.95; per-class thresholds | The teacher's weights were also selected on validation, so its validation precision can be slightly optimistic | `docs/SSL.md`, `src/eurosat_le/ssl/pseudo_label.py` |
| 9 | Students start from fresh pretrained or random weights, not from the teacher | The comparison with the teacher stays like for like | Continue training the teacher | Students train on more images and steps; they run 10 epochs to limit extra compute | `docs/SSL.md` |
| 10 | Paired comparisons matched on seed, with a 95% bootstrap interval over test images | Removes variation due to different labelled subsets | Unpaired comparison of means | With 3 seeds, we report how many seeds agree and not p-values | `docs/EVALUATION.md` |
| 11 | Time cuts: 100% cells use one seed; ViT-Small has no learning-rate sweep; pseudo-labelling only at 1% and 10% | Needed to finish by the deadline | Full grid with 3 seeds everywhere | No standard deviation at 100%; ViT-Small may be under-tuned | `status.md` |

## Questions to consider while reviewing

- Which of the weaknesses above would matter most to someone reading the report?
- Is any decision missing from this list?
- Does the code do what the document says it does? Pick one row and check it.

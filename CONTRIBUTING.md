# Contributing workflow

## Branches

- `main` always runs: tests pass and the committed `splits/` files verify. Do not push directly to `main`.
- Start every piece of work on a short-lived branch from an up-to-date `main`:
  - `feat/<topic>`: new code, for example `feat/ms-loader` or `feat/pseudo-label`
  - `exp/<topic>`: experiment configs and aggregated results, for example `exp/resnet-label-curve`
  - `fix/<topic>`, `docs/<topic>`, `report/<section>`
- Open a pull request into `main`. Another group member reviews and merges it.

```bash
git switch main && git pull
git switch -c feat/ms-loader
# ... work ...
pytest -q && python scripts/make_splits.py --check
git push -u origin feat/ms-loader     # then open a PR
```

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): summary`, in the imperative mood, at most 72 characters.

```
feat(data): add 13-band GeoTIFF dataset with train-only normalisation
fix(splits): keep at least one sample per class in tiny subsets
exp(resnet): add 1/10/100% label runs, 3 seeds
docs(report): draft methodology section
```

Each commit should contain one logical change. Do not commit data, checkpoints or per-run logs; `.gitignore` covers these.

## Rules that keep results comparable

1. **Never regenerate `splits/`.** Every run loads the committed files. A training run checks them against the hashes in `manifest.json`.
2. **The test set is for final numbers only.** Use the validation set for early stopping, choosing hyper-parameters and picking pseudo-label thresholds.
3. **Same budget across conditions.** Within one comparison, every model gets the same epochs or steps and the same augmentation policy, unless the difference is documented.
4. **Seeds.** Report the mean ± std over the three labelled-subset seeds (0, 1, 2).
5. **Supervised and semi-supervised results go in separate tables.**

## Experiment logging

Each run writes `results/runs/<run_name>/` (git-ignored) with:

- `config.yaml`: the exact configuration
- `env.json`: library versions, GPU, git commit
- `metrics.json`: validation and test accuracy, macro-F1, per-class metrics
- `confusion_matrix.csv`

The run name format is `<model>_<input>_<frac>pct_seed<s>[_<method>]`, for example `resnet50_ms_1pct_seed0_pl`.

Aggregated tables and figures go in `results/summary/` and are committed. On Colab or Kaggle, copy `results/runs/` to Drive or a Kaggle dataset, so that whoever builds the tables can collect all the runs.

## Dividing work

Each phase is a separate module, so members can work in parallel without merge conflicts:

| Area | Files |
|---|---|
| Data / loaders | `src/eurosat_le/data/` |
| Models | `src/eurosat_le/models/` |
| Training / SSL | `src/eurosat_le/train/` |
| Evaluation / plots | `src/eurosat_le/eval/`, `scripts/plot_*.py` |
| Report | `report/` |

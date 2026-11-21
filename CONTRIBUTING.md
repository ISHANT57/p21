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
| Evaluation / plots | `src/eurosat_le/eval/`, `src/eurosat_le/figures.py`, `src/eurosat_le/tables.py` |
| Report | `report/` |

## Work split for four members

Replace "Member 1-4" with names. Every member gets about the same amount of GPU runs, report writing and repository work.

| | Member 1: Data | Member 2: Models | Member 3: Semi-supervised | Member 4: Evaluation and report |
|---|---|---|---|---|
| **Area to explain** (viva) | `src/eurosat_le/data/`, splits, `docs/DATA.md` | `src/eurosat_le/models/`, `train/engine.py`, `docs/MODELS.md` | `src/eurosat_le/ssl/`, `train/grid.py`, `docs/SSL.md` | `src/eurosat_le/eval/`, `figures.py`, `tables.py`, `docs/EVALUATION.md` |
| **GPU runs** | `--shard 0/4` of tune, main, pl; plus the baseline | `--shard 1/4` | `--shard 2/4` | `--shard 3/4` |
| **Report text** | Introduction, Related Work, Data and Splits | Method (models, training), Experimental Setup | Method (pseudo-labelling), Results: pseudo-labelling | Abstract, Results (supervised, bands, error analysis), Discussion, Limitations, Conclusion |
| **Repository** | Licence and `CITATION.cff`; fresh-clone check | Freeze `lr` after the tuning sweep; keep `requirements-lock.txt` accurate | Choose `PL_MODELS` after the main grid | Run `collect_results.py`, `evaluate.py`, `make_figures.py` and commit the summaries; build the PDF; final README; CI green |

Each shard is about a quarter of every grid: 54 supervised runs are 14 / 14 / 13 / 13, the 9 tuning runs 3 / 2 / 2 / 2, and the 30 pseudo-label students 8 / 8 / 7 / 7.

### Running the shards

1. **One shared Drive folder.** Member 1 creates `p21_runs` in Google Drive and shares it with the others as *Editor*. Everyone adds a shortcut to it in their *My Drive* (right-click, *Organize, Add shortcut*). The notebook then writes to the same folder from every account. Run names never collide, and finished runs are skipped.
2. **Order, with three sync points.** Run these in the notebook with your own `k`:
   - `--plan tune --shard k/4`. **Sync 1:** everyone waits until all 9 tuning runs exist. Member 2 then freezes `lr` in `configs/*.yaml`, commits and pushes, and the others pull (or re-clone) before step 3.
   - `--plan main --shard k/4`. **Sync 2:** everyone waits until all 54 runs exist. Member 3 picks the pretrained model for pseudo-labelling.
   - `--plan pl --shard k/4`, then `--plan pl --balance --models resnet50 --fractions 0.01 --shard k/4`. A student needs its teacher from any member's shard, so start this only after Sync 2.
3. **Then** Member 4 runs `collect_results.py`, `evaluate.py` and `make_figures.py` on the merged folder, commits `results/summary/`, `results/figures/` and `results/tables/`, and the Results text is written.
4. **Commits.** Each member commits their own work on their own branch with their own name. The git history records who wrote what, so do not commit another member's work under your name.


# Land-Use Classification from Satellite Imagery with Limited Labels (P21)

This project classifies Sentinel-2 image patches from **EuroSAT** into 10 land-use classes when only a few labelled examples are available. It asks two questions:

1. How does accuracy change as labels become scarce (1%, 10%, 100% of training labels) for a CNN trained from scratch, an ImageNet-pretrained CNN, and a pretrained ViT?
2. Do the 10 non-RGB Sentinel-2 bands help, compared with RGB-only input?

Reference: Helber et al., *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification*, IEEE JSTARS 2019, [arXiv:1709.00029](https://arxiv.org/abs/1709.00029).

## Status

Plan and done criteria for each phase: [phases.md](phases.md). Current progress, findings and open decisions: [status.md](status.md).

## Results (final)

Training is finished: 51 runs (42 supervised, 9 pseudo-label students). Test macro-F1 in percent, mean ± std over seeds (`(n=1)` is one seed, so no spread):

| | 1% labels | 10% labels | 100% labels |
|---|---|---|---|
| Scratch CNN, RGB | 74.6 ± 1.0 | 91.8 ± 0.3 | 97.5 (n=1) |
| Scratch CNN, 13 bands | 78.1 ± 0.7 | 95.9 ± 0.2 | 98.8 (n=1) |
| ResNet-50, RGB | 88.5 ± 0.0 | 96.4 ± 0.3 | 98.6 (n=1) |
| ResNet-50, 13 bands | 89.4 ± 0.6 | 97.3 ± 0.4 | 99.1 (n=1) |
| ViT-Small, RGB | 90.7 ± 1.0 | 97.4 ± 0.2 | 98.8 (n=1) |
| ViT-Small, 13 bands | 91.2 ± 1.3 | 97.6 ± 0.2 | 98.9 (n=1) |

- **Fewer labels:** the pretrained models lose 8 to 10 points from 100% to 1% of the labels, the scratch CNN 21 to 23.
- **Extra bands:** +3.5 to +4.2 points for the scratch CNN, about +0.9 for ResNet-50, 0.5 or less for ViT-Small (at 1% within the spread between seeds).
- **Pseudo-labelling (ResNet-50):** +2.3 to +2.4 points at 1% labels, +0.2 at 10% (two seeds). It is not done for ViT-Small, and only one scratch CNN student finished.
- **Limits:** the Colab lanes kept only their logs, so there are no bootstrap intervals (except ViT-Small RGB vs 13 bands), no calibration, and per-class results exist only for ViT-Small at 1% and 10% labels. See [docs/EVALUATION.md](docs/EVALUATION.md) ("Results taken from logs"). Figures: `results/figures/`, tables: `results/tables/`, the evidence lines from every notebook: `results/logs/`. The report is in `report/`.

## Experiment design

| Factor | Levels |
|---|---|
| Model | CNN from scratch · ResNet (ImageNet-pretrained) · ViT-Small (pretrained, 224×224) · *optional:* small ViT from scratch at 64×64 |
| Input | RGB (3 bands) · MS (13 bands); pretrained first layer widened to 13 channels |
| Labels | 1% · 10% · 100% of the training split (nested subsets) |
| Training | supervised · pseudo-labelling (semi-supervised, reported separately) |
| Seeds | 3 labelled-subset draws (0, 1, 2) |

The split is fixed: 70/15/15, stratified, seed 42. Validation and test are the same for every run. See [docs/DATA.md](docs/DATA.md) for the full procedure and the leakage checks.

## Repository layout

```
├── src/eurosat_le/          # importable package
│   ├── constants.py         # classes, bands, seeds, download mirrors + hashes
│   ├── utils.py             # seeding helpers
│   ├── viz.py               # shared figure style / colours
│   ├── models/              # scratch CNN, ResNet-50, ViT-Small, 13-band input adaptation
│   ├── train/               # TrainConfig + training engine
│   ├── eval/                # metrics, ECE, paired bootstrap, confusion matrices
│   ├── ssl/                 # pseudo-labelling: threshold, selection, student runs
│   └── data/
│       ├── download.py      # hash-verified download + safe extraction
│       ├── index.py         # sample index, RGB/MS id consistency check
│       ├── splits.py        # stratified split, nested label subsets, verification
│       ├── cache.py         # pack images into one .npy per input type
│       ├── transforms.py    # channel-agnostic weak/strong augmentation
│       ├── datasets.py      # EuroSATDataset, make_dataset, labelled/unlabelled ids
│       └── loaders.py       # DataConfig + build_loaders
├── scripts/                 # command-line entry points
│   ├── download_data.py
│   ├── make_splits.py
│   ├── inspect_dataset.py
│   ├── build_cache.py
│   ├── smoke_test.py        # overfit one batch per model x input
│   ├── train.py             # one training run
│   ├── run_grid.py          # tuning sweep / 54-run grid / pseudo-label students; resumable
│   ├── pseudo_label.py      # one pseudo-label student from a teacher run
│   ├── collect_results.py   # runs -> quick summary tables
│   ├── evaluate.py          # full evaluation from saved predictions
│   └── make_figures.py      # report figures and tables from the evaluation tables
├── configs/                 # one YAML per model
├── notebooks/colab_setup.ipynb  # Colab/Kaggle: clone, install, data, checks
├── splits/                  # COMMITTED split files: split.csv, subsets/, manifest.json, norm_stats.json
├── results/
│   ├── dataset_inspection/  # committed dataset figures + summary
│   ├── figures/, tables/    # report figures and tables (make_figures.py)
│   └── runs/                # per-run outputs (git-ignored)
├── report/                  # IEEE LaTeX report skeleton
├── tests/                   # pytest unit tests (no dataset needed)
├── docs/                    # DATA.md (data + splits), MODELS.md (models + training), SSL.md (pseudo-labelling), EVALUATION.md, FIGURES.md (every report figure and table: source CSV, columns, colour rules)
├── phases.md                # phase plan + task checklists
├── status.md                # progress, findings, open decisions
├── requirements.txt         # minimum versions (Colab/Kaggle friendly)
└── requirements-lock.txt    # exact versions of the local reference environment
```

## Setup

### Local (Ubuntu, CPU)

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate      # or: python3.12 -m venv .venv
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt -e .
pytest -q
```

**Windows:** follow [docs/WINDOWS.md](docs/WINDOWS.md).

To reproduce the exact local environment, use `uv pip install -r requirements-lock.txt -e .` instead.

### Google Colab / Kaggle (GPU)

Keep the preinstalled CUDA build of `torch`/`torchvision` and install only the rest:

```bash
!git clone <repo-url> p21 && cd p21
!pip install -q -r requirements.txt -e .
!python scripts/download_data.py --which all --delete-zip
!python scripts/make_splits.py --check
!python scripts/build_cache.py --which all
```

Or open [notebooks/colab_setup.ipynb](notebooks/colab_setup.ipynb), which does all of this and checks the GPU.

Each training run will save its own library versions and GPU type next to its results, so runs on different platforms can be traced.

## Data pipeline

```bash
python scripts/download_data.py --which all --delete-zip   # RGB ~95 MB, MS ~2.1 GB (needs ~3.5 GB free)
python scripts/make_splits.py --check                      # verify committed splits (do NOT regenerate)
python scripts/inspect_dataset.py                          # dataset figures + train-only normalisation stats
python scripts/build_cache.py --which all                  # pack images into data/cache/*.npy for fast loading
```

`make_splits.py` without `--check` was run once to create `splits/`. It refuses to overwrite the splits unless you pass `--force`. Regenerating the splits would make every earlier result incomparable.

## Training

```bash
python scripts/smoke_test.py                                    # every model x input memorises one batch
python scripts/train.py --config configs/resnet50.yaml --modality rgb --fraction 1.0 --seed 0
python scripts/train.py --config configs/vit_small.yaml --modality ms --fraction 0.01 --seed 1 --set lr=5e-5
```

```bash
python scripts/run_grid.py --plan tune                          # LR sweep (validation only), 9 runs
python scripts/run_grid.py --plan main --dry-run                # 54-run grid: what is done / to do
python scripts/run_grid.py --plan main --models resnet50 --max-hours 11   # resumable; --shard k/n to split
python scripts/run_grid.py --plan pl                            # pseudo-label students (Phase 5)
python scripts/collect_results.py                               # tables in results/summary/
python scripts/evaluate.py --examples resnet50_ms_1pct_seed0    # Phase 6: all metrics, comparisons, error grid
python scripts/make_figures.py                                  # Phase 7: results/figures/*.png, results/tables/*.tex
# Final results of this project come from the Colab logs (see docs/EVALUATION.md, 'Results taken from logs'):
#   python scripts/parse_colab_logs.py <notebooks...>; python scripts/eval_from_logs.py; python scripts/make_figures.py --fraction 0.01
```

Outputs go to `results/runs/<model>_<input>_<pct>pct_seed<s>/`. Pseudo-labelling is described in [docs/SSL.md](docs/SSL.md). See [docs/MODELS.md](docs/MODELS.md) for the models, how the first layer is set up for 13 bands, and the training details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch, commit and experiment-logging workflow.

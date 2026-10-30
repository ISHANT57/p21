# Land-Use Classification from Satellite Imagery with Limited Labels (P21)

This project classifies Sentinel-2 image patches from **EuroSAT** into 10 land-use classes when only a few labelled examples are available. It asks two questions:

1. How does accuracy change as labels become scarce (1%, 10%, 100% of training labels) for a CNN trained from scratch, an ImageNet-pretrained CNN, and a pretrained ViT?
2. Do the 10 non-RGB Sentinel-2 bands help, compared with RGB-only input?

Reference: Helber et al., *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification*, IEEE JSTARS 2019, [arXiv:1709.00029](https://arxiv.org/abs/1709.00029).

## Status

Plan and done criteria for each phase: [phases.md](phases.md). Current progress, findings and open decisions: [status.md](status.md).

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
│   ├── eval/                # metrics
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
│   └── train.py             # one training run
├── configs/                 # one YAML per model
├── notebooks/colab_setup.ipynb  # Colab/Kaggle: clone, install, data, checks
├── splits/                  # COMMITTED split files: split.csv, subsets/, manifest.json, norm_stats.json
├── results/
│   ├── dataset_inspection/  # committed dataset figures + summary
│   └── runs/                # per-run outputs (git-ignored)
├── tests/                   # pytest unit tests (no dataset needed)
├── docs/                    # DATA.md (data + splits), MODELS.md (models + training)
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

Outputs go to `results/runs/<model>_<input>_<pct>pct_seed<s>/`. See [docs/MODELS.md](docs/MODELS.md) for the models, how the first layer is set up for 13 bands, and the training details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch, commit and experiment-logging workflow.

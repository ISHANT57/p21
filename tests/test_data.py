"""Tests for transforms, datasets and loaders.

Unit tests use synthetic arrays. Tests marked `needs_cache` use the real packed
cache and are skipped when data/cache/ has not been built.
"""

import numpy as np
import pytest
import torch

from eurosat_le.data.cache import CACHE_DIR
from eurosat_le.data.datasets import EuroSATDataset
from eurosat_le.data.transforms import RandomD4, TwoViews, build_transform


@pytest.mark.parametrize("c", [3, 13])
@pytest.mark.parametrize("policy", ["weak", "strong"])
def test_transforms_keep_shape_for_any_channel_count(c, policy):
    x = torch.randn(c, 64, 64)
    y = build_transform(policy)(x)
    assert y.shape == (c, 64, 64) and torch.isfinite(y).all()


def test_d4_only_permutes_pixels():
    x = torch.arange(13 * 64 * 64, dtype=torch.float32).view(13, 64, 64)
    torch.manual_seed(0)
    for _ in range(10):
        y = RandomD4()(x)
        assert torch.equal(y.flatten().sort().values, x.flatten())
        assert torch.equal(y.mean(dim=(1, 2)), x.mean(dim=(1, 2)))  # band statistics untouched


def test_transforms_reproducible_under_seed():
    x = torch.randn(13, 64, 64)
    t = build_transform("strong")
    torch.manual_seed(1)
    a = t(x)
    torch.manual_seed(1)
    assert torch.equal(a, t(x))


def test_two_views():
    w, s = TwoViews(build_transform("weak"), build_transform("strong"))(torch.randn(3, 64, 64))
    assert w.shape == s.shape == (3, 64, 64)


def _toy_dataset(labels):
    src = np.random.default_rng(0).integers(0, 255, size=(20, 3, 64, 64)).astype(np.uint8)
    return src, EuroSATDataset(src, rows=[4, 7, 9], labels=labels, mean=[0.5] * 3, std=[0.25] * 3, scale=1 / 255)


def test_dataset_normalises_and_returns_labels():
    src, ds = _toy_dataset(labels=[1, 2, 3])
    x, y = ds[1]
    expected = (torch.from_numpy(src[7].astype(np.float32)) / 255 - 0.5) / 0.25
    assert torch.allclose(x, expected, atol=1e-5) and y == 2 and len(ds) == 3


def test_unlabelled_dataset_hides_labels():
    _, ds = _toy_dataset(labels=None)
    assert isinstance(ds[0], torch.Tensor)


# ---- integration tests on the real cache -----------------------------------------------------
needs_cache = pytest.mark.skipif(
    not ((CACHE_DIR / "rgb.npy").exists() and (CACHE_DIR / "ms.npy").exists()),
    reason="data/cache not built (run scripts/build_cache.py)",
)


@needs_cache
@pytest.mark.parametrize("modality,c", [("rgb", 3), ("ms", 13)])
def test_loaders_shapes_and_unlabelled_pool(modality, c):
    from eurosat_le.data.datasets import split_ids
    from eurosat_le.data.loaders import DataConfig, build_loaders

    cfg = DataConfig(modality=modality, fraction=0.01, subset_seed=0, batch_size=16, num_workers=0,
                     unlabelled=True, unlabelled_ratio=2)
    loaders = build_loaders(cfg)
    x, y = next(iter(loaders["train"]))
    assert x.shape == (16, c, 64, 64) and y.dtype == torch.int64
    uw, us = next(iter(loaders["unlabelled"]))
    assert uw.shape == us.shape == (32, c, 64, 64)

    lab, unl = set(loaders["train"].dataset.ids), set(loaders["unlabelled"].dataset.ids)
    assert not lab & unl, "labelled and unlabelled pools overlap"
    assert lab | unl == set(split_ids("train"))
    assert len(loaders["val"].dataset) == 4051 and len(loaders["test"].dataset) == 4050


@needs_cache
@pytest.mark.parametrize("modality", ["rgb", "ms"])
def test_train_split_is_standardised(modality):
    from eurosat_le.data.datasets import make_dataset, split_ids

    # Exact per-channel statistics over the whole train split, accumulated in float64.
    ds = make_dataset(modality, split_ids("train"))
    n, s, ss = 0, 0.0, 0.0
    for start in range(0, len(ds), 1000):
        x = torch.stack([ds.load(i) for i in range(start, min(start + 1000, len(ds)))]).double()
        n += x.numel() // x.shape[1]
        s = s + x.sum(dim=(0, 2, 3))
        ss = ss + (x**2).sum(dim=(0, 2, 3))
    mean = s / n
    std = (ss / n - mean**2).sqrt()
    assert mean.abs().max() < 1e-3, mean
    assert (std - 1).abs().max() < 1e-3, std


@needs_cache
@pytest.mark.parametrize("modality", ["rgb", "ms"])
def test_cache_matches_raw_files(modality):
    from eurosat_le.data.cache import load_cache, read_raw
    from eurosat_le.data.splits import load_split

    arr, ids = load_cache(modality), load_split()["id"]
    for row in [0, 13_500, 26_999]:
        assert np.array_equal(arr[row], read_raw(modality, ids[row]))

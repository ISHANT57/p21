"""Tests for transforms, datasets and loaders.

Unit tests use synthetic arrays. Tests marked `needs_cache` use the real packed
cache and are skipped when data/cache/ has not been built.

In plain words: the first half checks the image augmentations (flips, crops, noise) and the dataset
class on tiny made-up data, so it runs anywhere. The second half checks the real packed EuroSAT cache
(shapes, no overlap between labelled and unlabelled images, correct standardisation).
"""

# numpy: build fake image arrays; pytest: parametrise and skip tests; torch: tensors the code works on.
import numpy as np
import pytest
import torch

# CACHE_DIR: where the packed dataset lives (used to decide whether the real-data tests can run).
from eurosat_le.data.cache import CACHE_DIR
# EuroSATDataset: the class that normalises raw pixels and returns (image, label).
from eurosat_le.data.datasets import EuroSATDataset
# The augmentations under test: random flip/rotation, the weak+strong pair, and the policy factory.
from eurosat_le.data.transforms import RandomD4, TwoViews, build_transform


# Run every combination of: 3 or 13 channels (RGB / multispectral) and weak or strong augmentation.
@pytest.mark.parametrize("c", [3, 13])
@pytest.mark.parametrize("policy", ["weak", "strong"])
def test_transforms_keep_shape_for_any_channel_count(c, policy):
    """Augmentation must not change the image size and must not create NaN/inf, for RGB and 13 bands."""
    # A random fake image with c channels at the native 64x64 size (no real data needed).
    x = torch.randn(c, 64, 64)
    # Apply the weak or strong augmentation policy to it.
    y = build_transform(policy)(x)
    # Fails if a crop/rotation changes the shape or if noise/blur produces non-finite values.
    assert y.shape == (c, 64, 64) and torch.isfinite(y).all()


def test_d4_only_permutes_pixels():
    """Flips and 90-degree rotations only move pixels around; they must never change pixel values."""
    # A 13-band image where every value is different (0, 1, 2, ...), so any change in values is visible.
    x = torch.arange(13 * 64 * 64, dtype=torch.float32).view(13, 64, 64)
    # Fix the random generator so the test is repeatable.
    torch.manual_seed(0)
    # Try several random rotations/flips, because each call picks a different one of the 8 symmetries.
    for _ in range(10):
        y = RandomD4()(x)
        # Sorted values must be identical: the same pixels, just in a different order.
        assert torch.equal(y.flatten().sort().values, x.flatten())
        # Each band keeps the same mean, so the spectral statistics are untouched.
        assert torch.equal(y.mean(dim=(1, 2)), x.mean(dim=(1, 2)))  # band statistics untouched


def test_transforms_reproducible_under_seed():
    """With the same random seed the strong augmentation must give exactly the same output twice."""
    # A random fake 13-band image and the strong policy (it uses the most random operations).
    x = torch.randn(13, 64, 64)
    t = build_transform("strong")
    # Seed, augment once and remember the result.
    torch.manual_seed(1)
    a = t(x)
    # Re-seed with the same value and augment again; the result must be bit-for-bit equal.
    torch.manual_seed(1)
    assert torch.equal(a, t(x))


def test_two_views():
    """TwoViews must return a weak and a strong version of the image, both with the original shape."""
    # Build the pair from the weak and strong policies and apply it to a fake RGB image.
    w, s = TwoViews(build_transform("weak"), build_transform("strong"))(torch.randn(3, 64, 64))
    # Both views must have the same shape as the input so they can be stacked into one batch.
    assert w.shape == s.shape == (3, 64, 64)


def _toy_dataset(labels):
    """Helper: build a tiny fake EuroSATDataset (3 items taken from 20 random images) with given labels.

    Returns (raw array, dataset) so the test can compute the expected answer from the raw pixels.
    """
    # 20 fake 3-band 64x64 images with integer pixel values 0-254 (like raw uint8 data).
    src = np.random.default_rng(0).integers(0, 255, size=(20, 3, 64, 64)).astype(np.uint8)
    # The dataset exposes only rows 4, 7 and 9 of the source; mean 0.5, std 0.25, and pixels scaled by 1/255.
    return src, EuroSATDataset(src, rows=[4, 7, 9], labels=labels, mean=[0.5] * 3, std=[0.25] * 3, scale=1 / 255)


def test_dataset_normalises_and_returns_labels():
    """The dataset must scale and standardise pixels correctly, return the right label, and have the right length."""
    # labels=[1, 2, 3] belong to rows 4, 7, 9 respectively.
    src, ds = _toy_dataset(labels=[1, 2, 3])
    # Item 1 of the dataset is source row 7 with label 2.
    x, y = ds[1]
    # Expected result computed by hand: pixels / 255, minus the mean 0.5, divided by the std 0.25.
    expected = (torch.from_numpy(src[7].astype(np.float32)) / 255 - 0.5) / 0.25
    # Fails if the wrong row is picked, the maths is wrong, the label is wrong or the length is not 3.
    assert torch.allclose(x, expected, atol=1e-5) and y == 2 and len(ds) == 3


def test_unlabelled_dataset_hides_labels():
    """With labels=None the dataset must return only the image (no label), as needed for unlabelled data."""
    _, ds = _toy_dataset(labels=None)
    # A bare tensor, not an (image, label) tuple.
    assert isinstance(ds[0], torch.Tensor)


# ---- integration tests on the real cache -----------------------------------------------------
# Marker that skips a test when the packed real-data files (rgb.npy and ms.npy) have not been built yet.
needs_cache = pytest.mark.skipif(
    not ((CACHE_DIR / "rgb.npy").exists() and (CACHE_DIR / "ms.npy").exists()),
    reason="data/cache not built (run scripts/build_cache.py)",
)


@needs_cache
@pytest.mark.parametrize("modality,c", [("rgb", 3), ("ms", 13)])
def test_loaders_shapes_and_unlabelled_pool(modality, c):
    """On the real data: batches have the right shape, and labelled/unlabelled pools split train cleanly."""
    # Imported inside the test so that merely collecting the tests does not need the cache.
    # split_ids: ids of a split; DataConfig/build_loaders: describe and create the data loaders.
    from eurosat_le.data.datasets import split_ids
    from eurosat_le.data.loaders import DataConfig, build_loaders

    # Use 1% of the training labels, small batches, no worker processes, and also build an
    # unlabelled loader that serves twice as many images per step as the labelled one.
    cfg = DataConfig(modality=modality, fraction=0.01, subset_seed=0, batch_size=16, num_workers=0,
                     unlabelled=True, unlabelled_ratio=2)
    loaders = build_loaders(cfg)
    # Take one labelled batch: 16 images with c channels, and integer class labels.
    x, y = next(iter(loaders["train"]))
    assert x.shape == (16, c, 64, 64) and y.dtype == torch.int64
    # One unlabelled batch gives two views (weak, strong) of 16 * 2 = 32 images each.
    uw, us = next(iter(loaders["unlabelled"]))
    assert uw.shape == us.shape == (32, c, 64, 64)

    # Sets of image ids used as labelled and as unlabelled data.
    lab, unl = set(loaders["train"].dataset.ids), set(loaders["unlabelled"].dataset.ids)
    # Fails if any image is both labelled and unlabelled (that would leak labels into the pseudo-label pool).
    assert not lab & unl, "labelled and unlabelled pools overlap"
    # Together the two pools must cover exactly the whole train split: nothing lost, nothing extra.
    assert lab | unl == set(split_ids("train"))
    # The validation and test sets must have their fixed official sizes.
    assert len(loaders["val"].dataset) == 4051 and len(loaders["test"].dataset) == 4050


@needs_cache
@pytest.mark.parametrize("modality", ["rgb", "ms"])
def test_train_split_is_standardised(modality):
    """After normalisation the training images must have mean 0 and std 1 in every channel."""
    # Imported here so the cache is only needed when the test actually runs.
    from eurosat_le.data.datasets import make_dataset, split_ids

    # Exact per-channel statistics over the whole train split, accumulated in float64.
    # Build the (unaugmented) training dataset; its normalisation uses the saved train statistics.
    ds = make_dataset(modality, split_ids("train"))
    # n = number of pixels per channel, s = running sum of values, ss = running sum of squares.
    n, s, ss = 0, 0.0, 0.0
    # Go through the data in chunks of 1000 images so memory use stays small.
    for start in range(0, len(ds), 1000):
        # Stack one chunk into a (chunk, C, H, W) tensor in float64 for accurate sums.
        x = torch.stack([ds.load(i) for i in range(start, min(start + 1000, len(ds)))]).double()
        # Count the pixels per channel (total values divided by the number of channels).
        n += x.numel() // x.shape[1]
        # Add this chunk's per-channel sum ...
        s = s + x.sum(dim=(0, 2, 3))
        # ... and per-channel sum of squares (needed for the variance).
        ss = ss + (x**2).sum(dim=(0, 2, 3))
    # Mean = sum / count; std from the identity Var = E[x^2] - (E[x])^2.
    mean = s / n
    std = (ss / n - mean**2).sqrt()
    # Fails if any channel mean is not ~0, or any channel std is not ~1 (the printed value helps debugging).
    assert mean.abs().max() < 1e-3, mean
    assert (std - 1).abs().max() < 1e-3, std


@needs_cache
@pytest.mark.parametrize("modality", ["rgb", "ms"])
def test_cache_matches_raw_files(modality):
    """The packed cache must hold exactly the same pixels as the original image files."""
    # load_cache: read the packed array; read_raw: read one original image file; load_split: the split table.
    from eurosat_le.data.cache import load_cache, read_raw
    from eurosat_le.data.splits import load_split

    # The packed array and the list of image ids in the same row order as the cache.
    arr, ids = load_cache(modality), load_split()["id"]
    # Spot-check the first, a middle and the last row instead of all 27,000 (to keep the test fast).
    for row in [0, 13_500, 26_999]:
        # Fails if the cache row differs from the original file of the same id (wrong order or corrupt cache).
        assert np.array_equal(arr[row], read_raw(modality, ids[row]))

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


# ====================================================================================================
# CONCEPTS USED IN THIS FILE (study notes)
# ====================================================================================================
# - unit test: a small automatic check of ONE function or class, run on tiny made-up input. Most tests here
#     (transforms, TwoViews, EuroSATDataset) are unit tests.
# - integration test: a test that uses the REAL packed data and several project parts together. The three tests
#     marked needs_cache (loaders, standardisation, cache-vs-raw) are integration tests.
# - assert: a statement that must be true; if it is false pytest marks the test as FAILED and shows the values.
# - pytest.mark.parametrize: runs the same test function several times with different arguments. Two stacked
#     decorators give all combinations: 2 x 2 = 4 cases for the shape test (c = 3 or 13, policy = weak or strong).
# - skipif marker (needs_cache): a pytest marker that skips a test, with a reason, when a condition is true. Here
#     the condition is 'rgb.npy or ms.npy is missing in data/cache', so a fresh clone does not fail.
# - synthetic / fake data: made-up numbers (torch.randn, numpy random integers) used instead of real images. Enough
#     when a test checks shapes, value-preservation or arithmetic, not accuracy.
# - tensor and shape (C, H, W): a tensor is a PyTorch array. An image is (channels, height, width); here (3 or 13,
#     64, 64). 'Shape' is the tuple of its sizes.
# - channel / band: one layer of an image: RGB has 3 bands, the Sentinel-2 multispectral (MS) input has 13.
# - augmentation (weak / strong): random changes to training images. weak = random flip/rotation (RandomD4) + random
#     resized crop; strong = weak + brightness/contrast (gain and offset), Gaussian noise, blur and cutout
#     (transforms.build_transform).
# - D4 (dihedral group of a square): the 8 flips and 90-degree rotations of a square image. RandomD4 picks one at
#     random; it only moves pixels, never changes their values (test_d4_only_permutes_pixels).
# - NaN / inf (torch.isfinite): NaN = 'not a number', inf = infinity; both signal a broken calculation.
#     torch.isfinite(y).all() is True only when every value is a normal number.
# - seed / determinism / reproducibility: torch.manual_seed(n) fixes the random generator so the same 'random'
#     numbers come out again. Same seed -> same augmentation; needed to repeat experiments.
# - normalisation (standardisation): per channel: (pixel * scale - mean) / std, using the TRAIN-split statistics in
#     splits/norm_stats.json. Afterwards train data has mean 0 and std 1 in every channel (EuroSATDataset.load).
# - allclose / atol: torch.allclose(a, b, atol=1e-5) compares floats with a tiny allowed error, because float32
#     arithmetic is never exactly equal. torch.equal needs exact equality.
# - TwoViews (weak + strong view): returns (weak(x), strong(x)) for the same image; used for unlabelled data in
#     semi-supervised (consistency) training.
# - labelled vs unlabelled dataset: a labelled dataset returns (image, label). With labels=None the dataset returns
#     only the image, so the pseudo-label pool can never read true labels.
# - DataLoader / batch / DataConfig: a DataLoader hands the dataset to the model in batches (e.g. 16 images).
#     DataConfig holds the settings; build_loaders(cfg) creates the train/val/test(/unlabelled) loaders.
#     num_workers=0 means no extra processes.
# - label fraction / labelled subset: only a part of the train images (here 1%) is treated as labelled; the rest of
#     train is the unlabelled pool.
# - data leakage: information that should be hidden reaching the wrong place (e.g. an image being both labelled and
#     unlabelled, or test images in training). test_loaders_... checks the pools do not overlap.
# - cache (.npy): all images of one input type packed into one numpy file (data/cache/rgb.npy, ms.npy) so reading is
#     fast; rows follow splits/split.csv. A correct cache equals the original files pixel by pixel.
# - dtype int64: labels must be 64-bit integers, the type the loss function expects.
# - float64 accumulation / variance formula: sums are done in double precision for accuracy; the std is obtained
#     from Var = E[x^2] - (E[x])^2 (test_train_split_is_standardised).
# - import inside a test: the real-data tests import their helpers inside the function, so merely collecting the
#     tests never needs the cache.

# ====================================================================================================
# Q&A ABOUT THIS FILE (viva practice)
# ====================================================================================================
# Q: Which part of the project does this file protect?
# A: The data pipeline: the augmentations (data/transforms.py), the EuroSATDataset class (data/datasets.py), the
#    loaders (data/loaders.py) and the packed cache (data/cache.py). A silent bug here would change every experiment
#    result, so it is tested first.
# Q: How do you run it?
# A: From the repo root: pytest tests/test_data.py (pyproject.toml sets testpaths = tests). The eurosat_le package
#    must be importable, for example after pip install -e .
# Q: Why is no dataset needed?
# A: The six unit tests use random tensors and a 20-image fake numpy array. The three integration tests need the
#    packed cache and are skipped automatically when data/cache/rgb.npy or ms.npy does not exist (they also read the
#    committed files in splits/).

# Q: What does test_transforms_keep_shape_for_any_channel_count check?
# A: That build_transform('weak') and build_transform('strong') return a tensor with exactly the input shape (c, 64,
#    64) and only finite values, for c = 3 and c = 13 (4 parametrised cases).
# Q: What bug would make it fail?
# A: A crop or rotation that changes the image size (the batch could no longer be stacked), noise/blur/gain
#    producing NaN or inf, or an augmentation written only for 3-band RGB that breaks on 13 bands.
# Q: How is the input built and why is it enough?
# A: torch.randn(c, 64, 64): random numbers with mean 0 and std 1, like normalised pixels. The test checks only
#    shape and finiteness, so the picture content does not matter.
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


# Q: What does this test check?
# A: RandomD4 (random 90-degree rotation + optional flip) only re-arranges pixels: sorted values are identical and
#    every band keeps the same mean.
# Q: Why does it matter?
# A: Overhead images have no 'up', so flips and rotations are label-preserving. If this transform changed pixel
#    values it would also change the spectral information the 13-band models rely on.
# Q: How is the input built?
# A: torch.arange(13*64*64) reshaped to (13, 64, 64): every value is different, so any change of values or
#    duplication is visible. The seed is fixed to 0 and 10 random draws are tried, because each call picks one of 8
#    symmetries.
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


# Q: What does this test check?
# A: With the same torch seed, the 'strong' augmentation of the same image gives bit-for-bit the same output.
# Q: Why does it matter?
# A: The project's random numbers all come from torch's global generator, so a seed makes an experiment repeatable.
#    If a transform used an unseeded random source, two runs with the same seed would differ.
# Q: Why the strong policy?
# A: It uses the most random operations (D4, crop, gain/offset, noise, blur, cutout); if all of them are
#    reproducible the weak policy is too.
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


# Q: What does this test check?
# A: TwoViews(weak, strong) applied to a (3, 64, 64) image returns a pair whose two parts both have the shape (3,
#    64, 64).
# Q: Why does it matter?
# A: Semi-supervised training feeds both views of the same unlabelled image to the model, and the two views are
#    stacked into batches, so their shapes must be identical.
# Q: What does it NOT check?
# A: That the two views differ; only the shapes are asserted.
def test_two_views():
    """TwoViews must return a weak and a strong version of the image, both with the original shape."""
    # Build the pair from the weak and strong policies and apply it to a fake RGB image.
    w, s = TwoViews(build_transform("weak"), build_transform("strong"))(torch.randn(3, 64, 64))
    # Both views must have the same shape as the input so they can be stacked into one batch.
    assert w.shape == s.shape == (3, 64, 64)


# Q: What does this helper do?
# A: Builds a tiny fake EuroSATDataset: 20 random uint8 images (3 x 64 x 64, values 0-254) held in a plain numpy
#    array, of which the dataset exposes only rows 4, 7 and 9, with the given labels, mean 0.5, std 0.25 and scale
#    1/255. It returns (src, dataset) so a test can recompute the expected answer.
# Q: Why can a numpy array serve as the data source?
# A: EuroSATDataset only needs source[row] to return a (C, H, W) array (the real code uses NpySource or RawSource),
#    so no files are needed.
def _toy_dataset(labels):
    """Helper: build a tiny fake EuroSATDataset (3 items taken from 20 random images) with given labels.

    Returns (raw array, dataset) so the test can compute the expected answer from the raw pixels.
    """
    # 20 fake 3-band 64x64 images with integer pixel values 0-254 (like raw uint8 data).
    src = np.random.default_rng(0).integers(0, 255, size=(20, 3, 64, 64)).astype(np.uint8)
    # The dataset exposes only rows 4, 7 and 9 of the source; mean 0.5, std 0.25, and pixels scaled by 1/255.
    return src, EuroSATDataset(src, rows=[4, 7, 9], labels=labels, mean=[0.5] * 3, std=[0.25] * 3, scale=1 / 255)


# Q: What does this test check?
# A: ds[1] is source row 7 (rows = [4, 7, 9]), scaled by 1/255, minus mean 0.5, divided by std 0.25; its label is 2
#    and len(ds) == 3.
# Q: What bug would make it fail?
# A: Using the wrong row (i instead of rows[i]), forgetting the scale, subtracting the mean before scaling, dividing
#    by the wrong std, or mixing up the labels.
# Q: Which function does it exercise?
# A: EuroSATDataset.__getitem__ and .load in data/datasets.py, with transform=None (so no random augmentation).
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


# Q: What does this test check?
# A: A dataset built with labels=None returns a bare torch.Tensor from ds[0], not an (image, label) tuple.
# Q: Why does it matter?
# A: The unlabelled pool of the semi-supervised methods must never expose true labels; hiding them at construction
#    time makes accidental use impossible (no label leakage).
def test_unlabelled_dataset_hides_labels():
    """With labels=None the dataset must return only the image (no label), as needed for unlabelled data."""
    _, ds = _toy_dataset(labels=None)
    # A bare tensor, not an (image, label) tuple.
    assert isinstance(ds[0], torch.Tensor)


# Q: What is needs_cache and why is it used?
# A: A pytest skipif marker: the tests carrying it are skipped (not failed) when data/cache/rgb.npy or ms.npy does
#    not exist. A fresh clone has no data; the cache is built with scripts/build_cache.py.
# Q: Where is the condition evaluated?
# A: At import time of this file, using CACHE_DIR from eurosat_le.data.cache.
# ---- integration tests on the real cache -----------------------------------------------------
# Marker that skips a test when the packed real-data files (rgb.npy and ms.npy) have not been built yet.
needs_cache = pytest.mark.skipif(
    not ((CACHE_DIR / "rgb.npy").exists() and (CACHE_DIR / "ms.npy").exists()),
    reason="data/cache not built (run scripts/build_cache.py)",
)


# Q: What does test_loaders_shapes_and_unlabelled_pool check?
# A: On the real data, for rgb (3 channels) and ms (13): a train batch is (16, c, 64, 64) with int64 labels; an
#    unlabelled batch gives weak and strong views of 32 images; labelled and unlabelled id sets do not overlap;
#    together they equal the whole train split; val has 4051 and test has 4050 images.
# Q: What bug would make it fail?
# A: An image used both as labelled and unlabelled (data leakage), lost or extra train images, a wrong batch size or
#    ratio (2 x 16 = 32), or changed val/test sizes.
# Q: How is the config chosen?
# A: DataConfig with fraction 0.01 (1% labels), subset_seed 0, batch_size 16, num_workers 0 (no worker processes:
#    simpler and faster for a test), unlabelled=True with unlabelled_ratio=2. Function under test: build_loaders in
#    data/loaders.py.
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


# Q: What does test_train_split_is_standardised check?
# A: That after EuroSATDataset normalisation, the whole train split has mean about 0 and std about 1 in every
#    channel (error below 1e-3), for rgb and ms.
# Q: Why does it matter?
# A: It proves that splits/norm_stats.json (computed on TRAIN only) matches the cache and the SCALE used at load
#    time. A wrong scale or stale statistics would make the mean or std differ from 0 / 1.
# Q: How is it computed?
# A: Images are loaded without augmentation in chunks of 1000 (small memory), converted to float64, and per-channel
#    sums and sums of squares give mean = s/n and std = sqrt(E[x^2] - mean^2).
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


# Q: What does test_cache_matches_raw_files check?
# A: Rows 0, 13500 and 26999 of the packed cache are exactly equal (np.array_equal) to the original image file with
#    the same id, for rgb and ms.
# Q: Why does it matter?
# A: The cache must be a lossless copy in the order of splits/split.csv. If the order or a pixel were wrong, every
#    image would be paired with the wrong label.
# Q: Why only three rows?
# A: Checking all 27,000 files would be slow; first, middle and last rows catch an order or packing error. Note it
#    also needs the original images in data/raw (read_raw), not only the cache.
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

"""DataLoader factory shared by every experiment.

Pipeline position: the last step of the data pipeline. build_loaders() combines the split files,
labelled subsets, datasets (eurosat_le.data.datasets) and augmentation (eurosat_le.data.transforms)
into train/val/test loaders and, optionally, an unlabelled loader for semi-supervised training.

Design decisions:
    - The data pipeline always yields 64x64 images; resizing to 224 px for the ImageNet models is
      done on the GPU inside the model. This keeps the 2-core CPUs of Colab/Kaggle from becoming
      the bottleneck.
    - Val and test loaders are fixed (all val / all test ids, no augmentation) for every
      condition, so results are comparable. Only the training set changes with the label fraction.
    - The labelled set is a nested subset of train picked by (fraction, subset_seed); the
      unlabelled pool is the rest of train and its labels are never loaded.
    - To make every run use the same number of optimiser steps regardless of label fraction,
      `steps_per_epoch` switches the train loader to sampling with replacement with a fixed length.
    - Shuffling, sampling and worker augmentation are all seeded from DataConfig.seed.
"""

# Allow modern type-hint syntax on older Python versions.
from __future__ import annotations

# dataclass: auto-generates the constructor for the settings class below.
from dataclasses import dataclass

# torch: random generators; DataLoader: batches data; RandomSampler: draws random samples with replacement.
import torch
from torch.utils.data import DataLoader, RandomSampler

# Dataset construction, id lists for each split, augmentation builders, and the worker seeding hook.
from eurosat_le.data.datasets import labelled_ids, make_dataset, split_ids, unlabelled_ids
from eurosat_le.data.transforms import TwoViews, build_transform
from eurosat_le.utils import seed_worker

# ====================================================================================================
# CONCEPTS USED IN THIS FILE
# - DataLoader: the PyTorch object that takes a Dataset and produces batches of items, optionally shuffled and loaded
#   by several worker processes. build_loaders returns one for train, val, test and optionally unlabelled.
# - Batch / batch_size / eval_batch_size: a batch is a group of images processed together. Training uses batch_size
#   (64 by default); val/test use a larger eval_batch_size (256) because no gradients are stored, so less memory is
#   needed.
# - num_workers / persistent_workers: worker processes that prepare batches in parallel (2 by default, matching 2-core
#   Colab/Kaggle CPUs); persistent workers stay alive between epochs to avoid start-up cost.
# - pin_memory: puts batches in page-locked memory so copying to the GPU is faster; only turned on when a GPU is
#   available.
# - Seed / torch.Generator / worker_init_fn: each loader has its own seeded generator (cfg.seed + offset) controlling
#   shuffling; seed_worker seeds the random state inside each worker. Together they make augmentation and ordering
#   reproducible.
# - RandomSampler with replacement: draws items at random and may pick the same item more than once; num_samples fixes
#   how many are drawn per epoch.
# - steps_per_epoch: forces a fixed number of optimiser steps per epoch regardless of how many labelled images there
#   are, so 1%, 10% and 100% runs get the same training budget.
# - drop_last / BatchNorm: drop_last discards an incomplete last batch. A batch of size 1 breaks BatchNorm (it cannot
#   compute a variance), so it is avoided.
# - Label fraction / labelled subset / subset_seed: a nested subset of train (1%, 10%, 100%) chosen by (fraction,
#   subset_seed). `seed` (shuffling and augmentation) and `subset_seed` (which subset file) are separate on purpose.
# - Unlabelled pool and semi-supervised learning: train ids outside the labelled subset are loaded without labels, as
#   (weak view, strong view) pairs. unlabelled_ratio = 7 means the unlabelled batch is 7x the labelled batch
#   (FixMatch's mu).
# - Pseudo-labels (extra_ids / extra_labels): labels predicted by a model for some unlabelled train images; they are
#   added to the labelled training set. They must not overlap the labelled subset.
# - Augmentation policies: 'none', 'weak' or 'strong' (see data.transforms); val and test never use augmentation.
# - Source: cache vs raw / in_memory: pixels come from the packed .npy cache (default, fast) or the original image
#   files (slow); in_memory loads the whole cache into RAM.
# - dataclass: a Python decorator that generates __init__ and similar methods from the annotated fields, used for the
#   settings class DataConfig.
# ====================================================================================================
#
# FILE-LEVEL Q&A
# Q: What is this file for?
# A: It is the single factory that turns one DataConfig into the DataLoaders (train, val, test and optionally
#    unlabelled) used by every experiment.
# Q: Where does it fit in the pipeline?
# A: It is the last step of the data pipeline: it combines the split files, labelled subsets, datasets (data.datasets)
#    and augmentation (data.transforms). The training code calls build_loaders.
# Q: What does it read and write?
# A: It reads through make_dataset (split files, cache, norm stats) and writes nothing.
# Q: Why are val and test loaders identical for every condition?
# A: So results at different label fractions are evaluated on exactly the same images and are comparable; only the
#    training set changes.


# Q: What does this class do?
# A: DataConfig bundles all settings needed by build_loaders (input type, label fraction, batch sizes, augmentation,
#    seeds, pseudo-label extras, ...) with default values.
# Q: Why a dataclass instead of many function arguments?
# A: One object can be created once, passed around, copied or saved, and its fields document the allowed values;
#    adding a setting does not change every function signature.
# Q: Why are seed and subset_seed separate?
# A: seed controls shuffling and augmentation; subset_seed only picks which labelled subset file is used. Changing one
#    without the other lets us vary the training randomness while keeping the same labelled images.
@dataclass
class DataConfig:
    """Everything build_loaders needs; the field comments describe the allowed values.

    `seed` controls shuffling/sampling and augmentation, while `subset_seed` only selects which
    labelled subset file is used. They are separate fields on purpose.
    """

    modality: str = "rgb"           # "rgb" or "ms"
    fraction: float = 1.0           # labelled fraction of train: 0.01, 0.1, 1.0
    subset_seed: int = 0            # which labelled-subset draw (0, 1, 2)
    batch_size: int = 64
    eval_batch_size: int = 256
    num_workers: int = 2            # Colab/Kaggle free tiers have 2 CPU cores
    aug: str = "weak"               # policy for labelled training images
    steps_per_epoch: int | None = None  # fixed optimiser steps per epoch (sampling with replacement)
    unlabelled: bool = False        # also build the weak/strong unlabelled loader
    unlabelled_ratio: int = 7       # unlabelled batch = ratio * batch_size (FixMatch's mu)
    source: str = "cache"           # "cache" (packed .npy) or "raw" (original files)
    in_memory: bool = False         # load the whole cache into RAM instead of memory-mapping
    seed: int = 0                   # seed for shuffling and augmentation
    extra_ids: list[str] | None = None     # pseudo-labelled train ids added to the labelled set
    extra_labels: list[int] | None = None  # their pseudo-labels (aligned with extra_ids)


# Q: What does this function do?
# A: It creates a DataLoader with the project's common settings (workers, pinned memory, persistent workers, seeded
#    generator, seed_worker).
# Q: Why does each loader get its own generator and an offset?
# A: The generator is seeded with cfg.seed + offset, so shuffling is reproducible and different loaders (train,
#    unlabelled) do not share one random stream.
# Q: What are the inputs and what does it return?
# A: ds, batch_size, cfg, shuffle, sampler, drop_last, offset. It returns a DataLoader. If a sampler is given it takes
#    precedence over shuffle (shuffle is then False).
# Q: Why pin_memory only if CUDA is available?
# A: Pinned memory only helps CPU-to-GPU copies; without a GPU it would just waste memory.
def _loader(ds, batch_size, cfg, shuffle=False, sampler=None, drop_last=False, offset=0):
    """DataLoader with the project's common settings.

    The loader's own generator is seeded with cfg.seed + offset (it drives shuffling and the per-worker
    base seed); different loaders use different offsets so they do not share a random stream. A sampler,
    if given, takes precedence over `shuffle`. Workers are persistent and seeded via seed_worker.

    Args:
        ds: the Dataset to load from.
        batch_size: number of items per batch.
        cfg: the DataConfig (supplies seed and num_workers).
        shuffle: reshuffle the data every epoch (ignored when a sampler is given).
        sampler: custom sampler that decides which items are drawn.
        drop_last: drop an incomplete final batch.
        offset: number added to cfg.seed so that different loaders get different random streams.
    """
    # Random generator owned by this loader; seeded so shuffling is reproducible.
    g = torch.Generator().manual_seed(cfg.seed + offset)
    # Create the loader. pin_memory speeds up CPU->GPU copies (only useful when a GPU exists), and
    # persistent_workers keeps worker processes alive between epochs to avoid start-up cost.
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle if sampler is None else False, sampler=sampler,
        drop_last=drop_last, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0, worker_init_fn=seed_worker, generator=g,
    )


# Q: What does this function do?
# A: It builds the datasets for the labelled train subset, the full val and test splits and, optionally, the
#    unlabelled pool, and wraps each in a DataLoader.
# Q: Why is steps_per_epoch implemented with RandomSampler(replacement=True)?
# A: 1% of train is only about 2 batches. Sampling with replacement lets a tiny labelled set still fill a fixed step
#    budget, so every label fraction trains for the same number of optimiser steps.
# Q: Why must pseudo-labelled extras be outside the labelled subset?
# A: Otherwise one image would appear twice, once with its true label and once with a pseudo-label. The labelled ids
#    keep their true labels; only the extras use the supplied ones.
# Q: Why is drop_last set only when the train set is larger than one batch?
# A: To avoid a final batch of size 1, which breaks BatchNorm; if the data is smaller than one batch, dropping would
#    leave no batches at all.
# Q: What are the inputs and what does it return?
# A: cfg (a DataConfig). It returns a dict with 'train', 'val', 'test' and, if cfg.unlabelled is set and the pool is
#    not empty, 'unlabelled' (batches of (weak, strong) image pairs with no labels).
def build_loaders(cfg: DataConfig) -> dict[str, DataLoader]:
    """Return loaders: train, val, test, and (if cfg.unlabelled and the pool is non-empty) unlabelled.

    - train: labelled subset (plus pseudo-labelled extras if given), with the `cfg.aug` policy.
    - val / test: all ids of that split, no augmentation, never shuffled.
    - unlabelled: yields (weak, strong) view pairs with no labels. Its length is tied to the train
      loader: each epoch draws len(train) batches of unlabelled_ratio * batch_size images, with
      replacement.

    Pseudo-labelled extras (cfg.extra_ids / extra_labels) must be train ids outside the labelled subset
    (checked); the labelled subset keeps its TRUE labels and only the extras use the given labels.

    Args:
        cfg: a DataConfig describing modality, label fraction, batch sizes, augmentation, seeds, etc.

    Returns:
        Dict mapping "train", "val", "test" (and optionally "unlabelled") to DataLoaders.
    """
    # Settings shared by every make_dataset call below (where pixels come from).
    kw = dict(source=cfg.source, in_memory=cfg.in_memory)
    # Ids of the labelled training subset for this (fraction, subset seed).
    lab = labelled_ids(cfg.fraction, cfg.subset_seed)
    if cfg.extra_ids:
        # Extras must come from outside the labelled set, otherwise one image would appear twice
        # (once with its true label and once with a pseudo-label).
        if set(cfg.extra_ids) & set(lab):
            raise ValueError("pseudo-labelled ids overlap the labelled subset")
        # Labelled ids keep their true labels; only the extra ids get the supplied pseudo-labels.
        # Look up the true labels of the labelled ids (from split.csv) as a plain list.
        true = make_dataset(cfg.modality, lab, **kw).labels.tolist()
        # Training set = labelled ids (true labels) followed by the extra ids (pseudo-labels),
        # with the chosen augmentation applied to every item.
        train_ds = make_dataset(cfg.modality, lab + list(cfg.extra_ids), labels=true + list(cfg.extra_labels),
                                transform=build_transform(cfg.aug), **kw)
    else:
        # Normal case: only the labelled subset, with its true labels and the chosen augmentation.
        train_ds = make_dataset(cfg.modality, lab, transform=build_transform(cfg.aug), **kw)
    # Validation set: all val ids, no augmentation (the default transform is None).
    val_ds = make_dataset(cfg.modality, split_ids("val"), **kw)
    # Test set: all test ids, no augmentation; identical for every experiment.
    test_ds = make_dataset(cfg.modality, split_ids("test"), **kw)

    if cfg.steps_per_epoch:
        # Fixed number of steps per epoch independent of the subset size (1% of train is only ~2 batches);
        # sampling with replacement lets a tiny labelled set fill the step budget.
        # Seeded generator so the sampled sequence is reproducible.
        g = torch.Generator().manual_seed(cfg.seed)
        # Each epoch draws exactly steps_per_epoch * batch_size random items (repeats allowed).
        sampler = RandomSampler(train_ds, replacement=True, num_samples=cfg.steps_per_epoch * cfg.batch_size,
                                generator=g)
        # Train loader that follows that sampler.
        train = _loader(train_ds, cfg.batch_size, cfg, sampler=sampler)
    else:
        # drop_last avoids a size-1 final batch, which breaks BatchNorm.
        train = _loader(train_ds, cfg.batch_size, cfg, shuffle=True, drop_last=len(train_ds) > cfg.batch_size)
    # Collect the loaders in a dict; val and test use a bigger batch (no gradients, so less memory is needed).
    loaders = {
        "train": train,
        "val": _loader(val_ds, cfg.eval_batch_size, cfg),
        "test": _loader(test_ds, cfg.eval_batch_size, cfg),
    }

    if cfg.unlabelled:
        # Train ids that are not labelled; empty when the fraction is 100%.
        pool = unlabelled_ids(cfg.fraction, cfg.subset_seed)
        if pool:
            # Unlabelled dataset: labels hidden, each item returned as a (weak view, strong view) pair.
            u_ds = make_dataset(cfg.modality, pool, labelled=False,
                                transform=TwoViews(build_transform("weak"), build_transform("strong")), **kw)
            # Unlabelled batch size = ratio x labelled batch size.
            u_bs = cfg.unlabelled_ratio * cfg.batch_size
            # Separate seeds (cfg.seed + 1 for the sampler, offset=2 for the loader) keep the unlabelled
            # stream independent of the labelled one.
            g = torch.Generator().manual_seed(cfg.seed + 1)
            # Draw enough random items for one unlabelled batch per labelled batch in an epoch.
            u_sampler = RandomSampler(u_ds, replacement=True, num_samples=len(train) * u_bs, generator=g)
            loaders["unlabelled"] = _loader(u_ds, u_bs, cfg, sampler=u_sampler, offset=2)
    return loaders

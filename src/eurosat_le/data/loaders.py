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

"""DataLoader factory shared by every experiment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, RandomSampler

from eurosat_le.data.datasets import labelled_ids, make_dataset, split_ids, unlabelled_ids
from eurosat_le.data.transforms import TwoViews, build_transform
from eurosat_le.utils import seed_worker


@dataclass
class DataConfig:
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


def _loader(ds, batch_size, cfg, shuffle=False, sampler=None, drop_last=False, offset=0):
    g = torch.Generator().manual_seed(cfg.seed + offset)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle if sampler is None else False, sampler=sampler,
        drop_last=drop_last, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0, worker_init_fn=seed_worker, generator=g,
    )


def build_loaders(cfg: DataConfig) -> dict[str, DataLoader]:
    """Return loaders: train, val, test, and (if cfg.unlabelled and the pool is non-empty) unlabelled."""
    kw = dict(source=cfg.source, in_memory=cfg.in_memory)
    train_ds = make_dataset(cfg.modality, labelled_ids(cfg.fraction, cfg.subset_seed),
                            transform=build_transform(cfg.aug), **kw)
    val_ds = make_dataset(cfg.modality, split_ids("val"), **kw)
    test_ds = make_dataset(cfg.modality, split_ids("test"), **kw)

    if cfg.steps_per_epoch:
        g = torch.Generator().manual_seed(cfg.seed)
        sampler = RandomSampler(train_ds, replacement=True, num_samples=cfg.steps_per_epoch * cfg.batch_size,
                                generator=g)
        train = _loader(train_ds, cfg.batch_size, cfg, sampler=sampler)
    else:
        # drop_last avoids a size-1 final batch, which breaks BatchNorm.
        train = _loader(train_ds, cfg.batch_size, cfg, shuffle=True, drop_last=len(train_ds) > cfg.batch_size)
    loaders = {
        "train": train,
        "val": _loader(val_ds, cfg.eval_batch_size, cfg),
        "test": _loader(test_ds, cfg.eval_batch_size, cfg),
    }

    if cfg.unlabelled:
        pool = unlabelled_ids(cfg.fraction, cfg.subset_seed)
        if pool:
            u_ds = make_dataset(cfg.modality, pool, labelled=False,
                                transform=TwoViews(build_transform("weak"), build_transform("strong")), **kw)
            u_bs = cfg.unlabelled_ratio * cfg.batch_size
            g = torch.Generator().manual_seed(cfg.seed + 1)
            u_sampler = RandomSampler(u_ds, replacement=True, num_samples=len(train) * u_bs, generator=g)
            loaders["unlabelled"] = _loader(u_ds, u_bs, cfg, sampler=u_sampler, offset=2)
    return loaders

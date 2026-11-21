"""Seeding helpers for reproducible runs.

Runs are seeded so that a (model, input type, label fraction, seed) cell can be
repeated and give comparable results. `seed_everything` seeds the main process;
`seed_worker` is passed to DataLoaders so that augmentation randomness in worker
processes is also derived from the seed (see eurosat_le.data.loaders).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all CUDA devices) with `seed`.

    Args:
        seed: the run seed.
        deterministic: if True, also make cuDNN pick deterministic kernels and turn
            off its autotuner. This is slower and is off by default, so runs are
            reproducible in the seeding sense but not guaranteed bit-identical on GPU.

    Note: setting PYTHONHASHSEED here only affects child processes started later;
    it does not change hashing in the already-running interpreter.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: derive numpy/random seeds from torch's per-worker seed."""
    # Inside a worker, torch.initial_seed() is base_seed + worker_id, where base_seed comes from the
    # DataLoader's generator. NumPy and `random` are not seeded by torch, hence this hook. The modulo keeps
    # the value in the 32-bit range np.random.seed accepts.
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)

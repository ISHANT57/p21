"""Seeding helpers for reproducible runs.

Runs are seeded so that a (model, input type, label fraction, seed) cell can be
repeated and give comparable results. `seed_everything` seeds the main process;
`seed_worker` is passed to DataLoaders so that augmentation randomness in worker
processes is also derived from the seed (see eurosat_le.data.loaders).
"""

# Allow modern type-hint syntax (e.g. "int | None") on older Python versions.
from __future__ import annotations

# os: set an environment variable; random: Python's built-in random number generator.
import os
import random

# numpy: the array library whose random generator we also seed; torch: PyTorch, whose generators we seed.
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
    # Seed Python's own random module (used by e.g. random.shuffle).
    random.seed(seed)
    # Seed NumPy's global random generator.
    np.random.seed(seed)
    # Seed PyTorch's CPU generator (weight initialisation, dropout, torch.rand in augmentations, ...).
    torch.manual_seed(seed)
    # Seed the random generators of every GPU too (does nothing if there is no GPU).
    torch.cuda.manual_seed_all(seed)
    # Record the seed for child processes started later, so their string hashing is also reproducible.
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Force cuDNN (NVIDIA's GPU library) to use only deterministic algorithms.
        torch.backends.cudnn.deterministic = True
        # Disable the autotuner, which may pick a different (non-reproducible) kernel on each run.
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: derive numpy/random seeds from torch's per-worker seed.

    Args:
        worker_id: index of the DataLoader worker process that is starting (passed in by PyTorch).

    Returns:
        Nothing; it only seeds the NumPy and `random` generators inside that worker.
    """
    # Inside a worker, torch.initial_seed() is base_seed + worker_id, where base_seed comes from the
    # DataLoader's generator. NumPy and `random` are not seeded by torch, hence this hook. The modulo keeps
    # the value in the 32-bit range np.random.seed accepts.
    # Take this worker's torch seed and shrink it to 32 bits.
    s = torch.initial_seed() % 2**32
    # Give NumPy the same seed so its random calls differ between workers but repeat between runs.
    np.random.seed(s)
    # Do the same for Python's random module.
    random.seed(s)

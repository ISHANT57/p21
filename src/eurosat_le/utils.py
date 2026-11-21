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

# ====================================================================================================
# CONCEPTS USED IN THIS FILE
# - Seed: a starting number for a random number generator; the same seed gives the same 'random' sequence, so a run
#   can be repeated. Used by seed_everything and seed_worker.
# - Reproducibility: being able to repeat an experiment and get comparable results. Here it means seeding every random
#   generator (Python, NumPy, PyTorch CPU/GPU).
# - Random number generators (RNGs): Python's `random`, NumPy's `np.random` and PyTorch's `torch` each keep their own
#   separate state, so each has to be seeded on its own.
# - CUDA / cuDNN: CUDA is NVIDIA's GPU platform; cuDNN is its library of neural-network routines. cuDNN can pick fast
#   but non-deterministic algorithms (benchmark) unless told otherwise (deterministic).
# - Deterministic vs benchmark mode: deterministic=True forces repeatable algorithms; benchmark=True lets cuDNN time
#   several algorithms and choose the fastest, which may differ between runs. Both are changed only when
#   deterministic=True is requested.
# - DataLoader worker: a separate process PyTorch starts to prepare batches in parallel (num_workers). Each worker has
#   its own random state, so it needs its own seeding (seed_worker).
# - PYTHONHASHSEED: an environment variable that fixes Python's string hashing (affects set/dict ordering of strings).
#   Set here only for child processes started later.
# - worker_init_fn: a function PyTorch's DataLoader calls once in each worker when it starts; seed_worker is passed
#   there.
# ====================================================================================================
#
# FILE-LEVEL Q&A
# Q: What is this file for?
# A: It holds the two seeding helpers that make runs repeatable: seed_everything for the main process and seed_worker
#    for DataLoader worker processes.
# Q: Where does it fit in the pipeline?
# A: The training engine (train/engine.py) and scripts/smoke_test.py call seed_everything; data.loaders passes
#    seed_worker to every DataLoader.
# Q: What does it read and write?
# A: No files. It only sets random-generator states, one environment variable, and (optionally) two cuDNN flags.


# Q: What does this function do?
# A: It seeds Python's random, NumPy's random, PyTorch's CPU generator and all GPU generators with the same number,
#    and records the seed in PYTHONHASHSEED.
# Q: Why is it needed? What would go wrong without it?
# A: Weight initialisation, shuffling and augmentation are random. Without seeding, two runs of the same (model, input
#    type, label fraction, seed) cell would give different results and could not be compared fairly.
# Q: What are the inputs and what does it return?
# A: seed (int) and deterministic (bool, default False). It returns None; it only changes global state.
# Q: Why is deterministic off by default?
# A: Deterministic cuDNN kernels and no autotuning are slower. So runs are reproducible in the seeding sense but not
#    guaranteed bit-identical on GPU.
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


# Q: What does this function do?
# A: It runs inside each DataLoader worker process and seeds NumPy and Python's random from that worker's PyTorch
#    seed.
# Q: Why is this step needed?
# A: PyTorch seeds only its own generator in workers, not NumPy or `random`. Without this hook, the workers' other
#    random calls would not be controlled by the run seed.
# Q: What are the inputs and what does it return?
# A: worker_id (int, supplied by PyTorch, unused here because torch.initial_seed() already includes it). It returns
#    None.
# Q: Why take the seed modulo 2**32?
# A: np.random.seed only accepts 32-bit values, while torch's seed can be larger.
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

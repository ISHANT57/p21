"""Supervised training loop with early stopping and complete run logging.

This is the single training routine of the project (Phases 3-5): the supervised grid, the tuning
sweep and the pseudo-label students (Phase 5) all go through `run`. A student differs only in its
config (`pseudo_labels` set), so its training set is the labelled subset plus pseudo-labelled images.

How a run works
- Optimiser: AdamW, no weight decay on biases / norm layers / position embeddings / class token.
- Schedule: linear warm-up, then cosine decay to 0, stepped once per batch (not per epoch).
- Budget: total steps = epochs x steps_per_epoch, where steps_per_epoch comes from the budget rule
  max(epochs x one pass, min_steps) in `TrainConfig.resolve_steps_per_epoch`.
- Precision: fp16 autocast with a GradScaler, only on CUDA (a CPU run silently uses fp32).
- Model selection: after every epoch the model is scored on the validation set; the weights with
  the best validation macro-F1 are kept and training stops early after `patience` epochs without
  improvement. The best weights are held in memory (a CPU copy of the state dict) rather than in
  a checkpoint file, so no large files are written unless `save_checkpoint` is set.
- Test set: predicted exactly once, after training, with the selected weights. Nothing in the loop
  reads it, and tuning runs (`evaluate_test: false`) never touch it.
- Seeding: `seed_everything(cfg.seed)` seeds Python, NumPy and PyTorch, but cuDNN determinism is
  not forced, so GPU runs with the same seed are close but not guaranteed to be bit-identical.

Skip-finished-runs behaviour: metrics.json is written last, so its presence marks a complete run.
`run` returns the stored metrics instead of retraining unless `overwrite=True`. A run that crashed
part-way leaves its other files behind but no metrics.json, so it is simply started again.

A run writes results/runs/<run_name>/:
    config.yaml      exact TrainConfig
    env.json         library versions, device, git commit
    history.csv      per-epoch train loss/accuracy and val accuracy/macro-F1
    preds_val.npz    logits + labels + ids of the best model on val
    preds_test.npz   same on test
    preds_unlabelled.npz  logits + ids on the unlabelled train pool (fraction < 1),
                     used as teacher predictions for pseudo-labelling
    metrics.json     final val/test metrics (written last: marks the run complete)
Student runs additionally contain pseudo_labels.csv and the other pseudo_label_* files written by
`eurosat_le.ssl.pseudo_label`, including pseudo_label_quality.json (see `run`). best.pt appears only
when `save_checkpoint` is true. Logits in the .npz files are stored as float16 to save space.
"""

# Makes type hints such as `int | None` work as plain text on older Python versions.
from __future__ import annotations

# Standard library: copy (deep copy of the best weights), json (write metrics), math (cosine),
# platform/subprocess (machine and git info), time (timing), Path (file paths).
import copy
import json
import math
import platform
import subprocess
import time
from pathlib import Path

# Third-party: numpy (arrays), pandas (history.csv), timm (model library, only for its version),
# torch (deep-learning framework), yaml (write config.yaml), nn (layers and loss), DataLoader (batching).
import numpy as np
import pandas as pd
import timm
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

# Project code: folder constants, dataset helpers (which ids are labelled / unlabelled), data loaders,
# split-file integrity check, metric functions, the model factory, pseudo-label helpers, the config class
# and the random-seed helper.
from eurosat_le.constants import REPO_ROOT, RESULTS_DIR
from eurosat_le.data.datasets import labelled_ids, make_dataset, unlabelled_ids
from eurosat_le.data.loaders import build_loaders
from eurosat_le.data.splits import check_integrity
from eurosat_le.eval.metrics import classification_metrics, summary
from eurosat_le.models import build_model, count_params
from eurosat_le.ssl.pseudo_label import load_pseudo_labels, quality
from eurosat_le.train.config import TrainConfig
from eurosat_le.utils import seed_everything

# ======================================== CONCEPTS USED IN THIS FILE ========================================
# epoch: one round of training (a number of optimiser steps) followed by one validation check. cfg.epochs is the maximum
#   number of rounds.
# batch / step: a batch is a group of images processed together; each batch gives one optimiser step (one weight
#   update). total_steps = epochs x steps_per_epoch.
# training budget rule: total steps = max(epochs x one pass, min_steps); small label fractions get stretched epochs
#   (sampling with replacement); decided in TrainConfig.resolve_steps_per_epoch.
# logits: the raw, unnormalised scores the network outputs, one per class. The predicted class is the argmax (index of
#   the largest logit).
# softmax / confidence: softmax turns logits into probabilities summing to 1; CrossEntropyLoss applies it internally.
#   Confidence = the largest probability (used later for pseudo-labels).
# cross-entropy loss: the classification loss; large when the model gives low probability to the true class. The
#   training signal that gradients come from.
# label smoothing: softens the one-hot targets slightly (cfg.label_smoothing, 0 = off) so the model is not pushed to be
#   over-confident.
# gradient / backpropagation: how much the loss changes with each weight; loss.backward() computes it, the optimiser
#   then moves the weights against it.
# optimiser / AdamW: the algorithm that updates weights from gradients. AdamW adapts the step per weight and applies
#   weight decay separately (decoupled) from the gradient.
# weight decay: a small pull of weights towards 0 each step that fights over-fitting. Not applied to biases, norm-layer
#   scales, pos_embed and cls_token (param_groups).
# learning rate (lr): how big each update is. Warm-up: lr rises linearly from ~0 over the first steps (warmup_frac of
#   all steps) so early noisy gradients do not wreck pretrained weights.
# cosine schedule / LambdaLR: after warm-up lr decays along a half cosine from lr to 0. LambdaLR multiplies the base lr
#   by factor(step); stepped once per batch, not per epoch.
# mixed precision / autocast / fp16 / GradScaler: autocast runs the forward pass in 16-bit where safe (faster, less
#   memory); GradScaler scales the loss so tiny fp16 gradients do not underflow to 0. CUDA only.
# model.train() vs model.eval(): train mode keeps dropout on and lets batch-norm update running statistics; eval mode
#   turns dropout off and uses the stored statistics.
# torch.no_grad: switches off gradient tracking during prediction to save memory and time.
# validation set vs test set: validation is scored every epoch to choose the best weights and to stop early; the test
#   set is predicted exactly once at the end with the selected weights and never used for any decision.
# accuracy vs macro-F1: accuracy = share correct; macro-F1 = unweighted mean of per-class F1, so every class counts
#   equally. Macro-F1 on validation is the model-selection metric.
# model selection / best weights / state_dict: after each epoch, if validation macro-F1 is a new best, a CPU copy of the
#   weights (state_dict) is kept; at the end the model is reloaded with them.
# early stopping / patience: stop training when validation macro-F1 has not improved for cfg.patience epochs in a row
#   (bad_epochs counter).
# seed / reproducibility: seed_everything fixes Python, NumPy and PyTorch randomness; env.json records versions, device
#   and git commit. GPU runs are close but not guaranteed bit-identical.
# labelled subset / unlabelled pool / pseudo-labels / teacher / student: a run trains on the labelled subset (fraction
#   of train images); the rest is the pool; a teacher predicts the pool and confident predictions become pseudo-labels;
#   a student trains on labelled + pseudo-labelled images.
# in_chans = 3 or 13: RGB or all 13 Sentinel-2 bands; pretrained: start from ImageNet weights (models/__init__.py adapts
#   the first layer for 13 bands).
# run folder / metrics.json as completion marker: each run writes results/runs/<run_name>/; metrics.json is written
#   last, so its presence means the run finished and it can be skipped.
# .npz / float16 logits: compressed NumPy file; logits stored in half precision to save space, ids kept so predictions
#   can be matched to images.
# DataLoader: PyTorch object that hands out batches of (images, labels); built by data/loaders.py.

# ======================================== Q&A ABOUT THIS FILE ========================================
# Q: What is this file for?
# A: It is the one training routine of the project: `run(cfg)` trains a model, picks the best epoch on validation,
#   predicts val/test (and the unlabelled pool) and writes everything into the run folder.
# Q: Where does it fit in the pipeline?
# A: scripts/run_grid.py (supervised grid and tuning) and ssl/self_training.py (pseudo-label students) both call `run`.
#   Its saved predictions are the input of eval/evaluate.py and ssl/pseudo_label.py.
# Q: What does it read and write?
# A: It reads the split files (via the data loaders) and, for a student, pseudo_labels.csv. It writes config.yaml,
#   env.json, history.csv, preds_val/test/unlabelled.npz, metrics.json (last) and optionally best.pt.
# Q: Why is the test set only used at the very end?
# A: If the test set influenced any choice (best epoch, stopping, learning rate) the final test score would be
#   optimistic. Here selection uses validation only, so the test score is an honest estimate.

# Parent folder under which every run gets its own sub-folder.
RUNS_DIR = RESULTS_DIR / "runs"


# Q: What does _read do?
# A: It returns the stripped text of a file, or "" if the file does not exist.
# Q: Why not just read the file directly?
# A: A missing COMMIT file (only present in code zips) must not crash the run; an empty string lets `environment` fall
#   back gracefully.
def _read(path: Path) -> str:
    """Stripped text of `path`, or "" if it does not exist.

    Args:
        path: the file to read.
    Returns:
        The file's text without leading/trailing whitespace, or an empty string if there is no such file.
    """
    # Read only if the file exists, so a missing file gives "" instead of an error.
    return path.read_text().strip() if path.exists() else ""


# Q: What does environment() return and why save it?
# A: A dict with Python / torch / timm / numpy versions, CUDA version, device name, git commit and whether the code has
#   uncommitted changes. It is saved as env.json so a result can be traced to the exact software and code.
# Q: What does git_dirty tell us?
# A: True means tracked files had uncommitted changes, so the commit hash alone would not reproduce the exact code used.
# Q: Why is there a fallback to a COMMIT file?
# A: Code zips made for the compute server have no .git folder, so scripts/make_code_zip.sh writes the commit hash into
#   a COMMIT file instead.
def environment() -> dict:
    """Versions, device and git state of this machine, saved as env.json for reproducibility.

    Returns:
        A dict with the Python / torch / timm / numpy versions, the CUDA version, the device name,
        the current git commit and whether there are uncommitted changes.
    """
    # Q: What does this inner helper do?
    # A: It runs `git <args>` in the repository folder and returns the printed text, or "" if git is not installed.
    # Q: Why catch OSError?
    # A: When git is missing, subprocess raises OSError; reporting "" is better than crashing a training run just for
    #   metadata.
    def git(*args):
        """Run `git <args>` in the repository and return its output text ("" if git is unavailable)."""
        try:
            # Run the git command, capture what it prints and strip the trailing newline.
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            # git is not installed (OSError): report nothing instead of crashing.
            return ""

    # True when a CUDA GPU can be used.
    cuda = torch.cuda.is_available()
    # Collect everything that could affect a result, so the run can be reproduced or explained later.
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda if cuda else None,
        "device": torch.cuda.get_device_name(0) if cuda else platform.processor() or "cpu",
        # COMMIT is written into code zips made by scripts/make_code_zip.sh (no .git there)
        "git_commit": git("rev-parse", "HEAD") or _read(REPO_ROOT / "COMMIT"),
        # True if tracked files have uncommitted changes (so the commit hash alone would not match the code).
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


# Q: What does param_groups do?
# A: It splits the model's trainable parameters into two lists for AdamW: ordinary weights (with weight decay) and 1-D /
#   embedding parameters (without decay).
# Q: Why not decay biases, norm scales, pos_embed and cls_token?
# A: Shrinking them towards 0 hurts more than it helps: they are shifts/scales or learned embeddings, not large weight
#   matrices that need regularising. This is the standard recipe for transformers.
# Q: How are those parameters recognised?
# A: Biases and norm scales have ndim <= 1; pos_embed and cls_token are higher-dimensional, so they are matched by name.
#   Frozen parameters (requires_grad False) are skipped.
# Q: What is returned?
# A: A list of two dicts in the per-group format torch.optim.AdamW accepts.
def param_groups(model: nn.Module, weight_decay: float):
    """Two AdamW parameter groups: weights with decay, and 1-D / embedding parameters without.

    Biases and norm-layer scales are 1-D (ndim <= 1); `pos_embed` and `cls_token` are matched by name
    because they are higher-dimensional but should not be decayed. Frozen parameters are skipped.

    Args:
        model: the network whose parameters are grouped.
        weight_decay: strength of the weight decay applied to the first group.
    Returns:
        A list of two dicts in the format `torch.optim.AdamW` accepts for per-group settings.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        # Frozen parameters are not trained, so the optimiser does not need them.
        if not p.requires_grad:
            continue
        # 1-D tensors (biases, norm-layer scales) and the ViT position embedding / class token are not
        # decayed: shrinking them towards 0 hurts more than it helps.
        if p.ndim <= 1 or name.endswith(("pos_embed", "cls_token")):
            no_decay.append(p)
        else:
            # Ordinary weight matrices / convolution filters get weight decay.
            decay.append(p)
    # Two groups: the first uses the configured weight decay, the second uses none.
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


# Q: What does this function do?
# A: It creates the learning-rate schedule: a linear warm-up to the base lr over warmup_steps, then a cosine decay down
#   to 0 at total_steps.
# Q: Why warm up, and why cosine decay?
# A: Warm-up avoids large destructive updates at the start (especially to pretrained weights); decaying to 0 lets the
#   model settle into a good solution at the end.
# Q: Why is the multiplier (step + 1) / warmup_steps?
# A: The "+1" means the very first step already has a non-zero learning rate instead of exactly 0.
# Q: When should scheduler.step() be called?
# A: Once per batch (optimiser step), not once per epoch. If early stopping ends the run, the decay has simply not
#   reached 0.
def cosine_with_warmup(optimizer, total_steps: int, warmup_steps: int):
    """LambdaLR schedule over optimiser steps: linear warm-up to the base lr, then cosine decay to 0.

    The multiplier is (step + 1) / warmup_steps during warm-up (so the very first step already has a
    non-zero lr), then 0.5 * (1 + cos(pi * progress)) with progress running from 0 to 1 over the
    remaining steps. Call `scheduler.step()` once per batch. If early stopping ends the run, the
    decay has simply not reached 0.

    Args:
        optimizer: the optimiser whose learning rate is scheduled.
        total_steps: planned number of optimiser steps in the whole run.
        warmup_steps: number of steps at the start spent raising the learning rate.
    Returns:
        A `LambdaLR` scheduler (a scheduler multiplies the base learning rate by a factor each step).
    """
    # Q: What does this inner function compute?
    # A: The factor (0 to 1) by which the base learning rate is multiplied at optimiser step `step`: warm-up ramp first,
    #   then 0.5 * (1 + cos(pi * progress)).
    # Q: Why min(1.0, progress) and max(1, ...)?
    # A: To keep the cosine within its range if training goes past the planned steps, and to avoid dividing by zero when
    #   warm-up covers all steps.
    def factor(step):
        """Learning-rate multiplier (between 0 and 1) for optimiser step number `step`."""
        # Warm-up: climb linearly so early, noisy gradients do not wreck the pretrained weights.
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        # After warm-up: how far (0..1) we are through the remaining steps.
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        # Cosine decay: starts at 1, smoothly falls to 0 at the end of training.
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    # Hand the multiplier function to PyTorch's LambdaLR scheduler.
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# Q: What does predict do?
# A: It runs the model over a data loader and returns all logits (float32 NumPy, N x classes) and the true labels (empty
#   array for an unlabelled loader).
# Q: Why model.eval() and torch.no_grad()?
# A: eval() turns off dropout and makes batch-norm use its stored statistics so predictions are deterministic; no_grad()
#   skips gradient bookkeeping, saving memory and time.
# Q: How does it tell a labelled loader from an unlabelled one?
# A: A labelled loader yields (images, labels) pairs; an unlabelled one yields bare tensors, which is checked with
#   isinstance(batch, torch.Tensor).
# Q: Why does it return logits rather than predicted classes?
# A: Logits keep all information: argmax gives the class, softmax gives confidence (needed for pseudo-labels and
#   calibration/ECE later).
@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits, labels) as NumPy arrays; labels is empty for an unlabelled loader.

    Logits are float32 (cast back from fp16 autocast), in the loader's order. An unlabelled loader
    yields bare tensors instead of (x, y) pairs, which is how the two cases are told apart.

    Args:
        model: the network to evaluate.
        loader: yields batches of images (with labels, or without for an unlabelled pool).
        device: where the model lives (CPU or CUDA GPU).
        amp: use mixed precision (fp16 autocast) during the forward pass.
    The `@torch.no_grad()` decorator switches off gradient tracking, which saves memory and time.
    """
    # eval mode turns off dropout and makes batch-norm use its stored statistics.
    model.eval()
    logits, labels = [], []
    for batch in loader:
        # A labelled loader gives (images, labels); an unlabelled one gives just the image tensor.
        x, y = (batch, None) if isinstance(batch, torch.Tensor) else batch
        # autocast runs the forward pass in half precision (fp16) where it is safe: faster, less memory.
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            # Forward pass: images in, logits (raw class scores, one per class) out.
            out = model(x.to(device, non_blocking=True))
        # Back to float32 on the CPU so the results can be stored as NumPy arrays.
        logits.append(out.float().cpu())
        if y is not None:
            labels.append(y)
    # Join all batches into single arrays; with no labels return an empty array.
    return torch.cat(logits).numpy(), (torch.cat(labels).numpy() if labels else np.array([]))


# Q: What does this function do?
# A: It trains the model for one epoch: for each batch, forward pass, loss, backpropagation, optimiser step, scheduler
#   step; it returns the mean training loss and accuracy.
# Q: Why zero_grad, scaler.scale(...).backward(), scaler.step and scaler.update in this order?
# A: Gradients accumulate by default, so they are cleared first; the loss is scaled before backward so fp16 gradients do
#   not vanish; scaler.step applies the update (skipped if values overflowed); update adjusts the scale for the next
#   batch.
# Q: Is the returned training accuracy a performance estimate?
# A: No. It is measured on augmented images in train mode while the weights change, so it is only a rough progress
#   signal. The real estimate is the validation / test score.
# Q: What is max_batches?
# A: A debug cap to stop an epoch early for quick tests; it is None in normal runs.
def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, device, amp, max_batches=None):
    """One pass over `loader` (an "epoch" may be a fixed number of sampled steps, see the budget rule).

    The scheduler is stepped after every batch. `max_batches` is a debug cap. Returns the mean
    training loss and accuracy; the accuracy is measured on the augmented batches in train mode, so
    it is only a rough progress signal, not a performance estimate.

    Args:
        model: the network being trained (its weights are updated in place).
        loader: yields (images, labels) training batches.
        optimizer: the AdamW optimiser that updates the weights.
        scheduler: the learning-rate schedule, advanced after every batch.
        scaler: GradScaler, which scales the loss in fp16 training so small gradients do not vanish.
        criterion: the loss function (cross-entropy).
        device: CPU or CUDA GPU.
        amp: use mixed precision.
        max_batches: optional cap on batches for quick debugging.
    Returns:
        {"loss": mean loss per image, "accuracy": fraction of images classified correctly}.
    """
    # train mode turns on dropout and lets batch-norm update its running statistics.
    model.train()
    # Running totals: summed loss, number of correct predictions, number of images seen.
    loss_sum, correct, n = 0.0, 0, 0
    for i, (x, y) in enumerate(loader):
        # Debug option: stop early after `max_batches` batches.
        if max_batches is not None and i >= max_batches:
            break
        # Move this batch of images and labels onto the GPU (or stay on CPU).
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # Forward pass under autocast: compute logits, then the loss between logits and true labels.
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out = model(x)
            loss = criterion(out, y)
        # Clear the gradients left over from the previous step (set_to_none is a faster way to do so).
        optimizer.zero_grad(set_to_none=True)
        # Backpropagation: compute how the loss changes with every weight (the gradients).
        scaler.scale(loss).backward()
        # Optimiser step: nudge every weight a little against its gradient (skipped if fp16 overflowed).
        scaler.step(optimizer)
        # Let the scaler adjust its scale factor for the next batch.
        scaler.update()
        # Move the learning rate one step along the warm-up / cosine schedule.
        scheduler.step()
        # Accumulate statistics: loss weighted by batch size, and the count of correct argmax predictions.
        loss_sum += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    # Epoch averages (max(n, 1) avoids dividing by zero for an empty loader).
    return {"loss": loss_sum / max(n, 1), "accuracy": correct / max(n, 1)}


# Q: What does _save_preds do?
# A: It writes logits (as float16), labels and sample ids to a compressed .npz file.
# Q: Why save logits instead of only metrics?
# A: Phase 6 re-computes all metrics (accuracy, F1, ECE, confusion matrices, paired bootstraps) from these files without
#   retraining, and the unlabelled-pool logits become pseudo-labels.
# Q: Why float16 and why save the ids?
# A: float16 halves the file size; the ids let predictions be matched to images and aligned across runs.
def _save_preds(path: Path, logits, labels, ids):
    """Save logits (as float16), labels and sample ids to a compressed .npz; Phase 6 re-computes all metrics from it.

    Args:
        path: output .npz file.
        logits: (N, classes) raw model scores.
        labels: (N,) true labels (an empty array for the unlabelled pool).
        ids: the N sample ids, in the same order as the logits.
    """
    # float16 halves the file size; the ids are stored so predictions can be matched to images later.
    np.savez_compressed(path, logits=logits.astype(np.float16), labels=labels, ids=np.asarray(ids))


# Q: What does run do, in order?
# A: Skips if finished; saves config and environment; checks split integrity and seeds everything; builds loaders,
#   model, AdamW, cosine schedule, GradScaler and loss; trains epoch by epoch with validation, best-weights tracking and
#   early stopping; reloads the best weights; predicts val (and test, and the unlabelled pool); writes metrics.json
#   last.
# Q: Why is metrics.json written last?
# A: Its presence marks a complete run, so a crashed run (no metrics.json) is simply started again while a finished one
#   is skipped.
# Q: Why are the best weights kept in memory and the test set predicted only after training?
# A: Keeping a deep CPU copy avoids a large checkpoint file; predicting the test set once, with the validation-selected
#   weights, means nothing about the test set can influence training or selection.
# Q: What are the inputs and the return value?
# A: cfg (TrainConfig), overwrite (retrain even if finished) and runs_dir (parent folder); it returns the metrics dict
#   that is also saved as metrics.json.
# Q: How does a pseudo-label student differ?
# A: Only through the config: cfg.pseudo_labels names a CSV in the run folder; its ids/labels are added to the training
#   set, and pseudo-label quality is written after training for analysis only.
def run(cfg: TrainConfig, overwrite: bool = False, runs_dir: Path = RUNS_DIR) -> dict:
    """Train one run described by `cfg`, write its run folder and return the metrics dict.

    Args:
        cfg: the run configuration (a student when `cfg.pseudo_labels` names a CSV in the run folder).
        overwrite: retrain even if metrics.json already exists.
        runs_dir: parent folder of run folders (default results/runs).
    Returns:
        The contents of metrics.json: best epoch, step counts, timing, and the val (and, unless
        `evaluate_test` is false, test) metrics of the best-validation-F1 weights. For a skipped run,
        the stored file is returned unchanged.
    Side effects: creates the run folder; writes config.yaml and env.json before training, history.csv
    after every epoch, and the prediction files and metrics.json at the end.
    """
    # Each run has its own folder named after the run.
    run_dir = runs_dir / cfg.run_name
    # metrics.json is written last, so if it exists the run is complete: skip it and return the stored result.
    if (run_dir / "metrics.json").exists() and not overwrite:
        print(f"[skip] {cfg.run_name} already finished")
        return json.loads((run_dir / "metrics.json").read_text())
    # Create the run folder and record exactly what was run: the config and the machine / library versions.
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    env = environment()
    (run_dir / "env.json").write_text(json.dumps(env, indent=2) + "\n")

    check_integrity()  # refuse to train if the committed split files no longer match splits/manifest.json
    # Fix the random seeds (weight initialisation, shuffling, augmentation) so the run is repeatable.
    seed_everything(cfg.seed)
    # Use the GPU if there is one, otherwise the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg.amp and device.type == "cuda"  # autocast / GradScaler are only enabled on CUDA
    n_lab = len(labelled_ids(cfg.fraction, cfg.seed))  # size of the nested labelled subset for this seed
    # A student loads its pseudo-labelled ids and labels from the CSV in its folder; a normal run has none.
    extra_ids, extra_labels = (load_pseudo_labels(run_dir / cfg.pseudo_labels) if cfg.pseudo_labels else ([], []))
    # The budget rule counts labelled + pseudo-labelled images, so a student usually gets more steps than its teacher.
    dc = cfg.data_config(cfg.resolve_steps_per_epoch(n_lab + len(extra_ids)))
    # Hand the pseudo-labelled images to the data loader as extra training examples (None when there are none).
    dc.extra_ids, dc.extra_labels = extra_ids or None, extra_labels or None
    # Build the train / val / test data loaders (they hand out batches of images and labels).
    loaders = build_loaders(dc)
    in_chans = 3 if cfg.modality == "rgb" else 13  # "ms" = all 13 Sentinel-2 bands
    # Create the network with the right number of input channels, and move it to the GPU if available.
    model = build_model(cfg.model, in_chans, pretrained=cfg.pretrained, init_mode=cfg.init_mode,
                        input_size=cfg.input_size).to(device)

    # Batches per epoch: what the train loader yields, optionally capped for debugging.
    steps_per_epoch = len(loaders["train"]) if cfg.max_train_batches is None else \
        min(len(loaders["train"]), cfg.max_train_batches)
    total_steps = cfg.epochs * steps_per_epoch  # the cosine schedule is laid out over the full planned budget
    # AdamW optimiser: the algorithm that updates the weights from the gradients (with decoupled weight decay).
    optimizer = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr)
    # Learning-rate schedule: warm up for `warmup_frac` of the steps (at least 1), then decay along a cosine.
    scheduler = cosine_with_warmup(optimizer, total_steps, max(1, int(cfg.warmup_frac * total_steps)))
    # GradScaler multiplies the loss before backprop so tiny fp16 gradients do not underflow to 0.
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    # Cross-entropy: the classification loss; it applies softmax to the logits internally and compares to labels.
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # One-line summary of what is about to be trained.
    print(f"[{cfg.run_name}] {count_params(model) / 1e6:.1f}M params, {n_lab} labelled + {len(extra_ids)} pseudo, "
          f"{steps_per_epoch} steps/epoch x {cfg.epochs} epochs on {env['device']}")

    # Bookkeeping for model selection: per-epoch log, best validation F1 so far (-1 = nothing yet), a copy of the
    # best weights, the epoch they came from, and how many epochs in a row have not improved.
    history, best_f1, best_state, best_epoch, bad_epochs = [], -1.0, None, -1, 0
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        te = time.time()
        # Train for one epoch and get the average training loss / accuracy.
        tr = train_one_epoch(model, loaders["train"], optimizer, scheduler, scaler, criterion, device, amp,
                             cfg.max_train_batches)
        # Score the model on the validation set: argmax picks each image's highest-scoring class.
        val_logits, val_y = predict(model, loaders["val"], device, amp)
        va = summary(val_y, val_logits.argmax(1))
        # Record this epoch's numbers (including the current learning rate and the time taken).
        history.append({"epoch": epoch, "train_loss": tr["loss"], "train_acc": tr["accuracy"],
                        "val_acc": va["accuracy"], "val_macro_f1": va["macro_f1"],
                        "lr": scheduler.get_last_lr()[0], "seconds": time.time() - te})
        # Model selection uses validation macro-F1: did this epoch beat the best so far?
        improved = va["macro_f1"] > best_f1
        if improved:
            # New best: remember its score and epoch, and reset the "no improvement" counter.
            best_f1, best_epoch, bad_epochs = va["macro_f1"], epoch, 0
            # Snapshot on CPU: the live model keeps training and its tensors are updated in place, so copy.
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            # No gain: count another epoch without improvement (used for early stopping).
            bad_epochs += 1
        # Print a progress line for this epoch; a trailing "*" marks a new best.
        print(f"  epoch {epoch:3d}  loss {tr['loss']:.4f}  train_acc {tr['accuracy']:.4f}  "
              f"val_acc {va['accuracy']:.4f}  val_f1 {va['macro_f1']:.4f}  {time.time() - te:5.1f}s"
              f"{'  *' if improved else ''}", flush=True)
        # Rewrite history.csv every epoch so progress is saved even if the run is interrupted.
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        # Early stopping: quit when validation F1 has not improved for `patience` epochs in a row.
        if bad_epochs >= cfg.patience:
            print(f"  early stop: no val macro-F1 gain for {cfg.patience} epochs")
            break
    train_seconds = time.time() - t0

    model.load_state_dict(best_state)  # from here on, every prediction uses the best-validation-F1 weights
    # Optionally store those weights on disk (big file, so off by default).
    if cfg.save_checkpoint:
        torch.save(best_state, run_dir / "best.pt")
    # Start the metrics dict with run facts; validation / test metrics are added below.
    results = {"run_name": cfg.run_name, "best_epoch": best_epoch, "epochs_run": len(history),
               "train_seconds": round(train_seconds, 1), "n_params": count_params(model),
               "n_labelled": n_lab, "n_pseudo": len(extra_ids),
               "steps_per_epoch": steps_per_epoch, "total_steps_planned": total_steps}
    # The test split is predicted here, once, with the selected weights; tuning runs skip it entirely.
    for split in ("val", "test") if cfg.evaluate_test else ("val",):
        # Predict, save the raw logits (for later analysis) and compute accuracy / macro-F1 / confusion matrix.
        logits, y = predict(model, loaders[split], device, amp)
        _save_preds(run_dir / f"preds_{split}.npz", logits, y, loaders[split].dataset.ids)
        results[split] = classification_metrics(y, logits.argmax(1))

    # Teacher predictions for pseudo-labelling (labels never loaded). Skipped at 100% labels (no pool) and for
    # tuning runs. These logits come from the best-validation weights, so no checkpoint has to be stored.
    if cfg.fraction < 1.0 and cfg.evaluate_test:
        # The "pool" is every training image outside the labelled subset, loaded without labels.
        pool = make_dataset(cfg.modality, unlabelled_ids(cfg.fraction, cfg.seed), labelled=False,
                            in_memory=cfg.in_memory)
        pool_loader = DataLoader(pool, batch_size=cfg.eval_batch_size, num_workers=cfg.num_workers)
        # Predict on the pool and save the logits; the student stage turns them into pseudo-labels.
        logits, _ = predict(model, pool_loader, device, amp)
        _save_preds(run_dir / "preds_unlabelled.npz", logits, np.array([]), pool.ids)

    # Analysis only: pseudo-label accuracy vs the hidden true labels. It is computed after training and its
    # output is never read by training or selection, so the hidden labels cannot influence the student.
    if cfg.pseudo_labels:
        (run_dir / "pseudo_label_quality.json").write_text(
            json.dumps(quality(run_dir / cfg.pseudo_labels), indent=2) + "\n")

    # Written last: its presence marks the run as complete (see the skip check at the top).
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    # Text for the final summary line; empty when the test set was not evaluated.
    test = f"| test acc {results['test']['accuracy']:.4f} f1 {results['test']['macro_f1']:.4f} " \
        if cfg.evaluate_test else ""
    print(f"[{cfg.run_name}] best epoch {best_epoch}: val acc {results['val']['accuracy']:.4f} "
          f"f1 {results['val']['macro_f1']:.4f} {test}| {train_seconds / 60:.1f} min")
    return results

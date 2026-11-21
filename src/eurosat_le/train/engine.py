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

from __future__ import annotations

import copy
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from eurosat_le.constants import REPO_ROOT, RESULTS_DIR
from eurosat_le.data.datasets import labelled_ids, make_dataset, unlabelled_ids
from eurosat_le.data.loaders import build_loaders
from eurosat_le.data.splits import check_integrity
from eurosat_le.eval.metrics import classification_metrics, summary
from eurosat_le.models import build_model, count_params
from eurosat_le.ssl.pseudo_label import load_pseudo_labels, quality
from eurosat_le.train.config import TrainConfig
from eurosat_le.utils import seed_everything

RUNS_DIR = RESULTS_DIR / "runs"


def _read(path: Path) -> str:
    """Stripped text of `path`, or "" if it does not exist."""
    return path.read_text().strip() if path.exists() else ""


def environment() -> dict:
    """Versions, device and git state of this machine, saved as env.json for reproducibility."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""

    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda if cuda else None,
        "device": torch.cuda.get_device_name(0) if cuda else platform.processor() or "cpu",
        # COMMIT is written into code zips made by scripts/make_code_zip.sh (no .git there)
        "git_commit": git("rev-parse", "HEAD") or _read(REPO_ROOT / "COMMIT"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def param_groups(model: nn.Module, weight_decay: float):
    """Two AdamW parameter groups: weights with decay, and 1-D / embedding parameters without.

    Biases and norm-layer scales are 1-D (ndim <= 1); `pos_embed` and `cls_token` are matched by name
    because they are higher-dimensional but should not be decayed. Frozen parameters are skipped.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(("pos_embed", "cls_token")):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def cosine_with_warmup(optimizer, total_steps: int, warmup_steps: int):
    """LambdaLR schedule over optimiser steps: linear warm-up to the base lr, then cosine decay to 0.

    The multiplier is (step + 1) / warmup_steps during warm-up (so the very first step already has a
    non-zero lr), then 0.5 * (1 + cos(pi * progress)) with progress running from 0 to 1 over the
    remaining steps. Call `scheduler.step()` once per batch. If early stopping ends the run, the
    decay has simply not reached 0.
    """
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits, labels) as NumPy arrays; labels is empty for an unlabelled loader.

    Logits are float32 (cast back from fp16 autocast), in the loader's order. An unlabelled loader
    yields bare tensors instead of (x, y) pairs, which is how the two cases are told apart.
    """
    model.eval()
    logits, labels = [], []
    for batch in loader:
        x, y = (batch, None) if isinstance(batch, torch.Tensor) else batch
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out = model(x.to(device, non_blocking=True))
        logits.append(out.float().cpu())
        if y is not None:
            labels.append(y)
    return torch.cat(logits).numpy(), (torch.cat(labels).numpy() if labels else np.array([]))


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, device, amp, max_batches=None):
    """One pass over `loader` (an "epoch" may be a fixed number of sampled steps, see the budget rule).

    The scheduler is stepped after every batch. `max_batches` is a debug cap. Returns the mean
    training loss and accuracy; the accuracy is measured on the augmented batches in train mode, so
    it is only a rough progress signal, not a performance estimate.
    """
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out = model(x)
            loss = criterion(out, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        loss_sum += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return {"loss": loss_sum / max(n, 1), "accuracy": correct / max(n, 1)}


def _save_preds(path: Path, logits, labels, ids):
    """Save logits (as float16), labels and sample ids to a compressed .npz; Phase 6 re-computes all metrics from it."""
    np.savez_compressed(path, logits=logits.astype(np.float16), labels=labels, ids=np.asarray(ids))


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
    run_dir = runs_dir / cfg.run_name
    if (run_dir / "metrics.json").exists() and not overwrite:
        print(f"[skip] {cfg.run_name} already finished")
        return json.loads((run_dir / "metrics.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    env = environment()
    (run_dir / "env.json").write_text(json.dumps(env, indent=2) + "\n")

    check_integrity()  # refuse to train if the committed split files no longer match splits/manifest.json
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg.amp and device.type == "cuda"  # autocast / GradScaler are only enabled on CUDA
    n_lab = len(labelled_ids(cfg.fraction, cfg.seed))  # size of the nested labelled subset for this seed
    extra_ids, extra_labels = (load_pseudo_labels(run_dir / cfg.pseudo_labels) if cfg.pseudo_labels else ([], []))
    # The budget rule counts labelled + pseudo-labelled images, so a student usually gets more steps than its teacher.
    dc = cfg.data_config(cfg.resolve_steps_per_epoch(n_lab + len(extra_ids)))
    dc.extra_ids, dc.extra_labels = extra_ids or None, extra_labels or None
    loaders = build_loaders(dc)
    in_chans = 3 if cfg.modality == "rgb" else 13  # "ms" = all 13 Sentinel-2 bands
    model = build_model(cfg.model, in_chans, pretrained=cfg.pretrained, init_mode=cfg.init_mode,
                        input_size=cfg.input_size).to(device)

    steps_per_epoch = len(loaders["train"]) if cfg.max_train_batches is None else \
        min(len(loaders["train"]), cfg.max_train_batches)
    total_steps = cfg.epochs * steps_per_epoch  # the cosine schedule is laid out over the full planned budget
    optimizer = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr)
    scheduler = cosine_with_warmup(optimizer, total_steps, max(1, int(cfg.warmup_frac * total_steps)))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    print(f"[{cfg.run_name}] {count_params(model) / 1e6:.1f}M params, {n_lab} labelled + {len(extra_ids)} pseudo, "
          f"{steps_per_epoch} steps/epoch x {cfg.epochs} epochs on {env['device']}")

    history, best_f1, best_state, best_epoch, bad_epochs = [], -1.0, None, -1, 0
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        te = time.time()
        tr = train_one_epoch(model, loaders["train"], optimizer, scheduler, scaler, criterion, device, amp,
                             cfg.max_train_batches)
        val_logits, val_y = predict(model, loaders["val"], device, amp)
        va = summary(val_y, val_logits.argmax(1))
        history.append({"epoch": epoch, "train_loss": tr["loss"], "train_acc": tr["accuracy"],
                        "val_acc": va["accuracy"], "val_macro_f1": va["macro_f1"],
                        "lr": scheduler.get_last_lr()[0], "seconds": time.time() - te})
        improved = va["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, bad_epochs = va["macro_f1"], epoch, 0
            # Snapshot on CPU: the live model keeps training and its tensors are updated in place, so copy.
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad_epochs += 1
        print(f"  epoch {epoch:3d}  loss {tr['loss']:.4f}  train_acc {tr['accuracy']:.4f}  "
              f"val_acc {va['accuracy']:.4f}  val_f1 {va['macro_f1']:.4f}  {time.time() - te:5.1f}s"
              f"{'  *' if improved else ''}", flush=True)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if bad_epochs >= cfg.patience:
            print(f"  early stop: no val macro-F1 gain for {cfg.patience} epochs")
            break
    train_seconds = time.time() - t0

    model.load_state_dict(best_state)  # from here on, every prediction uses the best-validation-F1 weights
    if cfg.save_checkpoint:
        torch.save(best_state, run_dir / "best.pt")
    results = {"run_name": cfg.run_name, "best_epoch": best_epoch, "epochs_run": len(history),
               "train_seconds": round(train_seconds, 1), "n_params": count_params(model),
               "n_labelled": n_lab, "n_pseudo": len(extra_ids),
               "steps_per_epoch": steps_per_epoch, "total_steps_planned": total_steps}
    # The test split is predicted here, once, with the selected weights; tuning runs skip it entirely.
    for split in ("val", "test") if cfg.evaluate_test else ("val",):
        logits, y = predict(model, loaders[split], device, amp)
        _save_preds(run_dir / f"preds_{split}.npz", logits, y, loaders[split].dataset.ids)
        results[split] = classification_metrics(y, logits.argmax(1))

    # Teacher predictions for pseudo-labelling (labels never loaded). Skipped at 100% labels (no pool) and for
    # tuning runs. These logits come from the best-validation weights, so no checkpoint has to be stored.
    if cfg.fraction < 1.0 and cfg.evaluate_test:
        pool = make_dataset(cfg.modality, unlabelled_ids(cfg.fraction, cfg.seed), labelled=False,
                            in_memory=cfg.in_memory)
        pool_loader = DataLoader(pool, batch_size=cfg.eval_batch_size, num_workers=cfg.num_workers)
        logits, _ = predict(model, pool_loader, device, amp)
        _save_preds(run_dir / "preds_unlabelled.npz", logits, np.array([]), pool.ids)

    # Analysis only: pseudo-label accuracy vs the hidden true labels. It is computed after training and its
    # output is never read by training or selection, so the hidden labels cannot influence the student.
    if cfg.pseudo_labels:
        (run_dir / "pseudo_label_quality.json").write_text(
            json.dumps(quality(run_dir / cfg.pseudo_labels), indent=2) + "\n")

    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    test = f"| test acc {results['test']['accuracy']:.4f} f1 {results['test']['macro_f1']:.4f} " \
        if cfg.evaluate_test else ""
    print(f"[{cfg.run_name}] best epoch {best_epoch}: val acc {results['val']['accuracy']:.4f} "
          f"f1 {results['val']['macro_f1']:.4f} {test}| {train_seconds / 60:.1f} min")
    return results

"""Overfit one small real batch with every model and input type (CPU-friendly sanity check).

A model that cannot drive the loss on 8 images close to zero has a wiring bug
(wrong input layer, frozen weights, broken normalisation, ...).

Usage:
    python scripts/smoke_test.py                       # all models, rgb + ms, pretrained weights
    python scripts/smoke_test.py --models scratch_cnn --no-pretrained

What is tested: for each (model, input type) the script builds the model with build_model, takes one
mixed-label batch of real images (no augmentation, so the batch can be memorised), trains on it for
--steps AdamW steps with dropout switched off, and checks that the loss falls below a quarter of
its starting value and that the batch is classified perfectly. This exercises the pieces that can
silently go wrong: the 13-band first-layer adaptation, the upsampling to 224 px, the data loading
and the normalisation. It is a wiring check, not a measure of accuracy, and uses no validation
or test data (the images come from the 1% labelled subset of seed 0). The model and batch are never
moved to a GPU in this script, so it runs on the CPU.

Exit status is non-zero if any combination fails; the last line says which.
"""

# argparse reads the command-line options; time measures how long each model takes.
import argparse
import time

# torch is the deep-learning library; nn gives access to layers and the loss function.
import torch
from torch import nn

# labelled_ids lists the ids of a labelled subset; make_dataset builds a dataset over those ids.
from eurosat_le.data.datasets import labelled_ids, make_dataset
# ARCHS is the table of available model architectures; build_model creates one of them.
from eurosat_le.models import ARCHS, build_model
# seed_everything makes the random numbers (and so the model initialisation) repeatable.
from eurosat_le.utils import seed_everything

# Per-model learning rate for the overfitting test only (unrelated to the tuned training learning rates).
LR = {"scratch_cnn": 2e-3, "resnet50": 5e-4, "vit_small": 2e-4, "vit_scratch": 1e-3}


def main() -> None:
    """Try to overfit one tiny batch with every model/input combination and report PASS or FAIL for each.

    Takes no arguments (options come from sys.argv) and returns nothing. Raises SystemExit with a message
    listing the failed combinations if any of them cannot fit the batch.
    """
    # Build the argument parser; the module docstring doubles as the --help text.
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Which architectures to test (default: all known ones).
    p.add_argument("--models", nargs="*", default=list(ARCHS))
    # Which input types to test: RGB and/or 13-band multispectral.
    p.add_argument("--modalities", nargs="*", default=["rgb", "ms"])
    # Number of images in the test batch.
    p.add_argument("--batch", type=int, default=8)
    # Number of optimiser steps used to try to memorise the batch.
    p.add_argument("--steps", type=int, default=30)
    # Flag: start from random weights instead of downloading pretrained ones.
    p.add_argument("--no-pretrained", action="store_true")
    # Parse sys.argv into the args object.
    args = p.parse_args()

    # Collect the names of the combinations that fail, so they can all be reported at the end.
    failures = []
    for modality in args.modalities:
        ids = labelled_ids(0.01, 0)  # 1% labelled subset, seed 0: small, and never val/test data
        # Dataset without augmentation, so the same images come out every time and can be memorised.
        ds = make_dataset(modality, ids)
        # one image from each of the first `batch` classes -> a mixed-label batch
        # Map class number -> index of the first image of that class (setdefault keeps only the first one).
        picks = {}
        for i in range(len(ds)):
            picks.setdefault(int(ds.labels[i]), i)
        # Keep at most `batch` of those images, each from a different class.
        idx = list(picks.values())[: args.batch]
        # Load the chosen images and stack them into one input tensor.
        x = torch.stack([ds.load(i) for i in idx])  # (batch, 3 or 13, 64, 64); the model upsamples itself if needed
        # Matching class labels for the batch.
        y = torch.tensor([int(ds.labels[i]) for i in idx])
        for name in args.models:
            seed_everything(0)  # same init for every model/input combination on every invocation
            # Create the model; x.shape[1] is the number of input channels (3 for RGB, 13 for multispectral).
            model = build_model(name, x.shape[1], pretrained=not args.no_pretrained)
            # Dropout/stochastic depth off: we test the ability to fit, not regularisation.
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = 0.0
            # AdamW optimiser with this model's test learning rate and no weight decay (pure fitting test).
            opt = torch.optim.AdamW(model.parameters(), lr=LR[name], weight_decay=0.0)
            # Training mode (BatchNorm uses batch statistics, gradients are tracked).
            model.train()
            # Start the timer and prepare the list that records the loss after every step.
            t0, losses = time.time(), []
            # Repeatedly train on the very same batch.
            for _ in range(args.steps):
                # Forward pass: the model's class scores (logits) for the batch.
                out = model(x)
                # How wrong the predictions are (cross-entropy against the true labels).
                loss = nn.functional.cross_entropy(out, y)
                # Clear the gradients from the previous step.
                opt.zero_grad()
                # Backward pass: compute the gradient of the loss for every weight.
                loss.backward()
                # Update the weights using those gradients.
                opt.step()
                # Remember this step's loss as a plain number.
                losses.append(loss.item())
            # Train-mode accuracy of the last step: BatchNorm running stats need far more than
            # a few steps to settle, so eval mode would test the stats, not the wiring.
            acc = (out.argmax(1) == y).float().mean().item()
            # pass = loss fell to under a quarter of its start value AND all samples correct
            ok = losses[-1] < 0.25 * losses[0] and acc == 1.0
            # One result line per combination: verdict, model, input, loss before -> after, accuracy and time.
            print(f"{'PASS' if ok else 'FAIL'}  {name:12s} {modality:3s}  loss {losses[0]:.3f} -> {losses[-1]:.4f}  "
                  f"batch acc {acc:.2f}  {time.time() - t0:5.1f}s", flush=True)
            if not ok:
                failures.append(f"{name}/{modality}")
    # Exit with a non-zero status (and list the culprits) if anything failed, so scripts and CI notice.
    if failures:
        raise SystemExit(f"smoke test failed: {failures}")
    print("all smoke tests passed")


# Run main() only when the file is executed as a script, not when it is imported.
if __name__ == "__main__":
    main()

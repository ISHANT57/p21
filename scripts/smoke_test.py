"""Overfit one small real batch with every model and input type (CPU-friendly sanity check).

A model that cannot drive the loss on 8 images close to zero has a wiring bug
(wrong input layer, frozen weights, broken normalisation, ...).

Usage:
    python scripts/smoke_test.py                       # all models, rgb + ms, pretrained weights
    python scripts/smoke_test.py --models scratch_cnn --no-pretrained
"""

import argparse
import time

import torch
from torch import nn

from eurosat_le.data.datasets import labelled_ids, make_dataset
from eurosat_le.models import ARCHS, build_model
from eurosat_le.utils import seed_everything

LR = {"scratch_cnn": 2e-3, "resnet50": 5e-4, "vit_small": 2e-4, "vit_scratch": 1e-3}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", default=list(ARCHS))
    p.add_argument("--modalities", nargs="*", default=["rgb", "ms"])
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--no-pretrained", action="store_true")
    args = p.parse_args()

    failures = []
    for modality in args.modalities:
        ids = labelled_ids(0.01, 0)
        ds = make_dataset(modality, ids)
        # one image from each of the first `batch` classes -> a mixed-label batch
        picks = {}
        for i in range(len(ds)):
            picks.setdefault(int(ds.labels[i]), i)
        idx = list(picks.values())[: args.batch]
        x = torch.stack([ds.load(i) for i in idx])
        y = torch.tensor([int(ds.labels[i]) for i in idx])
        for name in args.models:
            seed_everything(0)
            model = build_model(name, x.shape[1], pretrained=not args.no_pretrained)
            # Dropout/stochastic depth off: we test the ability to fit, not regularisation.
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = 0.0
            opt = torch.optim.AdamW(model.parameters(), lr=LR[name], weight_decay=0.0)
            model.train()
            t0, losses = time.time(), []
            for _ in range(args.steps):
                out = model(x)
                loss = nn.functional.cross_entropy(out, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            # Train-mode accuracy of the last step: BatchNorm running stats need far more than
            # a few steps to settle, so eval mode would test the stats, not the wiring.
            acc = (out.argmax(1) == y).float().mean().item()
            ok = losses[-1] < 0.25 * losses[0] and acc == 1.0
            print(f"{'PASS' if ok else 'FAIL'}  {name:12s} {modality:3s}  loss {losses[0]:.3f} -> {losses[-1]:.4f}  "
                  f"batch acc {acc:.2f}  {time.time() - t0:5.1f}s", flush=True)
            if not ok:
                failures.append(f"{name}/{modality}")
    if failures:
        raise SystemExit(f"smoke test failed: {failures}")
    print("all smoke tests passed")


if __name__ == "__main__":
    main()

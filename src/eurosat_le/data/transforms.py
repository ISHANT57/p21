"""Channel-agnostic augmentations on normalised (C, H, W) float tensors.

Every transform here works the same for 3-band RGB and 13-band MS input, so
RGB-vs-MS comparisons use an identical augmentation policy. Images stay at the
native 64x64; models that need 224x224 upsample on the GPU (see models/).

Policies:
    none    no augmentation (evaluation, pseudo-label prediction)
    weak    random flip + 90-degree rotation (the 8 symmetries of a square)
            + random resized crop (scale 0.75-1.0) back to 64x64
    strong  weak + random brightness/contrast, Gaussian noise and blur (each
            p=0.5) + cutout. Brightness/contrast use one gain and offset for all
            bands, so the spectral shape of each pixel is preserved up to scale.
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2


class RandomD4:
    """Random element of the dihedral group D4: rotate by k*90 degrees, then maybe flip."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        k = int(torch.randint(4, (1,)))
        x = torch.rot90(x, k, dims=(-2, -1))
        if torch.rand(1) < 0.5:
            x = x.flip(-1)
        return x


class RandomGainOffset:
    """x * a + b with one (a, b) shared by all channels (works in normalised units)."""

    def __init__(self, gain=(0.75, 1.25), offset=(-0.25, 0.25), p=0.5):
        self.gain, self.offset, self.p = gain, offset, p

    def __call__(self, x):
        if torch.rand(1) >= self.p:
            return x
        a = torch.empty(1).uniform_(*self.gain)
        b = torch.empty(1).uniform_(*self.offset)
        return x * a + b


class RandomGaussianNoise:
    def __init__(self, sigma=0.1, p=0.5):
        self.sigma, self.p = sigma, p

    def __call__(self, x):
        if torch.rand(1) >= self.p:
            return x
        return x + torch.randn_like(x) * self.sigma


def build_transform(policy: str = "none", size: int = 64):
    if policy == "none":
        return None
    weak = [RandomD4(), v2.RandomResizedCrop(size, scale=(0.75, 1.0), antialias=True)]
    if policy == "weak":
        return v2.Compose(weak)
    if policy == "strong":
        return v2.Compose([
            *weak,
            RandomGainOffset(),
            RandomGaussianNoise(),
            v2.RandomApply([v2.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.5),
            v2.RandomErasing(p=1.0, scale=(0.02, 0.15), value=0.0),  # cutout; 0 = band mean
        ])
    raise ValueError(f"unknown augmentation policy {policy!r}")


class TwoViews:
    """Return (weak(x), strong(x)) for consistency-based semi-supervised methods."""

    def __init__(self, weak, strong):
        self.weak, self.strong = weak, strong

    def __call__(self, x):
        return self.weak(x), self.strong(x)

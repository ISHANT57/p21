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

Pipeline position: build_transform() is called by eurosat_le.data.loaders. The transforms act on
tensors that eurosat_le.data.datasets has already normalised (per-channel train-only mean/std),
so "0" means the band mean. In the strong policy the cutout fill value of 0 and the additive
offset and noise are all expressed in these normalised units.

Design decisions:
    - Overhead images have no natural "up", so flips and 90-degree rotations are label-preserving.
    - Nothing here is RGB-specific (no hue/saturation jitter), so 3-band and 13-band runs get the
      same augmentation and any difference between them is not due to the augmentation policy.
    - The weak policy is used for labelled training; the strong one only (together with weak,
      through TwoViews) for unlabelled data in semi-supervised training.
    - Randomness comes from torch's global RNG (torch.rand / randint), which is seeded per
      DataLoader worker (see eurosat_le.utils.seed_worker and the loader generator).
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2


class RandomD4:
    """Random element of the dihedral group D4: rotate by k*90 degrees, then maybe flip.

    The 4 rotations x 2 flip choices give the 8 symmetries of a square, each with probability 1/8.
    Works on any (..., H, W) tensor, so any number of channels.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        k = int(torch.randint(4, (1,)))
        x = torch.rot90(x, k, dims=(-2, -1))
        if torch.rand(1) < 0.5:
            x = x.flip(-1)
        return x


class RandomGainOffset:
    """x * a + b with one (a, b) shared by all channels (works in normalised units).

    Applied with probability `p`. Using the same gain and offset for every band avoids changing
    the relative spectral shape of a pixel, which matters for the 13-band input.
    """

    def __init__(self, gain=(0.75, 1.25), offset=(-0.25, 0.25), p=0.5):
        self.gain, self.offset, self.p = gain, offset, p

    def __call__(self, x):
        if torch.rand(1) >= self.p:
            return x
        a = torch.empty(1).uniform_(*self.gain)
        b = torch.empty(1).uniform_(*self.offset)
        return x * a + b


class RandomGaussianNoise:
    """With probability `p`, add i.i.d. Gaussian noise (std `sigma`, in normalised units) to every value."""

    def __init__(self, sigma=0.1, p=0.5):
        self.sigma, self.p = sigma, p

    def __call__(self, x):
        if torch.rand(1) >= self.p:
            return x
        return x + torch.randn_like(x) * self.sigma


def build_transform(policy: str = "none", size: int = 64):
    """Return the augmentation callable for `policy` ("none", "weak" or "strong").

    "none" returns None (the dataset then applies no transform). `size` is the output side of the
    random resized crop; the default 64 matches the native patch size, so the image size never
    changes. Raises ValueError for an unknown policy.
    """
    if policy == "none":
        return None
    # The crop keeps 75-100% of the image area (torchvision's default aspect-ratio range) and is
    # resized back to `size`, so every output has the same shape.
    weak = [RandomD4(), v2.RandomResizedCrop(size, scale=(0.75, 1.0), antialias=True)]
    if policy == "weak":
        return v2.Compose(weak)
    if policy == "strong":
        return v2.Compose([
            *weak,
            RandomGainOffset(),
            RandomGaussianNoise(),
            # 3x3 kernel: mild blur, since the patches are only 64x64.
            v2.RandomApply([v2.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.5),
            v2.RandomErasing(p=1.0, scale=(0.02, 0.15), value=0.0),  # cutout; 0 = band mean
        ])
    raise ValueError(f"unknown augmentation policy {policy!r}")


class TwoViews:
    """Return (weak(x), strong(x)) for consistency-based semi-supervised methods.

    Both views are made from the same input tensor but with independent random draws. The dataset
    returns this tuple directly for unlabelled items (no label is attached).
    """

    def __init__(self, weak, strong):
        self.weak, self.strong = weak, strong

    def __call__(self, x):
        return self.weak(x), self.strong(x)

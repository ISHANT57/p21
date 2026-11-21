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

# Allow modern type-hint syntax on older Python versions.
from __future__ import annotations

# torch: tensor operations and random numbers; torchvision v2: ready-made crop/blur/erase transforms.
import torch
from torchvision.transforms import v2


class RandomD4:
    """Random element of the dihedral group D4: rotate by k*90 degrees, then maybe flip.

    The 4 rotations x 2 flip choices give the 8 symmetries of a square, each with probability 1/8.
    Works on any (..., H, W) tensor, so any number of channels.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a random rotation (0/90/180/270 degrees) and a random horizontal flip.

        Args:
            x: image tensor of shape (C, H, W).

        Returns:
            The transformed tensor with the same shape.
        """
        # Pick how many quarter turns to make: 0, 1, 2 or 3, all equally likely.
        k = int(torch.randint(4, (1,)))
        # Rotate in the plane of the last two dimensions (height and width).
        x = torch.rot90(x, k, dims=(-2, -1))
        # With probability 1/2, mirror the image left-right.
        if torch.rand(1) < 0.5:
            x = x.flip(-1)
        return x


class RandomGainOffset:
    """x * a + b with one (a, b) shared by all channels (works in normalised units).

    Applied with probability `p`. Using the same gain and offset for every band avoids changing
    the relative spectral shape of a pixel, which matters for the 13-band input.
    """

    def __init__(self, gain=(0.75, 1.25), offset=(-0.25, 0.25), p=0.5):
        """Store the settings.

        Args:
            gain: (low, high) range from which the multiplier `a` is drawn.
            offset: (low, high) range from which the added constant `b` is drawn.
            p: probability that the transform is applied at all.
        """
        # Keep the ranges and the probability for use in __call__.
        self.gain, self.offset, self.p = gain, offset, p

    def __call__(self, x):
        """Randomly change brightness (offset b) and contrast (gain a) of the whole image.

        Args:
            x: normalised image tensor (C, H, W).

        Returns:
            The adjusted tensor, or x unchanged if the random draw says to skip.
        """
        # Skip with probability 1 - p (a uniform random number in [0, 1) must fall below p to apply).
        if torch.rand(1) >= self.p:
            return x
        # Draw one random gain and one random offset; the same pair is used for all channels.
        a = torch.empty(1).uniform_(*self.gain)
        b = torch.empty(1).uniform_(*self.offset)
        # Apply to every value of every channel.
        return x * a + b


class RandomGaussianNoise:
    """With probability `p`, add i.i.d. Gaussian noise (std `sigma`, in normalised units) to every value."""

    def __init__(self, sigma=0.1, p=0.5):
        """Store the settings.

        Args:
            sigma: standard deviation of the noise, in normalised units.
            p: probability that noise is added.
        """
        # Keep for use in __call__.
        self.sigma, self.p = sigma, p

    def __call__(self, x):
        """Add random noise to the image with probability p.

        Args:
            x: normalised image tensor (C, H, W).

        Returns:
            The noisy tensor, or x unchanged if the random draw says to skip.
        """
        # Skip with probability 1 - p.
        if torch.rand(1) >= self.p:
            return x
        # Standard normal noise with the same shape as x (one independent value per pixel and band),
        # scaled to the wanted standard deviation and added to the image.
        return x + torch.randn_like(x) * self.sigma


def build_transform(policy: str = "none", size: int = 64):
    """Return the augmentation callable for `policy` ("none", "weak" or "strong").

    "none" returns None (the dataset then applies no transform). `size` is the output side of the
    random resized crop; the default 64 matches the native patch size, so the image size never
    changes. Raises ValueError for an unknown policy.

    Args:
        policy: "none", "weak" or "strong" (see the module docstring).
        size: side length of the output image in pixels.

    Returns:
        A callable that maps a (C, H, W) tensor to an augmented tensor, or None for "none".
    """
    if policy == "none":
        # No augmentation requested.
        return None
    # The crop keeps 75-100% of the image area (torchvision's default aspect-ratio range) and is
    # resized back to `size`, so every output has the same shape.
    # Building blocks of the weak policy: random rotation/flip, then a random crop resized back to `size`.
    weak = [RandomD4(), v2.RandomResizedCrop(size, scale=(0.75, 1.0), antialias=True)]
    if policy == "weak":
        # Chain the steps so they run one after the other.
        return v2.Compose(weak)
    if policy == "strong":
        # Strong = everything in weak, plus brightness/contrast, noise, blur and cutout.
        return v2.Compose([
            *weak,
            RandomGainOffset(),
            RandomGaussianNoise(),
            # 3x3 kernel: mild blur, since the patches are only 64x64.
            v2.RandomApply([v2.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.5),
            # Cutout: always blank one random rectangle covering 2-15% of the image.
            v2.RandomErasing(p=1.0, scale=(0.02, 0.15), value=0.0),  # cutout; 0 = band mean
        ])
    # Any other name is a mistake by the caller.
    raise ValueError(f"unknown augmentation policy {policy!r}")


class TwoViews:
    """Return (weak(x), strong(x)) for consistency-based semi-supervised methods.

    Both views are made from the same input tensor but with independent random draws. The dataset
    returns this tuple directly for unlabelled items (no label is attached).
    """

    def __init__(self, weak, strong):
        """Store the two augmentation callables.

        Args:
            weak: transform producing the weakly augmented view.
            strong: transform producing the strongly augmented view.
        """
        # Keep both for use in __call__.
        self.weak, self.strong = weak, strong

    def __call__(self, x):
        """Return the pair (weak view, strong view) of the same image x.

        Args:
            x: normalised image tensor (C, H, W).
        """
        # Each call draws fresh random numbers, so the two views differ.
        return self.weak(x), self.strong(x)

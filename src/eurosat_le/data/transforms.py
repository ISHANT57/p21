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

# ====================================================================================================
# CONCEPTS USED IN THIS FILE
# - Data augmentation: randomly changing a training image (flip, crop, noise, ...) so the model sees many variants and
#   over-fits less. The label stays the same.
# - Weak vs strong augmentation: weak = random flip/90-degree rotation + random resized crop; strong = weak +
#   brightness/contrast, Gaussian noise, blur and cutout. 'none' = no augmentation. Weak is used for labelled
#   training, strong (with weak, via TwoViews) for unlabelled data.
# - Normalised (C, H, W) tensor: a float image with channels first, already standardised per channel by data.datasets,
#   so 0 means the band mean. All transforms here work in these units.
# - Channel-agnostic: no transform assumes 3 colour channels (no hue or saturation changes), so RGB and 13-band MS get
#   the identical policy and comparisons between them are fair.
# - Dihedral group D4: the 8 symmetries of a square: 4 rotations (0/90/180/270 degrees) times flip or no flip.
#   Overhead images have no natural 'up', so these do not change the label.
# - torch.rot90 / flip: rot90 rotates in the height/width plane by k quarter turns; flip(-1) mirrors the last axis
#   (left-right).
# - RandomResizedCrop: cuts a random region (here 75-100% of the area) and resizes it back to 64x64, so image shape
#   never changes. antialias=True smooths the resize.
# - Gain and offset (brightness/contrast): x * a + b with one a and one b shared by all bands, so the relative
#   spectral shape of each pixel is preserved up to scale.
# - Gaussian noise: random values from a normal distribution (std sigma, in normalised units) added to every value
#   independently.
# - Gaussian blur: smooths the image; a 3x3 kernel with sigma between 0.1 and 1.0 is mild because the patches are only
#   64x64.
# - Cutout (RandomErasing): blanks one random rectangle (2-15% of the image) with value 0, which in normalised units
#   is the band mean.
# - Probability p: the chance that a random transform is applied at all, e.g. p=0.5 means half of the images get
#   noise.
# - torch's global RNG: torch.rand / randint / randn draw from PyTorch's random generator, which is seeded per
#   DataLoader worker (utils.seed_worker plus the loader's generator).
# - v2.Compose / v2.RandomApply: torchvision helpers: Compose runs transforms one after another; RandomApply applies
#   the wrapped transform with probability p.
# - Consistency-based semi-supervised learning (TwoViews): the same unlabelled image is shown in a weak and a strong
#   version; the model should predict the same for both. TwoViews returns that pair.
# ====================================================================================================
#
# FILE-LEVEL Q&A
# Q: What is this file for?
# A: It defines the augmentation transforms and build_transform, which creates the 'none', 'weak' or 'strong' policy,
#    plus TwoViews for semi-supervised training.
# Q: Where does it fit in the pipeline?
# A: data.loaders calls build_transform; data.datasets applies the result to the already-normalised tensor of each
#    item.
# Q: What does it read and write?
# A: Nothing on disk. Every class takes a tensor in and returns a tensor (or a tuple of two) out.
# Q: Why are the transforms written by hand instead of using colour jitter?
# A: torchvision's colour jitter assumes 3 RGB channels. The custom gain/offset and noise work on any number of bands,
#    so 3-band and 13-band runs share one policy.


# Q: What does this class do?
# A: It applies a random rotation by 0, 90, 180 or 270 degrees followed by a 50% chance of a left-right flip: one of
#    the 8 symmetries of a square, each with probability 1/8.
# Q: Why is it label-preserving?
# A: Satellite images seen from above have no natural up or down, so rotating or flipping a forest or a river does not
#    change what it is.
# Q: Why does it work for any number of channels?
# A: It rotates and flips only the last two dimensions (height and width) and leaves the channel axis alone.
class RandomD4:
    """Random element of the dihedral group D4: rotate by k*90 degrees, then maybe flip.

    The 4 rotations x 2 flip choices give the 8 symmetries of a square, each with probability 1/8.
    Works on any (..., H, W) tensor, so any number of channels.
    """

    # Q: What does this method do?
    # A: It rotates the image by a random multiple of 90 degrees and then flips it left-right with probability 1/2.
    # Q: What are the inputs and what does it return?
    # A: x, a tensor (C, H, W). It returns a tensor of the same shape.
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


# Q: What does this class do?
# A: It randomly changes brightness and contrast of the whole image: x * a + b, with a (gain) and b (offset) drawn
#    from given ranges, applied with probability p.
# Q: Why is the same gain and offset used for all channels?
# A: Using different ones per band would change the spectral shape of a pixel (the ratios between bands), which
#    matters for the 13-band input. One shared pair keeps it up to scale.
# Q: Why does it work in normalised units?
# A: The data is already standardised, so the ranges (gain 0.75-1.25, offset -0.25 to 0.25) mean the same thing for
#    every band.
class RandomGainOffset:
    """x * a + b with one (a, b) shared by all channels (works in normalised units).

    Applied with probability `p`. Using the same gain and offset for every band avoids changing
    the relative spectral shape of a pixel, which matters for the 13-band input.
    """

    # Q: What does this constructor do?
    # A: It stores the gain range, the offset range and the probability p; nothing random happens yet.
    def __init__(self, gain=(0.75, 1.25), offset=(-0.25, 0.25), p=0.5):
        """Store the settings.

        Args:
            gain: (low, high) range from which the multiplier `a` is drawn.
            offset: (low, high) range from which the added constant `b` is drawn.
            p: probability that the transform is applied at all.
        """
        # Keep the ranges and the probability for use in __call__.
        self.gain, self.offset, self.p = gain, offset, p

    # Q: What does this method do?
    # A: With probability p it multiplies the whole image by a random gain and adds a random offset; otherwise it
    #    returns the image unchanged.
    # Q: What are the inputs and what does it return?
    # A: x, a normalised (C, H, W) tensor. It returns a tensor of the same shape, or x itself when skipped.
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


# Q: What does this class do?
# A: With probability p it adds independent Gaussian (normal) noise with standard deviation sigma to every value of
#    the image.
# Q: Why is it useful?
# A: It makes the model robust to small sensor or compression noise; it is part of the strong policy only.
# Q: Why is sigma 0.1 reasonable?
# A: The image is standardised, so a std of 0.1 is small compared with the data's own spread of about 1.
class RandomGaussianNoise:
    """With probability `p`, add i.i.d. Gaussian noise (std `sigma`, in normalised units) to every value."""

    # Q: What does this constructor do?
    # A: It stores the noise standard deviation (in normalised units) and the probability of adding noise.
    def __init__(self, sigma=0.1, p=0.5):
        """Store the settings.

        Args:
            sigma: standard deviation of the noise, in normalised units.
            p: probability that noise is added.
        """
        # Keep for use in __call__.
        self.sigma, self.p = sigma, p

    # Q: What does this method do?
    # A: With probability p it adds noise drawn with torch.randn_like(x) times sigma; otherwise it returns x
    #    unchanged.
    # Q: What are the inputs and what does it return?
    # A: x, a normalised (C, H, W) tensor. It returns a tensor of the same shape.
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


# Q: What does this function do?
# A: It returns the augmentation for a policy name: None for 'none', a Compose of flip/rotate and random resized crop
#    for 'weak', and weak plus gain/offset, noise, blur and cutout for 'strong'.
# Q: Why does 'none' return None instead of an identity transform?
# A: The dataset checks `transform is not None`, so with None it skips the augmentation step entirely.
# Q: What are the inputs and what does it return?
# A: policy ('none', 'weak', 'strong') and size (output side, default 64 = native patch size). It returns a callable
#    mapping a (C, H, W) tensor to an augmented tensor, or None. An unknown policy raises ValueError.
# Q: Why is the blur kernel only 3x3 and the crop scale 0.75-1.0?
# A: The patches are just 64x64 pixels, so a larger blur would erase too much detail; keeping 75-100% of the area in
#    the crop leaves most of the scene visible.
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


# Q: What does this class do?
# A: It calls a weak and a strong transform on the same image and returns both results as a pair (weak(x), strong(x)).
# Q: Why are two views needed?
# A: Consistency-based semi-supervised methods (such as FixMatch) use the weakly augmented view to make a prediction
#    and train the model to give the same answer for the strongly augmented view; no label is needed.
# Q: Are the two views identical?
# A: No. Each call draws fresh random numbers, so they differ even though they come from the same input.
class TwoViews:
    """Return (weak(x), strong(x)) for consistency-based semi-supervised methods.

    Both views are made from the same input tensor but with independent random draws. The dataset
    returns this tuple directly for unlabelled items (no label is attached).
    """

    # Q: What does this constructor do?
    # A: It stores the two transforms (weak and strong) for later use in __call__.
    def __init__(self, weak, strong):
        """Store the two augmentation callables.

        Args:
            weak: transform producing the weakly augmented view.
            strong: transform producing the strongly augmented view.
        """
        # Keep both for use in __call__.
        self.weak, self.strong = weak, strong

    # Q: What does this method do?
    # A: It applies the weak transform and the strong transform to the same input and returns them as a tuple.
    # Q: What are the inputs and what does it return?
    # A: x, a normalised (C, H, W) tensor. It returns (weak view, strong view), two tensors of the same shape.
    def __call__(self, x):
        """Return the pair (weak view, strong view) of the same image x.

        Args:
            x: normalised image tensor (C, H, W).
        """
        # Each call draws fresh random numbers, so the two views differ.
        return self.weak(x), self.strong(x)

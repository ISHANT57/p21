"""Model registry: build_model(name, in_chans, pretrained) -> Classifier.

| name         | backbone                                   | input  | init                 |
|--------------|--------------------------------------------|--------|----------------------|
| scratch_cnn  | ScratchCNN (small ResNet, 2.8M params)     | 64 px  | random               |
| resnet50     | timm resnet50.a1_in1k                      | 224 px | ImageNet-1k          |
| vit_small    | timm vit_small_patch16_224.augreg_in21k_ft_in1k | 224 px | ImageNet-21k -> 1k |
| vit_scratch  | timm ViT-Tiny, patch 8, at 64 px (optional) | 64 px  | random               |

For 13-band input, the first convolution of a pretrained model (ResNet conv1 or
the ViT patch embedding) is widened with `adapt_input_conv`; see docs/MODELS.md.

Where it fits (Phase 3): `eurosat_le.train.engine.run` calls `build_model` once per run, and
`scripts/smoke_test.py` calls it for every model / input-type pair.

Key design decisions
- Every backbone is wrapped in `Classifier`, which upsamples the 64x64 batch to the backbone's
  input size (224 px for the pretrained models) on the device. The data pipeline therefore
  always stays at 64 px.
- Pretrained weights are loaded as the original 3-channel RGB checkpoint first, and only then is
  the first convolution replaced by a wider one (`_replace_first_conv`). The new filters are
  derived from the pretrained RGB filters by `adapt_input_conv`:
    * mode "mean" (default): the R, G, B filters are copied to the Sentinel-2 channels B04, B03,
      B02 (positions 3, 2, 1 of the 13-band file, `MS_RGB_IDX`); each of the other 10 bands gets
      the mean of the three RGB filters; all 13 filters are then scaled by 3/13. Every band is
      standardised to unit variance, so this keeps the summed first-layer response at about the
      size the pretrained network saw with 3 channels.
    * mode "zero" (ablation): RGB filters copied unscaled, the other 10 bands start at zero, so
      the 13-band network initially behaves exactly like the RGB network.
  Nothing in ImageNet pretraining knows what NIR / red-edge / SWIR mean, so the extra bands are
  only learned during fine-tuning (see the limitations in docs/MODELS.md).
- The first-layer bias (if any) is copied unchanged; every other layer keeps its pretrained weights.
- `vit_scratch` and `scratch_cnn` are randomly initialised; their first layer is created directly
  with `in_chans` channels, so `adapt_input_conv` is not used for them.
"""

# Makes type hints such as `int | None` work as plain text on older Python versions.
from __future__ import annotations

# timm: library of ready-made (and pretrained) image models; torch: tensors;
# F: functional operations such as image resizing; nn: network layers and the Module base class.
import timm
import torch
import torch.nn.functional as F
from torch import nn

# MS_RGB_IDX: positions of the red, green, blue bands inside the 13-band image; ScratchCNN: our own small CNN.
from eurosat_le.constants import MS_RGB_IDX
from eurosat_le.models.scratch_cnn import ScratchCNN

# ======================================== CONCEPTS USED IN THIS FILE ========================================
# model registry (ARCHS): a dict mapping a model name to its settings (timm model name, first-conv path, input size), so
#   build_model can create any model from a string.
# backbone: the main network that turns an image into class scores; here ScratchCNN, a ResNet50 or a ViT. Classifier
#   wraps it.
# pretrained weights / ImageNet: weights already learned on a large image data set; fine-tuning them on EuroSAT usually
#   works better than random weights when labels are few.
# timm: a library of ready-made image models (ResNet50, ViT ...) with pretrained weights, created by timm.create_model.
# ResNet50: a 50-layer convolutional network with residual connections. ViT (Vision Transformer): cuts the image into
#   patches and processes them with attention; "small" is a mid-size ViT, ViT-Tiny with patch 8 is the from-scratch one.
# patch embedding (patch_embed.proj): the ViT's first layer, a convolution that turns each image patch into a vector; it
#   plays the role of ResNet's conv1.
# first-layer adaptation for 13 bands: pretrained models expect 3 input channels (RGB). For 13 Sentinel-2 bands the
#   first conv is replaced by a 13-channel one whose filters are derived from the pretrained RGB filters
#   (adapt_input_conv); all later layers keep their pretrained weights.
# MS_RGB_IDX / Sentinel-2 bands B04, B03, B02: positions (3, 2, 1) of red, green, blue inside the 13-band image; R, G, B
#   pretrained filters are copied there.
# init mode "mean" vs "zero": "mean" gives the other 10 bands the mean RGB filter, then scales every filter by 3/13 to
#   keep the first-layer response at its pretrained size; "zero" copies RGB filters unscaled and starts other bands at
#   0, so the net begins exactly as the RGB model (ablation).
# bilinear interpolation (F.interpolate): enlarges an image by smoothly mixing neighbouring pixels; used to upsample
#   64x64 images to 224x224 for pretrained models, on the device, so the data pipeline stays at 64 px.
# logits: raw class scores the model outputs; no softmax inside the model. num_classes = 10 EuroSAT classes; a fresh
#   output layer replaces the ImageNet one.
# parameters: the learned weights; count_params adds them all up (numel).
# torch.no_grad: used when copying weights, because that is initialisation and not learning (no gradients tracked).
# nn.Module: PyTorch base class of every network; assigning a layer as an attribute registers its weights.

# ======================================== Q&A ABOUT THIS FILE ========================================
# Q: What is this file for?
# A: It builds the four possible networks by name (scratch_cnn, resnet50, vit_small, vit_scratch), wraps each so it
#   accepts 64x64 images, and adapts pretrained first layers to 13 bands.
# Q: Where does it fit in the pipeline?
# A: train/engine.py `run` calls build_model once per run; scripts/smoke_test.py calls it for every model / input-type
#   pair. The returned model is then trained by the engine.
# Q: What does it read and write?
# A: It writes nothing. For pretrained models timm downloads the ImageNet weights on first use. The result is a
#   Classifier on CPU.
# Q: Why load the 3-channel checkpoint first and widen the conv afterwards?
# A: The published checkpoints only exist for 3 channels. Loading them unchanged and then replacing only the first conv
#   keeps every other pretrained layer intact and makes the band handling explicit and documented.

# Registry of the available architectures: the timm model name (if any), which layer takes the image
# (`first_conv`, as a dotted path), the input resolution the model expects, and extra constructor arguments.
ARCHS = {
    "scratch_cnn": {"input_size": 64},
    "resnet50": {"timm": "resnet50.a1_in1k", "first_conv": "conv1", "input_size": 224},
    "vit_small": {"timm": "vit_small_patch16_224.augreg_in21k_ft_in1k", "first_conv": "patch_embed.proj",
                  "input_size": 224},
    "vit_scratch": {"timm": "vit_tiny_patch16_224", "input_size": 64,
                    "kwargs": {"img_size": 64, "patch_size": 8}},
}


# Q: What is Classifier?
# A: A thin wrapper around any backbone that first upsamples the input batch to the size the backbone expects (e.g. 64
#   -> 224 px), then calls the backbone and returns raw logits.
# Q: Why upsample inside the model instead of in the data pipeline?
# A: It keeps the data pipeline, files and memory at 64 px for every model; the resize runs on the GPU only when needed,
#   so the loaders never produce 224 px images.
# Q: What happens for the scratch CNN?
# A: Its input size is already 64, so the batch passes through untouched.
class Classifier(nn.Module):
    """Upsamples the 64x64 input on the device to the backbone's input size, then classifies.

    Args:
        backbone: a network mapping (N, C, input_size, input_size) to (N, num_classes) logits.
        input_size: side length (px) the backbone expects. If a batch already has this size
            (e.g. the scratch CNN at 64 px) it is passed through untouched.

    The forward pass returns raw logits (no softmax). Interpolation is bilinear and runs on the
    same device as the batch, so the loaders never have to produce 224 px images.
    """

    # Q: What does Classifier.__init__ store?
    # A: The wrapped backbone and the image side length it expects. Because the backbone is an nn.Module attribute, its
    #   weights are registered and trained.
    def __init__(self, backbone: nn.Module, input_size: int):
        """Store the wrapped network and the image size it expects (see the class docstring)."""
        super().__init__()
        # Keep both as attributes; assigning a Module attribute also registers its weights with PyTorch.
        self.backbone, self.input_size = backbone, input_size

    # Q: What does Classifier.forward do?
    # A: If the batch's width differs from input_size it enlarges it with bilinear interpolation, then returns
    #   backbone(x): logits of shape (N, num_classes).
    # Q: Why bilinear and align_corners=False?
    # A: Bilinear is a smooth, standard resizing; align_corners=False is the usual PyTorch convention that keeps pixel
    #   centres consistent.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Resize the batch `x` (N, C, H, W) if needed and return the backbone's logits (N, num_classes)."""
        # Only resize when the image size differs from what the backbone wants (e.g. 64 -> 224 px).
        if x.shape[-1] != self.input_size:
            # Bilinear interpolation smoothly enlarges the image by mixing neighbouring pixels.
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return self.backbone(x)


# Q: What does adapt_input_conv do?
# A: It turns a pretrained RGB first-layer filter bank (out, 3, k, k) into one for in_chans channels (out, 13, k, k);
#   for 3 channels it just returns a copy.
# Q: Why is this needed?
# A: A pretrained network expects 3 channels; our multispectral images have 13. We cannot feed 13 channels into a
#   3-channel layer, and starting the new filters randomly would throw away what the pretrained filters know about
#   colour.
# Q: What do the "mean" and "zero" modes do, and why scale by 3 / in_chans?
# A: Both copy R, G, B filters to the B04, B03, B02 channels. "mean" gives every other band the mean RGB filter and
#   scales all filters by 3/13 so the summed response keeps the size the pretrained network saw with 3 channels; "zero"
#   leaves other bands at 0 so the model starts exactly as the RGB model.
# Q: Are the inputs modified?
# A: No. It builds a new tensor; `weight` itself is never changed. An unknown mode raises ValueError.
def adapt_input_conv(weight: torch.Tensor, in_chans: int, rgb_idx=MS_RGB_IDX, mode: str = "mean") -> torch.Tensor:
    """Widen a pretrained RGB filter bank (out, 3, k, k) to `in_chans` input channels.

    Args:
        weight: pretrained first-layer weights, shape (out, 3, k, k), channel order R, G, B.
        in_chans: number of input channels wanted (13 for the multispectral input).
        rgb_idx: destination channel of R, G, B in the wide input; default (B04, B03, B02).
        mode: "mean" or "zero" (see below).
    Returns:
        A new tensor of shape (out, in_chans, k, k); `weight` itself is never modified.

    in_chans == 3: unchanged. Otherwise:
      mode="mean": the R, G, B filters go to the channels in `rgb_idx` (B04, B03, B02);
                   every other band gets the mean of the three RGB filters; all filters
                   are then scaled by 3 / in_chans so the summed response keeps the
                   magnitude the pretrained network expects.
      mode="zero": R, G, B filters copied unscaled, other bands zero, so the network
                   starts exactly as the RGB model and must learn to use the new bands.
    """
    # Plain RGB input needs no change: return a copy of the pretrained filters.
    if in_chans == 3:
        return weight.clone()
    # Unpack the shape: number of filters, input channels (must be 3), and kernel height / width.
    out, cin, kh, kw = weight.shape
    assert cin == 3, f"expected an RGB filter bank, got {cin} input channels"
    # Start from an all-zero filter bank with the wider number of input channels.
    new = torch.zeros(out, in_chans, kh, kw, dtype=weight.dtype)
    if mode == "mean":
        new[:] = weight.mean(dim=1, keepdim=True)  # (out,1,k,k) broadcast: every band starts as the mean filter
    elif mode != "zero":
        raise ValueError(f"unknown init mode {mode!r}")
    for rgb_channel, band in enumerate(rgb_idx):  # overwrite the mean filter with the real R/G/B filter
        # Copy the pretrained R, G or B filter into the position of that colour among the 13 bands.
        new[:, band] = weight[:, rgb_channel]
    if mode == "mean":
        # Applied to ALL channels, RGB included, so the RGB filters end up weaker than the pretrained ones.
        new *= 3 / in_chans
    return new


# Q: What does _replace_first_conv do?
# A: It swaps the first Conv2d of a model (found by a dotted path such as "conv1" or "patch_embed.proj") for a new one
#   with in_chans input channels, filled from the pretrained filters, in place.
# Q: Why keep kernel size, stride, padding and bias?
# A: Only the number of input channels should change; everything else must match the pretrained design so the rest of
#   the network still works.
# Q: Why torch.no_grad() while copying?
# A: Copying initial values is not learning, so no gradient tracking is needed. The bias does not depend on the input
#   channels and is copied as is.
def _replace_first_conv(model: nn.Module, path: str, in_chans: int, mode: str) -> None:
    """Swap the Conv2d at the dotted `path` (e.g. "conv1", "patch_embed.proj") for an `in_chans` one.

    Modifies `model` in place. Kernel size, stride, padding, dilation, groups and bias presence are
    kept; the weights come from `adapt_input_conv` and the bias (if any) is copied unchanged.

    Args:
        model: the pretrained network to modify.
        path: dotted attribute path to the first convolution layer.
        in_chans: the new number of input channels.
        mode: "mean" or "zero" initialisation of the extra channels (see `adapt_input_conv`).
    """
    # Split "patch_embed.proj" into the parent module path ("patch_embed") and the layer name ("proj").
    parent_path, _, attr = path.rpartition(".")
    # Find the parent module (or the model itself when the layer sits at the top level, like "conv1").
    parent = model.get_submodule(parent_path) if parent_path else model
    # The existing 3-channel convolution layer.
    old: nn.Conv2d = getattr(parent, attr)
    # A new convolution identical to the old one except for the number of input channels.
    new = nn.Conv2d(in_chans, old.out_channels, old.kernel_size, old.stride, old.padding,
                    dilation=old.dilation, groups=old.groups, bias=old.bias is not None)
    # Copy weights without tracking gradients (this is initialisation, not learning).
    with torch.no_grad():
        # Fill the new filters from the pretrained RGB ones.
        new.weight.copy_(adapt_input_conv(old.weight, in_chans, mode=mode))
        # The bias does not depend on the number of input channels, so it is copied as is.
        if old.bias is not None:
            new.bias.copy_(old.bias)
    # Put the new layer in place of the old one inside the model.
    setattr(parent, attr, new)


# Q: What does build_model do?
# A: It creates the named architecture (scratch_cnn, resnet50, vit_small or vit_scratch), widens the first conv for
#   13-band input when the model is pretrained, and returns it wrapped in a Classifier.
# Q: Why different branches for scratch models and pretrained models?
# A: Scratch models are randomly initialised, so their first layer is created with in_chans channels directly.
#   Pretrained models must load the 3-channel checkpoint first, and only then is the first layer replaced.
# Q: Why pass img_size to the ViTs?
# A: The ViT's position embeddings depend on image size, so the model must be told the size it will receive (224, or 64
#   for vit_scratch).
# Q: What are the inputs and the return value?
# A: name, in_chans (3 or 13), num_classes (10), pretrained, init_mode, an optional input_size override, and **kwargs
#   for the backbone. It returns the model on CPU; the caller moves it to the device.
def build_model(name: str, in_chans: int = 3, num_classes: int = 10, pretrained: bool = True,
                init_mode: str = "mean", input_size: int | None = None, **kwargs) -> Classifier:
    """Build the named architecture (a key of `ARCHS`) wrapped in a `Classifier`.

    Args:
        name: "scratch_cnn", "resnet50", "vit_small" or "vit_scratch".
        in_chans: 3 for RGB, 13 for multispectral input.
        num_classes: size of the output layer (10 EuroSAT classes).
        pretrained: load ImageNet weights for resnet50 / vit_small (timm downloads them on first use);
            ignored for the two randomly initialised models.
        init_mode: 13-band first-layer init, "mean" or "zero"; see `adapt_input_conv`.
        input_size: override the backbone's input resolution (default: the value in `ARCHS`).
        **kwargs: forwarded to the backbone constructor.
    Returns:
        The model on CPU; the caller moves it to the device.
    """
    # Look up this architecture's settings and decide the input size (user override wins).
    spec = ARCHS[name]
    size = input_size or spec["input_size"]
    if name == "scratch_cnn":
        # Our own small CNN, randomly initialised, built directly with `in_chans` input channels.
        backbone = ScratchCNN(in_chans, num_classes, **kwargs)
    elif name == "vit_scratch":
        # A small Vision Transformer from timm with random weights and the right number of input channels.
        backbone = timm.create_model(spec["timm"], pretrained=False, num_classes=num_classes, in_chans=in_chans,
                                     **{**spec["kwargs"], **kwargs})
    else:
        # Load the RGB checkpoint, then widen its first conv ourselves (band-aware, documented).
        if name.startswith("vit"):
            # ViTs need to know the image size so their position embeddings match.
            kwargs = {"img_size": size, **kwargs}
        # Download / load the pretrained network with a fresh output layer of `num_classes` scores.
        backbone = timm.create_model(spec["timm"], pretrained=pretrained, num_classes=num_classes, **kwargs)
        # For multispectral input, replace the first convolution with a 13-channel version.
        if in_chans != 3:
            _replace_first_conv(backbone, spec["first_conv"], in_chans, init_mode)
    # Wrap with the resizing step so the data can stay at 64x64.
    return Classifier(backbone, size)


# Q: What does count_params return and what is it used for?
# A: The total number of weights in the model (sum of numel over all parameters). engine.run prints it, and stores it in
#   metrics.json as n_params, to compare model sizes (about 2.8M for the scratch CNN).
def count_params(model: nn.Module) -> int:
    """Total number of weights (parameters) in `model`, e.g. about 2.8 million for the scratch CNN."""
    # numel() = number of elements in one parameter tensor; add them all up.
    return sum(p.numel() for p in model.parameters())

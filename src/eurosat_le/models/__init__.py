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

from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import nn

from eurosat_le.constants import MS_RGB_IDX
from eurosat_le.models.scratch_cnn import ScratchCNN

ARCHS = {
    "scratch_cnn": {"input_size": 64},
    "resnet50": {"timm": "resnet50.a1_in1k", "first_conv": "conv1", "input_size": 224},
    "vit_small": {"timm": "vit_small_patch16_224.augreg_in21k_ft_in1k", "first_conv": "patch_embed.proj",
                  "input_size": 224},
    "vit_scratch": {"timm": "vit_tiny_patch16_224", "input_size": 64,
                    "kwargs": {"img_size": 64, "patch_size": 8}},
}


class Classifier(nn.Module):
    """Upsamples the 64x64 input on the device to the backbone's input size, then classifies.

    Args:
        backbone: a network mapping (N, C, input_size, input_size) to (N, num_classes) logits.
        input_size: side length (px) the backbone expects. If a batch already has this size
            (e.g. the scratch CNN at 64 px) it is passed through untouched.

    The forward pass returns raw logits (no softmax). Interpolation is bilinear and runs on the
    same device as the batch, so the loaders never have to produce 224 px images.
    """

    def __init__(self, backbone: nn.Module, input_size: int):
        super().__init__()
        self.backbone, self.input_size = backbone, input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return self.backbone(x)


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
    if in_chans == 3:
        return weight.clone()
    out, cin, kh, kw = weight.shape
    assert cin == 3, f"expected an RGB filter bank, got {cin} input channels"
    new = torch.zeros(out, in_chans, kh, kw, dtype=weight.dtype)
    if mode == "mean":
        new[:] = weight.mean(dim=1, keepdim=True)  # (out,1,k,k) broadcast: every band starts as the mean filter
    elif mode != "zero":
        raise ValueError(f"unknown init mode {mode!r}")
    for rgb_channel, band in enumerate(rgb_idx):  # overwrite the mean filter with the real R/G/B filter
        new[:, band] = weight[:, rgb_channel]
    if mode == "mean":
        # Applied to ALL channels, RGB included, so the RGB filters end up weaker than the pretrained ones.
        new *= 3 / in_chans
    return new


def _replace_first_conv(model: nn.Module, path: str, in_chans: int, mode: str) -> None:
    """Swap the Conv2d at the dotted `path` (e.g. "conv1", "patch_embed.proj") for an `in_chans` one.

    Modifies `model` in place. Kernel size, stride, padding, dilation, groups and bias presence are
    kept; the weights come from `adapt_input_conv` and the bias (if any) is copied unchanged.
    """
    parent_path, _, attr = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    old: nn.Conv2d = getattr(parent, attr)
    new = nn.Conv2d(in_chans, old.out_channels, old.kernel_size, old.stride, old.padding,
                    dilation=old.dilation, groups=old.groups, bias=old.bias is not None)
    with torch.no_grad():
        new.weight.copy_(adapt_input_conv(old.weight, in_chans, mode=mode))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    setattr(parent, attr, new)


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
    spec = ARCHS[name]
    size = input_size or spec["input_size"]
    if name == "scratch_cnn":
        backbone = ScratchCNN(in_chans, num_classes, **kwargs)
    elif name == "vit_scratch":
        backbone = timm.create_model(spec["timm"], pretrained=False, num_classes=num_classes, in_chans=in_chans,
                                     **{**spec["kwargs"], **kwargs})
    else:
        # Load the RGB checkpoint, then widen its first conv ourselves (band-aware, documented).
        if name.startswith("vit"):
            kwargs = {"img_size": size, **kwargs}
        backbone = timm.create_model(spec["timm"], pretrained=pretrained, num_classes=num_classes, **kwargs)
        if in_chans != 3:
            _replace_first_conv(backbone, spec["first_conv"], in_chans, init_mode)
    return Classifier(backbone, size)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

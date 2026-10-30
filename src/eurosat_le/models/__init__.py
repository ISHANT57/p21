"""Model registry: build_model(name, in_chans, pretrained) -> Classifier.

| name         | backbone                                   | input  | init                 |
|--------------|--------------------------------------------|--------|----------------------|
| scratch_cnn  | ScratchCNN (small ResNet, 2.8M params)     | 64 px  | random               |
| resnet50     | timm resnet50.a1_in1k                      | 224 px | ImageNet-1k          |
| vit_small    | timm vit_small_patch16_224.augreg_in21k_ft_in1k | 224 px | ImageNet-21k -> 1k |
| vit_scratch  | timm ViT-Tiny, patch 8, at 64 px (optional) | 64 px  | random               |

For 13-band input, the first convolution of a pretrained model (ResNet conv1 or
the ViT patch embedding) is widened with `adapt_input_conv`; see docs/MODELS.md.
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
    """Upsamples the 64x64 input on the device to the backbone's input size, then classifies."""

    def __init__(self, backbone: nn.Module, input_size: int):
        super().__init__()
        self.backbone, self.input_size = backbone, input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return self.backbone(x)


def adapt_input_conv(weight: torch.Tensor, in_chans: int, rgb_idx=MS_RGB_IDX, mode: str = "mean") -> torch.Tensor:
    """Widen a pretrained RGB filter bank (out, 3, k, k) to `in_chans` input channels.

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
        new[:] = weight.mean(dim=1, keepdim=True)
    elif mode != "zero":
        raise ValueError(f"unknown init mode {mode!r}")
    for rgb_channel, band in enumerate(rgb_idx):
        new[:, band] = weight[:, rgb_channel]
    if mode == "mean":
        new *= 3 / in_chans
    return new


def _replace_first_conv(model: nn.Module, path: str, in_chans: int, mode: str) -> None:
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

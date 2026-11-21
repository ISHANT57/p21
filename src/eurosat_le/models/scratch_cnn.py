"""Small ResNet-style CNN trained from scratch at the native 64x64 resolution.

This is the "no pretraining" baseline of the label-efficiency study (see docs/MODELS.md). It takes
3 (RGB) or 13 (multispectral) input channels directly: the first convolution is created with that
many channels and randomly initialised, so no band-specific adaptation is needed.

Layout and size (about 2.8M parameters):
    stem 3x3 conv (32 ch, stride 1) -> stage 1 (32 ch, 64x64) -> stage 2 (64 ch, 32x32)
    -> stage 3 (128 ch, 16x16) -> stage 4 (256 ch, 8x8) -> global average pool -> dropout -> linear.
Each stage is two `BasicBlock`s. The stem has no stride or pooling because the input is only 64x64.
"""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    """Standard ResNet basic block: two 3x3 conv+BN layers with a residual connection.

    Args:
        cin, cout: input / output channels.
        stride: stride of the first conv (2 halves the spatial size at the start of a stage).
    When the shape changes (stride != 1 or cin != cout) the shortcut is a 1x1 conv + BN; otherwise
    it is the identity. Convs have no bias because BatchNorm follows each of them.
    """

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ScratchCNN(nn.Module):
    """Stem (64x64) -> 4 stages of 2 BasicBlocks (64x64, 32x32, 16x16, 8x8) -> GAP -> linear.

    About 2.8M parameters; the stem has no stride or pooling because the input is only 64x64.

    Args:
        in_chans: 3 (RGB) or 13 (multispectral).
        num_classes: size of the output layer.
        widths: channels of the four stages.
        drop: dropout probability before the linear head (smoke_test.py sets it to 0 to test fitting).
    Input (N, in_chans, 64, 64) -> output (N, num_classes) logits.
    """

    def __init__(self, in_chans: int = 3, num_classes: int = 10, widths=(32, 64, 128, 256), drop: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_chans, widths[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        layers, cin = [], widths[0]
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2  # stage 1 keeps 64x64; each later stage halves the resolution
            layers += [BasicBlock(cin, w, stride), BasicBlock(w, w)]
            cin = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(drop), nn.Linear(cin, num_classes))
        for m in self.modules():  # Kaiming-normal init for every conv; BatchNorm and linear keep PyTorch defaults
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(self.stem(x)))

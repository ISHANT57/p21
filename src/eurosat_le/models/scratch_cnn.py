"""Small ResNet-style CNN trained from scratch at the native 64x64 resolution."""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
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
    """

    def __init__(self, in_chans: int = 3, num_classes: int = 10, widths=(32, 64, 128, 256), drop: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_chans, widths[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        layers, cin = [], widths[0]
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2
            layers += [BasicBlock(cin, w, stride), BasicBlock(w, w)]
            cin = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(drop), nn.Linear(cin, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(self.stem(x)))

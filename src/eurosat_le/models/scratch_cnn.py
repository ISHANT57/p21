"""Small ResNet-style CNN trained from scratch at the native 64x64 resolution.

This is the "no pretraining" baseline of the label-efficiency study (see docs/MODELS.md). It takes
3 (RGB) or 13 (multispectral) input channels directly: the first convolution is created with that
many channels and randomly initialised, so no band-specific adaptation is needed.

Layout and size (about 2.8M parameters):
    stem 3x3 conv (32 ch, stride 1) -> stage 1 (32 ch, 64x64) -> stage 2 (64 ch, 32x32)
    -> stage 3 (128 ch, 16x16) -> stage 4 (256 ch, 8x8) -> global average pool -> dropout -> linear.
Each stage is two `BasicBlock`s. The stem has no stride or pooling because the input is only 64x64.
"""

# Makes type hints such as `int | None` work as plain text on older Python versions.
from __future__ import annotations

# torch: the deep-learning framework (tensors); nn: ready-made network layers and the Module base class.
import torch
from torch import nn


class BasicBlock(nn.Module):
    """Standard ResNet basic block: two 3x3 conv+BN layers with a residual connection.

    Args:
        cin, cout: input / output channels.
        stride: stride of the first conv (2 halves the spatial size at the start of a stage).
    When the shape changes (stride != 1 or cin != cout) the shortcut is a 1x1 conv + BN; otherwise
    it is the identity. Convs have no bias because BatchNorm follows each of them.

    The "residual connection" adds the block's input to its output, which lets gradients flow
    straight through and makes deep networks much easier to train.
    """

    def __init__(self, cin: int, cout: int, stride: int = 1):
        """Create the layers of one block (see the class docstring for the arguments)."""
        # Initialise the nn.Module machinery so PyTorch can track the layers defined below.
        super().__init__()
        # First 3x3 convolution: looks for local patterns; stride 2 shrinks the image to half size.
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        # Batch normalisation rescales activations to a stable range, which speeds up and steadies training.
        self.bn1 = nn.BatchNorm2d(cout)
        # Second 3x3 convolution (stride 1) keeps the size and refines the features.
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        # ReLU keeps positive values and zeroes negative ones; this non-linearity lets the network learn curves.
        self.relu = nn.ReLU(inplace=True)
        # By default the shortcut passes the input through unchanged.
        self.shortcut = nn.Identity()
        # If the output shape differs from the input shape, the shortcut needs a 1x1 conv to match it
        # (otherwise the two could not be added together).
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        """Run the block on a batch `x` of shape (N, cin, H, W); returns (N, cout, H/stride, W/stride)."""
        # conv -> batch-norm -> ReLU.
        out = self.relu(self.bn1(self.conv1(x)))
        # second conv -> batch-norm (ReLU is applied after the addition).
        out = self.bn2(self.conv2(out))
        # Residual connection: add the (possibly reshaped) input back, then apply ReLU.
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
        """Build the stem, the four stages and the classification head (arguments: see class docstring)."""
        super().__init__()
        # Stem: first convolution turns the raw input bands into `widths[0]` feature maps, then BN and ReLU.
        self.stem = nn.Sequential(nn.Conv2d(in_chans, widths[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        # `layers` collects all blocks; `cin` tracks how many channels flow into the next block.
        layers, cin = [], widths[0]
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2  # stage 1 keeps 64x64; each later stage halves the resolution
            # Each stage is two blocks: the first may change size/width, the second keeps them.
            layers += [BasicBlock(cin, w, stride), BasicBlock(w, w)]
            cin = w
        # Chain all the blocks into one sequential feature extractor.
        self.features = nn.Sequential(*layers)
        # Head: average each feature map to one number (global average pooling), flatten to a vector,
        # randomly drop some values while training (dropout, against over-fitting), then a linear layer
        # that produces one score (logit) per class.
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(drop), nn.Linear(cin, num_classes))
        for m in self.modules():  # Kaiming-normal init for every conv; BatchNorm and linear keep PyTorch defaults
            if isinstance(m, nn.Conv2d):
                # Random starting weights scaled so signals neither explode nor vanish through the ReLU layers.
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of images (N, in_chans, 64, 64) to class logits (N, num_classes)."""
        # stem -> stages -> pooling/dropout/linear head.
        return self.head(self.features(self.stem(x)))

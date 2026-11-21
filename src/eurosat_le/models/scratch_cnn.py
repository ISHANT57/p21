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


# ======================================== CONCEPTS USED IN THIS FILE ========================================
# training from scratch: the weights start random (no pretrained weights), so this model is the "no pretraining"
#   baseline.
# CNN (convolutional neural network): a network that slides small filters over the image to find local patterns (edges,
#   textures, then larger shapes).
# convolution (Conv2d): a small learnable filter (here 3x3, or 1x1 for the shortcut) slid over the image; `stride` is
#   the step size, stride 2 halves the width and height.
# channels / bands / in_chans: depth of the image. 3 for RGB, 13 for multispectral; the first conv is created with this
#   many input channels, so no band adaptation is needed.
# stem, stage, block: the stem is the first conv layer; a stage is a group of blocks working at one resolution; a block
#   is the repeated building unit (BasicBlock).
# batch normalisation (BatchNorm2d): rescales each channel's activations to a stable range per batch; speeds up and
#   steadies training. That is why the convs have bias=False.
# ReLU: activation function max(0, x); the non-linearity that lets the network learn curved decision boundaries.
# residual (skip) connection / shortcut: the block adds its input to its output (out + shortcut(x)), so gradients flow
#   straight through and deeper nets train more easily. A 1x1 conv shortcut is used when the shape changes.
# global average pooling (AdaptiveAvgPool2d(1)): averages every feature map to one number, giving a fixed-length vector
#   regardless of image size.
# dropout (drop): randomly zeroes some values during training only, so the network cannot rely on single features
#   (fights over-fitting).
# linear layer / logits: the final layer gives one raw score (logit) per class; softmax/cross-entropy are applied later,
#   not in the model.
# Kaiming (He) initialisation: random starting weights scaled so signals neither explode nor vanish through ReLU layers;
#   mode "fan_out" uses the output size for the scale.
# parameters: all learned numbers (weights); this model has about 2.8 million.
# 64x64 resolution: EuroSAT images are 64x64 pixels, so the stem has no stride and no pooling (no early shrinking).

# ======================================== Q&A ABOUT THIS FILE ========================================
# Q: What is this file for?
# A: It defines our own small ResNet-style CNN (ScratchCNN) and its building block BasicBlock; it is the no-pretraining
#   baseline of the study.
# Q: Where does it fit in the pipeline?
# A: models/__init__.py build_model creates ScratchCNN and wraps it in Classifier; train/engine.py then trains it like
#   any other model.
# Q: What does it read and write?
# A: Nothing on disk. It takes an image batch (N, 3 or 13, 64, 64) and returns logits (N, 10).
# Q: Why a small CNN from scratch instead of only pretrained models?
# A: To have a baseline that shows how much ImageNet pretraining helps when labels are scarce; it also works directly on
#   13 bands and at native 64x64.

# Q: What is BasicBlock?
# A: The standard ResNet block: two 3x3 conv + batch-norm layers, plus a residual (skip) connection that adds the input
#   back before the final ReLU.
# Q: Why a residual connection?
# A: It gives gradients a direct path, which makes deeper networks much easier to train; without it the stacked layers
#   would be harder to optimise.
# Q: Why is the shortcut sometimes a 1x1 conv?
# A: When stride != 1 or the channel count changes, the input and output shapes differ and could not be added, so a 1x1
#   conv + BN reshapes the input; otherwise it is the identity.
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

    # Q: What does BasicBlock.__init__ do?
    # A: It creates the layers (conv1, bn1, conv2, bn2, ReLU and the shortcut); the actual computation happens later in
    #   forward.
    # Q: Why bias=False on the convolutions?
    # A: Batch norm follows each conv and has its own shift parameter, so a conv bias would be redundant.
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

    # Q: What does forward do for a block?
    # A: conv -> BN -> ReLU -> conv -> BN, then adds the shortcut of the input and applies ReLU. Shape (N, cin, H, W)
    #   in, (N, cout, H/stride, W/stride) out.
    # Q: Why is ReLU applied after the addition rather than before?
    # A: That is the standard ResNet design: the sum of branch and shortcut is passed through the non-linearity once.
    def forward(self, x):
        """Run the block on a batch `x` of shape (N, cin, H, W); returns (N, cout, H/stride, W/stride)."""
        # conv -> batch-norm -> ReLU.
        out = self.relu(self.bn1(self.conv1(x)))
        # second conv -> batch-norm (ReLU is applied after the addition).
        out = self.bn2(self.conv2(out))
        # Residual connection: add the (possibly reshaped) input back, then apply ReLU.
        return self.relu(out + self.shortcut(x))


# Q: What is ScratchCNN?
# A: The whole network: a 3x3 conv stem, four stages of two BasicBlocks (widths 32, 64, 128, 256; resolutions 64, 32,
#   16, 8), global average pooling, dropout and a linear layer giving 10 logits.
# Q: Why does the stem have no stride or pooling?
# A: The input is only 64x64, so shrinking it right away would throw away detail needed to tell similar land-use classes
#   apart.
# Q: What are the constructor arguments?
# A: in_chans (3 RGB or 13 multispectral), num_classes (10), widths (channels per stage) and drop (dropout probability,
#   0.2 by default).
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

    # Q: What does ScratchCNN.__init__ build, step by step?
    # A: The stem; then for each width a stage of two BasicBlocks (stride 1 for the first stage, 2 for later ones); then
    #   the pooling/dropout/linear head; finally Kaiming initialisation of every conv.
    # Q: Why is in_chans a parameter?
    # A: The first conv is built with exactly 3 or 13 input channels and trained from random weights, so the same class
    #   handles RGB and multispectral without any special adaptation.
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

    # Q: What does ScratchCNN.forward return?
    # A: Raw class scores (logits) of shape (N, num_classes) for a batch of shape (N, in_chans, 64, 64). No softmax
    #   here; the loss function applies it.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of images (N, in_chans, 64, 64) to class logits (N, num_classes)."""
        # stem -> stages -> pooling/dropout/linear head.
        return self.head(self.features(self.stem(x)))

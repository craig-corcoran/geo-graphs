"""U-Net for binary road segmentation, and the loss it trains against.

Deliberately small and plain. The interesting part of this project is the graph
extraction and the topology-aware metric, not the architecture, so this is a
textbook U-Net with a configurable width rather than anything clever.

Convention: the model emits **logits**, not probabilities. Keeping the sigmoid
out of the forward pass is what lets the loss use the numerically stable
``binary_cross_entropy_with_logits``; callers wanting a mask apply
:func:`predict_mask`.
"""

import itertools
import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two 3x3 convolutions with batch norm and ReLU, the U-Net repeating unit."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Encoder-decoder with skip connections, emitting one logit per pixel.

    Args:
        in_channels: Input image channels.
        widths: Channel count at each encoder level. Depth is
            ``len(widths) - 1`` downsamples, so the input side must be
            divisible by ``2 ** (len(widths) - 1)``.
    """

    def __init__(
        self, in_channels: int = 3, widths: Sequence[int] = (32, 64, 128, 256)
    ) -> None:
        super().__init__()
        if len(widths) < 2:
            raise ValueError("widths needs at least an encoder and a bottleneck")

        self.widths = tuple(widths)
        self.pool = nn.MaxPool2d(2)

        channels = [in_channels, *self.widths[:-2]]
        self.encoders = nn.ModuleList(
            conv_block(a, b) for a, b in zip(channels, self.widths[:-1], strict=True)
        )
        self.bottleneck = conv_block(self.widths[-2], self.widths[-1])

        reversed_widths = list(reversed(self.widths))
        self.ups = nn.ModuleList(
            nn.ConvTranspose2d(a, b, 2, stride=2)
            for a, b in itertools.pairwise(reversed_widths)
        )
        self.decoders = nn.ModuleList(conv_block(2 * b, b) for b in reversed_widths[1:])
        self.head = nn.Conv2d(self.widths[0], 1, 1)

    @property
    def depth(self) -> int:
        """Number of downsampling steps; input dimensions must divide by 2**depth."""
        return len(self.widths) - 1

    def forward(self, x: Tensor) -> Tensor:
        """Map a batch of images to per-pixel logits.

        Args:
            x: ``(N, C, H, W)`` float tensor.

        Returns:
            ``(N, 1, H, W)`` logits.
        """
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, decoder, skip in zip(
            self.ups, self.decoders, reversed(skips), strict=True
        ):
            x = up(x)
            x = decoder(torch.cat([skip, x], dim=1))

        return self.head(x)


def soft_dice_loss(logits: Tensor, targets: Tensor, eps: float = 1.0) -> Tensor:
    """Dice loss on sigmoid probabilities.

    Roads occupy under a tenth of a tile, so cross-entropy alone is dominated by
    easy background and a model can score well while predicting almost nothing.
    Dice is computed on the overlap ratio, which does not let that pass.

    Args:
        logits: ``(N, 1, H, W)`` raw model outputs.
        targets: ``(N, 1, H, W)`` float targets in ``{0, 1}``.
        eps: Smoothing added to numerator and denominator, which also defines
            the loss as 0 when both prediction and target are empty.

    Returns:
        Scalar loss in ``[0, 1]``.
    """
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dims)
    total = probs.sum(dims) + targets.sum(dims)
    return 1.0 - ((2.0 * intersection + eps) / (total + eps)).mean()


def segmentation_loss(
    logits: Tensor, targets: Tensor, dice_weight: float = 0.5
) -> Tensor:
    """Binary cross-entropy plus soft Dice.

    Args:
        logits: ``(N, 1, H, W)`` raw model outputs.
        targets: ``(N, 1, H, W)`` float targets in ``{0, 1}``.
        dice_weight: Blend factor; 0 is pure cross-entropy, 1 is pure Dice.

    Returns:
        Scalar loss.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    return (1.0 - dice_weight) * bce + dice_weight * soft_dice_loss(logits, targets)


def logit(probability: float) -> float:
    """Inverse of the sigmoid: the raw output that yields this probability.

    Args:
        probability: A value in ``[0, 1]``.

    Returns:
        The corresponding logit, infinite at the endpoints.

    Raises:
        ValueError: If the probability lies outside ``[0, 1]``.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1]; got {probability}")
    if probability == 0.0:
        return -math.inf
    if probability == 1.0:
        return math.inf
    return math.log(probability / (1.0 - probability))


def predict_mask[ArrayT: (np.ndarray, Tensor)](
    logits: ArrayT, threshold: float = 0.5
) -> ArrayT:
    """Threshold logits into a boolean road mask.

    The comparison happens in logit space rather than probability space. Both
    give the same answer because the sigmoid is monotonic, but this way there is
    no sigmoid pass over the array and the same expression works on a numpy
    array or a torch tensor without branching on type. That matters because
    inference returns numpy while the training loop stays in torch, and both
    have to reach the same decision.

    This is the single definition of "is this pixel road", deliberately. The
    threshold is a real hyperparameter, not a formality: raising it thins the
    mask, so fewer connections are invented and more roads are severed. It
    trades the two APLS directions directly against each other and should be
    tuned on APLS rather than on pixel overlap.

    Args:
        logits: Raw model outputs, any shape.
        threshold: Probability above which a pixel counts as road.

    Returns:
        A boolean array of the same shape and type as ``logits``.

    Raises:
        ValueError: If the threshold lies outside ``[0, 1]``.
    """
    return logits > logit(threshold)

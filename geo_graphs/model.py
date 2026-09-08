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


#: Peeling iterations for :func:`soft_skeleton`.
#:
#: Each iteration erodes by one pixel of radius, so the skeleton of a bar is
#: only complete once the iteration count reaches its half-width. Roads here
#: rasterize to roughly 10-14 px at 1 m/px, a half-width of 5-7, and 10 leaves
#: headroom for the widest of them and for a model that predicts a road thicker
#: than the label. Under-counting silently returns a hollow skeleton -- the
#: centreline of a wide road never appears -- and that failure looks like a
#: weak loss rather than a mis-set knob, so the committed default is the
#: correct one and a smoke run turns it down at the call site.
SKELETON_ITERATIONS = 10


def _soft_erode(x: Tensor) -> Tensor:
    """Grayscale erosion by a 3x3 box, as a differentiable min-pool."""
    return -nn.functional.max_pool2d(-x, 3, stride=1, padding=1)


def _soft_dilate(x: Tensor) -> Tensor:
    """Grayscale dilation by a 3x3 box."""
    return nn.functional.max_pool2d(x, 3, stride=1, padding=1)


def _soft_open(x: Tensor) -> Tensor:
    """Erosion followed by dilation; anti-extensive, so ``open(x) <= x``."""
    return _soft_dilate(_soft_erode(x))


def soft_skeleton(probs: Tensor, iterations: int = SKELETON_ITERATIONS) -> Tensor:
    """Differentiable morphological skeleton of a probability map.

    Peels the shape one pixel of radius at a time. At each level the residue
    ``relu(x - open(x))`` holds the parts too thin to survive an opening — the
    medial axis of what is left — and those residues accumulate without ever
    exceeding one, since ``skel + relu(delta - skel * delta)`` adds
    ``delta * (1 - skel)`` while both stay in ``[0, 1]``.

    Ends erode too, so the skeleton of a bar is shorter than the bar by about
    its half-width at each end. That shortening is not an artifact to correct:
    it is what makes a gap a larger fraction of the centreline than of the area,
    which is the sensitivity :func:`soft_cldice_loss` is built on.

    Args:
        probs: ``(N, C, H, W)`` values in ``[0, 1]`` — probabilities or a binary
            mask, not logits.
        iterations: Peeling steps. Must reach the shape's half-width in pixels
            or the skeleton is hollow; see :data:`SKELETON_ITERATIONS`. This is
            the cost knob: each step is three pooling passes over the batch.

    Returns:
        ``(N, C, H, W)`` skeleton membership in ``[0, 1]``, differentiable
        with respect to ``probs``.
    """
    relu = nn.functional.relu
    skeleton = relu(probs - _soft_open(probs))
    eroded = probs
    for _ in range(iterations):
        eroded = _soft_erode(eroded)
        delta = relu(eroded - _soft_open(eroded))
        skeleton = skeleton + relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(
    logits: Tensor,
    targets: Tensor,
    iterations: int = SKELETON_ITERATIONS,
    eps: float = 1.0,
) -> Tensor:
    """Centreline Dice loss: Dice measured on skeletons rather than volumes.

    Dice and cross-entropy are both pixel measures, and both are nearly
    indifferent to the one failure that dominates APLS. A few pixels missing
    under a tree shadow barely move an overlap ratio, but they sever a route.
    clDice scores against the *centreline*, which runs straight through such a
    gap and — because skeleton ends retract by roughly the road half-width —
    loses far more than the gap's own length when the road is cut.

    Two quantities, following Shit et al., "clDice - a Novel
    Topology-Preserving Loss Function for Tubular Structure Segmentation"
    (CVPR 2021):

    - *Topological precision*: how much of the predicted skeleton lies on real
      road, which punishes invented connections.
    - *Topological sensitivity*: how much of the true skeleton the prediction
      covers, which punishes severed ones.

    They combine by **harmonic mean**, the same combination APLS uses over its
    two directions, so neither direction can be bought by sacrificing the other.

    What this does not do, and should not be sold as doing: it is a *local,
    soft* proxy. It rewards overlapping skeletons, not connected routes — two
    fragments 20 px apart overlap nothing and generate no gradient pulling them
    together, because the pooling that builds the skeleton only ever sees a 3x3
    neighbourhood per step. And it cannot touch the ~0.97 non-planarity ceiling,
    which is a representation limit of a 2D mask rather than a loss one; the
    number it can move is ``fraction_of_ceiling``.

    Args:
        logits: ``(N, 1, H, W)`` raw model outputs. Logits, not probabilities —
            the sigmoid is applied here, as in :func:`soft_dice_loss`.
        targets: ``(N, 1, H, W)`` float targets in ``{0, 1}``.
        iterations: Skeletonization steps; see :data:`SKELETON_ITERATIONS`.
        eps: Smoothing added to numerator and denominator of both ratios,
            matching :func:`soft_dice_loss`. It also keeps every division
            finite: an empty skeleton scores 1 rather than dividing by zero, so
            the loss is 0 when prediction and target are both empty.

    Returns:
        Scalar loss in ``[0, 1]``.
    """
    probs = torch.sigmoid(logits)
    predicted_skeleton = soft_skeleton(probs, iterations)
    target_skeleton = soft_skeleton(targets, iterations)

    dims = tuple(range(1, probs.ndim))
    precision = ((predicted_skeleton * targets).sum(dims) + eps) / (
        predicted_skeleton.sum(dims) + eps
    )
    sensitivity = ((target_skeleton * probs).sum(dims) + eps) / (
        target_skeleton.sum(dims) + eps
    )
    return 1.0 - (2.0 * precision * sensitivity / (precision + sensitivity)).mean()


def segmentation_loss(
    logits: Tensor,
    targets: Tensor,
    dice_weight: float = 0.5,
    cldice_weight: float = 0.0,
    cldice_iterations: int = SKELETON_ITERATIONS,
) -> Tensor:
    """Binary cross-entropy plus soft Dice, optionally blended with clDice.

    The pixel terms come first: ``dice_weight`` mixes cross-entropy with soft
    Dice, and ``cldice_weight`` then mixes that pixel loss with the centreline
    term. Nested convex blends rather than an extra additive term, so the total
    stays on the same scale whatever the weights and two runs remain comparable.

    clDice blends, it does not replace: the paper reports it is unstable used
    alone, so ``cldice_weight`` is meant for the low end of its range.

    Args:
        logits: ``(N, 1, H, W)`` raw model outputs.
        targets: ``(N, 1, H, W)`` float targets in ``{0, 1}``.
        dice_weight: Blend factor; 0 is pure cross-entropy, 1 is pure Dice.
        cldice_weight: Share of the centreline term. Exactly 0 skips the
            skeletonization entirely rather than multiplying it away, since the
            iterated pooling is the expensive part of the loss.
        cldice_iterations: Skeletonization steps, unused at weight 0; see
            :data:`SKELETON_ITERATIONS`.

    Returns:
        Scalar loss.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    pixel = (1.0 - dice_weight) * bce + dice_weight * soft_dice_loss(logits, targets)
    if cldice_weight == 0.0:
        return pixel
    cldice = soft_cldice_loss(logits, targets, iterations=cldice_iterations)
    return (1.0 - cldice_weight) * pixel + cldice_weight * cldice


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

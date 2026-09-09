"""U-Net for binary road segmentation, and the loss it trains against.

Deliberately small and plain. The interesting part of this project is the graph
extraction and the topology-aware metric, not the architecture, so this is a
textbook U-Net with a configurable width rather than anything clever.

The one pluggable piece is how an auxiliary channel stack — lidar height and
ground-return intensity, with an availability indicator — reaches the network.
That goes through :data:`FUSION_REGISTRY`, because the choice is a measurement
rather than a preference.

Convention: the model emits **logits**, not probabilities. Keeping the sigmoid
out of the forward pass is what lets the loss use the numerically stable
``binary_cross_entropy_with_logits``; callers wanting a mask apply
:func:`predict_mask`.
"""

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

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


def mask_unobserved(aux: Tensor) -> Tensor:
    """Zero the aux data channels wherever the trailing validity channel is 0.

    The one guarantee that makes the fill value unobservable. Whatever a data
    pipeline wrote into unmeasured positions — zero, a sentinel, uninitialized
    memory — is multiplied away here, so a model cannot learn to read the
    coverage mask off the data channels or to treat one particular fill as
    evidence about the ground. Every fusion implementation calls this before
    the aux stack reaches a convolution.

    Args:
        aux: ``(N, K + 1, H, W)``, data channels followed by the availability
            indicator in ``{0, 1}``.

    Returns:
        The same tensor with unobserved data zeroed and the indicator kept.

    Raises:
        ValueError: If the stack has no room for both data and an indicator.
    """
    if aux.ndim != 4 or aux.shape[1] < 2:
        raise ValueError(f"expected (N, K + 1, H, W) with K >= 1; got {tuple(aux.shape)}")
    return torch.cat([aux[:, :-1] * aux[:, -1:], aux[:, -1:]], dim=1)


@runtime_checkable
class Fusion(Protocol):
    """How an auxiliary channel stack reaches the segmentation network.

    A stage rather than a fixed choice because the alternatives are not
    equivalent under domain shift, and that cannot be settled in-domain. nDSM
    and roughness are physical measurements in metres: they do not move with
    atmosphere, sun angle, sensor calibration or season, and RGB moves with all
    of them. Early fusion entangles the two in the first convolution, so they
    cannot be regularized or augmented differently; a two-encoder alternative
    can, at the cost of parameters. Which trade wins is a measurement, so it
    goes through a registry.
    """

    def stem_channels(self, image_channels: int, aux_channels: int) -> int:
        """Input channels the encoder stem must be built to accept."""
        ...

    def fuse(self, image: Tensor, aux: Tensor) -> Tensor:
        """Combine imagery and the aux stack into the stem's input tensor."""
        ...


@dataclass(frozen=True, slots=True)
class EarlyFusion:
    """Concatenate the aux stack onto the input stem.

    The cheap end of the design space: one wider first convolution and no other
    change to the network. It ties imagery and geometry together immediately,
    which is efficient and is exactly the property that makes it suspect across
    areas — see :class:`Fusion`.
    """

    def stem_channels(self, image_channels: int, aux_channels: int) -> int:
        """Image channels plus the whole aux stack, indicator included."""
        return image_channels + aux_channels

    def fuse(self, image: Tensor, aux: Tensor) -> Tensor:
        """Concatenate along the channel axis, fill masked out first."""
        return torch.cat([image, mask_unobserved(aux)], dim=1)


# Structural typing is checked where a class enters a Protocol-typed slot, and
# the registry's values are factories rather than instances, so nothing else
# here would check EarlyFusion at all.
_: type[Fusion] = EarlyFusion

#: Selects a fusion strategy by config key. Values are factories, so each
#: lookup yields a fresh instance rather than a shared one. ``dual`` — separate
#: encoders fused at each decoder scale — is deliberately absent until there
#: are coverage curves to compare it against.
FUSION_REGISTRY: dict[str, Callable[..., Fusion]] = {
    "early": lambda **kw: EarlyFusion(**kw),
}


def seed_stem_channels(stem: nn.Conv2d, image_channels: int) -> None:
    """Initialize a widened stem's extra inputs from the mean of the image ones.

    Concatenating channels onto the stem changes the shape of its weight, so a
    pretrained RGB encoder cannot be loaded into it unmodified and a fresh
    random block would swamp the pretrained response on the first steps.
    Seeding each new channel with the mean over the RGB weights keeps the
    layer's output distribution close to what it was, which is the standard
    recipe for widening a pretrained stem.

    Nothing happens when the stem is no wider than the imagery, so a fusion
    strategy that does not touch the stem needs no special case.

    Args:
        stem: The network's first convolution, modified in place.
        image_channels: How many of its input channels are imagery.
    """
    if stem.weight.shape[1] <= image_channels:
        return
    with torch.no_grad():
        stem.weight[:, image_channels:] = stem.weight[:, :image_channels].mean(
            dim=1, keepdim=True
        )


def _first_conv(block: nn.Module) -> nn.Conv2d:
    """The first convolution inside a block, which for encoder 0 is the stem."""
    for module in block.modules():
        if isinstance(module, nn.Conv2d):
            return module
    raise ValueError("block contains no convolution")


class UNet(nn.Module):
    """Encoder-decoder with skip connections, emitting one logit per pixel.

    Args:
        in_channels: Input image channels.
        widths: Channel count at each encoder level. Depth is
            ``len(widths) - 1`` downsamples, so the input side must be
            divisible by ``2 ** (len(widths) - 1)``.
        aux_channels: Channels in the auxiliary stack, availability indicator
            included. Zero is the imagery-only network.
        fusion: How that stack reaches the network. Required when
            ``aux_channels`` is non-zero and meaningless otherwise.

    Raises:
        ValueError: If ``widths`` is degenerate, or if ``aux_channels`` and
            ``fusion`` disagree about whether there is anything to fuse.
    """

    def __init__(
        self,
        in_channels: int = 3,
        widths: Sequence[int] = (32, 64, 128, 256),
        aux_channels: int = 0,
        fusion: Fusion | None = None,
    ) -> None:
        super().__init__()
        if len(widths) < 2:
            raise ValueError("widths needs at least an encoder and a bottleneck")
        if (aux_channels > 0) != (fusion is not None):
            raise ValueError(
                f"aux_channels={aux_channels} and fusion={fusion!r} disagree; "
                "supply both or neither"
            )

        self.widths = tuple(widths)
        self.in_channels = in_channels
        self.aux_channels = aux_channels
        self.fusion = fusion
        self.pool = nn.MaxPool2d(2)

        stem_in = (
            fusion.stem_channels(in_channels, aux_channels)
            if fusion is not None
            else in_channels
        )
        channels = [stem_in, *self.widths[:-2]]
        self.encoders = nn.ModuleList(
            conv_block(a, b) for a, b in zip(channels, self.widths[:-1], strict=True)
        )
        seed_stem_channels(_first_conv(self.encoders[0]), in_channels)
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

    def forward(self, x: Tensor, aux: Tensor | None = None) -> Tensor:
        """Map a batch of images, and optionally lidar, to per-pixel logits.

        Args:
            x: ``(N, C, H, W)`` float tensor.
            aux: ``(N, K + 1, H, W)`` auxiliary stack, the last channel being
                the availability indicator. Required exactly when the network
                was built with a fusion stage.

        Returns:
            ``(N, 1, H, W)`` logits.

        Raises:
            ValueError: If ``aux`` and the configured fusion disagree.
        """
        if self.fusion is None:
            if aux is not None:
                raise ValueError("aux supplied to a network built without fusion")
        elif aux is None:
            raise ValueError("network was built with fusion but got no aux stack")
        else:
            x = self.fusion.fuse(x, aux)

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

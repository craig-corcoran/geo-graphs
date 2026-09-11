"""Assigning samples to training and validation, at random or by ground position.

A random split over chips is the wrong default for any task whose label is a
property of a *road* rather than of a pixel. SpaceNet chips are adjacent tiles,
so a street cut by a chip boundary is seen from both sides of a random split;
measured over the Las Vegas AOI, 876 of the 2,453 OSM ways touching a validation
chip also touch a training chip. For segmentation that leaks texture. For
attribute inference, where the class is constant along the way, it leaks the
answer.

The alternative is to hold out whole squares of ground, so that a road can only
straddle the split where it crosses a block boundary. Block size trades leakage
against everything else: larger blocks cut fewer roads, and also make the
validation set a smaller number of larger, more correlated places, which costs
statistical power and lets the class mix drift away from the city's.
``scripts/split_sweep.py`` measures that trade rather than assuming a value.

Buffering goes further and drops the training samples nearest the validation
blocks, opening a gap no road can cross. It is the only option here that removes
adjacency rather than merely reducing it, and it pays for that in training
samples.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.spatial import KDTree

from .tiles import Tile


@dataclass(frozen=True, slots=True)
class Placement:
    """Where each sample sits on the ground.

    Attributes:
        ids: Sample ids, aligned with the coordinate arrays.
        x: Easting of each sample's centre, in metres.
        y: Northing of each sample's centre, in metres.
    """

    ids: tuple[str, ...]
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True, slots=True)
class Assignment:
    """Which side of a split each sample landed on.

    Attributes:
        train: Samples to fit on.
        val: Samples held out.
        dropped: Samples on neither side. A buffered splitter uses these to open
            a gap between train and validation; every other splitter leaves it
            empty, and a caller that ignores it silently trains on the buffer.
    """

    train: tuple[str, ...]
    val: tuple[str, ...]
    dropped: tuple[str, ...]


@runtime_checkable
class Splitter(Protocol):
    """How samples are divided between training and validation."""

    def split(self, placement: Placement, val_fraction: float, seed: int) -> Assignment:
        """Assign every sample in ``placement`` to a side."""
        ...


def placement_from_tiles(ids: Sequence[str], tile_seq: Sequence[Tile]) -> Placement:
    """Take sample centres from their tiles.

    How a tile was obtained is the caller's business: reading a chip header is
    three orders of magnitude cheaper than loading its pixels, and only the
    caller knows whether it already has the pixels for another reason.

    Args:
        ids: Sample ids.
        tile_seq: Each sample's tile, in the same order and one CRS.

    Returns:
        The placement.

    Raises:
        ValueError: If the tiles span several CRSs, which would put the
            coordinates on incomparable grids and silently scramble any
            distance between them.
    """
    if len({str(t.crs) for t in tile_seq}) > 1:
        raise ValueError(
            "tiles span several CRSs; reproject to one before placing them, "
            "since block and buffer distances are measured in its units"
        )
    return Placement(
        ids=tuple(ids),
        x=np.array([t.x_min + t.width * t.resolution / 2.0 for t in tile_seq]),
        y=np.array([t.y_max - t.height * t.resolution / 2.0 for t in tile_seq]),
    )


def block_index(placement: Placement, block_m: float) -> np.ndarray:
    """Which square block of ground each sample falls in.

    The grid's origin is the south-west corner of the samples themselves, so a
    block is defined relative to the data rather than to the CRS origin. That
    keeps the number of blocks independent of where in the world the AOI sits.

    Args:
        placement: Sample positions.
        block_m: Block side length in metres.

    Returns:
        ``(n,)`` block label per sample, dense from zero.

    Raises:
        ValueError: If ``block_m`` is not positive.
    """
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")
    corner = np.column_stack(
        [
            np.floor((placement.x - placement.x.min()) / block_m),
            np.floor((placement.y - placement.y.min()) / block_m),
        ]
    )
    return np.unique(corner, axis=0, return_inverse=True)[1].ravel()


def hold_out_groups(group: np.ndarray, val_fraction: float, seed: int) -> np.ndarray:
    """Take whole groups for validation until the requested share is reached.

    Groups are consumed in a shuffled order and never split, so the achieved
    share overshoots the target by at most the size of the last group taken. No
    group is skipped for being too large: skipping would bias the validation set
    toward sparse places, which is exactly the kind of silent selection this
    function exists to avoid. A caller that cares about the overshoot should
    report the achieved fraction, not correct it.

    Args:
        group: ``(n,)`` group label per sample.
        val_fraction: Share of *samples* to aim for.
        seed: Seeds the group order.

    Returns:
        ``(n,)`` boolean, true where the sample is held out. At least one group
        is always taken.
    """
    labels = np.unique(group)
    np.random.default_rng(seed).shuffle(labels)
    sizes = np.bincount(group, minlength=int(labels.max()) + 1)

    target = max(round(len(group) * val_fraction), 1)
    taken: list[int] = []
    count = 0
    for label in labels:
        taken.append(int(label))
        count += int(sizes[label])
        if count >= target:
            break
    return np.isin(group, taken)


def _assign(placement: Placement, is_val: np.ndarray, drop: np.ndarray) -> Assignment:
    """Turn per-sample flags into an assignment, preserving input order."""
    ids = np.asarray(placement.ids)
    return Assignment(
        train=tuple(ids[~is_val & ~drop]),
        val=tuple(ids[is_val]),
        dropped=tuple(ids[drop]),
    )


class RandomSplitter:
    """Each sample independently, ignoring where it is.

    The baseline, and wrong for anything that reads a road as one object.

    Both sides come back in the shuffled order rather than the input order.
    That is not cosmetic: :func:`train.build_dataset` draws crop windows for
    each sample in turn from one seeded stream, so permuting the ids changes
    every window in the run.
    """

    def split(self, placement: Placement, val_fraction: float, seed: int) -> Assignment:
        """Shuffle the samples and hold out a fraction; see :class:`Splitter`."""
        shuffled = list(placement.ids)
        np.random.default_rng(seed).shuffle(shuffled)
        n_val = max(round(len(shuffled) * val_fraction), 1)
        return Assignment(
            train=tuple(shuffled[n_val:]), val=tuple(shuffled[:n_val]), dropped=()
        )


class BlockedSplitter:
    """Whole squares of ground, so a road straddles the split only at a block edge."""

    def __init__(self, block_m: float = 1280.0) -> None:
        """
        Args:
            block_m: Block side length in metres. Below a chip's own size this
                degenerates to a random split; far above it, the validation set
                becomes a handful of neighbourhoods.
        """
        self.block_m = block_m

    def split(self, placement: Placement, val_fraction: float, seed: int) -> Assignment:
        """Hold out whole blocks; see :class:`Splitter`."""
        is_val = hold_out_groups(block_index(placement, self.block_m), val_fraction, seed)
        return _assign(placement, is_val, np.zeros(len(is_val), dtype=bool))


class BufferedBlockSplitter:
    """Blocks, minus the training samples nearest them.

    Blocking alone leaves training and validation samples touching along block
    boundaries. Dropping a margin removes that contact outright, at the cost of
    the dropped samples. The margin comes out of *training*: taking it out of
    validation instead would shrink the set every reported number is measured
    on, and evaluation noise is already the binding constraint.
    """

    def __init__(self, block_m: float = 1280.0, buffer_m: float = 500.0) -> None:
        """
        Args:
            block_m: Block side length in metres.
            buffer_m: Minimum distance between a kept training sample's centre
                and any validation sample's centre. Measured centre to centre,
                so a value below a sample's own width drops nothing.
        """
        self.block_m = block_m
        self.buffer_m = buffer_m

    def split(self, placement: Placement, val_fraction: float, seed: int) -> Assignment:
        """Hold out whole blocks, then drop the training margin; see :class:`Splitter`."""
        is_val = hold_out_groups(block_index(placement, self.block_m), val_fraction, seed)
        points = np.column_stack([placement.x, placement.y])
        distance, _ = KDTree(points[is_val]).query(points, k=1)
        return _assign(placement, is_val, ~is_val & (distance < self.buffer_m))


_: type[Splitter] = RandomSplitter
_: type[Splitter] = BlockedSplitter
_: type[Splitter] = BufferedBlockSplitter

#: Selects a splitter by config key. Values are factories, so each lookup yields
#: a fresh instance rather than a shared one.
SPLIT_REGISTRY: dict[str, Callable[..., Splitter]] = {
    "random": lambda **kwargs: RandomSplitter(**kwargs),
    "blocked": lambda **kwargs: BlockedSplitter(**kwargs),
    "buffered": lambda **kwargs: BufferedBlockSplitter(**kwargs),
}

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

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
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


def digest(ids: Sequence[str]) -> str:
    """A content hash over a list of sample ids, order included.

    Order is part of the identity rather than noise: :func:`train.build_dataset`
    draws crop windows per sample from one seeded stream, so two assignments
    over the same ids in different orders are two different training runs.
    """
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenSplit:
    """One assignment recorded so that it cannot drift.

    A split is reproducible from its splitter, its seed and the list of samples
    it was drawn over. The last of those lives in an untracked data directory,
    so a chip added, removed or reordered silently produces a different split
    under the same seed. Recording the assignment, and hashing what produced it,
    is what turns "reproducible" into "the same".

    Attributes:
        name: The configuration's label, e.g. ``"buffered-2560+1000"``.
        splitter: Key of :data:`SPLIT_REGISTRY` that drew it.
        params: The splitter's own arguments, plus ``val_fraction`` and
            ``seed``.
        source: Where the samples came from: ``aoi_root``, ``resolution``,
            ``n_chips``, and ``chips_sha256`` over the id list in the order the
            source listed them.
        assignment: The assignment itself.
        digest: SHA-256 over the three sides, which
            :func:`read_frozen` checks on the way in.
    """

    name: str
    splitter: str
    params: Mapping[str, float]
    source: Mapping[str, str | float | int]
    assignment: Assignment
    digest: str


def _assignment_digest(assignment: Assignment) -> str:
    """Hash all three sides together, so moving one id between them shows up."""
    return digest(
        [
            *assignment.train,
            "--val--",
            *assignment.val,
            "--dropped--",
            *assignment.dropped,
        ]
    )


def freeze(
    name: str,
    splitter: str,
    kwargs: Mapping[str, float],
    placement: Placement,
    val_fraction: float,
    seed: int,
    source: Mapping[str, str | float | int],
) -> FrozenSplit:
    """Draw a split and record everything needed to recognise it again.

    Args:
        name: Label for the configuration.
        splitter: Key of :data:`SPLIT_REGISTRY`.
        kwargs: The splitter's own arguments.
        placement: Where the samples sit.
        val_fraction: Share of samples to hold out.
        seed: Seeds the draw.
        source: Provenance of the sample list; see :class:`FrozenSplit`.

    Returns:
        The frozen split.
    """
    assignment = SPLIT_REGISTRY[splitter](**kwargs).split(placement, val_fraction, seed)
    return FrozenSplit(
        name=name,
        splitter=splitter,
        params=MappingProxyType(
            dict(kwargs) | {"val_fraction": val_fraction, "seed": seed}
        ),
        source=MappingProxyType(dict(source)),
        assignment=assignment,
        digest=_assignment_digest(assignment),
    )


def write_frozen(path: Path | str, frozen: FrozenSplit) -> None:
    """Write a frozen split as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(
            {
                "name": frozen.name,
                "splitter": frozen.splitter,
                "params": dict(frozen.params),
                "source": dict(frozen.source),
                "digest": frozen.digest,
                "train": list(frozen.assignment.train),
                "val": list(frozen.assignment.val),
                "dropped": list(frozen.assignment.dropped),
            },
            indent=2,
        )
    )


def read_frozen(path: Path | str) -> FrozenSplit:
    """Read a frozen split, refusing one whose ids no longer hash to its digest.

    Args:
        path: File written by :func:`write_frozen`.

    Returns:
        The frozen split.

    Raises:
        ValueError: If the recorded digest does not match the recorded ids,
            which means the file was edited rather than redrawn.
    """
    raw = json.loads(Path(path).read_text())
    assignment = Assignment(
        train=tuple(raw["train"]),
        val=tuple(raw["val"]),
        dropped=tuple(raw["dropped"]),
    )
    found = _assignment_digest(assignment)
    if found != raw["digest"]:
        raise ValueError(
            f"{path} records digest {raw['digest']} but its ids hash to {found}; "
            "the file was edited by hand rather than redrawn"
        )
    return FrozenSplit(
        name=raw["name"],
        splitter=raw["splitter"],
        params=MappingProxyType(raw["params"]),
        source=MappingProxyType(raw["source"]),
        assignment=assignment,
        digest=raw["digest"],
    )


def rebuild(frozen: FrozenSplit, placement: Placement) -> Assignment:
    """Redraw a frozen split from its recorded recipe.

    What the freeze is checked against: equality with :attr:`FrozenSplit.assignment`
    says the recipe still produces the file, and inequality says the sample list
    moved underneath it.

    Args:
        frozen: The recorded split.
        placement: Where the samples sit now.

    Returns:
        The assignment the recipe produces today.
    """
    params = dict(frozen.params)
    val_fraction = float(params.pop("val_fraction"))
    seed = int(params.pop("seed"))
    return SPLIT_REGISTRY[frozen.splitter](**params).split(placement, val_fraction, seed)

"""Training data: tile sources, synthetic imagery, and crop sampling.

The segmentation model consumes ``(image, mask)`` crops. The mask comes from
:mod:`raster`, which renders the OSM ground-truth graph — that path is already
in place. The *image* is the part that does not exist yet, because there is no
imagery downloaded.

:class:`TileSource` is the seam. ``SyntheticTileSource`` fabricates an image
from the mask so the training loop, the loss and the evaluation path can all be
built and debugged now; a SpaceNet source drops in later behind the same
Protocol with no change to anything downstream.

**A synthetic image proves plumbing, never quality.** It is derived from the
answer, so a model will fit it easily and the resulting numbers mean nothing
about road extraction. Never report them.

This module is deliberately free of torch: the crop logic is array work, and
keeping it that way means it is testable without pulling a framework into the
test.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import networkx as nx
import numpy as np
from scipy import ndimage

from . import raster, tiles
from .osm import ground_truth_graph
from .tiles import Tile


@dataclass(frozen=True, slots=True)
class TileSample:
    """One tile's imagery, label mask, and the graph the mask came from.

    Attributes:
        tile: The tile these arrays cover.
        image: ``(H, W, C)`` float32 in ``[0, 1]``.
        mask: ``(H, W)`` bool, True on road.
        truth: Ground-truth road graph in tile pixel coordinates, carried
            along because evaluation scores against the graph rather than the
            mask.
    """

    tile: Tile
    image: np.ndarray
    mask: np.ndarray
    truth: nx.MultiGraph


@runtime_checkable
class TileSource(Protocol):
    """Supplies imagery and labels for a tile."""

    def load(self, tile: Tile) -> TileSample: ...


def synthesize_image(
    mask: np.ndarray,
    rng: np.random.Generator,
    noise: float = 0.08,
    blur: float = 1.2,
    n_distractors: int = 40,
) -> np.ndarray:
    """Fabricate a plausible overhead image from a road mask.

    Built to be non-trivially segmentable rather than realistic. A model handed
    the mask itself would learn the identity function and exercise nothing, so
    the roads are blurred and noised, the background is textured, and rectangular
    distractors are laid down at road-like brightness. That is enough to catch a
    channel-order slip or a broken normalization, and nothing more.

    Args:
        mask: ``(H, W)`` boolean road mask.
        rng: Source of randomness, for reproducible samples.
        noise: Standard deviation of per-pixel Gaussian noise.
        blur: Gaussian sigma applied to road edges, so they are not crisp.
        n_distractors: Number of road-coloured rectangles to scatter as
            confusers.

    Returns:
        ``(H, W, 3)`` float32 image in ``[0, 1]``.
    """
    height, width = mask.shape

    # Low-frequency ground texture: smooth noise, not flat.
    coarse = rng.normal(0.0, 1.0, (max(height // 32, 2), max(width // 32, 2)))
    stretched = np.asarray(
        ndimage.zoom(
            coarse, (height / coarse.shape[0], width / coarse.shape[1]), order=3
        ),
        dtype=np.float32,
    )
    background = 0.45 + 0.10 * stretched[:height, :width]

    surface = ndimage.gaussian_filter(mask.astype(np.float32), blur)

    distractors = np.zeros((height, width), dtype=np.float32)
    for _ in range(n_distractors):
        h, w = rng.integers(6, 26, size=2)
        r, c = rng.integers(0, height - h), rng.integers(0, width - w)
        distractors[r : r + h, c : c + w] = rng.uniform(0.3, 0.7)
    distractors = ndimage.gaussian_filter(distractors, blur)

    grey = background + 0.35 * surface + 0.18 * distractors
    # Slight per-channel tint so the model cannot rely on channels being equal.
    tint = np.array([1.0, 0.97, 0.92], dtype=np.float32)
    image = grey[..., None] * tint

    image = image + rng.normal(0.0, noise, image.shape)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


class SyntheticTileSource:
    """Real OSM labels, fabricated imagery.

    Lets the whole training and evaluation path be exercised before any
    imagery is downloaded. The labels are genuine; only the pixels are invented.
    """

    def __init__(
        self,
        seed: int = 0,
        half_width_px: int = 2,
        network_type: str = "drive",
    ) -> None:
        """
        Args:
            seed: Seed for image fabrication, so samples are reproducible.
            half_width_px: Road half-width passed to the rasterizer.
            network_type: osmnx network filter for the ground-truth query.
        """
        self.seed = seed
        self.half_width_px = half_width_px
        self.network_type = network_type

    def load(self, tile: Tile) -> TileSample:
        """Fetch ground truth for a tile and fabricate matching imagery."""
        truth = ground_truth_graph(tile, network_type=self.network_type)
        mask = raster.rasterize(truth, tile, half_width_px=self.half_width_px)
        # Seed off the tile so the same tile always yields the same image.
        rng = np.random.default_rng((self.seed, int(tile.x_min), int(tile.y_max)))
        return TileSample(
            tile=tile, image=synthesize_image(mask, rng), mask=mask, truth=truth
        )


_: type[TileSource] = SyntheticTileSource

#: Selects a tile source by config key. Values are factories, so each lookup
#: yields a fresh instance rather than a shared one.
TILE_SOURCE_REGISTRY: dict[str, Callable[[], TileSource]] = {
    "synthetic": lambda: SyntheticTileSource(),
}


@dataclass(frozen=True, slots=True)
class CropSpec:
    """A square window into a tile, in pixel coordinates.

    Attributes:
        row: Top edge.
        col: Left edge.
        size: Side length.
    """

    row: int
    col: int
    size: int


def crop_specs(
    mask: np.ndarray,
    size: int,
    count: int,
    rng: np.random.Generator,
    min_road_fraction: float = 0.01,
    max_attempts_per_crop: int = 32,
) -> tuple[CropSpec, ...]:
    """Choose crop windows, preferring ones that contain road.

    Roads cover a small share of a tile, so uniform sampling yields many crops
    with almost no positive pixels. Rejection sampling keeps the batch
    informative without hard-filtering to only dense crops, which would bias the
    model against sparse outskirts.

    The windows are returned rather than the pixels: they are small, hashable and
    reproducible, so a dataset can be described by its specs and materialized on
    demand.

    Args:
        mask: ``(H, W)`` boolean road mask used to judge road content.
        size: Crop side length in pixels.
        count: Number of crops to choose.
        rng: Source of randomness.
        min_road_fraction: Road share a crop should have. Falls back to
            accepting a crop once attempts run out, so sparse tiles still yield
            the requested count.
        max_attempts_per_crop: Rejection attempts before accepting whatever came
            up.

    Returns:
        ``count`` crop specs.

    Raises:
        ValueError: If the crop size does not fit inside the mask.
    """
    height, width = mask.shape
    if size > height or size > width:
        raise ValueError(f"crop size {size} exceeds mask shape {mask.shape}")

    def draw() -> tuple[int, int]:
        return (
            int(rng.integers(0, height - size + 1)),
            int(rng.integers(0, width - size + 1)),
        )

    def road_share(row: int, col: int) -> float:
        return float(mask[row : row + size, col : col + size].mean())

    specs = []
    for _ in range(count):
        row, col = draw()
        for _retry in range(max(max_attempts_per_crop - 1, 0)):
            if road_share(row, col) >= min_road_fraction:
                break
            row, col = draw()
        specs.append(CropSpec(row=row, col=col, size=size))
    return tuple(specs)


def take_crop(sample: TileSample, spec: CropSpec) -> tuple[np.ndarray, np.ndarray]:
    """Cut one crop out of a sample.

    Args:
        sample: Tile to cut from.
        spec: Window to cut.

    Returns:
        ``(image, mask)`` where image is ``(size, size, C)`` float32 and mask is
        ``(size, size)`` bool.
    """
    rows = slice(spec.row, spec.row + spec.size)
    cols = slice(spec.col, spec.col + spec.size)
    return sample.image[rows, cols], sample.mask[rows, cols]


def load_tile(
    lat: float,
    lon: float,
    size_m: float = 1024.0,
    resolution: float = 1.0,
    source: str = "synthetic",
) -> TileSample:
    """Load one tile through the registry.

    Args:
        lat: Tile centre latitude in degrees.
        lon: Tile centre longitude in degrees.
        size_m: Tile side length in metres.
        resolution: Metres per pixel.
        source: Registry key naming the tile source.

    Returns:
        The loaded sample.

    Raises:
        KeyError: If ``source`` is not registered.
    """
    if source not in TILE_SOURCE_REGISTRY:
        raise KeyError(
            f"unknown tile source {source!r}; have {sorted(TILE_SOURCE_REGISTRY)}"
        )
    tile = tiles.tile_from_center(lat, lon, size_m=size_m, resolution=resolution)
    return TILE_SOURCE_REGISTRY[source]().load(tile)

"""Training data: tile sources, synthetic imagery and lidar, crops, augmentation.

The segmentation model consumes ``(image, mask)`` crops. The mask comes from
:mod:`raster`, which renders the OSM ground-truth graph — that path is already
in place. The *image* is the part that does not exist yet, because there is no
imagery downloaded.

:class:`TileSource` is the seam. ``SyntheticTileSource`` fabricates an image
from the mask so the training loop, the loss and the evaluation path can all be
built and debugged now; a SpaceNet source drops in later behind the same
Protocol with no change to anything downstream.

The same seam carries lidar. ``TileSample.aux`` is an optional channel stack
with its own validity mask, ``synthesize_height`` fabricates one from the truth
graph, and ``CoverageSampler`` decides how much of it a given crop is allowed
to see — spatially, in bands and blocks, because that is the shape real
collection comes in.

**A synthetic image proves plumbing, never quality.** It is derived from the
answer, so a model will fit it easily and the resulting numbers mean nothing
about road extraction. Never report them. The same holds twice over for a
synthetic lidar channel, whose only job is to make the fusion path executable
and its relative ordering visible.

This module is deliberately free of torch: the crop logic is array work, and
keeping it that way means it is testable without pulling a framework into the
test.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import networkx as nx
import numpy as np
from scipy import ndimage
from shapely.geometry import LineString
from shapely.strtree import STRtree

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
        aux: ``(H, W, K)`` float32 auxiliary measurements in physical units,
            channel order :data:`AUX_CHANNEL_NAMES`. ``None`` means *this
            source carries no lidar at all*.
        aux_valid: ``(H, W)`` bool, True where ``aux`` was actually measured.
            ``None`` exactly when ``aux`` is.

    ``aux is None`` and an all-``False`` ``aux_valid`` are deliberately
    different states, and nothing may collapse them. ``None`` is a source-level
    fact — SpaceNet ships no lidar, so no coverage mask applies and no
    availability channel is meaningful. An all-``False`` ``aux_valid`` is a
    per-pixel measurement outcome: this source *does* carry lidar and none of
    it landed here. They differ in what a caller may do next (a coverage mask
    can only be drawn against a source that has lidar) and in what a run is
    reporting (an imagery-only baseline against the zero-coverage endpoint of a
    conditioned model). The same rule that forbids one boolean standing for
    both "not X" and "could not determine X" forbids reusing zeros here.
    """

    tile: Tile
    image: np.ndarray
    mask: np.ndarray
    truth: nx.MultiGraph
    aux: np.ndarray | None = None
    aux_valid: np.ndarray | None = None


@runtime_checkable
class TileSource(Protocol):
    """Supplies imagery and labels, enumerated by opaque sample id.

    Sources enumerate rather than accept a caller-chosen :class:`Tile`, because
    real imagery comes as fixed chips on the provider's own grid. Only a
    synthetic source is free to put a tile wherever it likes, so the interface
    is written for the constrained case.
    """

    def ids(self) -> tuple[str, ...]: ...

    def load(self, sample_id: str) -> TileSample: ...


def _smooth_field(
    shape: tuple[int, int], rng: np.random.Generator, scale: float
) -> np.ndarray:
    """Spatially correlated noise: coarse Gaussian draws stretched to full size."""
    height, width = shape
    coarse = rng.normal(
        0.0, 1.0, (max(int(height // scale), 2), max(int(width // scale), 2))
    )
    stretched = np.asarray(
        ndimage.zoom(
            coarse, (height / coarse.shape[0], width / coarse.shape[1]), order=3
        ),
        dtype=np.float32,
    )
    return stretched[:height, :width]


def _unit_field(
    shape: tuple[int, int], rng: np.random.Generator, scale: float
) -> np.ndarray:
    """:func:`_smooth_field` rescaled to zero mean and unit standard deviation.

    Cubic interpolation between coarse draws costs most of their variance, and
    by an amount that depends on ``scale``. Anything that wants a field of a
    *known* amplitude — heights in metres, reflectances in ``[0, 1]`` — has to
    standardize first or the knob it exposes will not mean what it says.
    """
    field = _smooth_field(shape, rng, scale)
    return (field - field.mean()) / (field.std() + 1e-6)


def canopy_patches(
    shape: tuple[int, int],
    rng: np.random.Generator,
    fraction: float = 0.12,
    scale: float = 20.0,
) -> np.ndarray:
    """Contiguous blobs of tree canopy covering a share of the tile.

    Thresholding a smooth field rather than scattering discs, so the patches
    have ragged outlines and a range of sizes, and so the covered share is
    exactly the requested fraction by construction.

    The canopy is the whole reason fusion has anything to gain here: it is
    generated once per tile and handed to *both* :func:`synthesize_image`,
    where it occludes the road, and :func:`synthesize_height`, where the ground
    return survives it.

    Args:
        shape: ``(H, W)`` grid to cover.
        rng: Source of randomness.
        fraction: Share of the tile under canopy. Zero returns bare ground.
        scale: Feature size of the blobs in pixels.

    Returns:
        ``(H, W)`` bool, True under canopy.
    """
    if fraction <= 0.0:
        return np.zeros(shape, dtype=bool)
    field = _unit_field(shape, rng, scale)
    return field >= np.quantile(field, 1.0 - fraction)


def synthesize_image(
    mask: np.ndarray,
    rng: np.random.Generator,
    noise: float = 0.08,
    blur: float = 1.2,
    n_distractors: int = 40,
    occlusion: np.ndarray | None = None,
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
        occlusion: ``(H, W)`` bool canopy cover from :func:`canopy_patches`.
            Where it is set the road surface is erased from the image and the
            ground darkened, which is the failure an overhead camera actually
            has and a lidar ground return does not. ``None`` leaves the image
            unoccluded.

    Returns:
        ``(H, W, 3)`` float32 image in ``[0, 1]``.
    """
    height, width = mask.shape

    # Low-frequency ground texture: smooth noise, not flat.
    background = 0.45 + 0.10 * _smooth_field((height, width), rng, 32.0)

    surface = ndimage.gaussian_filter(mask.astype(np.float32), blur)

    distractors = np.zeros((height, width), dtype=np.float32)
    for _ in range(n_distractors):
        h, w = rng.integers(6, 26, size=2)
        r, c = rng.integers(0, height - h), rng.integers(0, width - w)
        distractors[r : r + h, c : c + w] = rng.uniform(0.3, 0.7)
    distractors = ndimage.gaussian_filter(distractors, blur)

    # The road is removed under canopy rather than merely dimmed: a darkened
    # road keeps the contrast a segmentation model needs, so the occlusion
    # would cost it nothing and fusion would have nothing to recover.
    shade = (
        ndimage.gaussian_filter(occlusion.astype(np.float32), blur)
        if occlusion is not None
        else np.zeros((height, width), dtype=np.float32)
    )
    grey = background + 0.35 * surface * (1.0 - shade) + 0.18 * distractors - 0.12 * shade

    # Slight per-channel tint so the model cannot rely on channels being equal.
    tint = np.array([1.0, 0.97, 0.92], dtype=np.float32)
    image = grey[..., None] * tint

    image = image + rng.normal(0.0, noise, image.shape)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


#: Channel order of :attr:`TileSample.aux`, and the only place it is written
#: down. ``ndsm`` is height above ground in metres; ``intensity`` is the
#: ground-return reflectance in ``[0, 1]``. The plan's third data channel,
#: roughness, is not synthesized yet: it appends here and nothing else changes.
AUX_CHANNEL_NAMES: tuple[str, ...] = ("ndsm", "intensity")

#: Per-channel divisors bringing :attr:`TileSample.aux` onto the ``[0, 1]``
#: scale the imagery already uses. Kept beside :data:`AUX_CHANNEL_NAMES` so the
#: two cannot drift. The stack is stored in physical units because that is what
#: a real sensor delivers and what makes nDSM comparable across areas; the
#: rescale belongs at the tensor boundary, not in the measurement.
AUX_CHANNEL_SCALES: tuple[float, ...] = (10.0, 1.0)

#: Value written into the aux data channels wherever nothing was observed.
#:
#: Arbitrary, and required to stay unobservable: the validity channel is what
#: carries "no measurement here", and a model that could read this constant
#: would be reading the coverage mask through the data channels instead.
#: :func:`geo_graphs.model.mask_unobserved` is what enforces that.
AUX_FILL = 0.0


def _overpass_lift(
    shape: tuple[int, int],
    truth: nx.MultiGraph,
    height_m: float,
    radius_px: float,
    half_width_px: int,
) -> np.ndarray:
    """Height added where one road crosses another without meeting it.

    Two edges that intersect geometrically but share no node are a grade
    separation: an overpass, an underpass, or a road over a rail line. That is
    the one place a 2D mask is provably wrong and height is provably right, so
    the synthetic stack has to contain it even though the current graph
    representation cannot yet score it.

    Args:
        shape: ``(H, W)`` grid.
        truth: Ground-truth graph in pixel coordinates.
        height_m: Deck height above grade at the crossing.
        radius_px: Distance over which the ramp falls back to grade.
        half_width_px: Half-width of the lifted deck.

    Returns:
        ``(H, W)`` float32 height to add, zero away from crossings.
    """
    lift = np.zeros(shape, dtype=np.float32)
    edges = sorted(truth.edges(keys=True))
    geometry = [np.asarray(truth.edges[e]["pts"], dtype=float) for e in edges]
    keep = [i for i, pts in enumerate(geometry) if len(pts) >= 2]
    if len(keep) < 2:
        return lift

    lines = [LineString(geometry[i]) for i in keep]
    tree = STRtree(lines)
    rows, cols = np.indices(shape, dtype=np.float32)

    for a, line in enumerate(lines):
        nodes_a = set(edges[keep[a]][:2])
        centres = []
        for b in (int(j) for j in tree.query(line)):
            if b <= a or nodes_a & set(edges[keep[b]][:2]):
                continue
            meeting = line.intersection(lines[b])
            if not meeting.is_empty:
                # The centroid collapses a point, a cluster of points and a
                # short shared run to one place, which is all a ramp needs.
                centres.append(meeting.centroid)
        if not centres:
            continue

        taper = np.max(
            [
                np.exp(-0.5 * (np.hypot(cols - c.x, rows - c.y) / radius_px) ** 2)
                for c in centres
            ],
            axis=0,
        )
        deck = raster.draw_polylines(shape, [geometry[keep[a]]], half_width_px)
        lift = np.maximum(lift, height_m * deck * taper)
    return lift


def synthesize_height(
    mask: np.ndarray,
    truth: nx.MultiGraph,
    rng: np.random.Generator,
    canopy: np.ndarray | None = None,
    relief_m: float = 1.6,
    relief_scale: float = 24.0,
    road_relief: float = 0.3,
    material_scale: float = 18.0,
    material_spread: float = 0.25,
    asphalt: float = 0.22,
    asphalt_spread: float = 0.12,
    canopy_height_m: float = 9.0,
    canopy_noise: float = 0.12,
    overpass_height_m: float = 6.0,
    overpass_radius_px: float = 14.0,
    half_width_px: int = 2,
    noise: float = 0.03,
    blur: float = 1.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Fabricate a lidar height field and ground-return intensity from a graph.

    Two properties are load-bearing, and both are things real lidar has.

    *It carries information the imagery does not.* ``canopy`` occludes the road
    in :func:`synthesize_image`, but a multi-return sensor still gets a ground
    echo under the trees, so ``intensity`` keeps the dark asphalt signature
    right through a patch where the image has none. That gap is the entire
    thing a fusion model can win and an imagery-only model cannot.

    *It is not a noiseless copy of the label.* Off-road ground carries
    correlated relief at ``relief_m``, road surfaces keep a share of it, and
    ground reflectance varies enough that bare soil and gravel reach asphalt's
    darkness. If height were a clean function of the mask the model would read
    the answer off it, every coverage sweep would come out trivially monotone,
    and nothing measured here would transfer.

    Args:
        mask: ``(H, W)`` boolean road mask.
        truth: Graph the mask was rendered from, used to find grade
            separations — crossing edges that share no node.
        rng: Source of randomness, for reproducible samples.
        canopy: ``(H, W)`` bool cover from :func:`canopy_patches`, raised in
            ``ndsm`` and noisier in ``intensity``. ``None`` is bare ground.
        relief_m: Standard deviation, in metres, of off-road ground relief.
        relief_scale: Feature size of that relief in pixels.
        road_relief: Share of local relief a road surface keeps. Zero would
            make ``ndsm`` a clean function of the mask at the road edge.
        material_scale: Feature size of ground reflectance variation.
        material_spread: Standard deviation of off-road reflectance.
        asphalt: Mean ground-return reflectance of a road surface.
        asphalt_spread: Standard deviation of it, over wet, dry and concrete.
        canopy_height_m: Height of the canopy top above ground.
        canopy_noise: Extra intensity noise under canopy, where fewer pulses
            reach the ground.
        overpass_height_m: Deck height at a grade separation.
        overpass_radius_px: Distance over which an overpass returns to grade.
        half_width_px: Half-width of a lifted deck, matching the rasterizer.
        noise: Standard deviation of per-pixel noise on both channels.
        blur: Gaussian sigma softening the road edge, matching the imagery.

    Returns:
        ``(ndsm, intensity)``, both ``(H, W)`` float32. ``ndsm`` is metres
        above ground; ``intensity`` is reflectance in ``[0, 1]``.
    """
    shape = mask.shape
    road = ndimage.gaussian_filter(mask.astype(np.float32), blur)

    # Ground relief, kept non-negative because a normalized surface model sits
    # on the terrain rather than through it. Roads are graded flat but not
    # perfectly: they keep road_relief of whatever the ground around them does.
    ground = relief_m * np.clip(1.0 + _unit_field(shape, rng, relief_scale), 0.0, None)
    ndsm = ground * (1.0 - (1.0 - road_relief) * road)

    material = 0.5 + material_spread * _unit_field(shape, rng, material_scale)
    surface = asphalt + asphalt_spread * _unit_field(shape, rng, material_scale / 2)
    intensity = road * surface + (1.0 - road) * material

    if canopy is not None:
        shade = ndimage.gaussian_filter(canopy.astype(np.float32), blur)
        ndsm = np.maximum(ndsm, canopy_height_m * shade)
        intensity = intensity + canopy_noise * shade * rng.normal(0.0, 1.0, shape)

    ndsm = ndsm + _overpass_lift(
        shape, truth, overpass_height_m, overpass_radius_px, half_width_px
    )

    ndsm = np.clip(ndsm + rng.normal(0.0, noise, shape), 0.0, None)
    intensity = np.clip(intensity + rng.normal(0.0, noise, shape), 0.0, 1.0)
    return ndsm.astype(np.float32), intensity.astype(np.float32)


class SyntheticTileSource:
    """Real OSM labels, fabricated imagery.

    Lets the whole training and evaluation path be exercised before any
    imagery is downloaded. The labels are genuine; only the pixels are invented.
    """

    #: Tile centres used when none are given: downtown Las Vegas and three
    #: neighbours, inside SpaceNet AOI 2 so the ground matches the real data.
    DEFAULT_CENTERS = (
        (36.1699, -115.1398),
        (36.1560, -115.1560),
        (36.1820, -115.1250),
        (36.1440, -115.1700),
    )

    def __init__(
        self,
        centers: Sequence[tuple[float, float]] | None = None,
        size_m: float = 1024.0,
        resolution: float = 1.0,
        seed: int = 0,
        half_width_px: int = 2,
        network_type: str = "drive",
        lidar: bool = False,
        canopy_fraction: float = 0.12,
    ) -> None:
        """
        Args:
            centers: ``(lat, lon)`` tile centres to offer. Defaults to
                :data:`DEFAULT_CENTERS`.
            size_m: Tile side length in metres.
            resolution: Metres per pixel.
            seed: Seed for image fabrication, so samples are reproducible.
            half_width_px: Road half-width passed to the rasterizer.
            network_type: osmnx network filter for the ground-truth query.
            lidar: Populate ``aux`` and ``aux_valid`` from
                :func:`synthesize_height`. Off by default, which is the
                imagery-only source this project started with.
            canopy_fraction: Share of each tile under tree canopy. Applied to
                the *imagery* whether or not ``lidar`` is set, deliberately:
                if turning lidar on also made the pictures easier, a fused run
                and an imagery-only run would no longer be comparable and the
                measured gain would be partly the change in the images.
        """
        self.centers = tuple(centers if centers is not None else self.DEFAULT_CENTERS)
        self.size_m = size_m
        self.resolution = resolution
        self.seed = seed
        self.half_width_px = half_width_px
        self.network_type = network_type
        self.lidar = lidar
        self.canopy_fraction = canopy_fraction

    def ids(self) -> tuple[str, ...]:
        """Return one id per configured tile centre."""
        return tuple(f"tile_{i}" for i in range(len(self.centers)))

    def load(self, sample_id: str) -> TileSample:
        """Fetch ground truth for one tile and fabricate matching imagery.

        Args:
            sample_id: An id from :meth:`ids`.

        Returns:
            The loaded sample.

        Raises:
            KeyError: If the id is not one this source offers.
        """
        if sample_id not in self.ids():
            raise KeyError(f"unknown sample {sample_id!r}; have {self.ids()}")
        lat, lon = self.centers[int(sample_id.removeprefix("tile_"))]
        tile = tiles.tile_from_center(
            lat, lon, size_m=self.size_m, resolution=self.resolution
        )

        truth = ground_truth_graph(tile, network_type=self.network_type)
        mask = raster.rasterize(truth, tile, half_width_px=self.half_width_px)
        # Seed off the tile so the same tile always yields the same sample, and
        # split into independent streams so that adding the lidar channels does
        # not shift the imagery draws out from under an earlier run.
        canopy_rng, image_rng, lidar_rng = np.random.default_rng(
            (self.seed, int(tile.x_min), int(tile.y_max))
        ).spawn(3)

        canopy = canopy_patches(mask.shape, canopy_rng, fraction=self.canopy_fraction)
        image = synthesize_image(mask, image_rng, occlusion=canopy)
        if not self.lidar:
            return TileSample(tile=tile, image=image, mask=mask, truth=truth)

        ndsm, intensity = synthesize_height(
            mask,
            truth,
            lidar_rng,
            canopy=canopy,
            half_width_px=self.half_width_px,
        )
        return TileSample(
            tile=tile,
            image=image,
            mask=mask,
            truth=truth,
            aux=np.stack([ndsm, intensity], axis=-1),
            # This source flies the whole tile: every pixel is measured, and
            # what a run actually sees is decided by the coverage mask drawn
            # per crop rather than by holes in the collection.
            aux_valid=np.ones(mask.shape, dtype=bool),
        )


_: type[TileSource] = SyntheticTileSource

#: Selects a tile source by config key. Values are factories, so each lookup
#: yields a fresh instance rather than a shared one.
TILE_SOURCE_REGISTRY: dict[str, Callable[..., TileSource]] = {
    "synthetic": lambda **kw: SyntheticTileSource(**kw),
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


@dataclass(frozen=True, slots=True)
class Crop:
    """Every array belonging to one training example, kept registered.

    A crop is now five arrays of three different ranks, and the failure they
    invite is silent: transform four of them and forget the fifth and the model
    trains against a label rotated away from its imagery, which reads as a bad
    model rather than a bad pipeline. Bundling them means the transform takes
    one argument and cannot be applied unevenly.

    Attributes:
        image: ``(H, W, C)`` float32 imagery.
        mask: ``(H, W)`` bool label.
        aux: ``(H, W, K)`` float32 lidar in physical units, or ``None`` when
            the source has none. See :class:`TileSample`.
        aux_valid: ``(H, W)`` bool, True where ``aux`` was measured. Present
            exactly when ``aux`` is.
        coverage: ``(H, W)`` bool, True where this run *collected* lidar.
            Distinct from ``aux_valid``, which is what the sensor managed to
            measure; the two are intersected only at :func:`aux_stack`, so a
            simulated flight line never gets confused with a sensor dropout.

    Raises:
        ValueError: If the arrays disagree on their spatial shape, if ``aux``
            and ``aux_valid`` are not supplied together, or if a coverage mask
            is attached to a crop with no lidar to mask.
    """

    image: np.ndarray
    mask: np.ndarray
    aux: np.ndarray | None = None
    aux_valid: np.ndarray | None = None
    coverage: np.ndarray | None = None

    def __post_init__(self) -> None:
        shape = self.image.shape[:2]
        mismatched = [
            name
            for name, array in (
                ("mask", self.mask),
                ("aux", self.aux),
                ("aux_valid", self.aux_valid),
                ("coverage", self.coverage),
            )
            if array is not None and array.shape[:2] != shape
        ]
        if mismatched:
            raise ValueError(
                f"{', '.join(mismatched)} and image {shape} disagree on shape"
            )
        if (self.aux is None) != (self.aux_valid is None):
            raise ValueError("aux and aux_valid must be supplied together")
        if self.coverage is not None and self.aux is None:
            raise ValueError("a coverage mask on a crop with no aux masks nothing")


def take_crop(sample: TileSample, spec: CropSpec) -> Crop:
    """Cut one crop out of a sample, lidar channels included.

    Args:
        sample: Tile to cut from.
        spec: Window to cut.

    Returns:
        The crop, carrying whichever channels the sample had.
    """
    rows = slice(spec.row, spec.row + spec.size)
    cols = slice(spec.col, spec.col + spec.size)
    return Crop(
        image=sample.image[rows, cols],
        mask=sample.mask[rows, cols],
        aux=None if sample.aux is None else sample.aux[rows, cols],
        aux_valid=(None if sample.aux_valid is None else sample.aux_valid[rows, cols]),
    )


def whole_tile(sample: TileSample) -> Crop:
    """The whole sample as a single crop, for inference over a full tile."""
    return Crop(
        image=sample.image,
        mask=sample.mask,
        aux=sample.aux,
        aux_valid=sample.aux_valid,
    )


#: Spatial arrangements a coverage mask can take.
CoverageMode = Literal["full", "none", "strips", "blocks"]


@dataclass(frozen=True, slots=True)
class CoverageSampler:
    """How much lidar a crop is given, and in what shape.

    The deployment condition is contiguous corridors shaped like flight lines,
    so the patterns here are spatial. Bernoulli-per-sample dropout would train
    for all-or-nothing and per-pixel dropout would train an inpainter; a model
    fit to either looks right at both endpoints of the coverage sweep and sags
    through the middle, which is the only part worth measuring.

    Attributes:
        mode: Which pattern to draw. ``full`` and ``none`` are the endpoints
            and ignore the fraction curriculum entirely.
        endpoint_mass: Total probability placed on fractions exactly 0 and
            exactly 1, split evenly between them. A uniform draw puts no mass
            at either, and those are the two numbers being reported.
        min_fraction: Lower bound of the interior uniform draw.
        max_fraction: Upper bound of the interior uniform draw.
        strip_period_px: Inclusive range the band period is drawn from.
        block_px: Inclusive range the block side is drawn from.

    Raises:
        ValueError: If the mode is unknown, the endpoint mass is not a
            probability, or the fraction bounds are inverted or outside
            ``[0, 1]``.
    """

    mode: CoverageMode = "full"
    endpoint_mass: float = 0.4
    min_fraction: float = 0.05
    max_fraction: float = 0.95
    strip_period_px: tuple[int, int] = (16, 48)
    block_px: tuple[int, int] = (8, 32)

    def __post_init__(self) -> None:
        if self.mode not in COVERAGE_PATTERNS:
            raise ValueError(
                f"unknown coverage mode {self.mode!r}; have {sorted(COVERAGE_PATTERNS)}"
            )
        if not 0.0 <= self.endpoint_mass <= 1.0:
            raise ValueError(
                f"endpoint_mass must be a probability; got {self.endpoint_mass}"
            )
        if not 0.0 <= self.min_fraction <= self.max_fraction <= 1.0:
            raise ValueError(
                f"need 0 <= min_fraction <= max_fraction <= 1; "
                f"got [{self.min_fraction}, {self.max_fraction}]"
            )


def _coverage_full(
    config: CoverageSampler,
    shape: tuple[int, int],
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Everything collected: the upper endpoint of the sweep."""
    return np.ones(shape, dtype=bool)


def _coverage_none(
    config: CoverageSampler,
    shape: tuple[int, int],
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Nothing collected: the lower endpoint, and the imagery-only deployment."""
    return np.zeros(shape, dtype=bool)


def _coverage_strips(
    config: CoverageSampler,
    shape: tuple[int, int],
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Parallel bands at a sampled orientation, phase and period: flight lines."""
    height, width = shape
    angle = float(rng.uniform(0.0, np.pi))
    period = float(rng.integers(config.strip_period_px[0], config.strip_period_px[1] + 1))
    phase = float(rng.uniform(0.0, period))

    rows = np.arange(height, dtype=np.float32)[:, None]
    cols = np.arange(width, dtype=np.float32)[None, :]
    along = cols * np.cos(angle) + rows * np.sin(angle)
    return ((along + phase) % period) < fraction * period


def _coverage_blocks(
    config: CoverageSampler,
    shape: tuple[int, int],
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Axis-aligned tiles, for collection flown as a grid rather than swathed."""
    height, width = shape
    side = int(rng.integers(config.block_px[0], config.block_px[1] + 1))
    n_rows = max(round(height / side), 1)
    n_cols = max(round(width / side), 1)

    chosen = np.zeros(n_rows * n_cols, dtype=bool)
    chosen[rng.permutation(n_rows * n_cols)[: round(fraction * n_rows * n_cols)]] = True

    # Index rather than repeat, so blocks absorb the remainder a pixel at a
    # time instead of leaving a narrow strip at two edges that biases the
    # realized fraction away from the target.
    row_of = np.arange(height) * n_rows // height
    col_of = np.arange(width) * n_cols // width
    return chosen.reshape(n_rows, n_cols)[row_of[:, None], col_of[None, :]]


#: Coverage mode to the pattern that draws it. Pure functions of
#: ``(config, shape, fraction, rng)`` rather than factories, because a pattern
#: has no state to keep between draws.
COVERAGE_PATTERNS: dict[
    str,
    Callable[[CoverageSampler, tuple[int, int], float, np.random.Generator], np.ndarray],
] = {
    "full": _coverage_full,
    "none": _coverage_none,
    "strips": _coverage_strips,
    "blocks": _coverage_blocks,
}

#: The upper endpoint, as the default wherever coverage is not being swept.
FULL_COVERAGE = CoverageSampler(mode="full")


def coverage_fraction(config: CoverageSampler, rng: np.random.Generator) -> float:
    """Draw one target coverage fraction from the curriculum.

    ``full`` and ``none`` are fixed at 1 and 0. Every other mode draws the two
    endpoints with ``endpoint_mass`` between them and is uniform in between, so
    the degenerate cases stay common enough to report.

    Args:
        config: Coverage settings.
        rng: Source of randomness.

    Returns:
        A fraction in ``[0, 1]``.
    """
    if config.mode == "full":
        return 1.0
    if config.mode == "none":
        return 0.0
    draw = float(rng.random())
    if draw < config.endpoint_mass / 2.0:
        return 0.0
    if draw < config.endpoint_mass:
        return 1.0
    return float(rng.uniform(config.min_fraction, config.max_fraction))


def sample_coverage(
    config: CoverageSampler,
    shape: tuple[int, int],
    rng: np.random.Generator,
    fraction: float | None = None,
) -> np.ndarray:
    """Draw one coverage mask.

    Args:
        config: Coverage settings.
        shape: ``(H, W)`` grid to cover.
        rng: Source of randomness.
        fraction: Target share of the grid to cover. ``None`` draws it from
            the curriculum; an explicit value is what a held-out sweep passes
            when it wants a named point rather than a sample.

    Returns:
        ``(H, W)`` bool, True where lidar was collected.
    """
    drawn = coverage_fraction(config, rng) if fraction is None else fraction
    return COVERAGE_PATTERNS[config.mode](config, shape, drawn, rng)


def aux_stack(
    crop: Crop,
    scales: Sequence[float] = AUX_CHANNEL_SCALES,
    fill: float = AUX_FILL,
) -> np.ndarray:
    """Flatten a crop's lidar into the ``(H, W, K + 1)`` stack a model consumes.

    The trailing channel is the availability indicator, and it is not optional.
    Without it a zero in ``ndsm`` means both "measured, at grade" and "never
    measured", so zero-filling would teach the model that absent lidar is
    positive evidence of flat ground — worse than supplying no lidar at all.

    Args:
        crop: The crop to flatten. Must carry lidar; a source with none is a
            different case, handled by the caller rather than by an empty
            stack that a network would still have to be built around.
        scales: Per-channel divisors, one per ``aux`` channel.
        fill: Written into the data channels wherever nothing was observed.

    Returns:
        ``(H, W, K + 1)`` float32.

    Raises:
        ValueError: If the crop has no lidar, or ``scales`` does not have one
            entry per aux channel.
    """
    if crop.aux is None or crop.aux_valid is None:
        raise ValueError("crop carries no lidar to stack")
    if len(scales) != crop.aux.shape[-1]:
        raise ValueError(f"{len(scales)} scales for {crop.aux.shape[-1]} aux channels")

    observed = (
        crop.aux_valid if crop.coverage is None else (crop.aux_valid & crop.coverage)
    )
    scaled = crop.aux / np.asarray(scales, dtype=np.float32)
    values = np.where(observed[..., None], scaled, fill)
    return np.concatenate([values, observed[..., None]], axis=-1).astype(np.float32)


#: Size of the dihedral group of the square: four quarter turns, each optionally
#: composed with a flip.
#:
#: Overhead imagery has no canonical orientation, so all eight are *exact*
#: label-preserving symmetries of the problem rather than approximations of one,
#: the way they are for natural photographs. That makes the group an eight-fold
#: effective dataset for no extra bytes on disk.
DIHEDRAL_ORDER = 8


def dihedral(array: np.ndarray, index: int) -> np.ndarray:
    """Apply one of the eight dihedral transforms to a crop array.

    Acts on the leading two axes only, so an ``(H, W, C)`` image and its
    ``(H, W)`` mask are transformed identically and the channel axis is left
    alone.

    Both operations permute pixels rather than resample them, so the result is
    exact on the grid: a mask keeps its road pixel count to the pixel, and an
    image picks up no interpolation blur. That is the whole reason to stop at
    the dihedral group instead of allowing arbitrary angles.

    Args:
        array: ``(H, W)`` or ``(H, W, C)``, any dtype.
        index: Which group element, in ``[0, DIHEDRAL_ORDER)``. The low bit
            selects a left-right flip; the rest is the number of quarter turns.

    Returns:
        A new contiguous array of the same dtype. Square inputs keep their
        shape; the odd quarter turns swap ``H`` and ``W`` on non-square ones.

    Raises:
        ValueError: If ``index`` falls outside the group, or the array has
            fewer than two axes.
    """
    if not 0 <= index < DIHEDRAL_ORDER:
        raise ValueError(f"dihedral index {index} outside [0, {DIHEDRAL_ORDER})")
    if array.ndim < 2:
        raise ValueError(f"expected at least two axes, got shape {array.shape}")

    quarter_turns, flipped = divmod(index, 2)
    turned = np.rot90(array, k=quarter_turns, axes=(0, 1))
    # Both axes are named explicitly: np.rot90 would otherwise be free to pick
    # its own pair, and np.flip defaults to reversing *every* axis, which on an
    # (H, W, C) image would reverse the channels too.
    flopped = np.flip(turned, axis=1) if flipped else turned
    # rot90 and flip return views; copy so the caller cannot write through to
    # the tile the crop came from, and so downstream tensor conversion has the
    # contiguous buffer it wants.
    return np.ascontiguousarray(flopped)


def augment_crop(crop: Crop, index: int) -> Crop:
    """Apply one dihedral transform to every array in a crop together.

    Exists so the registration cannot drift. The arrays differ in rank and in
    optionality, and transforming them at separate call sites is exactly how an
    image ends up rotated away from the label — or a coverage mask away from
    the geometry it is supposed to be hiding.

    Args:
        crop: The crop to transform. :class:`Crop` has already checked that its
            arrays agree on shape.
        index: Which group element; see :func:`dihedral`.

    Returns:
        A new crop with every present array under the same transform.
    """

    def moved(array: np.ndarray | None) -> np.ndarray | None:
        return None if array is None else dihedral(array, index)

    return Crop(
        image=dihedral(crop.image, index),
        mask=dihedral(crop.mask, index),
        aux=moved(crop.aux),
        aux_valid=moved(crop.aux_valid),
        coverage=moved(crop.coverage),
    )


def load_sample(sample_id: str, source: str = "synthetic", **kwargs) -> TileSample:
    """Load one sample through the registry.

    Args:
        sample_id: Identifier the chosen source recognizes.
        source: Registry key naming the tile source.
        **kwargs: Passed to the source's constructor.

    Returns:
        The loaded sample.

    Raises:
        KeyError: If ``source`` is not registered.
    """
    if source not in TILE_SOURCE_REGISTRY:
        raise KeyError(
            f"unknown tile source {source!r}; have {sorted(TILE_SOURCE_REGISTRY)}"
        )
    return TILE_SOURCE_REGISTRY[source](**kwargs).load(sample_id)

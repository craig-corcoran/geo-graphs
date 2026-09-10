"""Cross-check SpaceNet Vegas labels against OSM, to explain low stub purity.

The endpoint census found that 47% of candidate endpoints have a stub whose
polyline never touches the SpaceNet label mask at all. Two explanations fit and
the census could not separate them: the model invents dead ends, or SpaceNet
Vegas does not label every paved way and those stubs are real roads absent from
the truth. Which one is right decides whether every ``prop_to_gt`` number in
this project has been penalising the model for finding real roads.

OSM carries service roads, alleys, tracks and driveways that SpaceNet
frequently omits, and ``osm.ground_truth_graph`` already returns them in the
tile's own pixel frame, so the two are directly comparable with no new
plumbing. This script rebuilds stub purity against OSM instead of SpaceNet and
reports how much of the zero-purity mass moves.

Three OSM networks are carried rather than the two osmnx names, because neither
name is the right comparison on its own. ``drive`` excludes ``service``, which
is most of what SpaceNet labels beyond the arterials. ``all`` includes footways
and steps, and a stub lying on a sidewalk is not a road the model correctly
found. ``osm_drivable`` is ``all`` minus the pedestrian classes, and it is the
one to read when the question is whether a stub is a real road.

Hits the network. Every chip costs two Overpass queries, one for the ``drive``
network and one for ``all``; osmnx caches responses under ``cache/``, so a
rerun costs the reprojection and clipping but no fetch.

``--overpass-url`` exists because ``overpass-api.de`` is a round-robin over
several backends and osmnx pins one of them for the whole process:
``_http._config_dns`` mutates ``socket.getaddrinfo`` to the single address
``socket.gethostbyname`` returned, deliberately, so that its slot accounting
and its query reach the same machine. The cost is that a backend which is down
takes every request down with it, since the fallback across addresses that
``create_connection`` would normally do never happens. Naming a backend
directly routes around a dead one. The committed default is osmnx's own.

``--no-overpass-rate-limit`` goes with it. osmnx paces itself by parsing the
endpoint's ``/status``, and a mirror that publishes no such endpoint sends it
to a flat 60 s pause before every single request. Turning the pacing off is
correct only against a mirror whose status cannot be read, and only while the
run stays sequential; against the main API, leave it on, because the pacing is
what keeps the run inside the server's slot allowance.

**Step 0 gates everything.** Before any comparison the two sources have to
register in the same pixel frame, or every number below is noise. The gate runs
first and alone, before the checkpoint is even loaded, so a failure costs no
inference and emits no length or purity numbers at all.

The direction it reads is ``osm_drive_on_spacenet``: the share of OSM ``drive``
length lying within a few pixels of a SpaceNet way. The reverse direction is
reported too and is *not* the gate, because it cannot separate the two failures
step 0 has to tell apart. SpaceNet Vegas labels ways that osmnx's ``drive``
filter excludes -- ``service`` above all -- so the share of *SpaceNet* length
near a ``drive`` way is bounded above by roughly the ratio of the two networks'
lengths, and is low on residential chips however well the frames register.
``drive`` is the arterial subset SpaceNet does label, so the other direction is
high when the frames agree and collapses when they do not. A null control
recomputes it with the OSM samples translated diagonally, which distinguishes
real registration from dense road cover.

**(a) is biased upward and the bias is not corrected.** SpaceNet Vegas imagery
is from roughly 2015-2017; OSM is read today, and Vegas grew. A road built
after the capture appears in OSM and could not have been found by the model, so
the length ratios overstate how much SpaceNet omitted. A historical OSM query
would price that and is deliberately out of scope here.

**(b) is much less exposed to it.** A stub the *model* produced has to
correspond to something visible in the 2015-2017 imagery, so a stub landing on
an OSM way is evidence about a way that was already there, not about one built
since.

Retrains nothing. The checkpoint is the frozen prior stage, so the whole
measurement costs one inference pass per chip plus graph work.

Candidate generation, the labeller and stub purity are **imported from**
``scripts/endpoint_census.py`` rather than restated, so the endpoints measured
here are by construction the ones the census counted.
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np
import osmnx as ox
from endpoint_census import (
    Label,
    endpoint_edge_candidates,
    endpoint_endpoint_candidates,
    endpoint_purities,
    endpoints,
    interior_flags,
    label_candidate,
    proposal_distance,
)
from loguru import logger
from osmnx._errors import InsufficientResponseError
from scipy import ndimage

from geo_graphs import cleanup, data, geograph, osm, raster, skeleton, train
from geo_graphs.model import predict_mask
from geo_graphs.tiles import Tile

#: Longest wait between fetch retries, in seconds.
#:
#: The doubling is capped because the usual failure is one unhealthy Overpass
#: address rather than rate limiting, and waiting longer does not make the next
#: address any healthier.
MAX_FETCH_BACKOFF = 60.0

#: Below this many seconds, a fetch is taken to have been served from the cache.
#:
#: Pacing exists to hold the request rate to Overpass down, and a cache hit
#: makes no request, so sleeping after one buys nothing. osmnx reads its cache
#: from local disk in milliseconds while a real query takes seconds, so the gap
#: is wide enough that the exact threshold does not matter.
CACHE_HIT_SECONDS = 0.25

#: Tolerances, in pixels (1 px = 1 m), the step 0 agreement is measured at.
DEFAULT_TOLERANCES = (4.0, 8.0, 16.0)

#: Extra dilations of every truth mask that stub purity is recomputed at.
#:
#: ``raster.rasterize`` already draws roads about 5 px wide, and the census used
#: 6 further iterations to reach the 17 px band that left 41% of endpoints at
#: zero purity. Both values run by default so the headline can be read as
#: mask-width-independent or not.
DEFAULT_PURITY_DILATES = (0.0, 6.0)

#: ``highway`` values that are not a drivable surface.
#:
#: A stub landing on a footway is not a road the model correctly found, so
#: pooling those with service roads would answer the wrong question. Everything
#: not listed here -- ``service``, ``track``, ``living_street`` and the whole
#: ``drive`` network -- is a surface a vehicle uses.
PEDESTRIAN_TAGS = frozenset(
    {
        "footway",
        "path",
        "steps",
        "cycleway",
        "pedestrian",
        "bridleway",
        "corridor",
        "elevator",
        "platform",
        "proposed",
        "construction",
    }
)

#: Which road network a purity or length number was measured against.
#:
#: ``osm_drivable`` is ``all`` minus :data:`PEDESTRIAN_TAGS`, and it is the one
#: to read when the question is whether a stub is a real road: ``osm_drive``
#: excludes the service roads SpaceNet does label, and ``osm_all`` includes
#: sidewalks it never would.
Source = Literal["spacenet", "osm_drive", "osm_drivable", "osm_all"]

SOURCES: tuple[Source, ...] = ("spacenet", "osm_drive", "osm_drivable", "osm_all")

#: Which set of purity values a distribution covers.
#:
#: ``endpoint`` is one value per interior endpoint holding a candidate, which is
#: the population the census's 47% is quoted over. ``candidate`` and
#: ``positive_candidate`` are one value per proposed reconnection, scored as the
#: weaker of its two stubs, matching ``endpoint_census.candidate_purity``.
Population = Literal["endpoint", "candidate", "positive_candidate"]

POPULATIONS: tuple[Population, ...] = ("endpoint", "candidate", "positive_candidate")


@dataclass(frozen=True, slots=True)
class Agreement:
    """How far one network's length lies from the other, at one tolerance.

    Four numbers rather than two, because coverage and registration both move a
    one-directional fraction and only one of them is what step 0 is asking
    about. A frame offset drives *every* direction to zero; a coverage
    difference drives only the direction whose denominator is the bigger
    network.

    Attributes:
        tolerance: Distance in pixels within which a sample counts as agreeing.
        spacenet_on_osm_drive: Share of SpaceNet truth length within
            ``tolerance`` of an OSM ``drive`` way. Bounded above by roughly
            ``osm_drive_length / spacenet_length``, so on a chip where SpaceNet
            labels ways osmnx's ``drive`` filter excludes -- ``service`` above
            all -- this is low no matter how well the frames register.
        osm_drive_on_spacenet: Share of OSM ``drive`` length within
            ``tolerance`` of a SpaceNet truth way. The registration measurement:
            ``drive`` is the arterial subset SpaceNet does label, so this is
            high when the frames agree and collapses when they do not.
        spacenet_on_osm_all: Share of SpaceNet truth length within ``tolerance``
            of any OSM way. The coverage-matched forward direction: what
            SpaceNet labels that OSM does not map at all.
        osm_drive_on_spacenet_shifted: ``osm_drive_on_spacenet`` recomputed with
            the OSM samples translated diagonally by a fixed offset. The null
            control. If a high ``osm_drive_on_spacenet`` were an artifact of
            dense road cover rather than real registration, this would stay
            high too.
    """

    tolerance: float
    spacenet_on_osm_drive: float
    osm_drive_on_spacenet: float
    spacenet_on_osm_all: float
    osm_drive_on_spacenet_shifted: float


@dataclass(frozen=True, slots=True)
class ChipAlignment:
    """Step 0 for one chip.

    Attributes:
        sample_id: Which chip this is.
        spacenet_length_m: Total SpaceNet truth road length, in metres.
        osm_drive_length_m: Total OSM ``drive`` length over the same tile.
        n_spacenet_samples: Samples taken along the SpaceNet centrelines.
        n_osm_samples: Samples taken along the OSM ``drive`` centrelines.
        agreements: One entry per tolerance.
    """

    sample_id: str
    spacenet_length_m: float
    osm_drive_length_m: float
    n_spacenet_samples: int
    n_osm_samples: int
    agreements: tuple[Agreement, ...]


@dataclass(frozen=True, slots=True)
class ChipLengths:
    """Road length from all three sources over one chip.

    Attributes:
        sample_id: Which chip this is.
        spacenet_m: SpaceNet truth length, in metres.
        osm_drive_m: OSM ``drive`` length.
        osm_drivable_m: OSM ``all`` length minus the pedestrian classes.
        osm_all_m: OSM ``all`` length, which adds service, track, path and
            footway classes to the driving network.
        drive_ratio: ``osm_drive_m / spacenet_m``, or ``None`` when the chip
            carries no labelled road at all.
        drivable_ratio: ``osm_drivable_m / spacenet_m``, or ``None`` likewise.
            The ratio to read for under-labelling: sidewalks do not inflate it.
        all_ratio: ``osm_all_m / spacenet_m``, or ``None`` likewise.
        drive_tag_lengths: OSM ``drive`` length by ``highway`` value.
        all_tag_lengths: OSM ``all`` length by ``highway`` value.
        n_multi_valued_ways: Clipped ``all`` ways whose ``highway`` tag held
            several values, so the tag they were counted under is the first of
            them rather than the only one.
    """

    sample_id: str
    spacenet_m: float
    osm_drive_m: float
    osm_drivable_m: float
    osm_all_m: float
    drive_ratio: float | None
    drivable_ratio: float | None
    all_ratio: float | None
    drive_tag_lengths: dict[str, float]
    all_tag_lengths: dict[str, float]
    n_multi_valued_ways: int


@dataclass(frozen=True, slots=True)
class EndpointPurity:
    """One candidate endpoint's stub, read against all three networks.

    Attributes:
        sample_id: Which chip this endpoint is on.
        node: Node id in the pre-prune proposal graph.
        dilate: Extra dilation applied to every mask before reading.
        purity: Share of the stub polyline on road, keyed by source.
    """

    sample_id: str
    node: int
    dilate: float
    purity: dict[Source, float]


@dataclass(frozen=True, slots=True)
class CandidatePurity:
    """One proposed reconnection, read against all three networks.

    Attributes:
        sample_id: Which chip this candidate is on.
        dilate: Extra dilation applied to every mask before reading.
        label: What the SpaceNet truth graph says at ``label_snap``.
        purity: Weaker of the candidate's two stubs, keyed by source.
    """

    sample_id: str
    dilate: float
    label: Label
    purity: dict[Source, float]


@dataclass(frozen=True, slots=True)
class AgreementSummary:
    """Step 0 pooled over the chips, at one tolerance.

    Every direction of :class:`Agreement` is carried through as a mean, a
    median and a first quartile, since a handful of misregistered chips would
    show in the quartile long before the mean.

    Attributes:
        tolerance: Distance in pixels within which a sample counts as agreeing.
        n_chips: Chips contributing.
        direction: Which of :class:`Agreement`'s fields this row describes.
        mean: Mean over chips.
        median: Median over chips.
        q1: First quartile over chips.
        gate_fraction: Value a chip must reach to count as passing.
        n_below_gate: Chips under ``gate_fraction`` on this direction.
        chips_below_gate: Which ones, so a failure names them.
    """

    tolerance: float
    n_chips: int
    direction: str
    mean: float
    median: float
    q1: float
    gate_fraction: float
    n_below_gate: int
    chips_below_gate: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LengthSummary:
    """Road length from all three sources, pooled over the chips.

    Attributes:
        n_chips: Chips contributing.
        spacenet_total_m: SpaceNet truth length summed over the chips.
        osm_drive_total_m: OSM ``drive`` length summed the same way.
        osm_drivable_total_m: OSM drivable length summed the same way.
        osm_all_total_m: OSM ``all`` length summed the same way.
        drive_ratio_mean: Mean over chips of ``osm_drive_m / spacenet_m``.
        drive_ratio_median: Median of the same.
        drivable_ratio_mean: Mean over chips of ``osm_drivable_m / spacenet_m``.
        drivable_ratio_median: Median of the same.
        all_ratio_mean: Mean over chips of ``osm_all_m / spacenet_m``.
        all_ratio_median: Median of the same.
        n_chips_rated: Chips with a non-zero SpaceNet length, which is what the
            per-chip ratios could be computed on.
    """

    n_chips: int
    spacenet_total_m: float
    osm_drive_total_m: float
    osm_drivable_total_m: float
    osm_all_total_m: float
    drive_ratio_mean: float | None
    drive_ratio_median: float | None
    drivable_ratio_mean: float | None
    drivable_ratio_median: float | None
    all_ratio_mean: float | None
    all_ratio_median: float | None
    n_chips_rated: int


@dataclass(frozen=True, slots=True)
class TagSummary:
    """One ``highway`` class's contribution to the OSM surplus over ``drive``.

    Attributes:
        highway: The tag value.
        all_length_m: Length carrying this tag in the OSM ``all`` network.
        drive_length_m: Length carrying it in the ``drive`` network.
        surplus_m: ``all_length_m - drive_length_m``, which is what ``all`` adds.
        share_of_all: This tag's share of total ``all`` length.
        share_of_surplus: This tag's share of the total surplus. Negative
            entries are possible in principle, since the two networks are
            separate Overpass queries rather than nested sets, and are reported
            rather than clipped.
    """

    highway: str
    all_length_m: float
    drive_length_m: float
    surplus_m: float
    share_of_all: float | None
    share_of_surplus: float | None


@dataclass(frozen=True, slots=True)
class PuritySummary:
    """The stub purity distribution under one truth source.

    Attributes:
        source: Which network purity was read against.
        population: Which set of values; see :data:`Population`.
        dilate: Extra dilation applied to the mask.
        n: Values contributing.
        mean: Mean purity.
        q1: First quartile.
        median: Median.
        q3: Third quartile.
        fraction_at_zero: Share whose stub touches no road at all.
        fraction_at_least_half: Share at or above 0.5, which the census reads as
            a genuinely severed road rather than an invented stub.
    """

    source: Source
    population: Population
    dilate: float
    n: int
    mean: float | None
    q1: float | None
    median: float | None
    q3: float | None
    fraction_at_zero: float | None
    fraction_at_least_half: float | None


@dataclass(frozen=True, slots=True)
class ConversionSummary:
    """How much of SpaceNet's zero-purity mass lands on OSM road.

    The headline of the whole script. Every row is conditioned on SpaceNet
    purity being exactly zero, so it counts only stubs the label mask never
    touches, and asks what OSM says about those same stubs.

    Attributes:
        source: Which OSM network the conditioned purity is read against.
        population: Which set of values; see :data:`Population`.
        dilate: Extra dilation applied to every mask.
        threshold: Purity at or above which a stub reads as lying on road.
        n_zero_spacenet: Stubs with zero SpaceNet purity.
        n_converted: How many of those reach ``threshold`` against ``source``.
        n_nonzero: How many reach any purity above zero against ``source``,
            reported apart because a stub clipping the corner of an alley is a
            weaker claim than one running along it.
        converted_fraction: ``n_converted / n_zero_spacenet``.
        nonzero_fraction: ``n_nonzero / n_zero_spacenet``.
        median_purity: Median purity against ``source`` over the conditioned
            set, which says whether the mass moved a little or a lot.
    """

    source: Source
    population: Population
    dilate: float
    threshold: float
    n_zero_spacenet: int
    n_converted: int
    n_nonzero: int
    converted_fraction: float | None
    nonzero_fraction: float | None
    median_purity: float | None


def resample(pts: np.ndarray, spacing: float) -> np.ndarray:
    """Points along a polyline at roughly uniform arc-length spacing.

    Uniform in arc length is what makes a fraction of *samples* stand in for a
    fraction of *length*: the polyline's own vertices cluster where it bends, so
    counting those would weight curves over straights.

    Args:
        pts: ``(N, 2)`` polyline in ``(x, y)`` pixel coordinates.
        spacing: Nominal distance between samples, in pixels.

    Returns:
        ``(M, 2)`` array of sample positions.
    """
    pts = np.asarray(pts, dtype=float)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    pts = pts[np.concatenate([[True], steps > 0.0])]
    if len(pts) < 2:
        return pts.reshape(-1, 2)

    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    n_samples = max(int(np.ceil(cum[-1] / spacing)) + 1, 2)
    arc = np.linspace(0.0, cum[-1], n_samples)
    return np.column_stack(
        [np.interp(arc, cum, pts[:, 0]), np.interp(arc, cum, pts[:, 1])]
    )


def edge_polylines(G: nx.MultiGraph) -> Iterable[np.ndarray]:
    """Every edge's polyline, oriented so geometry does not depend on numbering."""
    return (geograph.oriented_pts(G, u, v, k) for u, v, k in G.edges(keys=True))


def centreline_distance(G: nx.MultiGraph, shape: tuple[int, int]) -> np.ndarray | None:
    """Distance from every pixel to the nearest centreline pixel of a graph.

    Drawn at zero width rather than at the road width the label mask uses, so
    the tolerance in :func:`agreement` is the whole tolerance and is not
    silently widened by a dilation radius.

    Args:
        G: Graph in tile pixel coordinates.
        shape: ``(height, width)`` of the chip.

    Returns:
        ``(H, W)`` float distances, or ``None`` when the graph draws nothing and
        no distance is defined.
    """
    drawn = raster.draw_polylines(shape, edge_polylines(G), half_width_px=0)
    if not drawn.any():
        return None
    return np.asarray(ndimage.distance_transform_edt(~drawn), dtype=float)


def sample_points(G: nx.MultiGraph, spacing: float) -> np.ndarray:
    """Arc-length samples along every edge of a graph, pooled.

    Args:
        G: Graph whose length is being measured.
        spacing: Nominal distance between samples, in pixels.

    Returns:
        ``(M, 2)`` array of ``(x, y)`` sample positions.
    """
    samples = [resample(pts, spacing) for pts in edge_polylines(G)]
    return np.vstack(samples) if samples else np.zeros((0, 2))


def lookup(distance: np.ndarray, xy: np.ndarray, offset: float = 0.0) -> np.ndarray:
    """Read a distance field at sample positions, optionally translated.

    Samples that the offset pushes off the grid are dropped rather than clamped
    to the border, where they would read whatever road happens to run along the
    chip edge.

    Args:
        distance: ``(H, W)`` field from :func:`centreline_distance`.
        xy: ``(M, 2)`` sample positions in ``(x, y)`` pixel coordinates.
        offset: Pixels to translate the samples by, on both axes. The null
            control uses this; the real measurements pass zero.

    Returns:
        ``(K,)`` distances, ``K <= M``.
    """
    if len(xy) == 0:
        return np.zeros(0)
    rows = np.rint(xy[:, 1] + offset).astype(int)
    cols = np.rint(xy[:, 0] + offset).astype(int)
    inside = (
        (rows >= 0)
        & (rows < distance.shape[0])
        & (cols >= 0)
        & (cols < distance.shape[1])
    )
    return distance[rows[inside], cols[inside]]


def agreement(
    truth: nx.MultiGraph,
    osm_drive: nx.MultiGraph,
    osm_all: nx.MultiGraph,
    shape: tuple[int, int],
    tolerances: tuple[float, ...],
    spacing: float,
    control_shift: float,
) -> tuple[tuple[Agreement, ...], int, int]:
    """Measure how much of each network's length lies near the other.

    Args:
        truth: SpaceNet truth graph in tile pixel coordinates.
        osm_drive: OSM ``drive`` graph over the same tile.
        osm_all: OSM ``all`` graph over the same tile.
        shape: ``(height, width)`` of the chip.
        tolerances: Distances in pixels to report at.
        spacing: Nominal distance between samples along a polyline, in pixels.
        control_shift: Pixels to translate the OSM samples by for the null
            control; see :class:`Agreement`.

    Returns:
        ``(agreements, n_truth_samples, n_osm_drive_samples)``. The agreements
        are empty when either the truth or the ``drive`` network draws nothing
        on this chip, since no fraction is defined against an absent network.
    """
    to_drive = centreline_distance(osm_drive, shape)
    to_all = centreline_distance(osm_all, shape)
    to_truth = centreline_distance(truth, shape)
    truth_xy = sample_points(truth, spacing)
    drive_xy = sample_points(osm_drive, spacing)
    if to_drive is None or to_truth is None or len(truth_xy) == 0 or len(drive_xy) == 0:
        return (), len(truth_xy), len(drive_xy)

    forward_drive = lookup(to_drive, truth_xy)
    forward_all = (
        lookup(to_all, truth_xy) if to_all is not None else np.full(len(truth_xy), np.inf)
    )
    reverse = lookup(to_truth, drive_xy)
    reverse_shifted = lookup(to_truth, drive_xy, control_shift)

    def share(values: np.ndarray, tolerance: float) -> float:
        return float((values <= tolerance).mean()) if len(values) else 0.0

    return (
        tuple(
            Agreement(
                tolerance=tolerance,
                spacenet_on_osm_drive=share(forward_drive, tolerance),
                osm_drive_on_spacenet=share(reverse, tolerance),
                spacenet_on_osm_all=share(forward_all, tolerance),
                osm_drive_on_spacenet_shifted=share(reverse_shifted, tolerance),
            )
            for tolerance in tolerances
        ),
        len(truth_xy),
        len(drive_xy),
    )


def tag_lengths(ways: Sequence[osm.TaggedWay]) -> dict[str, float]:
    """Total clipped length per ``highway`` value, in metres."""
    totals: Counter[str] = Counter()
    for way in ways:
        totals[way.highway] += geograph.polyline_length(way.pts)
    return dict(totals)


def fetch_ways(
    tile: Tile,
    network_type: str,
    pad_m: float,
    retries: int,
    backoff: float,
    pace: float = 0.0,
) -> tuple[osm.TaggedWay, ...]:
    """Fetch one network for a tile, pacing the request and retrying a refusal.

    Overpass allocates a small number of query slots per client and drops the
    connection outright when they are exhausted, so exhaustion arrives as
    ``Connection refused`` rather than as a 429 osmnx would handle itself. It is
    indistinguishable at the socket from an unhealthy backend, and the two need
    opposite responses: a dead address wants an immediate retry, an exhausted
    allocation wants the caller to slow down.

    ``pace`` is what addresses the second, and it is the one that matters. A run
    that paces its successful requests does not reach the limit; a run that only
    backs off after a refusal has already reached it, and retrying harder from
    inside the penalty window does not recover. Observed directly: an unpaced
    run fetched normally for four minutes and was then refused by every backend,
    each of which answered a plain status probe moments earlier.

    A chip dropped for either reason is a chip missing from the sample, so the
    retry is still worth having underneath the pacing.

    Args:
        tile: Tile to fetch ways for.
        network_type: osmnx network filter.
        pad_m: Margin on the query extent, in metres.
        retries: Attempts after the first before giving up.
        backoff: Seconds before the first retry, doubling up to
            :data:`MAX_FETCH_BACKOFF`.
        pace: Seconds to wait after a successful fetch, holding the request
            rate below Overpass's allocation. Zero disables pacing.

    Returns:
        The clipped, tagged ways, empty when the extent genuinely holds no way
        of this class. That is an answer rather than an error, so it is neither
        retried nor counted as a failed chip.

    Raises:
        Exception: Whatever the last attempt raised, once the retries are spent.
    """
    for attempt in range(retries + 1):
        started = time.perf_counter()
        try:
            ways = osm.tagged_ways(tile, network_type, pad_m)
        except InsufficientResponseError:
            # Not a failure. A padded chip extent holding no way of this class
            # is a real answer -- a neighbourhood mapped entirely as ``service``
            # returns nothing for ``drive`` -- and retrying cannot change it.
            logger.info(f"{network_type}: no ways in this chip's extent")
            return ()
        except Exception as error:
            if attempt == retries:
                raise
            delay = min(backoff * 2**attempt, MAX_FETCH_BACKOFF)
            logger.warning(
                f"{network_type} fetch failed ({error}); retrying in {delay:.0f} s"
            )
            time.sleep(delay)
        else:
            served_from_cache = time.perf_counter() - started < CACHE_HIT_SECONDS
            if pace and not served_from_cache:
                time.sleep(pace)
            return ways
    raise AssertionError("unreachable")


def source_masks(
    sample: data.TileSample,
    osm_drive: nx.MultiGraph,
    osm_drivable: nx.MultiGraph,
    osm_all: nx.MultiGraph,
    tile: Tile,
    half_width_px: int,
    dilate: float,
) -> dict[Source, np.ndarray]:
    """Rasterize all three networks onto the chip grid at one dilation.

    ``half_width_px`` matches what ``spacenet.SpaceNetTileSource`` used to build
    ``sample.mask``, so the OSM masks are drawn to the same road width as the
    SpaceNet one and a purity difference between them is a difference in what
    was mapped rather than in how wide it was painted.

    Args:
        sample: Chip carrying the SpaceNet label mask.
        osm_drive: OSM ``drive`` graph in the chip's pixel frame.
        osm_drivable: OSM ``all`` minus the pedestrian classes, same frame.
        osm_all: OSM ``all`` graph in the same frame.
        tile: Tile defining the output grid.
        half_width_px: Dilation radius the centrelines are drawn with.
        dilate: Further binary dilation iterations applied to every mask,
            including the SpaceNet one, so the three stay comparable.

    Returns:
        One boolean mask per source.
    """
    masks: dict[Source, np.ndarray] = {
        "spacenet": np.asarray(sample.mask, dtype=bool),
        "osm_drive": raster.rasterize(osm_drive, tile, half_width_px=half_width_px),
        "osm_drivable": raster.rasterize(osm_drivable, tile, half_width_px=half_width_px),
        "osm_all": raster.rasterize(osm_all, tile, half_width_px=half_width_px),
    }
    if dilate <= 0:
        return masks
    structure = ndimage.generate_binary_structure(2, 2)
    return {
        source: ndimage.binary_dilation(mask, structure, int(dilate))
        for source, mask in masks.items()
    }


def crosscheck_chip(
    model,
    sample: data.TileSample,
    sample_id: str,
    threshold: float,
    radius: float,
    border_margin: float,
    tau: float,
    eps: float,
    label_snap: float,
    purity_spacing: float,
    purity_dilates: tuple[float, ...],
    half_width_px: int,
    simplify_tolerance: float,
    snap_tolerance: float,
    osm_drive: nx.MultiGraph,
    osm_drivable: nx.MultiGraph,
    osm_all: nx.MultiGraph,
    device: str,
) -> tuple[tuple[EndpointPurity, ...], tuple[CandidatePurity, ...]]:
    """Recompute stub purity on one chip against all three networks.

    Args:
        model: Frozen checkpoint.
        sample: The loaded chip, carrying its imagery, label mask and truth graph.
        sample_id: Recorded on each row so results stay traceable.
        threshold: Probability above which a pixel counts as road.
        radius: Proposal radius in pixels.
        border_margin: Endpoints this close to the chip edge are chipping
            artifacts rather than gaps.
        tau: Directness bound on the truth route.
        eps: How much worse the proposal route must be to count as a gap.
        label_snap: Snap radius the labeller runs at.
        purity_spacing: Polyline resampling step for stub purity, in pixels.
        purity_dilates: Extra dilations to recompute every purity at.
        half_width_px: Road half-width the OSM masks are drawn with.
        simplify_tolerance: Douglas-Peucker tolerance in pixels.
        snap_tolerance: Junction merge radius in pixels.
        osm_drive: OSM ``drive`` graph in this chip's pixel frame.
        osm_drivable: OSM ``all`` minus the pedestrian classes, same frame.
        osm_all: OSM ``all`` graph in the same frame.
        device: Device string for inference.

    Returns:
        ``(endpoint_rows, candidate_rows)``, one row per dilation per endpoint
        and per candidate.
    """
    logits = train.predict_tile_logits(model, sample, device=device)
    raw = skeleton.graph_from_mask(predict_mask(logits, threshold))
    preprune = cleanup.snap_junctions(
        cleanup.simplify_edges(raw, simplify_tolerance), snap_tolerance
    )

    nodes, pos = endpoints(preprune)
    keep = interior_flags(pos, sample.mask.shape, border_margin)
    interior_nodes = [n for n, k in zip(nodes, keep, strict=True) if k]
    interior_pos = pos[keep]

    candidates = endpoint_endpoint_candidates(
        preprune, interior_nodes, interior_pos, radius
    ) + endpoint_edge_candidates(preprune, interior_nodes, interior_pos, radius)
    if not candidates:
        return (), ()

    stub_nodes = sorted(
        {c.node_a for c in candidates}
        | {c.node_b for c in candidates if c.node_b is not None}
    )
    lengths = {
        n: nx.single_source_dijkstra_path_length(preprune, n, weight="length")
        for n in {c.node_a for c in candidates}
    }
    labels = [
        label_candidate(
            sample.truth,
            c,
            proposal_distance(preprune, lengths[c.node_a], c),
            tau,
            eps,
            label_snap,
        )[0]
        for c in candidates
    ]

    endpoint_rows: list[EndpointPurity] = []
    candidate_rows: list[CandidatePurity] = []
    for dilate in purity_dilates:
        masks = source_masks(
            sample, osm_drive, osm_drivable, osm_all, sample.tile, half_width_px, dilate
        )
        purities = {
            key: endpoint_purities(preprune, mask, stub_nodes, purity_spacing)
            for key, mask in masks.items()
        }
        endpoint_rows.extend(
            EndpointPurity(
                sample_id=sample_id,
                node=node,
                dilate=dilate,
                purity={s: purities[s][node] for s in SOURCES},
            )
            for node in stub_nodes
        )
        candidate_rows.extend(
            CandidatePurity(
                sample_id=sample_id,
                dilate=dilate,
                label=label,
                # The weaker of the two stubs decides, as in
                # endpoint_census.candidate_purity: a pair is only a severed
                # road if both halves are real road.
                purity={
                    s: (
                        purities[s][c.node_a]
                        if c.node_b is None
                        else min(purities[s][c.node_a], purities[s][c.node_b])
                    )
                    for s in SOURCES
                },
            )
            for c, label in zip(candidates, labels, strict=True)
        )
    return tuple(endpoint_rows), tuple(candidate_rows)


def _quantile(values: Sequence[float], q: float) -> float | None:
    """Percentile of a possibly-empty sample; ``None`` rather than ``nan``."""
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else None


def _fraction(numerator: int, denominator: int) -> float | None:
    """Share, or ``None`` when nothing was counted rather than a misleading 0."""
    return numerator / denominator if denominator else None


#: The directions of :class:`Agreement` a summary row can describe.
DIRECTIONS = (
    "spacenet_on_osm_drive",
    "osm_drive_on_spacenet",
    "spacenet_on_osm_all",
    "osm_drive_on_spacenet_shifted",
)


def summarize_agreement(
    alignments: Sequence[ChipAlignment],
    tolerance: float,
    direction: str,
    gate_fraction: float,
) -> AgreementSummary:
    """Pool one direction of step 0 over the chips at one tolerance.

    Args:
        alignments: Every chip's step 0 record.
        tolerance: Which tolerance to summarize.
        direction: Which field of :class:`Agreement` to read.
        gate_fraction: Value a chip must reach to count as passing.

    Returns:
        The distribution over chips plus which chips fall under the gate.
    """
    rows = [
        (r.sample_id, float(getattr(a, direction)))
        for r in alignments
        for a in r.agreements
        if a.tolerance == tolerance
    ]
    values = [v for _, v in rows]
    below = tuple(i for i, v in rows if v < gate_fraction)

    return AgreementSummary(
        tolerance=tolerance,
        n_chips=len(rows),
        direction=direction,
        mean=float(np.mean(values)) if values else 0.0,
        median=_quantile(values, 50) or 0.0,
        q1=_quantile(values, 25) or 0.0,
        gate_fraction=gate_fraction,
        n_below_gate=len(below),
        chips_below_gate=below,
    )


def summarize_lengths(lengths: Sequence[ChipLengths]) -> LengthSummary:
    """Pool road length from all three sources over the chips."""
    drive_ratios = [r.drive_ratio for r in lengths if r.drive_ratio is not None]
    drivable_ratios = [r.drivable_ratio for r in lengths if r.drivable_ratio is not None]
    all_ratios = [r.all_ratio for r in lengths if r.all_ratio is not None]
    return LengthSummary(
        n_chips=len(lengths),
        spacenet_total_m=float(sum(r.spacenet_m for r in lengths)),
        osm_drive_total_m=float(sum(r.osm_drive_m for r in lengths)),
        osm_drivable_total_m=float(sum(r.osm_drivable_m for r in lengths)),
        osm_all_total_m=float(sum(r.osm_all_m for r in lengths)),
        drive_ratio_mean=float(np.mean(drive_ratios)) if drive_ratios else None,
        drive_ratio_median=_quantile(drive_ratios, 50),
        drivable_ratio_mean=float(np.mean(drivable_ratios)) if drivable_ratios else None,
        drivable_ratio_median=_quantile(drivable_ratios, 50),
        all_ratio_mean=float(np.mean(all_ratios)) if all_ratios else None,
        all_ratio_median=_quantile(all_ratios, 50),
        n_chips_rated=len(drive_ratios),
    )


def summarize_tags(lengths: Sequence[ChipLengths]) -> tuple[TagSummary, ...]:
    """Attribute the OSM ``all`` surplus over ``drive`` to ``highway`` classes.

    Args:
        lengths: Every chip's length record.

    Returns:
        One row per tag, ordered by the surplus it carries, largest first. This
        is the table that separates "SpaceNet omits a class of way" from "OSM
        and SpaceNet disagree everywhere".
    """
    all_totals: Counter[str] = Counter()
    drive_totals: Counter[str] = Counter()
    for row in lengths:
        all_totals.update(row.all_tag_lengths)
        drive_totals.update(row.drive_tag_lengths)

    total_all = float(sum(all_totals.values()))
    total_surplus = total_all - float(sum(drive_totals.values()))
    rows = [
        TagSummary(
            highway=tag,
            all_length_m=float(all_totals.get(tag, 0.0)),
            drive_length_m=float(drive_totals.get(tag, 0.0)),
            surplus_m=float(all_totals.get(tag, 0.0) - drive_totals.get(tag, 0.0)),
            share_of_all=(all_totals.get(tag, 0.0) / total_all) if total_all else None,
            share_of_surplus=(
                (all_totals.get(tag, 0.0) - drive_totals.get(tag, 0.0)) / total_surplus
                if total_surplus
                else None
            ),
        )
        for tag in set(all_totals) | set(drive_totals)
    ]
    return tuple(sorted(rows, key=lambda r: r.surplus_m, reverse=True))


def _values(
    endpoint_rows: Sequence[EndpointPurity],
    candidate_rows: Sequence[CandidatePurity],
    population: Population,
    dilate: float,
) -> list[dict[Source, float]]:
    """Every purity record in one population at one dilation.

    Endpoints are deduplicated by ``(sample_id, node)``, matching the census, so
    an endpoint holding several candidates does not weigh more than one holding
    a single candidate.
    """
    match population:
        case "endpoint":
            by_node = {
                (r.sample_id, r.node): r.purity
                for r in endpoint_rows
                if r.dilate == dilate
            }
            return list(by_node.values())
        case "candidate":
            return [r.purity for r in candidate_rows if r.dilate == dilate]
        case "positive_candidate":
            return [
                r.purity
                for r in candidate_rows
                if r.dilate == dilate and r.label == "positive"
            ]


def summarize_purity(
    endpoint_rows: Sequence[EndpointPurity],
    candidate_rows: Sequence[CandidatePurity],
    source: Source,
    population: Population,
    dilate: float,
) -> PuritySummary:
    """Describe one population's purity distribution under one truth source.

    Args:
        endpoint_rows: Every endpoint's purity record.
        candidate_rows: Every candidate's purity record.
        source: Which network to read purity against.
        population: Which set of values; see :data:`Population`.
        dilate: Which dilation to report.

    Returns:
        Quantiles plus the two fractions the census reports.
    """
    values = [
        row[source] for row in _values(endpoint_rows, candidate_rows, population, dilate)
    ]
    return PuritySummary(
        source=source,
        population=population,
        dilate=dilate,
        n=len(values),
        mean=float(np.mean(values)) if values else None,
        q1=_quantile(values, 25),
        median=_quantile(values, 50),
        q3=_quantile(values, 75),
        fraction_at_zero=_fraction(sum(v == 0.0 for v in values), len(values)),
        fraction_at_least_half=_fraction(sum(v >= 0.5 for v in values), len(values)),
    )


def summarize_conversion(
    endpoint_rows: Sequence[EndpointPurity],
    candidate_rows: Sequence[CandidatePurity],
    source: Source,
    population: Population,
    dilate: float,
    threshold: float,
) -> ConversionSummary:
    """Count how much of SpaceNet's zero-purity mass lands on OSM road.

    Args:
        endpoint_rows: Every endpoint's purity record.
        candidate_rows: Every candidate's purity record.
        source: Which OSM network to read the conditioned purity against.
        population: Which set of values; see :data:`Population`.
        dilate: Which dilation to report.
        threshold: Purity at or above which a stub reads as lying on road.

    Returns:
        The conditioned counts and fractions; see :class:`ConversionSummary`.
    """
    conditioned = [
        row[source]
        for row in _values(endpoint_rows, candidate_rows, population, dilate)
        if row["spacenet"] == 0.0
    ]
    converted = sum(v >= threshold for v in conditioned)
    nonzero = sum(v > 0.0 for v in conditioned)
    return ConversionSummary(
        source=source,
        population=population,
        dilate=dilate,
        threshold=threshold,
        n_zero_spacenet=len(conditioned),
        n_converted=converted,
        n_nonzero=nonzero,
        converted_fraction=_fraction(converted, len(conditioned)),
        nonzero_fraction=_fraction(nonzero, len(conditioned)),
        median_purity=_quantile(conditioned, 50),
    )


def holdout_ids(run_json: Path, exclude_json: Path, skip_tiles: int) -> list[str]:
    """The reporting holdout: val ids past the threshold-selection prefix.

    Args:
        run_json: Run artifact carrying the checkpoint's own train/val split.
        exclude_json: Oracle artifact whose ``skipped_zero_ceiling`` names the
            chips a perfect mask still scores zero on. Those carry no usable
            ground truth and are excluded everywhere else in this project, so
            they are excluded here rather than recomputed.
        skip_tiles: Validation tiles spent selecting the mask threshold.

    Returns:
        Chip ids, in the split's own order.
    """
    val_ids = list(json.loads(run_json.read_text())["config"]["val_ids"])[skip_tiles:]
    if not exclude_json.exists():
        logger.warning(f"{exclude_json} missing; keeping every zero-ceiling chip")
        return val_ids
    excluded = set(json.loads(exclude_json.read_text()).get("skipped_zero_ceiling", []))
    return [i for i in val_ids if i not in excluded]


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/vegas_best.pt"))
    parser.add_argument(
        "--run-json",
        type=Path,
        default=Path("outputs/vegas_best.json"),
        help="run artifact supplying the val split the checkpoint was fit against",
    )
    parser.add_argument(
        "--exclude-json",
        type=Path,
        default=Path("outputs/link_oracle.json"),
        help=(
            "artifact whose skipped_zero_ceiling names the chips a perfect mask "
            "still scores zero on; they are dropped from the holdout here too"
        ),
    )
    parser.add_argument("--aoi-root", type=Path, default=Path("data/AOI_2_Vegas"))
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="the tuned operating point from outputs/threshold_sweep.json",
    )
    parser.add_argument(
        "--skip-tiles",
        type=int,
        default=40,
        help=(
            "validation tiles to skip; the first 40 were spent selecting the "
            "mask threshold, so the reporting holdout is what follows them"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=60,
        help=(
            "chips to measure; 0 runs the whole 155-chip holdout. Defaults to "
            "60 rather than 0 because every chip costs two Overpass queries and "
            "this is a diagnostic rather than a reported pipeline metric"
        ),
    )
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=list(DEFAULT_TOLERANCES),
        help="step 0 agreement tolerances, in pixels",
    )
    parser.add_argument(
        "--gate-tolerance",
        type=float,
        default=8.0,
        help="which tolerance the alignment gate reads",
    )
    parser.add_argument(
        "--gate-direction",
        default="osm_drive_on_spacenet",
        choices=DIRECTIONS,
        help=(
            "which agreement direction the gate reads. The default is the one "
            "that isolates registration from coverage; see the module "
            "docstring. Every direction is reported whichever is chosen"
        ),
    )
    parser.add_argument(
        "--gate-fraction",
        type=float,
        default=0.6,
        help=(
            "agreement a chip must reach on --gate-direction at "
            "--gate-tolerance to count as registered; the run emits no length "
            "or purity numbers when most chips fall under it, because "
            "misregistration would make them noise"
        ),
    )
    parser.add_argument(
        "--control-shift",
        type=float,
        default=25.0,
        help=(
            "pixels the OSM samples are translated by for the null control on "
            "the registration direction; a real registration collapses under it"
        ),
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=60.0,
        help=(
            "proposal radius in pixels; 60 is where the census quoted the 47%% "
            "zero-purity figure this script is trying to explain"
        ),
    )
    parser.add_argument(
        "--border-margin",
        type=float,
        default=10.0,
        help="endpoints this close to the chip edge are chipping artifacts",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.5,
        help="a truth route longer than tau x the straight-line gap is a different road",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.2,
        help="the proposal route must be worse than the truth route by more than this",
    )
    parser.add_argument(
        "--label-snap",
        type=float,
        default=10.0,
        help="the committed labeller snap radius the positive breakout is taken at",
    )
    parser.add_argument(
        "--purity-threshold",
        type=float,
        default=0.5,
        help="stub purity at or above which a stub reads as lying on road",
    )
    parser.add_argument(
        "--purity-spacing",
        type=float,
        default=1.0,
        help="polyline resampling step for stub purity, in pixels; the census value",
    )
    parser.add_argument(
        "--purity-dilate",
        type=float,
        nargs="+",
        default=list(DEFAULT_PURITY_DILATES),
        help=(
            "extra dilations every truth mask is widened by before purity is "
            "read; every value is applied to all three sources so the headline "
            "can be checked against a mask-width artifact"
        ),
    )
    parser.add_argument(
        "--sample-spacing",
        type=float,
        default=1.0,
        help="polyline resampling step for the step 0 agreement, in pixels",
    )
    parser.add_argument(
        "--half-width",
        type=float,
        default=2.0,
        help=(
            "road half-width the OSM masks are rasterized at; 2 is what "
            "spacenet.SpaceNetTileSource used to build the label mask, so the "
            "three masks are drawn to the same width"
        ),
    )
    parser.add_argument(
        "--overpass-url",
        default=ox.settings.overpass_url,
        help=(
            "Overpass endpoint. osmnx pins one backend address per process, so "
            "naming a specific server routes around a round-robin member that "
            "is refusing connections; see the module docstring"
        ),
    )
    parser.add_argument(
        "--no-overpass-rate-limit",
        dest="overpass_rate_limit",
        action="store_false",
        help=(
            "skip osmnx's status-endpoint pacing. Only for a mirror that "
            "publishes no parseable status, where the pacing degrades to a "
            "flat 60 s wait per request; see the module docstring"
        ),
    )
    parser.add_argument(
        "--fetch-retries",
        type=int,
        default=8,
        help=(
            "attempts after the first when Overpass refuses the connection; a "
            "dropped chip is a chip missing from the sample"
        ),
    )
    parser.add_argument(
        "--fetch-backoff",
        type=float,
        default=5.0,
        help=("seconds before the first fetch retry, doubling up to MAX_FETCH_BACKOFF"),
    )
    parser.add_argument(
        "--fetch-pace",
        type=float,
        default=4.0,
        help=(
            "seconds to wait after each successful Overpass query. Overpass "
            "drops connections when a client exhausts its slot allocation, and "
            "backing off afterwards does not recover from inside the penalty "
            "window; holding the rate down is what avoids it. Cached chips "
            "cost nothing, so this only paces real fetches. 0 disables"
        ),
    )
    parser.add_argument(
        "--way-source",
        choices=sorted(osm.WAY_SOURCE_REGISTRY),
        default="pbf",
        help=(
            "where OSM ways come from. 'pbf' reads a local Geofabrik extract: "
            "no rate limit, and a fixed file whose checksum can be folded into "
            "run identity, which an Overpass query cannot offer because the "
            "database moves under it. 'overpass' keeps the live path"
        ),
    )
    parser.add_argument(
        "--extract",
        type=Path,
        default=Path("data/nevada-latest.osm.pbf"),
        help="local .osm.pbf extract, read once and clipped per chip",
    )
    parser.add_argument(
        "--pad-m",
        type=float,
        default=250.0,
        help="margin on the OSM query extent so ways crossing the chip edge arrive whole",
    )
    parser.add_argument("--simplify-tolerance", type=float, default=2.0)
    parser.add_argument("--snap-tolerance", type=float, default=8.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("outputs/osm_crosscheck.json"))
    args = parser.parse_args()

    requested = holdout_ids(args.run_json, args.exclude_json, args.skip_tiles)[
        : args.limit or None
    ]
    ox.settings.overpass_url = args.overpass_url
    ox.settings.overpass_rate_limit = args.overpass_rate_limit
    tolerances = tuple(sorted(float(t) for t in args.tolerances))
    dilates = tuple(sorted({float(d) for d in args.purity_dilate}))
    half_width = int(args.half_width)

    source, _ = train.build_source(args.aoi_root, "spacenet", args.resolution)
    logger.info(
        f"{len(requested)} chips at threshold {args.threshold}, radius {args.radius}, "
        f"tolerances {tolerances}, purity dilates {dilates}"
    )

    # Step 0 and the length accounting run first and alone. Both need the OSM
    # fetch and the labels but no inference, so a failed gate costs no model
    # time and produces no downstream numbers.
    started = time.perf_counter()
    fetch_seconds = 0.0
    alignments: list[ChipAlignment] = []
    lengths: list[ChipLengths] = []
    osm_failures: list[str] = []
    # Held so the purity pass neither refetches nor reclips. Graphs only: the
    # samples they came from carry imagery and are reloaded instead.
    osm_graphs: dict[str, tuple[nx.MultiGraph, nx.MultiGraph, nx.MultiGraph]] = {}

    # Read-once, clip-many. Scanning the extract costs about the same whatever
    # extent is asked for -- the cost is the 123 MB file, not the bbox -- so one
    # read over every tile's covering bounds replaces a per-chip read that would
    # pay that scan 155 times.
    way_source: osm.WaySource
    extract_md5: str | None = None
    if args.way_source == "pbf":
        # Folded into the artifact so a run's OSM input is identified by content
        # rather than by filename. Geofabrik reuses "-latest" across dailies.
        extract_md5 = hashlib.md5(args.extract.read_bytes()).hexdigest()
        extent_started = time.perf_counter()
        tiles_for_extent = [source.load(i).tile for i in requested]
        extract = osm.read_extract(
            osm.covering_bounds(tiles_for_extent, args.pad_m), args.extract
        )
        way_source = osm.WAY_SOURCE_REGISTRY["pbf"](extract=extract)
        fetch_seconds += time.perf_counter() - extent_started
        logger.info(
            f"read {args.extract} in {fetch_seconds:.1f} s over "
            f"{len(tiles_for_extent)} tiles' covering bounds"
        )
    else:
        way_source = osm.WAY_SOURCE_REGISTRY["overpass"]()

    for n, sample_id in enumerate(requested, start=1):
        sample = source.load(sample_id)
        fetch_started = time.perf_counter()
        try:
            drive_ways = way_source.ways(sample.tile, "drive", args.pad_m)
            all_ways = way_source.ways(sample.tile, "all", args.pad_m)
        except Exception as error:
            # One bad chip must not end the run.
            logger.warning(f"{sample_id}: OSM fetch failed ({error}); skipping")
            osm_failures.append(sample_id)
            continue
        fetch_seconds += time.perf_counter() - fetch_started

        osm_drive = osm.graph_from_ways(drive_ways, sample.tile)
        osm_all = osm.graph_from_ways(all_ways, sample.tile)
        osm_drivable = osm.graph_from_ways(
            tuple(w for w in all_ways if w.highway not in PEDESTRIAN_TAGS), sample.tile
        )
        osm_graphs[sample_id] = (osm_drive, osm_drivable, osm_all)
        agreements, n_truth, n_osm = agreement(
            sample.truth,
            osm_drive,
            osm_all,
            sample.mask.shape,
            tolerances,
            args.sample_spacing,
            args.control_shift,
        )
        spacenet_m = geograph.total_length(sample.truth)
        drive_m = geograph.total_length(osm_drive)
        drivable_m = geograph.total_length(osm_drivable)
        all_m = geograph.total_length(osm_all)

        alignments.append(
            ChipAlignment(
                sample_id=sample_id,
                spacenet_length_m=spacenet_m,
                osm_drive_length_m=drive_m,
                n_spacenet_samples=n_truth,
                n_osm_samples=n_osm,
                agreements=agreements,
            )
        )
        lengths.append(
            ChipLengths(
                sample_id=sample_id,
                spacenet_m=spacenet_m,
                osm_drive_m=drive_m,
                osm_drivable_m=drivable_m,
                osm_all_m=all_m,
                drive_ratio=(drive_m / spacenet_m) if spacenet_m else None,
                drivable_ratio=(drivable_m / spacenet_m) if spacenet_m else None,
                all_ratio=(all_m / spacenet_m) if spacenet_m else None,
                drive_tag_lengths=tag_lengths(drive_ways),
                all_tag_lengths=tag_lengths(all_ways),
                n_multi_valued_ways=sum(w.multi_valued for w in all_ways),
            )
        )
        if n % 10 == 0 or n == len(requested):
            at_gate = next(
                (a for a in agreements if a.tolerance == args.gate_tolerance), None
            )
            reverse = "n/a" if at_gate is None else f"{at_gate.osm_drive_on_spacenet:.3f}"
            logger.info(
                f"[{n}/{len(requested)}] {sample_id}: spacenet {spacenet_m:.0f} m, "
                f"osm drive {drive_m:.0f} m, osm all {all_m:.0f} m, "
                f"drive-on-spacenet@{args.gate_tolerance:.0f} {reverse}"
            )

    rated = [r for r in alignments if r.agreements]
    agreement_summaries = tuple(
        summarize_agreement(rated, tolerance, direction, args.gate_fraction)
        for tolerance in tolerances
        for direction in DIRECTIONS
    )
    gate = next(
        (
            s
            for s in agreement_summaries
            if s.tolerance == args.gate_tolerance and s.direction == args.gate_direction
        ),
        None,
    )
    gate_passed = bool(gate and gate.n_chips and gate.n_below_gate * 2 <= gate.n_chips)

    logger.info(
        f"{'tol':>5} {'direction':>31} {'mean':>8} {'median':>8} {'q1':>8} {'below':>9}"
    )
    for s in agreement_summaries:
        logger.info(
            f"{s.tolerance:>5.0f} {s.direction:>31} {s.mean:>8.4f} {s.median:>8.4f} "
            f"{s.q1:>8.4f} {s.n_below_gate:>4}/{s.n_chips:<4}"
        )
    if gate and gate.chips_below_gate:
        logger.warning(
            f"chips under the gate ({args.gate_direction} at "
            f"{args.gate_tolerance:.0f} px < {args.gate_fraction}): "
            f"{list(gate.chips_below_gate)}"
        )

    length_summary: LengthSummary | None = None
    tag_summaries: tuple[TagSummary, ...] = ()
    purity_summaries: tuple[PuritySummary, ...] = ()
    conversion_summaries: tuple[ConversionSummary, ...] = ()
    endpoint_rows: tuple[EndpointPurity, ...] = ()
    candidate_rows: tuple[CandidatePurity, ...] = ()

    if not gate_passed:
        logger.error(
            "alignment gate FAILED: OSM and SpaceNet do not register in the same "
            "pixel frame on most chips, so no length or purity number is emitted"
        )
    else:
        length_summary = summarize_lengths(lengths)
        tag_summaries = summarize_tags(lengths)
        logger.info(
            f"spacenet {length_summary.spacenet_total_m:.0f} m, "
            f"osm drive {length_summary.osm_drive_total_m:.0f} m, "
            f"osm drivable {length_summary.osm_drivable_total_m:.0f} m, "
            f"osm all {length_summary.osm_all_total_m:.0f} m; "
            f"drive/spacenet mean {length_summary.drive_ratio_mean}, "
            f"drivable/spacenet mean {length_summary.drivable_ratio_mean}, "
            f"all/spacenet mean {length_summary.all_ratio_mean}"
        )
        for t in tag_summaries[:15]:
            logger.info(
                f"{t.highway:>16}: all {t.all_length_m:>10.0f} m, drive "
                f"{t.drive_length_m:>10.0f} m, surplus {t.surplus_m:>10.0f} m, "
                f"share of surplus {t.share_of_surplus}"
            )

        model = train.load_checkpoint(args.checkpoint, device=args.device)
        measured = [r.sample_id for r in alignments]
        endpoints_acc: list[EndpointPurity] = []
        candidates_acc: list[CandidatePurity] = []
        for n, sample_id in enumerate(measured, start=1):
            osm_drive, osm_drivable, osm_all = osm_graphs[sample_id]
            chip_endpoints, chip_candidates = crosscheck_chip(
                model,
                source.load(sample_id),
                sample_id,
                args.threshold,
                args.radius,
                args.border_margin,
                args.tau,
                args.eps,
                args.label_snap,
                args.purity_spacing,
                dilates,
                half_width,
                args.simplify_tolerance,
                args.snap_tolerance,
                osm_drive,
                osm_drivable,
                osm_all,
                args.device,
            )
            endpoints_acc.extend(chip_endpoints)
            candidates_acc.extend(chip_candidates)
            if n % 10 == 0 or n == len(measured):
                logger.info(
                    f"[{n}/{len(measured)}] {sample_id}: "
                    f"{len(endpoints_acc)} endpoint rows, "
                    f"{len(candidates_acc)} candidate rows"
                )

        endpoint_rows = tuple(endpoints_acc)
        candidate_rows = tuple(candidates_acc)
        purity_summaries = tuple(
            summarize_purity(endpoint_rows, candidate_rows, s, population, dilate)
            for dilate in dilates
            for population in POPULATIONS
            for s in SOURCES
        )
        conversion_summaries = tuple(
            summarize_conversion(
                endpoint_rows,
                candidate_rows,
                s,
                population,
                dilate,
                args.purity_threshold,
            )
            for dilate in dilates
            for population in POPULATIONS
            for s in SOURCES
        )

        logger.info(
            f"{'dilate':>7} {'population':>20} {'source':>10} {'n':>7} {'mean':>7} "
            f"{'q1':>7} {'med':>7} {'q3':>7} {'zero':>7} {'>=0.5':>7}"
        )
        for p in purity_summaries:
            logger.info(
                f"{p.dilate:>7.0f} {p.population:>20} {p.source:>10} {p.n:>7} "
                f"{p.mean or 0:>7.4f} {p.q1 or 0:>7.4f} {p.median or 0:>7.4f} "
                f"{p.q3 or 0:>7.4f} {p.fraction_at_zero or 0:>7.4f} "
                f"{p.fraction_at_least_half or 0:>7.4f}"
            )
        for c in conversion_summaries:
            if c.source == "spacenet":
                continue
            logger.info(
                f"dilate {c.dilate:>2.0f} {c.population:>20} zero-on-spacenet "
                f"n={c.n_zero_spacenet:>6} -> {c.source} >= {c.threshold}: "
                f"{c.n_converted} ({c.converted_fraction}), any purity "
                f"{c.n_nonzero} ({c.nonzero_fraction})"
            )

    elapsed = time.perf_counter() - started
    logger.info(
        f"wall time {elapsed / 60:.1f} min, of which OSM fetch and clip "
        f"{fetch_seconds / 60:.1f} min"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "run_json": str(args.run_json),
                "params": {
                    "threshold": args.threshold,
                    "radius": args.radius,
                    "border_margin": args.border_margin,
                    "tau": args.tau,
                    "eps": args.eps,
                    "label_snap": args.label_snap,
                    "purity_threshold": args.purity_threshold,
                    "purity_spacing": args.purity_spacing,
                    "purity_dilates": list(dilates),
                    "sample_spacing": args.sample_spacing,
                    "tolerances": list(tolerances),
                    "gate_tolerance": args.gate_tolerance,
                    "gate_direction": args.gate_direction,
                    "gate_fraction": args.gate_fraction,
                    "control_shift": args.control_shift,
                    "half_width_px": half_width,
                    "pad_m": args.pad_m,
                    "fetch_retries": args.fetch_retries,
                    "fetch_backoff": args.fetch_backoff,
                    "simplify_tolerance": args.simplify_tolerance,
                    "snap_tolerance": args.snap_tolerance,
                    "skip_tiles": args.skip_tiles,
                    "limit": args.limit,
                    "resolution": args.resolution,
                    "device": args.device,
                    "preprune_pipeline": "simplify_edges -> snap_junctions",
                    "osm_network_types": ["drive", "all"],
                    "overpass_url": args.overpass_url,
                    "overpass_rate_limit": args.overpass_rate_limit,
                    "pedestrian_tags": sorted(PEDESTRIAN_TAGS),
                    # Which OSM source produced these numbers, and its identity.
                    # Without this the artifact cannot say whether it came from
                    # a live Overpass query or a pinned extract, and only the
                    # extract is reproducible at all.
                    "way_source": args.way_source,
                    "extract": str(args.extract) if args.way_source == "pbf" else None,
                    "extract_md5": extract_md5,
                },
                "caveats": {
                    "temporal": (
                        "SpaceNet Vegas imagery is from roughly 2015-2017 and OSM "
                        "is read today. Roads built since appear in OSM and could "
                        "not have been found by the model, so every length ratio "
                        "in this artifact is biased upward. No historical OSM "
                        "query was made. The purity conversion is much less "
                        "exposed: a stub the model produced corresponds to "
                        "something visible in the 2015-2017 imagery."
                    ),
                },
                "elapsed_seconds": elapsed,
                "fetch_seconds": fetch_seconds,
                "n_requested": len(requested),
                "n_measured": len(alignments),
                "osm_failures": osm_failures,
                "sample_ids": [r.sample_id for r in alignments],
                "gate_passed": gate_passed,
                "agreement_summaries": [asdict(s) for s in agreement_summaries],
                "length_summary": asdict(length_summary) if length_summary else None,
                "tag_summaries": [asdict(t) for t in tag_summaries],
                "purity_summaries": [asdict(p) for p in purity_summaries],
                "conversion_summaries": [asdict(c) for c in conversion_summaries],
                "per_chip_alignment": [asdict(r) for r in alignments],
                "per_chip_lengths": [asdict(r) for r in lengths],
            },
            indent=2,
        )
    )
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()

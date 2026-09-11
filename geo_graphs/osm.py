"""Ground-truth road graphs from OpenStreetMap, in tile pixel coordinates.

Ways reach a tile through one of two sources, selected by config key through
:data:`WAY_SOURCE_REGISTRY`:

``overpass``
    Queries the live API through osmnx. One query per tile per network type,
    subject to the server's slot allowance, and not reproducible: the database
    changes under you, so a rerun is not the same input.
``pbf``
    Reads a local Geofabrik extract. The file is content-hashed, so a run
    identity can name the data it saw, and the whole extent is read once for a
    run rather than queried once per tile.

The two do not return the same shape of data, and the difference is silent.
osmnx returns a graph that is already noded and simplified; the ``lines`` layer
of an extract holds raw OSM ways, which run straight through their
intersections. :func:`pbf_ways` nodes them and :func:`tagged_ways` does not.
The network filters are the other place they can diverge, and
:data:`NETWORK_FILTERS` is pinned against osmnx's own by a test.
"""

import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import networkx as nx
import numpy as np
import osmnx as ox
import shapely
from loguru import logger
from pyogrio.raw import read as read_ogr
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, box
from shapely.strtree import STRtree

from . import geograph, tiles
from .tiles import Tile

#: Stand-in for a way that carries no ``highway`` value at all.
UNTAGGED = "untagged"

#: Local OSM extract the ``pbf`` source reads by default.
#:
#: Geofabrik publishes an MD5 beside every extract, so the file a run read is
#: identifiable after the fact, which an Overpass query is not.
DEFAULT_EXTRACT = Path("data/nevada-latest.osm.pbf")

#: Layer of an ``.osm.pbf`` holding ways with line geometry.
#:
#: GDAL's OSM driver splits a file into ``points``, ``lines``,
#: ``multilinestrings``, ``multipolygons`` and ``other_relations``. Roads are
#: ways, so ``lines`` is the whole road network; ``multilinestrings`` carries
#: route *relations*, which would double-count the ways they collect.
EXTRACT_LAYER = "lines"


@dataclass(frozen=True, slots=True)
class TaggedWay:
    """One clipped OSM way with the road class it was tagged as.

    Attributes:
        highway: The way's ``highway`` value, or :data:`UNTAGGED`.
        multi_valued: Whether the source tag held several values, so
            ``highway`` is the first of them rather than the only one. A caller
            tabulating length by class should report how often this is true
            rather than let the choice pass silently.
        pts: ``(N, 2)`` polyline in tile pixel coordinates.
        osm_id: The id of the OSM way this piece was cut from, or ``None`` when
            the source cannot name one. Noding splits a way at every junction
            along it, so pieces are not independent samples of anything; this
            is what groups them back into the way a mapper drew.
    """

    highway: str
    multi_valued: bool
    pts: np.ndarray
    osm_id: str | None = None


@dataclass(frozen=True, slots=True)
class NetworkFilter:
    """One osmnx network type, as the tag conditions that exclude a way.

    Copied from ``osmnx._overpass._get_network_filter`` rather than invented,
    because the two sources have to select the same ways or every length
    comparison between them measures the filter instead of the data.
    ``test_network_filters_match_osmnx`` parses osmnx's own filter strings and
    asserts these conditions are exactly theirs.

    Attributes:
        exclusions: ``(tag key, regex)`` pairs. A way is excluded when it
            carries the tag and the regex matches anywhere in its value.
            Overpass's ``!~`` is an unanchored regex match and a *missing* tag
            satisfies it, so an absent tag keeps the way.
    """

    exclusions: tuple[tuple[str, str], ...]


#: The osmnx network types the ``pbf`` source replicates.
#:
#: ``drive`` and ``all`` only: they are what the SpaceNet cross-check reads.
#: osmnx's ``walk``, ``bike``, ``drive_service`` and ``all_public`` are not
#: here, so the ``pbf`` source rejects them rather than quietly answering with
#: something close.
NETWORK_FILTERS: Mapping[str, NetworkFilter] = MappingProxyType(
    {
        "drive": NetworkFilter(
            exclusions=(
                (
                    "highway",
                    "abandoned|bridleway|bus_guideway|construction|corridor|"
                    "cycleway|elevator|escalator|footway|no|path|pedestrian|"
                    "planned|platform|proposed|raceway|razed|rest_area|service|"
                    "services|steps|track",
                ),
                ("area", "yes"),
                ("access", "private"),
                ("motor_vehicle", "no"),
                ("motorcar", "no"),
                (
                    "service",
                    "alley|driveway|emergency_access|parking|parking_aisle|private",
                ),
            )
        ),
        "all": NetworkFilter(
            exclusions=(
                (
                    "highway",
                    "abandoned|construction|no|planned|platform|proposed|raceway|"
                    "razed|rest_area|services",
                ),
                ("area", "yes"),
            )
        ),
    }
)

#: Tags the filters read beyond ``highway``, which GDAL leaves in ``other_tags``.
FILTER_TAG_KEYS: tuple[str, ...] = tuple(
    sorted(
        {
            key
            for network in NETWORK_FILTERS.values()
            for key, _ in network.exclusions
            if key != "highway"
        }
    )
)


def _edge_polylines(G: nx.MultiGraph | nx.MultiDiGraph) -> list[np.ndarray]:
    """Extract each edge's geometry, falling back to a straight run."""
    out = []
    for u, v, data in G.edges(data=True):
        geom = data.get("geometry")
        if geom is None:
            geom = LineString(
                [(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]
            )
        out.append(np.asarray(geom.coords, dtype=float))
    return out


def _clip(pts: np.ndarray, region) -> list[np.ndarray]:
    """Cut a polyline to a region, which may split it into several pieces."""
    clipped = LineString(pts).intersection(region)
    if clipped.is_empty:
        return []
    parts = clipped.geoms if isinstance(clipped, MultiLineString) else [clipped]
    return [
        np.asarray(p.coords, dtype=float)
        for p in parts
        if isinstance(p, LineString) and len(p.coords) >= 2
    ]


def _highway_value(data: dict) -> tuple[str, bool]:
    """The edge's ``highway`` tag as one string, and whether it held several.

    osmnx simplifies by default, merging consecutive degree-2 ways into one
    edge, and a merged run whose members disagree arrives as a list.
    """
    value = data.get("highway")
    if isinstance(value, list):
        return (str(value[0]) if value else UNTAGGED), True
    return (UNTAGGED if value is None else str(value)), False


def tagged_ways(
    tile: Tile, network_type: str = "drive", pad_m: float = 250.0
) -> tuple[TaggedWay, ...]:
    """Fetch OSM ways covering a tile from Overpass, clipped and keeping their class.

    The road class is what separates "OSM maps more than SpaceNet does" from
    "OSM maps a different *kind* of way than SpaceNet does", and
    :func:`ground_truth_graph` drops it, so this is the entry point for anything
    that needs to tabulate length by ``highway`` value.

    osmnx returns a noded, simplified graph and keeps only its largest weakly
    connected component, so no further noding happens here; see
    :func:`geograph.node_network` and :func:`pbf_ways` for the other source.

    Args:
        tile: Tile to fetch ways for.
        network_type: osmnx network filter, e.g. ``"drive"`` or ``"all"``.
        pad_m: Margin added to the query extent so ways crossing the tile edge
            arrive whole and are cut at the boundary here, rather than ending
            wherever OSM's own bbox filter truncated them.

    Returns:
        One entry per clipped piece. A way that leaves the tile and returns
        yields several, each carrying the same tag.
    """
    raw = ox.graph_from_bbox(
        tiles.lonlat_bounds(tile, pad_m=pad_m), network_type=network_type
    )
    raw = ox.project_graph(raw, to_crs=tile.crs)
    # OSM arrives directed; a two-way street is two edges and would otherwise
    # be rasterized and scored twice.
    raw = ox.convert.to_undirected(raw)

    region = box(0, 0, tile.width, tile.height)
    ways: list[TaggedWay] = []
    for (*_, data), world_pts in zip(
        raw.edges(data=True), _edge_polylines(raw), strict=True
    ):
        highway, multi_valued = _highway_value(data)
        ways.extend(
            TaggedWay(highway=highway, multi_valued=multi_valued, pts=pts)
            for pts in _clip(tiles.world_to_px(tile, world_pts), region)
        )
    return tuple(ways)


@dataclass(frozen=True, slots=True)
class WayExtract:
    """Raw OSM ways read out of a local extract, in EPSG:4326.

    Held whole and clipped per tile: a bbox read of the extract costs about
    four seconds whatever the extent, so one read serves a whole run and a
    per-tile read would cost that four seconds 155 times over.

    Attributes:
        lines: One WGS84 ``LineString`` per way.
        highway: Each way's ``highway`` value, aligned with ``lines``.
        passes: Per key of :data:`NETWORK_FILTERS`, whether each way survives
            that filter. Computed once at read because the filters are regex
            matches over every way, which is too slow to repeat per tile.
        osm_id: Each way's OSM id, aligned with ``lines``.
        bounds: ``(N, 4)`` of ``(minx, miny, maxx, maxy)`` per way, so the ways
            near a tile are one vectorized comparison.
        extent: The ``(west, south, east, north)`` degrees that were read. A
            tile outside it holds no ways for a reason that has nothing to do
            with the ground, so :func:`pbf_ways` refuses rather than returning
            an empty answer.
        path: Extract the ways came from.
    """

    lines: np.ndarray
    highway: np.ndarray
    osm_id: np.ndarray
    passes: Mapping[str, np.ndarray]
    bounds: np.ndarray
    extent: tuple[float, float, float, float]
    path: Path


def network_mask(
    highway: np.ndarray, tags: Mapping[str, np.ndarray], network_type: str
) -> np.ndarray:
    """Which ways one network filter keeps.

    Args:
        highway: ``(N,)`` of each way's ``highway`` value.
        tags: Per key of :data:`FILTER_TAG_KEYS`, ``(N,)`` of that tag's value,
            empty where the way does not carry it.
        network_type: A key of :data:`NETWORK_FILTERS`.

    Returns:
        ``(N,)`` boolean, true where the way survives the filter.

    Raises:
        ValueError: If ``network_type`` is not one of the replicated filters.
    """
    if network_type not in NETWORK_FILTERS:
        raise ValueError(
            f"unknown network type {network_type!r}; the local extract "
            f"replicates {sorted(NETWORK_FILTERS)}"
        )

    keep = np.ones(len(highway), dtype=bool)
    for key, pattern in NETWORK_FILTERS[network_type].exclusions:
        values = highway if key == "highway" else tags[key]
        search = re.compile(pattern).search
        # An absent tag satisfies Overpass's `!~`, so "" keeps the way.
        keep &= np.fromiter(
            (not value or search(value) is None for value in values),
            dtype=bool,
            count=len(values),
        )
    return keep


def _parse_other_tags(other_tags: np.ndarray, key: str) -> np.ndarray:
    """One tag's value per way, pulled out of GDAL's ``other_tags`` HSTORE."""
    search = re.compile(f'"{re.escape(key)}"=>"([^"]*)"').search
    return np.array(
        [
            match.group(1) if text and (match := search(text)) else ""
            for text in other_tags
        ],
        dtype=object,
    )


def read_extract(
    bbox: tuple[float, float, float, float], path: Path | str = DEFAULT_EXTRACT
) -> WayExtract:
    """Read every tagged way in an extent out of a local ``.osm.pbf``.

    Args:
        bbox: ``(west, south, east, north)`` in degrees, which is the order
            :func:`tiles.lonlat_bounds` returns and the order GDAL wants. Use
            :func:`covering_bounds` to cover a whole run's tiles at once.
        path: The extract to read. Also reads ``.osm`` XML, which is what the
            tests build fixtures in.

    Returns:
        The ways, their classes, and the filter masks, all in EPSG:4326.
    """
    started = time.perf_counter()
    _, _, geometry, fields = read_ogr(
        str(path),
        layer=EXTRACT_LAYER,
        bbox=bbox,
        columns=["osm_id", "highway", "other_tags"],
    )
    osm_id, highway, other_tags = fields

    lines = shapely.from_wkb(geometry)
    # A way with no `highway` is a railway, a wall or a stream. A way with
    # fewer than two coordinates has no geometry to place, and would otherwise
    # reach `node_network` as an empty line.
    keep = np.array([value is not None for value in highway], dtype=bool)
    keep &= shapely.get_num_coordinates(lines) >= 2
    lines = lines[keep]
    highway = highway[keep].astype(object)
    osm_id = osm_id[keep].astype(object)
    tags = {key: _parse_other_tags(other_tags[keep], key) for key in FILTER_TAG_KEYS}

    extract = WayExtract(
        lines=lines,
        highway=highway,
        osm_id=osm_id,
        passes=MappingProxyType(
            {key: network_mask(highway, tags, key) for key in NETWORK_FILTERS}
        ),
        bounds=shapely.bounds(lines),
        extent=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
        path=Path(path),
    )
    logger.info(
        f"{path}: {len(lines)} tagged ways in {bbox} in "
        f"{time.perf_counter() - started:.1f} s ("
        + ", ".join(f"{key} {int(mask.sum())}" for key, mask in extract.passes.items())
        + ")"
    )
    return extract


def covering_bounds(
    tile_seq: Sequence[Tile], pad_m: float = 250.0
) -> tuple[float, float, float, float]:
    """The lon/lat extent covering every tile, so one read serves them all.

    Args:
        tile_seq: Tiles the run will ask for. May span several UTM zones.
        pad_m: Margin each tile's extent is expanded by, matching the ``pad_m``
            the ways will be requested with.

    Returns:
        ``(west, south, east, north)`` in degrees.

    Raises:
        ValueError: If no tiles were given, which has no extent.
    """
    if not tile_seq:
        raise ValueError("no tiles to cover")
    corners = np.array([tiles.lonlat_bounds(t, pad_m=pad_m) for t in tile_seq])
    return (
        float(corners[:, 0].min()),
        float(corners[:, 1].min()),
        float(corners[:, 2].max()),
        float(corners[:, 3].max()),
    )


def _to_pixels(lines: np.ndarray, tile: Tile) -> list[LineString]:
    """Reproject WGS84 lines into a tile's pixel frame, all coordinates at once."""
    coords = shapely.get_coordinates(lines)
    to_tile = Transformer.from_crs(tiles.WGS84, tile.crs, always_xy=True)
    xs, ys = to_tile.transform(coords[:, 0], coords[:, 1])
    px = tiles.world_to_px(tile, np.column_stack([xs, ys]))
    ends = np.cumsum(shapely.get_num_coordinates(lines))[:-1]
    return [LineString(part) for part in np.split(px, ends)]


def node_sources(lines: Sequence[LineString]) -> list[tuple[int, np.ndarray]]:
    """Split lines where they meet, saying which input way each piece came from.

    :func:`geograph.node_network` dissolves its input into one geometry, which
    loses every per-way attribute. Each piece is a run of exactly one input
    way, so the source is recovered by asking which way a point inside the
    piece lies on. Collinear ways drawn twice are the exception: those dissolve
    into one piece, and it takes whichever of them the index reports first.

    Args:
        lines: Ways in tile pixel coordinates.

    Returns:
        ``(index into lines, polyline)`` per piece.
    """
    pieces = geograph.node_network(list(lines))
    if not pieces:
        return []

    # A point inside the piece's first segment: on its source way by
    # construction, and off the junctions where several ways share a point.
    probes = shapely.points(np.array([(p[0] + p[1]) / 2.0 for p in pieces]))
    probe_index, line_index = STRtree(list(lines)).query_nearest(
        probes, all_matches=False
    )
    source = np.zeros(len(pieces), dtype=int)
    source[probe_index] = line_index
    return [(int(source[i]), piece) for i, piece in enumerate(pieces)]


def pbf_ways(
    extract: WayExtract,
    tile: Tile,
    network_type: str = "drive",
    pad_m: float = 250.0,
) -> tuple[TaggedWay, ...]:
    """Clip a local extract to a tile, noding the ways as they arrive.

    The noding is the difference between this source and :func:`tagged_ways`,
    and leaving it out fails silently. A raw OSM way runs straight through the
    intersections along it, so a side street's endpoint lands on a main road's
    *interior*: a junction on the ground that shares no vertex.
    :func:`geograph.build` fuses coincident *endpoints* only, so without this
    pass the graph comes back as disconnected stubs whose length is right and
    whose topology is not.

    Args:
        extract: Ways already read, from :func:`read_extract`.
        tile: Tile to clip to.
        network_type: A key of :data:`NETWORK_FILTERS`.
        pad_m: Margin on the extent taken from the extract, so ways crossing
            the tile edge are noded whole and cut at the boundary afterwards.

    Returns:
        One entry per clipped piece, in tile pixel coordinates.

    Raises:
        ValueError: If the tile is not inside the extent that was read, which
            would otherwise return an empty answer that reads as "no roads
            here" rather than "these roads were never loaded".
    """
    west, south, east, north = tiles.lonlat_bounds(tile, pad_m=pad_m)
    ex_west, ex_south, ex_east, ex_north = extract.extent
    if west < ex_west or south < ex_south or east > ex_east or north > ex_north:
        raise ValueError(
            f"tile extent ({west}, {south}, {east}, {north}) reaches outside "
            f"{extract.path} as read over {extract.extent}"
        )

    if network_type not in extract.passes:
        raise ValueError(
            f"unknown network type {network_type!r}; the local extract "
            f"replicates {sorted(extract.passes)}"
        )
    keep = extract.passes[network_type] & (
        (extract.bounds[:, 0] <= east)
        & (extract.bounds[:, 2] >= west)
        & (extract.bounds[:, 1] <= north)
        & (extract.bounds[:, 3] >= south)
    )
    if not keep.any():
        return ()

    region = box(0, 0, tile.width, tile.height)
    highway = extract.highway[keep]
    osm_id = extract.osm_id[keep]
    return tuple(
        # A way in an extract carries one `highway` value; the several-valued
        # case is osmnx's simplification merging ways, not something OSM ships.
        TaggedWay(
            highway=str(highway[source]),
            multi_valued=False,
            pts=pts,
            osm_id=str(osm_id[source]),
        )
        for source, piece in node_sources(_to_pixels(extract.lines[keep], tile))
        for pts in _clip(piece, region)
    )


@runtime_checkable
class WaySource(Protocol):
    """Where a tile's OSM ways come from."""

    def ways(
        self, tile: Tile, network_type: str = "drive", pad_m: float = 250.0
    ) -> tuple[TaggedWay, ...]:
        """Return the ways covering a tile, clipped to it."""
        ...


class OverpassWaySource:
    """Ways from the live Overpass API, through osmnx."""

    def ways(
        self, tile: Tile, network_type: str = "drive", pad_m: float = 250.0
    ) -> tuple[TaggedWay, ...]:
        """Query Overpass for one tile; see :func:`tagged_ways`."""
        return tagged_ways(tile, network_type, pad_m)


class PbfWaySource:
    """Ways from a local extract, read once and clipped per tile."""

    def __init__(self, extract: WayExtract) -> None:
        """
        Args:
            extract: Ways covering every tile this source will be asked for,
                from :func:`read_extract`. Passed in rather than loaded here so
                one read is shared explicitly across a run instead of hiding in
                a module-level cache.
        """
        self.extract = extract

    def ways(
        self, tile: Tile, network_type: str = "drive", pad_m: float = 250.0
    ) -> tuple[TaggedWay, ...]:
        """Clip the extract to one tile; see :func:`pbf_ways`."""
        return pbf_ways(self.extract, tile, network_type, pad_m)


_: type[WaySource] = OverpassWaySource
_: type[WaySource] = PbfWaySource

#: Selects a way source by config key. Values are factories, so each lookup
#: yields a fresh instance rather than a shared one.
WAY_SOURCE_REGISTRY: dict[str, Callable[..., WaySource]] = {
    "overpass": lambda **kwargs: OverpassWaySource(**kwargs),
    "pbf": lambda **kwargs: PbfWaySource(**kwargs),
}


def graph_from_ways(ways: tuple[TaggedWay, ...], tile: Tile) -> nx.MultiGraph:
    """Assemble clipped ways into a graph in the :mod:`geograph` conventions.

    Does not node. Both sources hand over ways that are already split at their
    junctions, by osmnx upstream or by :func:`pbf_ways` on the way through, so
    noding here would additionally weld grade separations -- a bridge and the
    road under it cross in plan and meet nowhere.
    """
    G = geograph.build([way.pts for way in ways], tol=1e-3)
    G.graph["resolution"] = tile.resolution
    return G


def ground_truth_graph(
    tile: Tile,
    network_type: str = "drive",
    pad_m: float = 250.0,
    source: WaySource | None = None,
) -> nx.MultiGraph:
    """Fetch OSM roads covering a tile and return them in pixel coordinates.

    Args:
        tile: Tile to fetch roads for.
        network_type: Network filter, e.g. ``"drive"`` or ``"all"``.
        pad_m: Margin added to the query extent so roads crossing the tile edge
            arrive whole and are cut at the boundary here, rather than ending
            wherever OSM's own bbox filter truncated them.
        source: Where the ways come from; see :data:`WAY_SOURCE_REGISTRY`.
            Defaults to Overpass, which needs no local file.

    Returns:
        An undirected graph in the conventions described in :mod:`geograph`.
    """
    ways = (source or OverpassWaySource()).ways(tile, network_type, pad_m)
    return graph_from_ways(ways, tile)

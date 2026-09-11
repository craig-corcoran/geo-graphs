import re
from pathlib import Path
from types import MappingProxyType

import networkx as nx
import numpy as np
import pytest
import shapely
from osmnx import _overpass
from pyproj import Transformer
from shapely.geometry import LineString, box

from geo_graphs import geograph, tiles
from geo_graphs.osm import (
    DEFAULT_EXTRACT,
    FILTER_TAG_KEYS,
    NETWORK_FILTERS,
    UNTAGGED,
    WAY_SOURCE_REGISTRY,
    OverpassWaySource,
    PbfWaySource,
    TaggedWay,
    WayExtract,
    WaySource,
    _clip,
    _edge_polylines,
    _highway_value,
    _parse_other_tags,
    _to_pixels,
    covering_bounds,
    graph_from_ways,
    ground_truth_graph,
    network_mask,
    node_sources,
    pbf_ways,
    read_extract,
    tagged_ways,
)

UNIT_SQUARE = box(0, 0, 100, 100)

VEGAS = (36.1699, -115.1398)


def vegas_tile(size_m: float = 512.0):
    """The tile every local-extract test is expressed in pixels on."""
    return tiles.tile_from_center(*VEGAS, size_m=size_m)


def wgs84_line(tile, px_pts) -> LineString:
    """A WGS84 line that lands on given pixel coordinates of a tile.

    Writing the fixtures in pixels keeps the geometry legible: a junction at
    the tile centre is ``(256, 256)`` rather than six decimal places of
    longitude.
    """
    world = tiles.px_to_world(tile, np.asarray(px_pts, dtype=float))
    to_wgs = Transformer.from_crs(tile.crs, tiles.WGS84, always_xy=True)
    lons, lats = to_wgs.transform(world[:, 0], world[:, 1])
    return LineString(np.column_stack([lons, lats]))


def synthetic_extract(tile, ways, pad_m: float = 250.0) -> WayExtract:
    """A :class:`WayExtract` built by hand, so no ``.osm.pbf`` is needed.

    Args:
        tile: Tile the pixel coordinates are expressed on.
        ways: ``(highway, pixel polyline)`` pairs, or
            ``(highway, pixel polyline, extra tags)``.
        pad_m: Margin the recorded extent is taken at, matching what
            :func:`pbf_ways` will be called with.
    """
    lines = np.array([wgs84_line(tile, w[1]) for w in ways], dtype=object)
    highway = np.array([w[0] for w in ways], dtype=object)
    tags = {
        key: np.array(
            [(w[2].get(key, "") if len(w) > 2 else "") for w in ways], dtype=object
        )
        for key in FILTER_TAG_KEYS
    }
    return WayExtract(
        lines=lines,
        highway=highway,
        osm_id=np.array([str(i) for i in range(len(ways))], dtype=object),
        passes=MappingProxyType(
            {key: network_mask(highway, tags, key) for key in NETWORK_FILTERS}
        ),
        bounds=shapely.bounds(lines),
        extent=tiles.lonlat_bounds(tile, pad_m=pad_m),
        path=Path("synthetic"),
    )


def write_osm(path: Path, ways) -> Path:
    """Write a hand-built ``.osm`` XML file GDAL's OSM driver can read.

    The driver is read-only, so a fixture in its own format has to be XML
    rather than protobuf. Both go through the same driver and the same
    ``lines`` layer, which is what :func:`read_extract` reads.

    Args:
        path: File to write; the ``.osm`` suffix is what selects the driver.
        ways: ``(highway, [(lat, lon), ...], extra tags)`` per way. A
            ``highway`` of ``None`` writes no such tag, which is how a way that
            is not a road at all reaches the ``lines`` layer.
    """
    nodes: list[str] = []
    elements: list[str] = []
    for way_id, (highway, coords, extra) in enumerate(ways, start=1):
        refs = []
        for lat, lon in coords:
            node_id = len(nodes) + 1
            nodes.append(
                f'  <node id="{node_id}" lat="{float(lat)!r}" lon="{float(lon)!r}"/>'
            )
            refs.append(f'    <nd ref="{node_id}"/>')
        pairs = ({} if highway is None else {"highway": highway}) | extra
        tags = [f'    <tag k="{k}" v="{v}"/>' for k, v in pairs.items()]
        elements.append(
            f'  <way id="{1000 + way_id}">\n' + "\n".join(refs + tags) + "\n  </way>"
        )
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6" generator="test">\n'
        + "\n".join(nodes + elements)
        + "\n</osm>\n"
    )
    return path


def test_edge_polylines_uses_stored_geometry():
    G = nx.MultiGraph()
    G.add_node(0, x=0.0, y=0.0)
    G.add_node(1, x=10.0, y=0.0)
    curve = LineString([(0, 0), (5, 3), (10, 0)])
    G.add_edge(0, 1, geometry=curve)

    (pts,) = _edge_polylines(G)
    assert pts.tolist() == [[0, 0], [5, 3], [10, 0]]


def test_edge_polylines_falls_back_to_a_straight_run():
    """OSM omits `geometry` on edges that are straight between their nodes."""
    G = nx.MultiGraph()
    G.add_node(0, x=0.0, y=0.0)
    G.add_node(1, x=10.0, y=4.0)
    G.add_edge(0, 1)

    (pts,) = _edge_polylines(G)
    assert pts.tolist() == [[0, 0], [10, 4]]


def test_clip_keeps_an_interior_line_whole():
    pts = np.array([[10.0, 10.0], [90.0, 90.0]])
    (kept,) = _clip(pts, UNIT_SQUARE)
    assert geograph.polyline_length(kept) == pytest.approx(geograph.polyline_length(pts))


def test_clip_drops_a_line_entirely_outside():
    assert _clip(np.array([[200.0, 200.0], [300.0, 300.0]]), UNIT_SQUARE) == []


def test_clip_truncates_a_line_crossing_the_boundary():
    kept = _clip(np.array([[50.0, 50.0], [500.0, 50.0]]), UNIT_SQUARE)
    assert len(kept) == 1
    assert kept[0][-1][0] == pytest.approx(100.0)


def test_clip_splits_a_line_that_re_enters():
    """A road leaving and returning becomes two edges, not one with a gap."""
    pts = np.array([[50.0, 50.0], [50.0, 200.0], [80.0, 200.0], [80.0, 50.0]])
    assert len(_clip(pts, UNIT_SQUARE)) == 2


@pytest.mark.network
def test_ground_truth_graph_is_clipped_to_the_tile():
    tile = tiles.tile_from_center(36.1699, -115.1398, size_m=512)
    G = ground_truth_graph(tile)

    assert G.number_of_edges() > 0
    for u, v, k in G.edges(keys=True):
        pts = geograph.oriented_pts(G, u, v, k)
        assert pts[:, 0].min() >= -1.0
        assert pts[:, 1].min() >= -1.0
        assert pts[:, 0].max() <= tile.width + 1.0
        assert pts[:, 1].max() <= tile.height + 1.0


@pytest.mark.network
def test_ground_truth_graph_records_resolution():
    tile = tiles.tile_from_center(36.1699, -115.1398, size_m=512)
    assert ground_truth_graph(tile).graph["resolution"] == tile.resolution


def test_highway_value_reads_a_single_tag():
    assert _highway_value({"highway": "residential"}) == ("residential", False)


def test_highway_value_takes_the_first_of_a_merged_run():
    """osmnx simplification merges ways, and a disagreeing run arrives as a list."""
    assert _highway_value({"highway": ["service", "residential"]}) == ("service", True)


def test_highway_value_names_an_absent_tag():
    assert _highway_value({}) == (UNTAGGED, False)
    assert _highway_value({"highway": []}) == (UNTAGGED, True)


@pytest.mark.network
def test_tagged_ways_carry_a_highway_value():
    tile = tiles.tile_from_center(36.1699, -115.1398, size_m=512)
    ways = tagged_ways(tile, "drive")

    assert len(ways) > 0
    assert all(way.highway for way in ways)
    assert all(len(way.pts) >= 2 for way in ways)


@pytest.mark.network
def test_tagged_ways_rebuild_the_ground_truth_graph():
    """The tagged path is the same geometry, so the graph it builds is the same."""
    tile = tiles.tile_from_center(36.1699, -115.1398, size_m=512)
    rebuilt = graph_from_ways(tagged_ways(tile, "drive"), tile)
    direct = ground_truth_graph(tile)

    assert rebuilt.number_of_edges() == direct.number_of_edges()
    assert geograph.total_length(rebuilt) == pytest.approx(geograph.total_length(direct))


def tag_columns(highway, **tags):
    """Filter-tag columns for a hand-built set of ways, absent tags as ``""``."""
    return {
        key: np.array(tags.get(key, [""] * len(highway)), dtype=object)
        for key in FILTER_TAG_KEYS
    }


def test_network_filters_match_osmnx():
    """The whole point of two sources is that they select the same ways.

    Parsed out of osmnx's own filter strings rather than restated, so a change
    on their side fails here instead of quietly moving one source's road length.
    """
    for network_type, ours in NETWORK_FILTERS.items():
        theirs = _overpass._get_network_filter(network_type)
        assert set(re.findall(r'\["([^"]+)"!~"([^"]+)"\]', theirs)) == set(
            ours.exclusions
        )
        # Only the highway-presence test survives stripping the negations, so a
        # condition osmnx adds later cannot slip past unreplicated.
        assert re.sub(r'\["[^"]+"!~"[^"]+"\]', "", theirs) == '["highway"]'


def test_network_mask_keeps_a_plain_residential_street():
    highway = np.array(["residential"], dtype=object)
    assert network_mask(highway, tag_columns(highway), "drive").tolist() == [True]
    assert network_mask(highway, tag_columns(highway), "all").tolist() == [True]


def test_network_mask_drops_service_from_drive_but_not_from_all():
    """Service roads are most of what SpaceNet labels beyond the arterials."""
    highway = np.array(["service", "footway", "residential"], dtype=object)
    columns = tag_columns(highway)
    assert network_mask(highway, columns, "drive").tolist() == [False, False, True]
    assert network_mask(highway, columns, "all").tolist() == [True, True, True]


def test_network_mask_drops_a_private_way_from_drive_only():
    """`all` carries no access condition, so the two networks disagree here."""
    highway = np.array(["residential", "residential"], dtype=object)
    columns = tag_columns(highway, access=["private", "permissive"])
    assert network_mask(highway, columns, "drive").tolist() == [False, True]
    assert network_mask(highway, columns, "all").tolist() == [True, True]


def test_network_mask_reads_the_service_tag_on_a_non_service_road():
    highway = np.array(["residential", "residential"], dtype=object)
    columns = tag_columns(highway, service=["parking_aisle", ""])
    assert network_mask(highway, columns, "drive").tolist() == [False, True]


def test_network_mask_treats_an_absent_tag_as_passing():
    """Overpass's `!~` is satisfied by a missing tag, not only by a differing one."""
    highway = np.array(["residential"], dtype=object)
    columns = tag_columns(highway, motor_vehicle=[""])
    assert network_mask(highway, columns, "drive").tolist() == [True]


def test_network_mask_rejects_a_network_type_the_extract_does_not_replicate():
    highway = np.array(["residential"], dtype=object)
    with pytest.raises(ValueError, match="walk"):
        network_mask(highway, tag_columns(highway), "walk")


def test_parse_other_tags_reads_one_key():
    text = np.array(['"access"=>"private","surface"=>"asphalt"'], dtype=object)
    assert _parse_other_tags(text, "access").tolist() == ["private"]


def test_parse_other_tags_does_not_match_a_longer_key_ending_in_the_same_word():
    text = np.array(['"bicycle:access"=>"yes"'], dtype=object)
    assert _parse_other_tags(text, "access").tolist() == [""]


def test_parse_other_tags_handles_a_way_carrying_none():
    assert _parse_other_tags(np.array([None], dtype=object), "access").tolist() == [""]


def test_covering_bounds_spans_every_tile():
    east = tiles.tile_from_center(36.1699, -115.1000, size_m=512)
    west = tiles.tile_from_center(36.1699, -115.2000, size_m=512)
    covered = covering_bounds([east, west], pad_m=250.0)

    for tile in (east, west):
        w, s, e, n = tiles.lonlat_bounds(tile, pad_m=250.0)
        assert covered[0] <= w and covered[1] <= s
        assert covered[2] >= e and covered[3] >= n


def test_covering_bounds_refuses_an_empty_run():
    with pytest.raises(ValueError, match="no tiles"):
        covering_bounds([])


def test_to_pixels_round_trips_the_pixel_frame():
    tile = vegas_tile()
    pts = [[10.0, 20.0], [256.0, 256.0], [500.0, 400.0]]
    (line,) = _to_pixels(np.array([wgs84_line(tile, pts)], dtype=object), tile)
    assert np.asarray(line.coords) == pytest.approx(np.asarray(pts), abs=1e-4)


def test_node_sources_splits_a_crossing_and_keeps_both_sources():
    horizontal = LineString([(50.0, 256.0), (450.0, 256.0)])
    vertical = LineString([(256.0, 50.0), (256.0, 450.0)])
    pieces = node_sources([horizontal, vertical])

    assert len(pieces) == 4
    assert sorted(source for source, _ in pieces) == [0, 0, 1, 1]
    assert sum(geograph.polyline_length(pts) for _, pts in pieces) == pytest.approx(800.0)


def test_node_sources_leaves_ways_that_never_meet_alone():
    pieces = node_sources(
        [LineString([(0.0, 0.0), (10.0, 0.0)]), LineString([(0.0, 50.0), (10.0, 50.0)])]
    )
    assert sorted(source for source, _ in pieces) == [0, 1]


def test_node_sources_of_nothing_is_nothing():
    assert node_sources([]) == []


def test_graph_from_ways_does_not_node_its_input():
    """The asymmetry, pinned: noding belongs to the source, not to the assembly.

    Overpass hands over an already-noded graph, so noding here would weld grade
    separations together. The local extract does not, so `pbf_ways` nodes.
    """
    tile = vegas_tile()
    crossing = tuple(
        TaggedWay(highway="residential", multi_valued=False, pts=np.array(pts))
        for pts in (
            [[50.0, 256.0], [450.0, 256.0]],
            [[256.0, 50.0], [256.0, 450.0]],
        )
    )
    G = graph_from_ways(crossing, tile)

    assert G.number_of_edges() == 2
    assert nx.number_connected_components(G) == 2


def test_pbf_ways_nodes_the_junction_a_raw_way_runs_straight_through():
    tile = vegas_tile()
    extract = synthetic_extract(
        tile,
        [
            ("residential", [[50.0, 256.0], [450.0, 256.0]]),
            ("residential", [[256.0, 50.0], [256.0, 450.0]]),
        ],
    )
    ways = pbf_ways(extract, tile, "all")
    G = graph_from_ways(ways, tile)

    assert len(ways) == 4
    assert G.number_of_edges() == 4
    assert G.number_of_nodes() == 5
    assert nx.number_connected_components(G) == 1
    assert geograph.total_length(G) == pytest.approx(800.0, abs=1e-3)


def test_pbf_ways_applies_the_network_filter():
    tile = vegas_tile()
    extract = synthetic_extract(
        tile,
        [
            ("residential", [[50.0, 256.0], [450.0, 256.0]]),
            ("service", [[256.0, 50.0], [256.0, 450.0]]),
        ],
    )
    assert len(pbf_ways(extract, tile, "all")) == 4
    drive = pbf_ways(extract, tile, "drive")
    assert [way.highway for way in drive] == ["residential"]
    assert geograph.polyline_length(drive[0].pts) == pytest.approx(400.0, abs=1e-3)


def test_pbf_ways_clips_to_the_tile():
    tile = vegas_tile()
    extract = synthetic_extract(tile, [("primary", [[-200.0, 256.0], [700.0, 256.0]])])
    (way,) = pbf_ways(extract, tile, "drive")
    assert geograph.polyline_length(way.pts) == pytest.approx(tile.width, abs=1e-3)


def test_pbf_ways_carries_one_highway_value_per_way():
    """`multi_valued` is an osmnx simplification artifact, absent from raw ways."""
    tile = vegas_tile()
    extract = synthetic_extract(tile, [("primary", [[50.0, 256.0], [450.0, 256.0]])])
    assert not any(way.multi_valued for way in pbf_ways(extract, tile, "drive"))


def test_pbf_ways_refuses_a_tile_the_extract_never_covered():
    """An uncovered tile is empty for a reason that is not about the ground."""
    tile = vegas_tile()
    extract = synthetic_extract(tile, [("primary", [[50.0, 256.0], [450.0, 256.0]])])
    reno = tiles.tile_from_center(39.5296, -119.8138, size_m=512)

    with pytest.raises(ValueError, match="reaches outside"):
        pbf_ways(extract, reno, "drive")


def test_pbf_ways_is_empty_where_the_extract_holds_no_road():
    tile = vegas_tile()
    extract = synthetic_extract(tile, [("footway", [[50.0, 256.0], [450.0, 256.0]])])
    assert pbf_ways(extract, tile, "drive") == ()


def test_read_extract_keeps_tagged_ways_and_their_filters(tmp_path):
    path = write_osm(
        tmp_path / "tiny.osm",
        [
            ("residential", [(36.1690, -115.1410), (36.1690, -115.1390)], {}),
            (
                "service",
                [(36.1700, -115.1400), (36.1680, -115.1400)],
                {"service": "parking_aisle", "access": "private"},
            ),
            ("footway", [(36.1695, -115.1405), (36.1695, -115.1395)], {}),
            (None, [(36.1685, -115.1405), (36.1685, -115.1395)], {"waterway": "ditch"}),
        ],
    )
    extract = read_extract((-115.15, 36.16, -115.13, 36.18), path)
    drive = dict(zip(extract.highway, extract.passes["drive"], strict=True))
    keeps_all = dict(zip(extract.highway, extract.passes["all"], strict=True))

    # The ditch carries no `highway`, so it is not a way this module has an
    # opinion about and never reaches the arrays.
    assert sorted(extract.highway) == ["footway", "residential", "service"]
    assert drive == {"residential": True, "service": False, "footway": False}
    assert keeps_all == {"residential": True, "service": True, "footway": True}
    assert extract.extent == (-115.15, 36.16, -115.13, 36.18)
    assert extract.path == path


def test_read_extract_geometry_reaches_a_tile(tmp_path):
    """A way written to a file comes back on the pixel it was placed on.

    Read, reprojection and pixel conversion in one assertion: the fixture is
    written from pixel coordinates and has to return to them.
    """
    tile = vegas_tile(size_m=2048.0)
    placed = [[100.0, 1024.0], [900.0, 700.0], [1900.0, 1024.0]]
    lonlat = np.asarray(wgs84_line(tile, placed).coords)
    path = write_osm(
        tmp_path / "tiny.osm",
        [("primary", [(lat, lon) for lon, lat in lonlat], {})],
    )
    extract = read_extract(tiles.lonlat_bounds(tile, pad_m=250.0), path)
    (way,) = pbf_ways(extract, tile, "drive")

    # A pixel is a metre here, and OSM stores coordinates to 1e-7 degrees, so
    # the round trip is exact to about a centimetre and no further.
    assert way.pts == pytest.approx(np.asarray(placed), abs=0.02)


def test_pbf_ways_carry_the_id_of_the_way_they_were_cut_from(tmp_path):
    """Noding splits a way at every junction; the pieces still name their way.

    The id is what groups pieces back into the way a mapper drew, so an
    analysis that treats pieces as independent samples can tell that they
    are not.
    """
    tile = vegas_tile(size_m=2048.0)
    across = [[100.0, 1024.0], [1900.0, 1024.0]]
    down = [[1024.0, 100.0], [1024.0, 1900.0]]
    path = write_osm(
        tmp_path / "cross.osm",
        [
            ("primary", [(lat, lon) for lon, lat in wgs84_line(tile, across).coords], {}),
            (
                "residential",
                [(lat, lon) for lon, lat in wgs84_line(tile, down).coords],
                {},
            ),
        ],
    )
    extract = read_extract(tiles.lonlat_bounds(tile, pad_m=250.0), path)
    ways = pbf_ways(extract, tile, "drive")

    # write_osm numbers its ways from 1001, in the order they were given.
    by_id = {way.osm_id for way in ways}
    assert len(ways) == 4
    assert by_id == {"1001", "1002"}
    assert {w.highway for w in ways if w.osm_id == "1001"} == {"primary"}
    assert {w.highway for w in ways if w.osm_id == "1002"} == {"residential"}


def test_way_source_registry_holds_factories():
    assert sorted(WAY_SOURCE_REGISTRY) == ["overpass", "pbf"]
    first, second = WAY_SOURCE_REGISTRY["overpass"](), WAY_SOURCE_REGISTRY["overpass"]()
    assert first is not second
    assert isinstance(first, WaySource)
    assert isinstance(first, OverpassWaySource)


def test_pbf_source_reaches_the_same_ways_through_the_registry():
    tile = vegas_tile()
    extract = synthetic_extract(
        tile,
        [
            ("residential", [[50.0, 256.0], [450.0, 256.0]]),
            ("residential", [[256.0, 50.0], [256.0, 450.0]]),
        ],
    )
    source = WAY_SOURCE_REGISTRY["pbf"](extract=extract)

    assert isinstance(source, PbfWaySource)
    assert len(source.ways(tile, "all")) == len(pbf_ways(extract, tile, "all"))


def test_ground_truth_graph_accepts_a_local_source():
    tile = vegas_tile()
    extract = synthetic_extract(
        tile,
        [
            ("residential", [[50.0, 256.0], [450.0, 256.0]]),
            ("residential", [[256.0, 50.0], [256.0, 450.0]]),
        ],
    )
    G = ground_truth_graph(tile, "all", source=PbfWaySource(extract))

    assert G.graph["resolution"] == tile.resolution
    assert G.number_of_edges() == 4


@pytest.mark.skipif(
    not DEFAULT_EXTRACT.exists(), reason=f"{DEFAULT_EXTRACT} is not on disk"
)
def test_local_extract_gives_a_connected_vegas_network():
    """The real extract, end to end: read once, clip, node, assemble.

    Downtown Las Vegas is a connected grid, so a fragmented answer here is the
    noding failing, not the ground.
    """
    tile = vegas_tile()
    extract = read_extract(covering_bounds([tile]), DEFAULT_EXTRACT)
    drive = pbf_ways(extract, tile, "drive")
    every = pbf_ways(extract, tile, "all")
    G = graph_from_ways(drive, tile)

    assert len(drive) > 0
    assert len(every) > len(drive)
    for way in every:
        assert way.pts[:, 0].min() >= -1.0 and way.pts[:, 0].max() <= tile.width + 1.0
        assert way.pts[:, 1].min() >= -1.0 and way.pts[:, 1].max() <= tile.height + 1.0

    components = list(nx.connected_components(G))
    largest = max(
        geograph.total_length(G.subgraph(component)) for component in components
    )
    assert largest >= 0.8 * geograph.total_length(G)


@pytest.mark.network
@pytest.mark.skipif(
    not DEFAULT_EXTRACT.exists(), reason=f"{DEFAULT_EXTRACT} is not on disk"
)
def test_the_two_sources_agree_on_one_tile():
    """Overpass against the local extract, on the same tile and filter.

    The assertion is about noding. osmnx returns an already-noded graph and
    keeps only its largest component, so it is connected by construction; the
    extract's ways are raw and are noded on the way through. If that pass were
    missing the local graph would come back in dozens of pieces at roughly the
    same total length, which is exactly the failure that would make the model's
    dead-end stubs look like real unlabelled roads.
    """
    tile = vegas_tile()
    extract = read_extract(covering_bounds([tile]), DEFAULT_EXTRACT)
    local = graph_from_ways(pbf_ways(extract, tile, "drive"), tile)
    remote = graph_from_ways(tagged_ways(tile, "drive"), tile)

    assert geograph.total_length(local) == pytest.approx(
        geograph.total_length(remote), rel=0.4
    )
    assert (
        nx.number_connected_components(local)
        <= nx.number_connected_components(remote) + 3
    )

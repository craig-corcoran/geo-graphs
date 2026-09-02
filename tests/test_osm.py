import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, box

from geo_graphs import geograph, tiles
from geo_graphs.osm import _clip, _edge_polylines, ground_truth_graph

UNIT_SQUARE = box(0, 0, 100, 100)


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

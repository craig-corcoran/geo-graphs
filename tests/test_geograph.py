import networkx as nx
import numpy as np
import pytest

from geo_graphs import geograph
from tests.conftest import curvy_graph, grid_graph


def test_split_at_preserves_length():
    curvy = curvy_graph()
    for u, v, k in curvy.edges(keys=True):
        pts = geograph.oriented_pts(curvy, u, v, k)
        total = geograph.polyline_length(pts)
        pieces = geograph.split_at(pts, [total * f for f in (0.25, 0.5, 0.75)])
        assert sum(geograph.polyline_length(p) for p in pieces) == pytest.approx(total)


def test_oriented_pts_starts_at_u():
    """Regression: an undirected MultiGraph may report an edge either way round.

    Reading data['pts'] directly and assuming it starts at u silently produces
    mirrored geometry, which is invisible in edge lengths but wrecks anything
    that walks the edge from one endpoint to the other.
    """
    curvy = curvy_graph()
    for u, v, k in curvy.edges(keys=True):
        pts = geograph.oriented_pts(curvy, u, v, k)
        assert pts[0] == pytest.approx(curvy.nodes[u]["pos"])
        assert pts[-1] == pytest.approx(curvy.nodes[v]["pos"])


@pytest.mark.parametrize("sampling", ["uniform", "reference"])
def test_densify_preserves_length_and_connectivity(sampling):
    curvy = curvy_graph()
    D = geograph.densify(curvy, 25.0, sampling=sampling)
    assert geograph.total_length(D) == pytest.approx(geograph.total_length(curvy))
    assert nx.number_connected_components(D) == nx.number_connected_components(curvy)


def test_densify_respects_spacing():
    D = geograph.densify(curvy_graph(), 25.0, sampling="uniform")
    assert max(d["length"] for _, _, d in D.edges(data=True)) <= 25.0 + 1e-6


def test_reference_sampling_skips_straight_edges():
    """Every edge of a plain grid is straight, so none should be subdivided."""
    grid = grid_graph()
    D = geograph.densify(grid, 25.0, sampling="reference")
    assert D.number_of_nodes() == grid.number_of_nodes()
    straight = np.array([[0.0, 0.0], [200.0, 0.0]])
    assert geograph.control_cuts(straight, 50.0, "reference") == []


def test_inject_lands_points_exactly():
    """Injected control points must sit exactly where asked, not at a nearby node."""
    curvy = curvy_graph()
    D = geograph.densify(curvy, 37.0, sampling="uniform")
    xy = np.array([D.nodes[n]["pos"] for n in D.nodes])

    out, landed = geograph.inject_points(curvy, xy, max_dist=25.0)

    assert all(node is not None for node in landed)
    placed = np.array([out.nodes[n]["pos"] for n in landed])
    assert np.abs(placed - xy).max() == pytest.approx(0.0, abs=1e-9)
    assert geograph.total_length(out) == pytest.approx(geograph.total_length(curvy))


def test_inject_ignores_far_points():
    grid = grid_graph()
    out, landed = geograph.inject_points(
        grid, np.array([[9999.0, 9999.0]]), max_dist=25.0
    )
    assert landed == [None]
    assert out.number_of_nodes() == grid.number_of_nodes()

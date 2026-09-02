"""Offline coverage for the skeleton cleanup stage."""

import networkx as nx
import numpy as np
import pytest

from geo_graphs import cleanup, geograph
from tests.conftest import bow, curvy_graph, grid_edges, grid_graph


def endpoints_match_positions(G: nx.MultiGraph) -> bool:
    """Every edge polyline still runs from ``pos[u]`` to ``pos[v]``."""
    return all(
        np.allclose(geograph.oriented_pts(G, u, v, k)[0], G.nodes[u]["pos"])
        and np.allclose(geograph.oriented_pts(G, u, v, k)[-1], G.nodes[v]["pos"])
        for u, v, k in G.edges(keys=True)
    )


def lengths_are_consistent(G: nx.MultiGraph) -> bool:
    return all(
        data["length"] == pytest.approx(geograph.polyline_length(data["pts"]))
        for _, _, data in G.edges(data=True)
    )


def with_stub_chain() -> nx.MultiGraph:
    """A grid with two short dead-end segments chained off one corner.

    Pruning the outer stub turns the inner one into a new dead end, so a single
    pass would leave half the spur behind.
    """
    return geograph.build(
        [
            *grid_edges(),
            np.array([[0.0, 0.0], [0.0, -10.0]]),
            np.array([[0.0, -10.0], [0.0, -20.0]]),
        ]
    )


def test_simplify_reduces_vertices_without_moving_endpoints():
    curvy = curvy_graph()
    simplified = cleanup.simplify_edges(curvy, tolerance=5.0)

    before = sum(len(d["pts"]) for _, _, d in curvy.edges(data=True))
    after = sum(len(d["pts"]) for _, _, d in simplified.edges(data=True))
    assert after < before
    assert endpoints_match_positions(simplified)
    assert lengths_are_consistent(simplified)


def test_simplify_preserves_topology():
    curvy = curvy_graph()
    simplified = cleanup.simplify_edges(curvy, tolerance=5.0)
    assert simplified.number_of_nodes() == curvy.number_of_nodes()
    assert simplified.number_of_edges() == curvy.number_of_edges()


def test_simplify_with_zero_tolerance_keeps_length():
    curvy = curvy_graph()
    simplified = cleanup.simplify_edges(curvy, tolerance=0.0)
    assert geograph.total_length(simplified) == pytest.approx(
        geograph.total_length(curvy)
    )


def test_prune_removes_a_short_dead_end():
    G = geograph.build([*grid_edges(), np.array([[0.0, 0.0], [0.0, -10.0]])])
    pruned = cleanup.prune_spurs(G, min_length=20.0)

    assert pruned.number_of_edges() == G.number_of_edges() - 1
    assert all(pruned.degree(n) != 1 for n in pruned.nodes)


def test_prune_is_iterative():
    """One pass would strand the inner half of a chained spur."""
    pruned = cleanup.prune_spurs(with_stub_chain(), min_length=20.0)
    grid = grid_graph()
    assert pruned.number_of_edges() == grid.number_of_edges()
    assert pruned.number_of_nodes() == grid.number_of_nodes()


def test_prune_keeps_long_dead_ends():
    stub = np.array([[0.0, 0.0], [0.0, -80.0]])
    G = geograph.build([*grid_edges(), stub])
    assert cleanup.prune_spurs(G, min_length=20.0).number_of_edges() == (
        G.number_of_edges()
    )


def test_prune_never_removes_an_edge_between_two_junctions():
    grid = grid_graph()
    assert cleanup.prune_spurs(grid, min_length=1000.0).number_of_edges() == (
        grid.number_of_edges()
    )


def test_prune_is_idempotent():
    once = cleanup.prune_spurs(with_stub_chain(), min_length=20.0)
    twice = cleanup.prune_spurs(once, min_length=20.0)
    assert twice.number_of_edges() == once.number_of_edges()
    assert twice.number_of_nodes() == once.number_of_nodes()


def test_snap_merges_nearby_nodes_at_their_centroid():
    G = geograph.build(
        [
            np.array([[0.0, 0.0], [100.0, 0.0]]),
            np.array([[106.0, 0.0], [200.0, 0.0]]),
        ]
    )
    snapped = cleanup.snap_junctions(G, tolerance=8.0)

    assert snapped.number_of_nodes() == 3
    assert snapped.number_of_edges() == 2
    merged = [n for n in snapped.nodes if 100.0 < snapped.nodes[n]["pos"][0] < 106.0]
    assert len(merged) == 1
    assert snapped.nodes[merged[0]]["pos"][0] == pytest.approx(103.0)


def test_snap_leaves_distant_nodes_alone():
    grid = grid_graph()
    snapped = cleanup.snap_junctions(grid, tolerance=8.0)
    assert snapped.number_of_nodes() == grid.number_of_nodes()
    assert snapped.number_of_edges() == grid.number_of_edges()


def test_snap_leaves_no_dangling_references():
    G = geograph.build(
        [
            np.array([[0.0, 0.0], [100.0, 0.0]]),
            np.array([[104.0, 0.0], [200.0, 0.0]]),
            np.array([[104.0, 0.0], [104.0, 90.0]]),
        ]
    )
    snapped = cleanup.snap_junctions(G, tolerance=8.0)
    assert all(u in snapped.nodes and v in snapped.nodes for u, v in snapped.edges())


def test_snap_preserves_polyline_orientation():
    """Regression guard: snapping once rewrote endpoints onto mirrored polylines.

    ``snap_junctions`` pins each edge's first and last point to the merged
    centroids. Reading the polyline without reorienting it puts the wrong
    centroid on each end, which leaves lengths plausible and geometry wrong.
    """
    bowed = bow(np.array([[100.0, 0.0], [104.0, 90.0]]), amplitude=15.0)
    G = geograph.build(
        [
            np.array([[0.0, 0.0], [100.0, 0.0]]),
            bowed,
            np.array([[104.0, 90.0], [200.0, 90.0]]),
        ]
    )
    snapped = cleanup.snap_junctions(G, tolerance=8.0)
    assert endpoints_match_positions(snapped)


def test_clean_preserves_connectivity_on_a_grid():
    grid = grid_graph()
    cleaned = cleanup.clean(grid)
    assert nx.number_connected_components(cleaned) == nx.number_connected_components(grid)
    assert cleaned.number_of_edges() == grid.number_of_edges()


def test_clean_output_is_internally_consistent():
    cleaned = cleanup.clean(curvy_graph())
    assert endpoints_match_positions(cleaned)
    assert lengths_are_consistent(cleaned)
    assert cleaned.number_of_edges() > 0


def test_clean_removes_spurs_and_keeps_the_network():
    cleaned = cleanup.clean(with_stub_chain())
    grid = grid_graph()
    assert cleaned.number_of_edges() == grid.number_of_edges()
    assert nx.number_connected_components(cleaned) == 1

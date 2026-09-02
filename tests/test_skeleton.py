"""Offline coverage for mask -> skeleton -> graph.

Masks are built inline with numpy: the exact pixel geometry is what the
assertions are about, so it stays beside them rather than moving to
``conftest``.
"""

from collections.abc import Callable

import networkx as nx
import numpy as np
import pytest
from skimage.morphology import skeletonize

from geo_graphs import geograph, skeleton

# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------


def empty_mask() -> np.ndarray:
    return np.zeros((32, 32), bool)


def bar_mask() -> np.ndarray:
    m = np.zeros((64, 64), bool)
    m[30:34, 5:60] = True
    return m


def cross_mask() -> np.ndarray:
    m = np.zeros((64, 64), bool)
    m[30:34, 5:60] = True
    m[5:60, 30:34] = True
    return m


def tee_mask() -> np.ndarray:
    m = np.zeros((64, 64), bool)
    m[30:34, 5:60] = True
    m[30:60, 30:34] = True
    return m


RING_RADIUS = 21.0


def ring_mask() -> np.ndarray:
    rows, cols = np.mgrid[0:64, 0:64]
    r = np.hypot(rows - 32, cols - 32)
    return (r >= RING_RADIUS - 3.0) & (r <= RING_RADIUS + 3.0)


def two_bars_mask() -> np.ndarray:
    m = np.zeros((64, 64), bool)
    m[10:14, 5:30] = True
    m[40:44, 35:60] = True
    return m


def wide_bar_mask() -> np.ndarray:
    """Deliberately non-square: a row/column swap would put ``pos`` out of bounds."""
    m = np.zeros((20, 80), bool)
    m[8:12, 5:75] = True
    return m


def ring_and_bar_mask() -> np.ndarray:
    m = ring_mask()
    m[2:6, 5:60] = True
    return m


def lollipop_mask() -> np.ndarray:
    rows, cols = np.mgrid[0:80, 0:80]
    r = np.hypot(rows - 30, cols - 40)
    m = (r >= 14) & (r <= 18)
    m[46:75, 38:42] = True
    return m


def theta_mask() -> np.ndarray:
    rows, cols = np.mgrid[0:80, 0:80]
    r = np.hypot(rows - 40, cols - 40)
    m = (r >= 20) & (r <= 24)
    m[38:42, 20:60] = True
    return m


def street_grid_mask() -> np.ndarray:
    m = np.zeros((100, 100), bool)
    for row in (10, 50, 90):
        m[row - 2 : row + 2, 10:90] = True
    for col in (10, 50, 90):
        m[10:90, col - 2 : col + 2] = True
    return m


def symmetric_lens_mask() -> np.ndarray:
    """A diamond loop with a stub at each side vertex.

    Both arcs of the diamond span the same number of pixels between the same
    two single-pixel junctions, which is what makes it a dedup-key collision.
    """
    rows, cols = np.mgrid[0:21, 0:33]
    m = np.abs(rows - 10) + np.abs(cols - 16) == 8
    m[10, 4:8] = True
    m[10, 25:29] = True
    return m


def twin_loops_mask() -> np.ndarray:
    """Two identical diamond loops meeting at a single shared junction pixel."""
    rows, cols = np.mgrid[0:21, 0:37]
    left = np.abs(rows - 10) + np.abs(cols - 10) == 8
    right = np.abs(rows - 10) + np.abs(cols - 26) == 8
    return left | right


#: Shapes the tracer is expected to handle, used for the invariants that must
#: hold of every graph regardless of topology.
SHAPES: list[Callable[[], np.ndarray]] = [
    bar_mask,
    cross_mask,
    tee_mask,
    ring_mask,
    two_bars_mask,
    wide_bar_mask,
    ring_and_bar_mask,
    lollipop_mask,
    theta_mask,
    street_grid_mask,
]

shape_cases = pytest.mark.parametrize(
    "build", SHAPES, ids=[fn.__name__.removesuffix("_mask") for fn in SHAPES]
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def degrees(G: nx.MultiGraph) -> list[int]:
    return sorted(int(d) for _, d in G.degree())


def interior_pixels(skel: np.ndarray) -> set[tuple[int, int]]:
    """Skeleton pixels that lie strictly inside an edge (exactly two neighbours)."""
    counts = skeleton.neighbour_counts(skel)
    return {
        (int(r), int(c)) for r, c in zip(*np.nonzero(skel & (counts == 2)), strict=True)
    }


def polyline_pixels(G: nx.MultiGraph) -> set[tuple[int, int]]:
    """Every ``pts`` vertex of every edge, snapped back to (row, col)."""
    return {
        (int(y), int(x))
        for _, _, data in G.edges(data=True)
        for x, y in np.rint(np.asarray(data["pts"], dtype=float)).astype(int)
    }


# --------------------------------------------------------------------------
# neighbour_counts
# --------------------------------------------------------------------------


def test_neighbour_counts_is_zero_off_skeleton():
    skel = np.zeros((8, 8), bool)
    skel[3, 2:6] = True
    counts = skeleton.neighbour_counts(skel)
    assert counts.shape == skel.shape
    assert not counts[~skel].any()


def test_neighbour_counts_of_empty_skeleton_is_all_zero():
    assert not skeleton.neighbour_counts(np.zeros((6, 6), bool)).any()


def test_neighbour_counts_matches_hand_checked_tee():
    """A stem meeting a bar: the junction is a run of three high-count pixels.

    Thinning leaves the meeting point as a small blob rather than one pixel,
    which is exactly why ``graph_from_mask`` has to collapse blobs to a
    centroid instead of taking each ``>= 3`` pixel as its own node.
    """
    skel = np.zeros((5, 5), bool)
    skel[0:2, 2] = True
    skel[2, 0:5] = True

    expected = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 4, 0, 0],
            [1, 3, 3, 3, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    assert np.array_equal(skeleton.neighbour_counts(skel), expected)


def test_neighbour_counts_is_eight_connected():
    """A purely diagonal chain is connected: its middle pixel has two neighbours."""
    skel = np.eye(5, dtype=bool)
    counts = skeleton.neighbour_counts(skel)
    assert [int(counts[i, i]) for i in range(5)] == [1, 2, 2, 2, 1]


def test_neighbour_counts_of_isolated_pixel_is_zero():
    skel = np.zeros((3, 3), bool)
    skel[1, 1] = True
    assert int(skeleton.neighbour_counts(skel)[1, 1]) == 0


# --------------------------------------------------------------------------
# topology
# --------------------------------------------------------------------------


def test_empty_mask_yields_empty_graph():
    G = skeleton.graph_from_mask(empty_mask())
    assert G.number_of_nodes() == 0
    assert G.number_of_edges() == 0


def test_single_pixel_mask_yields_an_isolated_node():
    m = np.zeros((16, 16), bool)
    m[8, 8] = True
    G = skeleton.graph_from_mask(m)
    assert degrees(G) == [0]
    assert G.number_of_edges() == 0


def test_bar_yields_one_edge_between_two_endpoints():
    G = skeleton.graph_from_mask(bar_mask())
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1
    assert degrees(G) == [1, 1]
    assert geograph.total_length(G) == pytest.approx(54.0, abs=4.0)


def test_cross_yields_one_degree_four_junction_and_four_endpoints():
    G = skeleton.graph_from_mask(cross_mask())
    assert degrees(G) == [1, 1, 1, 1, 4]
    assert G.number_of_edges() == 4
    assert nx.number_connected_components(G) == 1


def test_tee_yields_one_degree_three_node():
    G = skeleton.graph_from_mask(tee_mask())
    assert degrees(G) == [1, 1, 1, 3]
    assert G.number_of_edges() == 3


def test_disjoint_bars_yield_two_components():
    G = skeleton.graph_from_mask(two_bars_mask())
    assert nx.number_connected_components(G) == 2
    assert degrees(G) == [1, 1, 1, 1]


def test_ring_has_no_endpoints_and_closes_into_a_cycle():
    """Every pixel of a ring has degree 2, so there is no node pixel to seed from.

    Without the isolated-loop pass the whole ring traces to nothing and is
    dropped, so the assertion that matters is that the geometry survives as a
    closed edge, not merely that some node exists.
    """
    G = skeleton.graph_from_mask(ring_mask())

    assert 1 not in degrees(G)
    assert nx.number_of_selfloops(G) == 1
    assert G.number_of_edges() == 1

    ((u, v, key),) = G.edges(keys=True)
    pts = geograph.oriented_pts(G, u, v, key)
    assert pts[0] == pytest.approx(pts[-1])
    circumference = 2.0 * np.pi * RING_RADIUS
    assert geograph.total_length(G) == pytest.approx(circumference, rel=0.1)


def test_ring_alongside_a_bar_keeps_both():
    G = skeleton.graph_from_mask(ring_and_bar_mask())
    assert nx.number_connected_components(G) == 2
    assert nx.number_of_selfloops(G) == 1
    assert sorted(degrees(G)) == [1, 1, 2]


def test_lollipop_attaches_its_ring_to_the_stem():
    G = skeleton.graph_from_mask(lollipop_mask())
    assert degrees(G) == [1, 3]
    assert nx.number_of_selfloops(G) == 1


def test_theta_keeps_all_three_parallel_edges():
    """Parallel edges between one pair of junctions are why this is a MultiGraph."""
    G = skeleton.graph_from_mask(theta_mask())
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 3
    assert degrees(G) == [3, 3]
    u, v = G.nodes
    assert len(G[u][v]) == 3


def test_street_grid_recovers_the_interior_junctions():
    """Grid corners are degree-2 bends, so only the four tees and the centre count."""
    G = skeleton.graph_from_mask(street_grid_mask())
    assert degrees(G) == [3, 3, 3, 3, 4]
    assert G.number_of_edges() == 8
    assert nx.number_connected_components(G) == 1


# --------------------------------------------------------------------------
# geometry conventions
# --------------------------------------------------------------------------


@shape_cases
def test_node_positions_lie_inside_mask_bounds(build: Callable[[], np.ndarray]):
    m = build()
    h, w = m.shape
    G = skeleton.graph_from_mask(m)
    assert G.number_of_nodes() > 0
    for n in G.nodes:
        x, y = G.nodes[n]["pos"]
        assert 0.0 <= x <= w - 1.0
        assert 0.0 <= y <= h - 1.0


@shape_cases
def test_edge_pts_run_from_pos_u_to_pos_v(build: Callable[[], np.ndarray]):
    G = skeleton.graph_from_mask(build())
    assert G.number_of_edges() > 0
    for u, v, key in G.edges(keys=True):
        pts = geograph.oriented_pts(G, u, v, key)
        assert pts.ndim == 2 and pts.shape[1] == 2
        assert pts[0] == pytest.approx(G.nodes[u]["pos"])
        assert pts[-1] == pytest.approx(G.nodes[v]["pos"])


@shape_cases
def test_edge_length_matches_its_polyline(build: Callable[[], np.ndarray]):
    G = skeleton.graph_from_mask(build())
    for _, _, data in G.edges(data=True):
        assert data["length"] == pytest.approx(geograph.polyline_length(data["pts"]))


@shape_cases
def test_edge_pts_lie_inside_mask_bounds(build: Callable[[], np.ndarray]):
    m = build()
    h, w = m.shape
    G = skeleton.graph_from_mask(m)
    for _, _, data in G.edges(data=True):
        pts = np.asarray(data["pts"], dtype=float)
        assert pts[:, 0].min() >= 0.0 and pts[:, 0].max() <= w - 1.0
        assert pts[:, 1].min() >= 0.0 and pts[:, 1].max() <= h - 1.0


@shape_cases
def test_no_interior_skeleton_pixel_is_dropped(build: Callable[[], np.ndarray]):
    """Every degree-2 pixel has to end up on some edge's polyline.

    An interior pixel that reaches no edge is geometry the tracer walked over
    and then threw away, which shows up downstream as a road that is simply
    missing rather than as a malformed graph.
    """
    m = build()
    G = skeleton.graph_from_mask(m)
    missing = interior_pixels(skeletonize(m)) - polyline_pixels(G)
    assert not missing


def test_non_square_mask_keeps_x_as_column_and_y_as_row():
    """``pos`` is (x, y) = (column, row); the transpose would leave the tile."""
    m = wide_bar_mask()
    G = skeleton.graph_from_mask(m)
    xs = [G.nodes[n]["pos"][0] for n in G.nodes]
    ys = [G.nodes[n]["pos"][1] for n in G.nodes]
    assert max(xs) - min(xs) > 60.0
    assert max(ys) - min(ys) < 5.0


def test_graph_carries_unit_resolution():
    assert skeleton.graph_from_mask(bar_mask()).graph["resolution"] == 1.0


def test_mask_is_not_modified():
    m = cross_mask()
    before = m.copy()
    skeleton.graph_from_mask(m)
    assert np.array_equal(m, before)


@pytest.mark.parametrize("dtype", [np.uint8, np.float64, np.int32])
def test_non_bool_masks_give_the_same_graph(dtype: type):
    m = tee_mask()
    G = skeleton.graph_from_mask(m)
    H = skeleton.graph_from_mask(m.astype(dtype))
    assert degrees(H) == degrees(G)
    assert geograph.total_length(H) == pytest.approx(geograph.total_length(G))


# --------------------------------------------------------------------------
# regression guards for the parallel-arc dedup bug
#
# Tracing visits every edge from both ends, so the second visit has to be
# recognized and dropped. Keying that on (start pixel, end pixel, path length)
# collides whenever two genuinely distinct routes share endpoints and size, and
# silently discards one of them along with its geometry. Keying on the pixel
# path itself cannot collide.
# --------------------------------------------------------------------------


def test_symmetric_parallel_arcs_are_both_traced():
    m = symmetric_lens_mask()
    G = skeleton.graph_from_mask(m)

    assert degrees(G) == [1, 1, 3, 3]
    assert G.number_of_edges() == 4
    assert not interior_pixels(skeletonize(m)) - polyline_pixels(G)


def test_twin_loops_at_one_junction_are_both_traced():
    m = twin_loops_mask()
    G = skeleton.graph_from_mask(m)

    assert degrees(G) == [4]
    assert nx.number_of_selfloops(G) == 2
    assert not interior_pixels(skeletonize(m)) - polyline_pixels(G)

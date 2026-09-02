"""Shared graph builders for the test suite.

These are plain functions, imported explicitly by the tests that use them, so a
reader can see where a fixture's contents came from without hunting for a
fixture definition. Test-specific variations stay local to their own module.
"""

import networkx as nx
import numpy as np

from geo_graphs import geograph


def grid_edges(n: int = 4, step: float = 100.0) -> list[np.ndarray]:
    """Straight polylines forming an ``n`` by ``n`` street grid."""
    return [
        np.array(pair)
        for i in range(n)
        for j in range(n - 1)
        for pair in (
            [[i * step, j * step], [i * step, (j + 1) * step]],
            [[j * step, i * step], [(j + 1) * step, i * step]],
        )
    ]


def bow(pts: np.ndarray, amplitude: float = 20.0, n: int = 25) -> np.ndarray:
    """Bow a straight segment sideways, keeping both endpoints where they were.

    Args:
        pts: Polyline whose first and last points define the chord.
        amplitude: Peak sideways displacement at the midpoint.
        n: Number of points in the result.

    Returns:
        An ``(n, 2)`` polyline between the same endpoints.
    """
    p0, p1 = np.asarray(pts[0], float), np.asarray(pts[-1], float)
    along = p1 - p0
    perp = np.array([-along[1], along[0]])
    perp /= np.linalg.norm(perp)
    t = np.linspace(0.0, 1.0, n)
    return p0 + np.outer(t, along) + np.outer(amplitude * np.sin(np.pi * t), perp)


def grid_graph() -> nx.MultiGraph:
    """A plain street grid. Every edge is straight."""
    return geograph.build(grid_edges())


def curvy_graph() -> nx.MultiGraph:
    """A grid where every third edge bows, exercising curved-edge handling.

    The reference skips midpoint insertion on straight edges, so an all-straight
    graph would never reach that path. Bowing in place keeps every edge attached
    to the same two junctions: a curve that merely *touches* another edge's
    interior would leave a dangling component overlapping the grid, where
    nearest-edge snapping is genuinely ambiguous.
    """
    edges = grid_edges()
    return geograph.build([bow(e) if i % 3 == 0 else e for i, e in enumerate(edges)])


def barbell_graph() -> nx.MultiGraph:
    """Two grids joined by a single bridge edge.

    A dense grid reroutes around any one removed edge, so it cannot show what
    losing connectivity costs. A bridge can.
    """
    left = grid_edges(n=4, step=100.0)
    right = [e + np.array([500.0, 0.0]) for e in grid_edges(n=4, step=100.0)]
    bridge = np.array([[300.0, 0.0], [500.0, 0.0]])
    return geograph.build(left + right + [bridge])


def drop_edges(G: nx.MultiGraph, frac: float, seed: int = 0) -> nx.MultiGraph:
    """Remove a fraction of edges at random, deterministically.

    Args:
        G: Graph to damage.
        frac: Fraction of edges to remove.
        seed: Seed controlling which edges are chosen.

    Returns:
        A copy with the chosen edges removed. Nodes are left in place.
    """
    H = G.copy()
    keys = list(H.edges(keys=True))
    rng = np.random.default_rng(seed)
    H.remove_edges_from(
        [keys[i] for i in rng.choice(len(keys), int(frac * len(keys)), replace=False)]
    )
    return H

"""Binary mask to graph, via morphological skeletonization.

The classification rule is the one from the project plan: on a thinned
skeleton, a pixel with exactly two neighbours is interior to an edge, one
neighbour marks an endpoint, and three or more marks a junction.
"""

from typing import cast

import networkx as nx
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from . import geograph

_NEIGHBOURHOOD = np.ones((3, 3), dtype=np.uint8)
_OFFSETS = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))


def neighbour_counts(skel: np.ndarray) -> np.ndarray:
    """Count 8-connected skeleton neighbours of every skeleton pixel."""
    counts = ndimage.convolve(
        skel.astype(np.uint8), _NEIGHBOURHOOD, mode="constant", cval=0
    )
    return np.where(skel, counts - 1, 0)


def _neighbours(skel: np.ndarray, r: int, c: int):
    h, w = skel.shape
    for dr, dc in _OFFSETS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and skel[rr, cc]:
            yield rr, cc


def _trace(skel, start, first, cluster_of, visited) -> tuple[list, tuple | None]:
    """Walk a chain of interior pixels until it reaches another node pixel."""
    path = [start, first]
    prev, cur = start, first
    while True:
        visited.add(cur)
        if cur in cluster_of:
            return path, cur
        step = next(
            (
                n
                for n in _neighbours(skel, *cur)
                if n != prev and (n in cluster_of or n not in visited)
            ),
            None,
        )
        if step is None:
            return path, None
        path.append(step)
        prev, cur = cur, step


def graph_from_mask(mask: np.ndarray) -> nx.MultiGraph:
    """Skeletonize a road mask and trace it into a geometric graph.

    Args:
        mask: Boolean array, True on road.

    Returns:
        A graph in pixel coordinates following the :mod:`geograph` conventions.
    """
    skel = skeletonize(mask.astype(bool))
    counts = neighbour_counts(skel)

    # Junctions arrive as small blobs of high-degree pixels rather than single
    # pixels, so each connected blob collapses to one node at its centroid.
    node_px = skel & (counts != 2)
    labels, n_clusters = cast(
        tuple[np.ndarray, int], ndimage.label(node_px, structure=np.ones((3, 3)))
    )
    centroids = ndimage.center_of_mass(node_px, labels, range(1, n_clusters + 1))

    cluster_of = {
        (int(r), int(c)): int(labels[r, c]) - 1
        for r, c in zip(*np.nonzero(node_px), strict=True)
    }

    G = nx.MultiGraph()
    for i, (r, c) in enumerate(centroids):
        G.add_node(i, pos=(float(c), float(r)))

    visited: set[tuple[int, int]] = set()
    seen_edges: set[tuple] = set()

    for start, cid in cluster_of.items():
        for first in _neighbours(skel, *start):
            if cluster_of.get(first) == cid or first in visited:
                continue
            path, end = _trace(skel, start, first, cluster_of, visited)
            if end is None:
                continue
            # Every edge is reachable from both ends, so it gets traced twice.
            # The pixel path itself is what identifies it: keying on endpoints
            # plus path *length* collides whenever two genuinely distinct routes
            # share endpoints and size — parallel arcs of a loop, say — and
            # silently discards the second one.
            key = min(tuple(path), tuple(reversed(path)))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            pts = _polyline(path, G, cid, cluster_of[end])
            G.add_edge(
                cid, cluster_of[end], pts=pts, length=geograph.polyline_length(pts)
            )

    _add_isolated_loops(skel, cluster_of, visited, G)
    G.graph["resolution"] = 1.0
    return G


def _polyline(path, G, u: int, v: int) -> np.ndarray:
    """Pixel path as (x, y), with ends pinned to the junction centroids."""
    pts = np.array([(c, r) for r, c in path], dtype=float)
    pts[0] = G.nodes[u]["pos"]
    pts[-1] = G.nodes[v]["pos"]
    return pts


def _add_isolated_loops(skel, cluster_of, visited, G: nx.MultiGraph) -> None:
    """Rings with no junction at all (every pixel degree 2) need a seed node."""
    counts = neighbour_counts(skel)
    remaining = {
        (int(r), int(c))
        for r, c in zip(*np.nonzero(skel & (counts == 2)), strict=True)
        if (int(r), int(c)) not in visited
    }
    while remaining:
        start = remaining.pop()
        node = G.number_of_nodes()
        G.add_node(node, pos=(float(start[1]), float(start[0])))
        loop_clusters = dict(cluster_of)
        loop_clusters[start] = node
        first = next(iter(_neighbours(skel, *start)), None)
        if first is None:
            continue
        path, end = _trace(skel, start, first, loop_clusters, visited)
        remaining -= visited
        if end is None:
            continue
        pts = _polyline(path, G, node, loop_clusters[end])
        G.add_edge(
            node, loop_clusters[end], pts=pts, length=geograph.polyline_length(pts)
        )

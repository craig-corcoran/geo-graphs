"""Undirected geometric graphs in tile pixel coordinates.

Conventions shared by every stage of the pipeline:

- the graph is an ``nx.MultiGraph`` (road networks have genuine parallel edges
  between the same pair of junctions, and roundabouts trace back to themselves)
- node attribute ``pos``: ``(x, y)`` float pixel coordinates
- edge attribute ``pts``: ``(N, 2)`` float array along the edge, running from
  ``pos[u]`` to ``pos[v]`` inclusive
- edge attribute ``length``: polyline length of ``pts``
"""

from collections.abc import Iterable
from typing import Literal

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree

#: Relative tolerance below which the reference treats an edge as straight.
REFERENCE_CURVED_EPS = 0.012

#: Shortest piece :func:`node_network` will keep, in pixels.
_MIN_SEGMENT_PX = 1e-6

#: How control points are spaced along edges.
#:
#: ``"reference"`` reproduces ``apls.create_graph_midpoints``: straight edges get
#: no interior points at all, because on a straight edge every interior point's
#: distances are determined by its two endpoints, so they add correlated routes
#: and over-weight long straight roads. ``"uniform"`` divides every edge, which
#: samples far more densely and is what you want when localizing *where* a
#: proposal loses score rather than reporting a comparable number.
Sampling = Literal["uniform", "reference"]


def polyline_length(pts: np.ndarray) -> float:
    """Total length of a polyline given as an ``(N, 2)`` array."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def node_network(lines: list[LineString]) -> list[np.ndarray]:
    """Split lines at every point where they meet, so junctions become nodes.

    :func:`build` fuses *endpoints* that coincide, so a source that runs a road
    straight through its intersections produces disconnected stubs rather than
    a network: a side street's endpoint lands on a main road's *interior*, which
    is a junction on the ground but shares no vertex. One Vegas chip of SpaceNet
    labels gives 33 edges in 30 components without this pass.

    Which sources need it is not obvious and the failure is silent, so it is
    worth stating per source. SpaceNet ships each road as one LineString and
    needs it. OSM read through ``osmnx`` arrives already noded and does not. OSM
    read from a ``.osm.pbf`` extract is raw ways and does, which is the same
    shape of problem as SpaceNet's.

    Args:
        lines: Road centerlines in tile pixel coordinates.

    Returns:
        Polylines split at every intersection, ready for :func:`build`.
    """
    if not lines:
        return []
    noded = unary_union(lines)
    parts = getattr(noded, "geoms", [noded])
    return [
        np.asarray(part.coords, dtype=float)
        for part in parts
        # unary_union can emit a degenerate zero-length piece at an
        # intersection. Kept, it becomes a self-loop that adds two to the
        # junction's degree and no geometry at all.
        if isinstance(part, LineString)
        and len(part.coords) >= 2
        and part.length > _MIN_SEGMENT_PX
    ]


def build(edges: Iterable[np.ndarray], tol: float = 1e-6) -> nx.MultiGraph:
    """Assemble a graph from polylines, fusing endpoints that coincide.

    Args:
        edges: Polylines as ``(N, 2)`` arrays of pixel coordinates.
        tol: Grid size endpoints are rounded to before being matched.

    Returns:
        A graph following this module's attribute conventions.
    """
    G = nx.MultiGraph()
    ids: dict[tuple[int, int], int] = {}

    def node_for(xy: np.ndarray) -> int:
        key = (round(xy[0] / tol), round(xy[1] / tol))
        if key not in ids:
            ids[key] = len(ids)
            G.add_node(ids[key], pos=(float(xy[0]), float(xy[1])))
        return ids[key]

    for pts in edges:
        pts = np.asarray(pts, dtype=float)
        if len(pts) < 2:
            continue
        u, v = node_for(pts[0]), node_for(pts[-1])
        G.add_edge(u, v, pts=pts, length=polyline_length(pts))
    return G


def oriented_pts(G: nx.MultiGraph, u, v, key) -> np.ndarray:
    """Edge polyline oriented to run from ``u`` to ``v``.

    An undirected MultiGraph reports each edge in whichever orientation its
    adjacency dict stores, so a polyline read back as ``(u, v, k)`` is not
    guaranteed to start at ``u``. Anything that walks an edge from one endpoint
    to the other has to reorient first or it silently builds mirrored geometry.
    """
    pts = np.asarray(G.edges[u, v, key]["pts"], dtype=float)
    if u == v:
        return pts
    start = np.asarray(G.nodes[u]["pos"], dtype=float)
    if np.linalg.norm(pts[0] - start) > np.linalg.norm(pts[-1] - start):
        pts = pts[::-1]
    return pts


def positions(G: nx.MultiGraph) -> np.ndarray:
    """(N, 2) array of node positions, ordered to match ``list(G.nodes)``."""
    return np.array([G.nodes[n]["pos"] for n in G.nodes], dtype=float)


def total_length(G: nx.MultiGraph) -> float:
    """Summed length of every edge in the graph."""
    return float(sum(d["length"] for _, _, d in G.edges(data=True)))


def split_at(pts: np.ndarray, distances) -> list[np.ndarray]:
    """Cut a polyline at the given arc lengths, which must lie strictly inside it."""
    pts = np.asarray(pts, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    cuts = sorted(d for d in distances if 0.0 < d < cum[-1])
    if not cuts:
        return [pts]

    pieces, start_pt, i = [], pts[0], 0
    for d in cuts:
        j = int(np.searchsorted(cum, d, side="right"))
        j = min(max(j, 1), len(pts) - 1)
        t = (d - cum[j - 1]) / max(seg[j - 1], 1e-12)
        cut_pt = pts[j - 1] + t * (pts[j] - pts[j - 1])
        pieces.append(np.vstack([start_pt, pts[i + 1 : j], cut_pt]))
        start_pt, i = cut_pt, j - 1
    pieces.append(np.vstack([start_pt, pts[i + 1 :]]))
    return [p for p in pieces if len(p) >= 2]


def split_polyline(pts: np.ndarray, spacing: float) -> list[np.ndarray]:
    """Cut a polyline into consecutive pieces of at most ``spacing`` length."""
    total = polyline_length(pts)
    return split_at(pts, np.arange(spacing, total, spacing))


def control_cuts(
    pts: np.ndarray, spacing: float, sampling: Sampling = "reference"
) -> list[float]:
    """Arc lengths along a polyline at which to place interior control points.

    Args:
        pts: Edge polyline as an ``(N, 2)`` array.
        spacing: Nominal distance between control points.
        sampling: Which rule to apply; see :data:`Sampling`.

    Returns:
        Arc lengths strictly inside the polyline, in increasing order.
    """
    total = polyline_length(pts)
    if total <= 0.0:
        return []
    if sampling == "uniform":
        return list(np.arange(spacing, total, spacing))

    # The reference measures "curvedness" against the bounding-box diagonal
    # rather than the endpoint separation. Crude, but it is what defines the
    # published numbers, so reproduce it rather than improve on it.
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    diagonal = float(np.hypot(*(hi - lo)))
    if abs(diagonal - total) / total < REFERENCE_CURVED_EPS:
        return []
    if total < 0.75 * spacing:
        return []
    if total <= spacing:
        return [0.5 * total]
    n_points = len(np.arange(0.0, total, spacing)) + 1
    return list(np.linspace(0.0, total, n_points)[1:-1])


def inject_points(
    G: nx.MultiGraph, points: np.ndarray, max_dist: float
) -> tuple[nx.MultiGraph, list]:
    """Add each point to ``G`` as a node on the nearest edge, splitting it.

    APLS needs the same physical location addressable in both graphs. Snapping
    to the nearest existing *node* instead would move the point by up to the
    node spacing, and that displacement shows up as bogus path-length error on
    short routes.

    Args:
        G: Graph to inject into; not modified.
        points: ``(N, 2)`` array of pixel coordinates.
        max_dist: Furthest an edge may be from a point and still receive it.

    Returns:
        The augmented graph, and per input point the node id it became, or
        ``None`` when no edge lay within ``max_dist``.
    """
    keys = list(G.edges(keys=True))
    if not keys:
        return G.copy(), [None] * len(points)

    oriented = [oriented_pts(G, *k) for k in keys]
    lines = [LineString(p) for p in oriented]
    tree = STRtree(lines)

    landed: list = [None] * len(points)
    per_edge: dict = {}
    for i, xy in enumerate(points):
        p = Point(float(xy[0]), float(xy[1]))
        hits = tree.query_nearest(p, max_distance=max_dist)
        if len(hits) == 0:
            continue
        j = min(hits, key=lambda j: lines[j].distance(p))
        per_edge.setdefault(j, []).append((lines[j].project(p), i))

    out = G.copy()
    next_id = max((n for n in G.nodes if isinstance(n, int)), default=-1) + 1
    tol = 1e-6

    for j, hits in per_edge.items():
        key = keys[j]
        u, v, _ = key
        pts = oriented[j]
        total = polyline_length(pts)

        interior = []
        for s, i in hits:
            if s <= tol:
                landed[i] = u
            elif s >= total - tol:
                landed[i] = v
            else:
                interior.append((s, i))
        if not interior:
            continue

        interior.sort()
        pieces = split_at(pts, [s for s, _ in interior])
        if len(pieces) != len(interior) + 1:
            continue

        out.remove_edge(*key)
        prev = u
        for piece, (_, i) in zip(pieces[:-1], interior, strict=True):
            node = next_id
            next_id += 1
            out.add_node(node, pos=(float(piece[-1][0]), float(piece[-1][1])))
            landed[i] = node
            out.add_edge(prev, node, pts=piece, length=polyline_length(piece))
            prev = node
        out.add_edge(prev, v, pts=pieces[-1], length=polyline_length(pieces[-1]))

    return out, landed


def densify(
    G: nx.MultiGraph, spacing: float, sampling: Sampling = "reference"
) -> nx.MultiGraph:
    """Insert degree-2 control nodes along edges.

    APLS compares path lengths between sampled points along the network, not
    just at junctions, so those sample points have to become real nodes first.

    Args:
        G: Graph to densify.
        spacing: Nominal distance between control points.
        sampling: Which spacing rule to apply; see :data:`Sampling`.

    Returns:
        A graph with the same geometry and extra degree-2 nodes.
    """
    D = nx.MultiGraph()
    D.add_nodes_from(G.nodes(data=True))
    next_id = max((n for n in G.nodes if isinstance(n, int)), default=-1) + 1

    for u, v, key in G.edges(keys=True):
        pts = oriented_pts(G, u, v, key)
        cuts = control_cuts(pts, spacing, sampling)
        pieces = split_at(pts, cuts) if cuts else [pts]
        prev = u
        for idx, piece in enumerate(pieces):
            if idx == len(pieces) - 1:
                nxt = v
            else:
                nxt = next_id
                next_id += 1
                D.add_node(nxt, pos=(float(piece[-1][0]), float(piece[-1][1])))
            D.add_edge(prev, nxt, pts=piece, length=polyline_length(piece))
            prev = nxt
    D.graph.update(G.graph)
    return D

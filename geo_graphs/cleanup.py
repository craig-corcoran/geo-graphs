"""Post-processing a traced skeleton into something graph-shaped.

Skeletonization leaves three characteristic artifacts: staircase wiggle along
every edge, short spurs where the mask frayed, and clusters of junction nodes a
few pixels apart. Each op here targets one of them.
"""

import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import LineString

from . import geograph


def simplify_edges(G: nx.MultiGraph, tolerance: float = 2.0) -> nx.MultiGraph:
    """Douglas-Peucker each edge polyline, leaving endpoints fixed.

    Args:
        G: Graph to simplify.
        tolerance: Maximum deviation, in pixels, from the original polyline.

    Returns:
        A graph with the same topology and fewer vertices per edge.
    """
    out = G.copy()
    for u, v, k, data in out.edges(keys=True, data=True):
        pts = np.asarray(data["pts"], dtype=float)
        if len(pts) > 2:
            pts = np.asarray(
                LineString(pts).simplify(tolerance, preserve_topology=False).coords
            )
        out.edges[u, v, k]["pts"] = pts
        out.edges[u, v, k]["length"] = geograph.polyline_length(pts)
    return out


def prune_spurs(G: nx.MultiGraph, min_length: float = 20.0) -> nx.MultiGraph:
    """Drop short dead-end edges, repeatedly, since pruning creates new ones.

    Args:
        G: Graph to prune.
        min_length: Dead-end edges shorter than this are removed.

    Returns:
        A graph with short spurs removed.
    """
    out = G.copy()
    while True:
        doomed = [
            (u, v, k)
            for n in out.nodes
            if out.degree(n) == 1
            for u, v, k, d in out.edges(n, keys=True, data=True)
            if d["length"] < min_length
        ]
        if not doomed:
            break
        out.remove_edges_from(doomed)
        out.remove_nodes_from([n for n in list(out.nodes) if out.degree(n) == 0])
    return out


def snap_junctions(G: nx.MultiGraph, tolerance: float = 8.0) -> nx.MultiGraph:
    """Fuse nodes closer than ``tolerance`` into a single node at the centroid.

    Args:
        G: Graph whose junction clusters should be merged.
        tolerance: Distance within which nodes are treated as one junction.

    Returns:
        A graph with one node per junction cluster.
    """
    nodes = list(G.nodes)
    if len(nodes) < 2:
        return G.copy()

    pos = geograph.positions(G)
    pairs = KDTree(pos).query_pairs(tolerance)

    clusters = nx.Graph()
    clusters.add_nodes_from(range(len(nodes)))
    clusters.add_edges_from(pairs)

    merged_of, centroid = {}, {}
    for group_id, group in enumerate(nx.connected_components(clusters)):
        idx = sorted(group)
        centroid[group_id] = pos[idx].mean(axis=0)
        for i in idx:
            merged_of[nodes[i]] = group_id

    out = nx.MultiGraph()
    for group_id, xy in centroid.items():
        out.add_node(group_id, pos=(float(xy[0]), float(xy[1])))

    for u, v, k in G.edges(keys=True):
        a, b = merged_of[u], merged_of[v]
        pts = geograph.oriented_pts(G, u, v, k).copy()
        pts[0], pts[-1] = centroid[a], centroid[b]
        if a == b and geograph.polyline_length(pts) < tolerance:
            continue
        out.add_edge(a, b, pts=pts, length=geograph.polyline_length(pts))

    out.graph.update(G.graph)
    return out


def clean(
    G: nx.MultiGraph,
    simplify_tolerance: float = 2.0,
    spur_length: float = 20.0,
    snap_tolerance: float = 8.0,
) -> nx.MultiGraph:
    """Run the full skeleton cleanup pipeline.

    Spurs are pruned twice: snapping junctions together can strand a new short
    dead end that the first pass could not have seen.

    Args:
        G: Traced skeleton graph.
        simplify_tolerance: Douglas-Peucker tolerance in pixels.
        spur_length: Shortest dead-end edge worth keeping.
        snap_tolerance: Junction merge radius in pixels.

    Returns:
        A cleaned graph.
    """
    G = simplify_edges(G, simplify_tolerance)
    G = prune_spurs(G, spur_length)
    G = snap_junctions(G, snap_tolerance)
    return prune_spurs(G, spur_length)

"""Adapter from our graph representation to the vendored reference APLS."""

import contextlib
import io
import os
import sys

import networkx as nx
from shapely.geometry import LineString

sys.path.insert(0, os.path.dirname(__file__))

import apls_reference as R

from geo_graphs import geograph

#: Offset applied to proposal node ids. The reference inserts each control
#: point into the *other* graph under its own id, so if the two graphs share an
#: id range those inserts land on unrelated existing nodes and silently corrupt
#: the comparison. SpaceNet never hits this because its truth and proposal
#: graphs carry disjoint namespaces (OSM ids vs generated ones).
PROPOSAL_ID_OFFSET = 1_000_000


def to_reference(G: nx.MultiGraph, id_offset: int = 0) -> nx.MultiGraph:
    """Re-express a geo_graphs graph in the attributes the reference reads.

    It wants ``x``/``y`` on nodes and ``geometry``/``length`` on edges, where we
    carry ``pos`` and ``pts``.
    """
    out = nx.MultiGraph()
    for n, data in G.nodes(data=True):
        x, y = data["pos"]
        out.add_node(n + id_offset, x=float(x), y=float(y))

    for u, v, k in G.edges(keys=True):
        pts = geograph.oriented_pts(G, u, v, k)
        geom = LineString(pts)
        out.add_edge(
            u + id_offset, v + id_offset, key=k, geometry=geom, length=float(geom.length)
        )
    return out


def reference_apls(
    truth: nx.MultiGraph,
    proposal: nx.MultiGraph,
    linestring_delta: float = 50.0,
    max_snap_dist: float = 25.0,
    min_path_length: float = 10.0,
) -> tuple[float, float, float]:
    """Score with the reference implementation. Returns (total, gt->prop, prop->gt)."""
    gt = to_reference(truth)
    prop = to_reference(proposal, id_offset=PROPOSAL_ID_OFFSET)
    with contextlib.redirect_stdout(io.StringIO()):
        return _score(gt, prop, linestring_delta, max_snap_dist, min_path_length)


def _score(gt, prop, linestring_delta, max_snap_dist, min_path_length):
    (
        _gt_cp,
        _p_cp,
        _gt_prime,
        _p_prime,
        control_gt,
        control_prop,
        len_gt_native,
        len_prop_native,
        len_gt_prime,
        len_prop_prime,
    ) = R.make_graphs(
        gt,
        prop,
        linestring_delta=linestring_delta,
        max_snap_dist=max_snap_dist,
        max_nodes_for_midpoints=10000,
        allow_renaming=False,
        verbose=False,
    )

    return R.compute_apls_metric(
        len_gt_native,
        len_prop_native,
        len_gt_prime,
        len_prop_prime,
        control_gt,
        control_prop,
        min_path_length=min_path_length,
        verbose=False,
    )

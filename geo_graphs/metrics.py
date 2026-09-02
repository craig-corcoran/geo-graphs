"""Scoring: pixel-level IoU, and the topology-aware APLS.

APLS (Average Path Length Similarity, from the SpaceNet challenge) samples
control points along the ground-truth network, snaps each to the nearest point
on the proposal, and compares shortest-path distances between every pair. A
proposal can score well on IoU and badly here: severing one edge leaves almost
every pixel intact while making a whole set of paths infinite.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np

from . import geograph

NO_PATH = float("inf")


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boolean masks.

    Args:
        a: Boolean mask.
        b: Boolean mask of the same shape.

    Returns:
        Overlap in ``[0, 1]``; 1.0 when both masks are empty.
    """
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0


@dataclass(frozen=True, slots=True)
class APLSResult:
    """A scored comparison of two road graphs.

    Attributes:
        score: Harmonic mean of the two directional scores, in ``[0, 1]``.
        gt_to_prop: Ground-truth routes measured on the proposal. Falls when
            real roads are missing.
        prop_to_gt: Proposal routes measured on the ground truth. Falls when
            roads are invented.
        n_control: Control points sampled in the forward direction.
    """

    score: float
    gt_to_prop: float
    prop_to_gt: float
    n_control: int

    def __repr__(self) -> str:
        return (
            f"APLS(score={self.score:.4f}, gt->prop={self.gt_to_prop:.4f}, "
            f"prop->gt={self.prop_to_gt:.4f}, n_control={self.n_control})"
        )


def _path_lengths(D: nx.MultiGraph, sources: list) -> dict:
    """Shortest-path length from each source to every reachable node."""
    return {
        s: nx.single_source_dijkstra_path_length(D, s, weight="length") for s in sources
    }


def _directional(
    A: nx.MultiGraph,
    B: nx.MultiGraph,
    spacing: float,
    max_snap: float,
    max_control: int | None,
    min_path_length: float,
    sampling: geograph.Sampling,
    rng: np.random.Generator,
) -> tuple[float, int]:
    """Score paths measured on A against the same paths measured on B.

    Args:
        A: Graph supplying control points and reference path lengths.
        B: Graph the control points are injected into for comparison.
        spacing: Distance between control points along each edge of A.
        max_snap: Furthest a control point may be moved to reach B.
        max_control: Cap on control points, subsampled when exceeded. ``None``
            uses them all, which is what agrees with the reference.
        min_path_length: Routes shorter than this are skipped, since the
            length ratio is dominated by snapping noise at short range.
        sampling: Control point spacing rule; see :data:`geograph.Sampling`.
        rng: Source of randomness for subsampling.

    Returns:
        The mean route score in ``[0, 1]``, and the control point count.
    """
    DA = geograph.densify(A, spacing, sampling)
    if DA.number_of_nodes() == 0:
        return 0.0, 0

    control = list(DA.nodes)
    if max_control is not None and len(control) > max_control:
        control = [control[i] for i in rng.choice(len(control), max_control, False)]

    xy = np.array([DA.nodes[n]["pos"] for n in control], dtype=float)
    DB, landed = geograph.inject_points(B, xy, max_snap)
    match = {c: node for c, node in zip(control, landed, strict=True) if node is not None}

    len_a = _path_lengths(DA, control)
    len_b = _path_lengths(DB, sorted(set(match.values())))

    scores = []
    for i, a1 in enumerate(control):
        for a2 in control[i + 1 :]:
            la = len_a[a1].get(a2, NO_PATH)
            if not np.isfinite(la) or la < min_path_length:
                continue  # unreachable, or too short for the ratio to be meaningful
            b1, b2 = match.get(a1), match.get(a2)
            lb = (
                len_b[b1].get(b2, NO_PATH)
                if b1 is not None and b2 is not None
                else NO_PATH
            )
            scores.append(
                0.0 if not np.isfinite(lb) else 1.0 - min(1.0, abs(la - lb) / la)
            )

    return (float(np.mean(scores)) if scores else 0.0), len(control)


def apls(
    truth: nx.MultiGraph,
    proposal: nx.MultiGraph,
    spacing: float = 50.0,
    max_snap: float = 25.0,
    max_control: int | None = None,
    min_path_length: float = 10.0,
    sampling: geograph.Sampling = "reference",
    seed: int = 0,
) -> APLSResult:
    """Score a proposed road graph against ground truth.

    The two directions catch different failures: ground-truth-to-proposal
    punishes missed roads, proposal-to-ground-truth punishes invented ones.
    Harmonic mean means a proposal has to do both: a graph that recovers half
    the network perfectly still scores near zero. The reference uses
    ``scipy.stats.hmean`` here, despite printing the result labelled "Mean".

    The defaults reproduce the reference implementation to within 3e-5; see
    tests/test_against_reference.py. Report numbers on those defaults, and
    switch ``sampling`` to ``"uniform"`` only to localize where score is lost.

    Args:
        truth: Ground-truth graph in pixel coordinates.
        proposal: Predicted graph in the same coordinates.
        spacing: Distance between control points along each edge.
        max_snap: Furthest a control point may be moved onto the other graph.
        max_control: Cap on control points per direction. ``None`` uses them
            all; setting it trades agreement with the reference for speed on
            large tiles.
        min_path_length: Shortest route worth scoring.
        sampling: Control point spacing rule; see :data:`geograph.Sampling`.
        seed: Seed for control point subsampling, unused when
            ``max_control`` is ``None``.

    Returns:
        The combined score and both directional scores.
    """
    rng = np.random.default_rng(seed)
    args = (spacing, max_snap, max_control, min_path_length, sampling)
    fwd, n_control = _directional(truth, proposal, *args, rng)
    rev, _ = _directional(proposal, truth, *args, rng)

    combined = 0.0 if fwd + rev == 0 else 2.0 * fwd * rev / (fwd + rev)
    return APLSResult(combined, fwd, rev, n_control)

"""Scoring: pixel-level IoU, the topology-aware APLS, and two coverage metrics.

Three metrics live here and they answer three different questions. Read the one
that matches the claim being made:

* :func:`apls` scores **routes**. An edge counts in proportion to how many
  shortest paths run through it, so a severed arterial is catastrophic and a
  missing cul-de-sac is nearly free.
* :func:`buffer_length_prf` scores **presence**, weighted by length. Every metre
  of road counts the same whether or not any route uses it. It is blind to
  topology: a road broken into two pieces one pixel apart is fully covered.
* :func:`junction_prf` scores **attachment**. It asks whether the junctions of
  the truth network exist in the proposal, near the right place, with the right
  number of arms. It says nothing about the roads between them.

None of the three is a substitute for another, and only :func:`apls` on its
committed defaults is comparable to a published leaderboard number.

APLS (Average Path Length Similarity, from the SpaceNet challenge) scores
*routes*, not roads. Control points are dropped every ``spacing`` metres along
the source graph and snapped onto the nearest edge of the target graph, within
``max_snap``. For a pair of them, ``l_a`` is the shortest path between the two
through the source and ``l_b`` the shortest path between their snapped
positions through the target, and the pair scores::

    1 - min(1, abs(l_a - l_b) / l_a)

A direction's score is the mean over pairs, and the two directions are combined
with a harmonic mean: ``gt_to_prop`` punishes roads that are missing,
``prop_to_gt`` punishes roads that were invented, and a proposal has to do both.

Three properties to know before reading a number off this module:

* **A route the target cannot make scores 0, not a partial penalty.** No path
  between the snapped points, or no edge within ``max_snap`` to snap to at all,
  and there is no ``l_b`` to subtract. This is why severing one edge is so
  expensive: it zeroes every pair whose route crossed it, while leaving almost
  every pixel intact.
* **The denominator is always the source length**, so the two directions are
  different measurements rather than one symmetric comparison. A route that
  comes back twice as long scores 0; one that comes back half as long keeps 0.5.
* **Geometry reaches the score only through snapping.** A road drawn nearly
  ``max_snap`` off its true line still snaps, still measures the same length,
  and still scores 1. Displacement below that threshold is deliberately
  invisible; see EXPERIMENT_LOG.md 2026-09-01 (later).

``make apls-explainer`` builds a page that draws real scored pairs, with their
routes, on real tiles.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
import shapely
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from . import geograph

NO_PATH = float("inf")

#: Degree at or above which a node is treated as a junction.
#:
#: Degree 2 is a bend in a road rather than a crossing, and degree 1 is a dead
#: end, so both are excluded: matching them would score geometry a second time
#: under a metric that exists to score attachment.
JUNCTION_MIN_DEGREE = 3


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
        n_control_gt: Control points sampled from the ground truth, scoring
            the ``gt_to_prop`` direction.
        n_control_prop: Control points sampled from the proposal, scoring the
            ``prop_to_gt`` direction. Running far above ``n_control_gt`` means
            the proposal carries more curved edges than the truth does, which
            is traced-geometry noise rather than extra road.
    """

    score: float
    gt_to_prop: float
    prop_to_gt: float
    n_control_gt: int
    n_control_prop: int

    def __repr__(self) -> str:
        return (
            f"APLS(score={self.score:.4f}, gt->prop={self.gt_to_prop:.4f}, "
            f"prop->gt={self.prop_to_gt:.4f}, "
            f"n_control={self.n_control_gt}/{self.n_control_prop})"
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
        The combined score, both directional scores, and each direction's
        control point count.
    """
    rng = np.random.default_rng(seed)
    args = (spacing, max_snap, max_control, min_path_length, sampling)
    fwd, n_control_gt = _directional(truth, proposal, *args, rng)
    rev, n_control_prop = _directional(proposal, truth, *args, rng)

    combined = 0.0 if fwd + rev == 0 else 2.0 * fwd * rev / (fwd + rev)
    return APLSResult(combined, fwd, rev, n_control_gt, n_control_prop)


def _f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall, 0.0 when both are 0."""
    total = precision + recall
    return 0.0 if total == 0.0 else 2.0 * precision * recall / total


@dataclass(frozen=True, slots=True)
class BufferLengthResult:
    """Length-weighted coverage of two road graphs at one buffer width.

    Attributes:
        buffer: Half-width of the matching corridor, in pixels (1 px = 1 m).
        precision: Share of proposal road length lying within ``buffer`` of
            some truth road. Falls when road is invented. 1.0 by convention
            when the proposal is empty, which :attr:`f1` then zeroes anyway.
        recall: Share of truth road length lying within ``buffer`` of some
            proposal road. Falls when road is missing. 1.0 by convention when
            the truth is empty.
        f1: Harmonic mean of the two, 0.0 when both are 0.
        truth_length: Total truth edge length, the ``recall`` denominator.
        proposal_length: Total proposal edge length, the ``precision``
            denominator.
        matched_truth_length: Truth length inside the proposal's corridor.
        matched_proposal_length: Proposal length inside the truth's corridor.
    """

    buffer: float
    precision: float
    recall: float
    f1: float
    truth_length: float
    proposal_length: float
    matched_truth_length: float
    matched_proposal_length: float

    def __repr__(self) -> str:
        return (
            f"BufferLength(buffer={self.buffer:.0f}, f1={self.f1:.4f}, "
            f"precision={self.precision:.4f}, recall={self.recall:.4f})"
        )


def road_lines(G: nx.MultiGraph) -> list[LineString]:
    """Each edge polyline as its own LineString, never dissolved together.

    Kept per edge rather than unioned so the summed length matches
    :func:`geograph.total_length`: a union would silently drop the second of two
    edges that share geometry.
    """
    return [
        LineString(np.asarray(d["pts"], dtype=float))
        for _, _, d in G.edges(data=True)
        if len(d["pts"]) >= 2
    ]


def _as_geometry_array(lines: Sequence[LineString]) -> np.ndarray:
    """Pack geometries into the object array shapely's vectorized ops take."""
    out = np.empty(len(lines), dtype=object)
    out[:] = list(lines)
    return out


def covered_length(lines: Sequence[LineString], region: BaseGeometry) -> float:
    """Total length of ``lines`` falling inside ``region``.

    Args:
        lines: Road polylines, one per edge.
        region: The corridor to clip against, typically a buffered graph.

    Returns:
        Summed length of the clipped pieces, in the same units as the input.
    """
    if not lines or region.is_empty:
        return 0.0
    clipped = shapely.intersection(_as_geometry_array(lines), region)
    return float(shapely.length(clipped).sum())


def corridor(
    lines: Sequence[LineString], buffer: float, quad_segs: int = 16
) -> BaseGeometry:
    """Dissolve road polylines into a single buffered region.

    Args:
        lines: Road polylines, one per edge.
        buffer: Corridor half-width in pixels.
        quad_segs: Segments per quarter circle on the round joins and caps.
            The buffer is an inscribed polygon, so it falls short of the true
            offset by ``1 - cos(pi / (4 * quad_segs))``: 0.12% of ``buffer`` at
            the default of 16, against 0.48% at shapely's 8.

    Returns:
        The buffered union, empty when ``lines`` is empty.
    """
    return shapely.union_all(_as_geometry_array(lines)).buffer(
        buffer, quad_segs=quad_segs
    )


def buffer_length_prf(
    truth: nx.MultiGraph,
    proposal: nx.MultiGraph,
    buffer: float = 10.0,
    quad_segs: int = 16,
) -> BufferLengthResult:
    """Score length-weighted road coverage inside a fixed corridor.

    Precision and recall are the two directions of "is this street here at
    all": what share of proposal road length lies within ``buffer`` of truth
    road, and what share of truth road length lies within ``buffer`` of
    proposal road. Every metre carries the same weight, so a residential
    cul-de-sac counts for its length exactly like an arterial of the same
    length.

    What it does **not** measure, stated because the number invites the
    stronger claim:

    * **Topology.** A road cut into two pieces one pixel apart is fully
      covered, and so is a network of disconnected fragments laid over the
      truth. Connectivity does not enter, so this cannot replace :func:`apls`.
    * **Displacement below the buffer.** Road drawn anywhere inside the
      corridor scores identically to road drawn exactly on the centreline.
      Displacement is visible only through the choice of ``buffer``, which is
      why it should be swept rather than fixed at one value.
    * **Direction of travel, junction structure, or road count.** Two proposal
      roads lying on one truth road are fully covered in both directions.

    Args:
        truth: Ground-truth graph in pixel coordinates (1 px = 1 m).
        proposal: Predicted graph in the same coordinates.
        buffer: Corridor half-width in pixels.
        quad_segs: Segments per quarter circle on the buffer's end caps.

    Returns:
        Precision, recall, F1, and the lengths they were computed from.
    """
    truth_lines = road_lines(truth)
    proposal_lines = road_lines(proposal)
    truth_total = float(shapely.length(_as_geometry_array(truth_lines)).sum())
    proposal_total = float(shapely.length(_as_geometry_array(proposal_lines)).sum())

    matched_truth = covered_length(
        truth_lines, corridor(proposal_lines, buffer, quad_segs)
    )
    matched_proposal = covered_length(
        proposal_lines, corridor(truth_lines, buffer, quad_segs)
    )

    # An empty side has nothing to be wrong about, so its own direction scores
    # 1.0 and the F1 collapses on the other direction instead.
    precision = matched_proposal / proposal_total if proposal_total > 0.0 else 1.0
    recall = matched_truth / truth_total if truth_total > 0.0 else 1.0

    return BufferLengthResult(
        buffer=buffer,
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        truth_length=truth_total,
        proposal_length=proposal_total,
        matched_truth_length=matched_truth,
        matched_proposal_length=matched_proposal,
    )


@dataclass(frozen=True, slots=True)
class JunctionResult:
    """A matched comparison of the junctions of two road graphs.

    Attributes:
        radius: Furthest two junctions may be apart and still match, in pixels.
        min_degree: Degree at or above which a node counts as a junction.
        n_truth: Truth junctions.
        n_proposal: Proposal junctions.
        n_matched: Pairs in the matching, one truth junction to one proposal
            junction.
        precision: ``n_matched / n_proposal``. Falls when junctions are
            invented. 1.0 by convention when the proposal has none.
        recall: ``n_matched / n_truth``. Falls when junctions are missed. 1.0
            by convention when the truth has none.
        f1: Harmonic mean of the two, 0.0 when both are 0.
        degree_agreement: Share of matched pairs whose degrees are equal, or
            ``None`` when nothing matched. ``None`` rather than 0.0 because
            "no junction agreed on degree" and "no junction was found at all"
            are different failures.
        mean_offset: Mean distance between matched junctions in pixels, or
            ``None`` when nothing matched.
    """

    radius: float
    min_degree: int
    n_truth: int
    n_proposal: int
    n_matched: int
    precision: float
    recall: float
    f1: float
    degree_agreement: float | None
    mean_offset: float | None

    def __repr__(self) -> str:
        agreement = (
            "n/a" if self.degree_agreement is None else f"{self.degree_agreement:.4f}"
        )
        return (
            f"Junction(radius={self.radius:.0f}, f1={self.f1:.4f}, "
            f"precision={self.precision:.4f}, recall={self.recall:.4f}, "
            f"degree_agreement={agreement})"
        )


def junctions(
    G: nx.MultiGraph, min_degree: int = JUNCTION_MIN_DEGREE
) -> tuple[np.ndarray, np.ndarray]:
    """Positions and degrees of the graph's junction nodes.

    Args:
        G: Graph to read.
        min_degree: Degree at or above which a node counts as a junction. A
            MultiGraph counts parallel edges separately and a self-loop twice,
            which is the intended reading: both are extra arms at the node.

    Returns:
        An ``(N, 2)`` array of positions and an ``(N,)`` array of degrees.
    """
    degrees = [(n, int(d)) for n, d in G.degree() if int(d) >= min_degree]
    pos = np.array([G.nodes[n]["pos"] for n, _ in degrees], dtype=float).reshape(-1, 2)
    return pos, np.array([d for _, d in degrees], dtype=int)


def match_junctions(
    truth_pos: np.ndarray, proposal_pos: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair up junctions one-to-one, maximizing pairs before minimizing distance.

    Greedy nearest-neighbour matching double-counts: one proposal junction can
    be the nearest to several truth junctions, which inflates recall. This
    solves the assignment instead, so every junction is used at most once.

    Out-of-radius pairs are priced at a constant larger than the whole valid
    matching could ever cost, which makes the optimum lexicographic: it takes
    the largest possible number of in-radius pairs first, and only then the
    shortest total distance among them.

    Args:
        truth_pos: ``(N, 2)`` truth junction positions.
        proposal_pos: ``(M, 2)`` proposal junction positions.
        radius: Furthest a pair may be apart and still count as matched.

    Returns:
        Truth indices, proposal indices, and the distance of each matched pair.
    """
    if len(truth_pos) == 0 or len(proposal_pos) == 0:
        empty_int = np.zeros(0, dtype=int)
        return empty_int, empty_int.copy(), np.zeros(0, dtype=float)

    distance = np.linalg.norm(truth_pos[:, None, :] - proposal_pos[None, :, :], axis=-1)
    # Any total over in-radius pairs is below (pairs x radius), so one extra
    # in-radius pair always beats any distance saving among the rest.
    penalty = radius * (min(len(truth_pos), len(proposal_pos)) + 1.0) + 1.0
    rows, cols = linear_sum_assignment(np.where(distance <= radius, distance, penalty))

    within = distance[rows, cols] <= radius
    rows, cols = rows[within], cols[within]
    return rows, cols, distance[rows, cols]


def junction_prf(
    truth: nx.MultiGraph,
    proposal: nx.MultiGraph,
    radius: float = 10.0,
    min_degree: int = JUNCTION_MIN_DEGREE,
) -> JunctionResult:
    """Score whether the truth's junctions exist in the proposal.

    This is the "attached to the right cross-streets" measurement. Junctions
    are matched one-to-one within ``radius`` by optimal assignment, and the
    matched pairs are then asked whether they agree on degree, which separates
    a four-way crossing recovered as a four-way crossing from one recovered as
    a T.

    What it does **not** measure:

    * **The roads between the junctions.** Every edge could be routed
      absurdly, or be missing entirely between two recovered junctions, and
      this score would not move. Pair it with :func:`buffer_length_prf`.
    * **Connectivity.** Two junctions can match perfectly while sitting in
      different components of the proposal.
    * **Anything about degree-1 and degree-2 nodes.** A network of nothing but
      dead ends and bends has no junctions at all, and scores 1.0 against a
      truth that has none either.
    * **Which arms match.** Degree agreement compares counts, not directions,
      so a T rotated 90 degrees agrees on degree.

    Args:
        truth: Ground-truth graph in pixel coordinates (1 px = 1 m).
        proposal: Predicted graph in the same coordinates.
        radius: Furthest two junctions may be apart and still match.
        min_degree: Degree at or above which a node counts as a junction.

    Returns:
        Counts, precision, recall, F1, degree agreement and mean offset.
    """
    truth_pos, truth_degree = junctions(truth, min_degree)
    proposal_pos, proposal_degree = junctions(proposal, min_degree)
    rows, cols, offsets = match_junctions(truth_pos, proposal_pos, radius)

    n_matched = len(rows)
    precision = n_matched / len(proposal_pos) if len(proposal_pos) else 1.0
    recall = n_matched / len(truth_pos) if len(truth_pos) else 1.0
    agreement = (
        float(np.mean(truth_degree[rows] == proposal_degree[cols])) if n_matched else None
    )

    return JunctionResult(
        radius=radius,
        min_degree=min_degree,
        n_truth=len(truth_pos),
        n_proposal=len(proposal_pos),
        n_matched=n_matched,
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        degree_agreement=agreement,
        mean_offset=float(offsets.mean()) if n_matched else None,
    )

"""Census the dead-end endpoints a frozen checkpoint's proposals leave behind.

A gap-closing stage can only be worth building if the gaps are there to close.
This counts them: how many degree-1 nodes the decoder produces per chip, how
many survive the border filter, how many reconnection candidates they generate
at a range of radii, and what fraction of those candidates the ground truth
says are genuine severed roads rather than real dead ends.

Retrains nothing. The checkpoint is the frozen prior stage, so the census costs
inference plus graph work rather than a training run.

Two proposal graphs are built per chip because ``cleanup.clean`` opens with
``prune_spurs(min_length=20)``, which deletes exactly the short stub a severed
road leaves behind. ``preprune`` (simplify then snap, no pruning) is where
candidates are generated; ``cleaned`` is the graph the pipeline actually ships.
The difference between their interior endpoint counts is how many candidates
the shipped cleanup destroys.

The train/val split is carried through every aggregate. The network has seen
the 785 training chips, so its proposals there have fewer and easier gaps than
on the 196 it has not; if the two positive-rate distributions match, a link
predictor can be trained on proposals generated in-fold, and if they diverge
the proposals have to come from out-of-fold inference instead.

Three measurements exist to bound how much of the label set is an artifact of
the label rule rather than a property of the roads.

**The labeller's snap radius is swept.** ``label_snap`` is the labeller's own
parameter, not ``metrics.apls``'s ``max_snap``, and defaults to a sweep over
10, 15 and 25 px. Every candidate is labelled once per value, and the headline
is the transition table: of the candidates positive at the largest radius, how
many stay positive, turn negative, or become undeterminable at the smaller
ones. Turning negative is a disagreement about the road; becoming
undeterminable is only an exclusion, and the two are never pooled.

**Positives are split by whether a route already exists.** ``component_join``
means ``prop_dist`` is infinite and the candidate is genuine connectivity
repair. ``detour_shortcut`` means the proposal already routes between the two
ends and the candidate cuts across that detour. A wrong shortcut costs
``prop_to_gt`` directly; a missed component join only fails to earn
``gt_to_prop``.

**Stub purity attacks the ambiguous bucket directly.** A candidate whose two
ends project onto the same point of the same truth road (``0 < truth_dist <=
1``) is a correct positive for a severed road and an incorrect one for a
hallucinated stub lying within the snap radius of a real road; the projections
cannot tell those apart. Sampling the truth mask along the endpoint's own
incident polyline can: a severed road's stub is road the model correctly found
and scores high, while an invented stub scores low.
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np
from loguru import logger
from scipy import ndimage
from scipy.spatial import KDTree
from shapely.geometry import LineString, Point, box
from shapely.strtree import STRtree

from geo_graphs import cleanup, data, geograph, skeleton, train
from geo_graphs.model import predict_mask

#: Radii, in pixels (1 px = 1 m), at which candidate pairs are proposed.
DEFAULT_RADII = (15.0, 25.0, 40.0, 60.0)

#: Snap radii, in pixels, at which the labeller is run.
#:
#: Independent of ``metrics.apls``'s ``max_snap``, which is 25.0 and is correct
#: there because it applies symmetrically to both graphs. As a labelling radius
#: on a 396x324 px chip it is loose, so the committed default labels every
#: candidate three times and reports how the positive class moves.
DEFAULT_LABEL_SNAPS = (10.0, 15.0, 25.0)

#: Which geometry a candidate connects.
#:
#: ``endpoint_endpoint`` joins two dead ends facing each other across a gap.
#: ``endpoint_edge`` joins a dead end to the nearest point on a road it does
#: not touch, which is the T-junction case: a side street whose stem was cut.
Kind = Literal["endpoint_endpoint", "endpoint_edge"]

KINDS: tuple[Kind, ...] = ("endpoint_endpoint", "endpoint_edge")

#: What the ground truth says about a candidate.
#:
#: Three-valued on purpose. ``undeterminable`` is not a weak ``negative``: it
#: means at least one of the two points had no truth edge within ``label_snap``,
#: so the truth graph holds no evidence either way and the candidate must be
#: excluded from a training loss rather than counted against it.
Label = Literal["positive", "negative", "undeterminable"]

#: What kind of repair a positive candidate is.
#:
#: ``component_join`` has no proposal route at all between its two ends;
#: ``detour_shortcut`` has one, but longer than the truth route by more than
#: ``eps``. Different modelling problems, and different failure costs.
PositiveKind = Literal["component_join", "detour_shortcut"]

POSITIVE_KINDS: tuple[PositiveKind, ...] = ("component_join", "detour_shortcut")

#: Candidate subsets whose stub purity is reported separately.
#:
#: ``coincident`` is the ambiguous bucket: ``0 < truth_dist <= 1`` px, both ends
#: on the same point of the same truth road.
Bucket = Literal["all", "positive", "coincident", "coincident_positive"]

BUCKETS: tuple[Bucket, ...] = ("all", "positive", "coincident", "coincident_positive")

Split = Literal["train", "val"]


@dataclass(frozen=True, slots=True)
class Candidate:
    """One proposed reconnection, before the truth graph has been consulted.

    Attributes:
        kind: Which geometry this joins; see :data:`Kind`.
        node_a: Interior degree-1 node the candidate starts from.
        a: ``(x, y)`` position of ``node_a``.
        b: ``(x, y)`` of the other end: another endpoint's position for
            ``endpoint_endpoint``, the projected point on the target edge for
            ``endpoint_edge``.
        gap: Euclidean distance between ``a`` and ``b``, in pixels.
        node_b: The other endpoint's node id, or ``None`` when ``b`` lies in
            the interior of an edge and is therefore not a node.
        edge: ``(u, v, key)`` of the target edge, or ``None`` for
            ``endpoint_endpoint``.
        arc: Arc length of ``b`` along that edge measured from ``u``, or
            ``None`` for ``endpoint_endpoint``.
    """

    kind: Kind
    node_a: int
    a: tuple[float, float]
    b: tuple[float, float]
    gap: float
    node_b: int | None = None
    edge: tuple[int, int, int] | None = None
    arc: float | None = None


@dataclass(frozen=True, slots=True)
class LabelledCandidate:
    """One candidate with everything the truth graph and truth mask say about it.

    Not serialized into the artifact. It is the working record the per-chip
    counters and every split-level distribution are both derived from, so the
    two can never disagree about what a candidate was.

    Attributes:
        sample_id: Chip this came from.
        split: Which side of the checkpoint's split that chip is on.
        candidate: The proposed reconnection.
        prop_dist: Shortest route the proposal already offers between the two
            ends, ``inf`` when they sit in different components.
        labels: Label per swept ``label_snap``.
        truth_dists: Truth route length per swept ``label_snap``: ``nan`` where
            the label is ``undeterminable``, ``inf`` where both ends landed but
            the truth graph offers no route between them.
        purity_a: Fraction of ``node_a``'s incident polyline lying on truth road.
        purity_b: The same for ``node_b``, or ``None`` when the far end is a
            point inside an edge rather than a dead end and so has no stub.
    """

    sample_id: str
    split: Split
    candidate: Candidate
    prop_dist: float
    labels: dict[float, Label]
    truth_dists: dict[float, float]
    purity_a: float
    purity_b: float | None


@dataclass(frozen=True, slots=True)
class RadiusCensus:
    """Candidate and label counts for one chip, kind, radius and label snap.

    Attributes:
        radius: Proposal radius in pixels.
        kind: Which geometry these candidates join.
        label_snap: Snap radius the labeller ran at.
        n_candidates: Candidates proposed at this radius.
        n_positive: Candidates the truth graph says are severed roads.
        n_negative: Candidates the truth graph rules out.
        n_undeterminable: Candidates with no truth evidence either way.
        n_component_join: Positives whose two ends are in different proposal
            components.
        n_detour_shortcut: Positives the proposal already routes between, more
            slowly than the truth does.
    """

    radius: float
    kind: Kind
    label_snap: float
    n_candidates: int
    n_positive: int
    n_negative: int
    n_undeterminable: int
    n_component_join: int
    n_detour_shortcut: int


@dataclass(frozen=True, slots=True)
class RadiusEndpoints:
    """How many of a chip's interior endpoints a radius actually reaches.

    Counted per source endpoint rather than per candidate, and pooled over both
    kinds. A single severed road can be proposed twice (once as the pair of
    facing dead ends, once as each dead end against the other's edge), so
    candidate totals overcount physical gaps while these do not.

    Attributes:
        radius: Proposal radius in pixels.
        label_snap: Snap radius the labeller ran at.
        n_with_candidate: Interior endpoints with at least one candidate. Does
            not depend on ``label_snap``; repeated across its values so the row
            joins to the positive count beside it.
        n_with_positive: Interior endpoints with at least one positive.
    """

    radius: float
    label_snap: float
    n_with_candidate: int
    n_with_positive: int


@dataclass(frozen=True, slots=True)
class TransitionCensus:
    """Where one chip's positives go when the labeller's snap radius shrinks.

    Attributes:
        radius: Proposal radius in pixels.
        from_snap: Snap radius the positives were selected at.
        to_snap: Snap radius they are re-read at.
        n_from_positive: Positives at ``from_snap`` within this radius.
        n_positive: How many are still positive at ``to_snap``.
        n_negative: How many turn negative, which is a genuine disagreement
            about whether the truth road is there.
        n_undeterminable: How many lose their truth evidence, which is an
            exclusion rather than a disagreement. Never pooled with the above.
    """

    radius: float
    from_snap: float
    to_snap: float
    n_from_positive: int
    n_positive: int
    n_negative: int
    n_undeterminable: int


@dataclass(frozen=True, slots=True)
class SnapDiagnostics:
    """Where the label rule degenerates, per chip and per snap radius.

    Pooled over every candidate, which is every candidate at the largest swept
    radius.

    Attributes:
        label_snap: Snap radius the labeller ran at.
        n_determinable: Candidates where both ends landed on a truth edge.
        zero_truth_dist: Candidates whose two points landed on the same truth
            node, so ``truth_dist`` is exactly zero and the positive test
            degenerates to ``prop_dist > 0``.
        coincident_truth_dist: Candidates with ``0 < truth_dist <= 1`` px. Both
            counters are diagnostics on the label rule rather than results: a
            truth route of a metre means the rule confirmed the two points sit
            on the same truth road, not that it traced the missing link.
    """

    label_snap: float
    n_determinable: int
    zero_truth_dist: int
    coincident_truth_dist: int


@dataclass(frozen=True, slots=True)
class ChipCensus:
    """Everything measured on one chip.

    Attributes:
        sample_id: Which chip this is.
        split: ``"train"`` or ``"val"``, from the checkpoint's own run artifact.
        height: Chip height in pixels.
        width: Chip width in pixels.
        truth_edges: Edges in the ground-truth graph. Zero means every label on
            this chip is ``undeterminable`` by construction.
        preprune_border: Degree-1 nodes within the border margin, before pruning.
        preprune_interior: Degree-1 nodes away from the border, before pruning.
        cleaned_border: Degree-1 nodes within the border margin, after ``clean``.
        cleaned_interior: Degree-1 nodes away from the border, after ``clean``.
        destroyed_interior: Interior ``preprune`` endpoints with no surviving
            degree-1 node nearby in ``cleaned``; what the shipped cleanup costs.
        radii: Per-kind, per-radius, per-snap candidate and label counts.
        endpoint_radii: Per-radius, per-snap counts of the source endpoints
            reached, pooled over both kinds.
        transitions: Where this chip's positives go as the snap radius shrinks.
        snap_diagnostics: Degenerate-truth-distance counts per snap radius.
    """

    sample_id: str
    split: Split
    height: int
    width: int
    truth_edges: int
    preprune_border: int
    preprune_interior: int
    cleaned_border: int
    cleaned_interior: int
    destroyed_interior: int
    radii: tuple[RadiusCensus, ...]
    endpoint_radii: tuple[RadiusEndpoints, ...]
    transitions: tuple[TransitionCensus, ...]
    snap_diagnostics: tuple[SnapDiagnostics, ...]


@dataclass(frozen=True, slots=True)
class EndpointSummary:
    """Endpoints per chip across one split.

    Attributes:
        split: Which split these chips come from.
        n_chips: Chips contributing.
        preprune_interior_mean: Mean interior endpoints per chip before pruning.
        preprune_interior_median: Median of the same.
        preprune_border_mean: Mean border endpoints per chip before pruning.
        preprune_border_median: Median of the same.
        cleaned_interior_mean: Mean interior endpoints per chip after ``clean``.
        cleaned_interior_median: Median of the same.
        cleaned_border_mean: Mean border endpoints per chip after ``clean``.
        cleaned_border_median: Median of the same.
        destroyed_interior_total: Interior endpoints ``clean`` destroys, summed.
        destroyed_interior_mean: Mean per chip.
        destroyed_interior_median: Median per chip.
        destroyed_interior_fraction: Share of interior ``preprune`` endpoints
            that ``clean`` destroys, pooled over the split.
    """

    split: Split
    n_chips: int
    preprune_interior_mean: float
    preprune_interior_median: float
    preprune_border_mean: float
    preprune_border_median: float
    cleaned_interior_mean: float
    cleaned_interior_median: float
    cleaned_border_mean: float
    cleaned_border_median: float
    destroyed_interior_total: int
    destroyed_interior_mean: float
    destroyed_interior_median: float
    destroyed_interior_fraction: float


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    """Candidates and labels for one split, kind, radius and label snap.

    Attributes:
        split: Which split these chips come from.
        kind: Which geometry these candidates join.
        radius: Proposal radius in pixels.
        label_snap: Snap radius the labeller ran at.
        n_chips: Chips contributing.
        n_candidates: Candidates pooled over the split.
        n_positive: Positives pooled over the split.
        n_negative: Negatives pooled over the split.
        n_undeterminable: Candidates with no truth evidence.
        n_component_join: Positives joining two proposal components.
        n_detour_shortcut: Positives shortcutting an existing proposal route.
        positive_rate: ``n_positive / (n_positive + n_negative)``, pooled. The
            undeterminable ones are excluded rather than counted as negatives,
            which is the whole reason the label is three-valued.
        candidates_per_chip_mean: Mean candidates per chip.
        candidates_per_chip_median: Median candidates per chip.
        positives_per_chip_mean: Mean positives per chip.
        positives_per_chip_median: Median positives per chip.
        chip_positive_rate_mean: Mean over chips of the per-chip positive rate,
            counting only chips with at least one determinable candidate. Weights
            every chip equally, unlike ``positive_rate``, so a handful of
            candidate-dense chips cannot carry the split.
        chip_positive_rate_median: Median of the same.
        n_chips_rated: Chips with at least one determinable candidate.
    """

    split: Split
    kind: Kind
    radius: float
    label_snap: float
    n_chips: int
    n_candidates: int
    n_positive: int
    n_negative: int
    n_undeterminable: int
    n_component_join: int
    n_detour_shortcut: int
    positive_rate: float
    candidates_per_chip_mean: float
    candidates_per_chip_median: float
    positives_per_chip_mean: float
    positives_per_chip_median: float
    chip_positive_rate_mean: float
    chip_positive_rate_median: float
    n_chips_rated: int


@dataclass(frozen=True, slots=True)
class EndpointCoverage:
    """What one radius reaches, counted in endpoints rather than candidates.

    Attributes:
        split: Which split these chips come from.
        radius: Proposal radius in pixels.
        label_snap: Snap radius the labeller ran at.
        n_chips: Chips contributing.
        interior_endpoints: Interior endpoints on the unpruned graph, pooled.
        with_candidate: How many of those got at least one candidate.
        with_positive: How many of those got at least one positive.
        with_candidate_fraction: ``with_candidate / interior_endpoints``.
        with_positive_fraction: ``with_positive / interior_endpoints``.
        with_positive_per_chip_mean: Mean per chip.
        with_positive_per_chip_median: Median per chip.
    """

    split: Split
    radius: float
    label_snap: float
    n_chips: int
    interior_endpoints: int
    with_candidate: int
    with_positive: int
    with_candidate_fraction: float
    with_positive_fraction: float
    with_positive_per_chip_mean: float
    with_positive_per_chip_median: float


@dataclass(frozen=True, slots=True)
class TransitionSummary:
    """Where a split's positives go when the labeller's snap radius shrinks.

    Attributes:
        split: Which split these chips come from.
        radius: Proposal radius in pixels.
        from_snap: Snap radius the positives were selected at.
        to_snap: Snap radius they are re-read at.
        n_from_positive: Positives at ``from_snap`` within this radius.
        n_positive: How many are still positive at ``to_snap``.
        n_negative: How many turn negative.
        n_undeterminable: How many become undeterminable.
        stayed_positive_fraction: ``n_positive / n_from_positive``.
        became_negative_fraction: ``n_negative / n_from_positive``. The alarming
            one: the truth graph now says these two ends are not the same road.
        became_undeterminable_fraction: ``n_undeterminable / n_from_positive``.
            Only an exclusion, so reported apart from the line above.
    """

    split: Split
    radius: float
    from_snap: float
    to_snap: float
    n_from_positive: int
    n_positive: int
    n_negative: int
    n_undeterminable: int
    stayed_positive_fraction: float
    became_negative_fraction: float
    became_undeterminable_fraction: float


@dataclass(frozen=True, slots=True)
class ShortcutRatioSummary:
    """How far a shortcut positive's existing route exceeds the truth route.

    ``eps`` is the only thing separating a ``detour_shortcut`` positive from a
    negative, so if the ratios pile up just above ``1 + eps`` the constant is
    carrying the classification rather than the geometry.

    Attributes:
        split: Which split these chips come from.
        radius: Proposal radius in pixels.
        label_snap: Snap radius the labeller ran at.
        n_shortcuts: ``detour_shortcut`` positives at this cell.
        n_ratio_defined: Of those, how many have ``truth_dist > 0``.
        n_ratio_undefined: How many have ``truth_dist == 0``, where the ratio is
            infinite and any positive ``prop_dist`` clears the test.
        ratio_median: Median ``prop_dist / truth_dist``, or ``None`` if none.
        ratio_p90: 90th percentile of the same, or ``None``.
        fraction_below_1_5: Share of defined ratios under 1.5.
    """

    split: Split
    radius: float
    label_snap: float
    n_shortcuts: int
    n_ratio_defined: int
    n_ratio_undefined: int
    ratio_median: float | None
    ratio_p90: float | None
    fraction_below_1_5: float | None


@dataclass(frozen=True, slots=True)
class EndpointPuritySummary:
    """Stub purity over every candidate endpoint in one split.

    One value per endpoint, not per candidate, so an endpoint that generated
    both a pair candidate and an edge candidate is not counted twice.

    Attributes:
        split: Which split these endpoints come from.
        n_endpoints: Interior endpoints holding at least one candidate.
        purity_mean: Mean stub purity.
        purity_min: Smallest.
        purity_q1: First quartile.
        purity_median: Median.
        purity_q3: Third quartile.
        purity_max: Largest.
        fraction_at_zero: Share whose incident polyline touches no truth road
            at all.
        fraction_at_one: Share lying entirely on truth road.
    """

    split: Split
    n_endpoints: int
    purity_mean: float | None
    purity_min: float | None
    purity_q1: float | None
    purity_median: float | None
    purity_q3: float | None
    purity_max: float | None
    fraction_at_zero: float | None
    fraction_at_one: float | None


@dataclass(frozen=True, slots=True)
class PurityBucketSummary:
    """One candidate subset cut by stub purity.

    The candidate's purity is the lower of its two stubs when both ends are
    dead ends, and ``purity_a`` alone when the far end is a point inside an
    edge: a pair is only a severed road if *both* halves are real road.

    Attributes:
        split: Which split these chips come from.
        radius: Proposal radius in pixels.
        label_snap: Snap radius the labeller ran at.
        bucket: Which subset; see :data:`Bucket`.
        purity_threshold: Cut between high and low purity.
        n: Candidates in the subset.
        n_high_purity: How many sit at or above the threshold, which the
            hypothesis reads as genuinely severed roads.
        n_low_purity: How many sit below, read as stubs the model invented
            beside a real road.
        high_purity_fraction: ``n_high_purity / n``.
        purity_median: Median candidate purity in the subset, or ``None``.
    """

    split: Split
    radius: float
    label_snap: float
    bucket: Bucket
    purity_threshold: float
    n: int
    n_high_purity: int
    n_low_purity: int
    high_purity_fraction: float | None
    purity_median: float | None


def interior_flags(pos: np.ndarray, shape: tuple[int, int], margin: float) -> np.ndarray:
    """Which positions sit far enough from the chip edge to be real endpoints.

    SpaceNet chips run about 396x324 px, so a 10 px perimeter is roughly 20% of
    the area. A road leaving the chip terminates at the frame, and that stub is
    an artifact of chipping rather than a gap anything could close.

    Args:
        pos: ``(N, 2)`` array of ``(x, y)`` pixel coordinates.
        shape: ``(height, width)`` of the chip.
        margin: Distance from any edge within which a node counts as border.

    Returns:
        ``(N,)`` bool array, True where the position is interior.
    """
    if len(pos) == 0:
        return np.zeros(0, dtype=bool)
    height, width = shape
    return (
        (pos[:, 0] >= margin)
        & (pos[:, 0] <= width - 1 - margin)
        & (pos[:, 1] >= margin)
        & (pos[:, 1] <= height - 1 - margin)
    )


def endpoints(G: nx.MultiGraph) -> tuple[list[int], np.ndarray]:
    """Degree-1 nodes and their positions.

    A self-loop counts twice toward degree in a MultiGraph, so a lone
    roundabout is degree 2 and correctly not an endpoint.

    Args:
        G: Proposal graph.

    Returns:
        ``(nodes, positions)`` with positions as an ``(N, 2)`` array ordered to
        match ``nodes``.
    """
    nodes = [n for n in G.nodes if G.degree(n) == 1]
    pos = np.array([G.nodes[n]["pos"] for n in nodes], dtype=float).reshape(-1, 2)
    return nodes, pos


def destroyed_count(
    preprune_interior_pos: np.ndarray, cleaned_pos: np.ndarray, match_radius: float
) -> int:
    """Interior endpoints present before pruning with no survivor after it.

    Matched by position rather than by node id: ``snap_junctions`` renumbers
    every node, so ids do not survive the comparison. A survivor may have moved
    a little, since pruning can change which nodes fall into a snap cluster and
    therefore where the cluster centroid lands.

    Args:
        preprune_interior_pos: ``(N, 2)`` interior endpoints of the unpruned graph.
        cleaned_pos: ``(M, 2)`` positions of every degree-1 node of the cleaned
            graph, border ones included, so a survivor that drifted across the
            margin is not miscounted as destroyed.
        match_radius: How far a survivor may have moved.

    Returns:
        How many interior endpoints the cleanup destroyed.
    """
    if len(preprune_interior_pos) == 0:
        return 0
    if len(cleaned_pos) == 0:
        return len(preprune_interior_pos)
    distance, _ = KDTree(cleaned_pos).query(preprune_interior_pos)
    return int((np.asarray(distance) > match_radius).sum())


def endpoint_endpoint_candidates(
    G: nx.MultiGraph, nodes: list[int], pos: np.ndarray, max_radius: float
) -> tuple[Candidate, ...]:
    """Pairs of interior endpoints within ``max_radius`` of each other.

    Pairs already joined by a single existing edge are dropped: a short edge
    between two dead ends is an isolated fragment, not a gap, and joining its
    ends would only duplicate the edge.

    Args:
        G: Proposal graph the endpoints came from.
        nodes: Interior degree-1 node ids.
        pos: ``(N, 2)`` positions matching ``nodes``.
        max_radius: Largest radius the sweep will ask for; smaller radii filter
            this set by :attr:`Candidate.gap` rather than re-querying.

    Returns:
        One candidate per surviving pair.
    """
    if len(nodes) < 2:
        return ()
    pairs = KDTree(pos).query_pairs(max_radius)
    return tuple(
        Candidate(
            kind="endpoint_endpoint",
            node_a=nodes[i],
            a=(float(pos[i, 0]), float(pos[i, 1])),
            b=(float(pos[j, 0]), float(pos[j, 1])),
            gap=float(np.hypot(*(pos[j] - pos[i]))),
            node_b=nodes[j],
        )
        for i, j in sorted(pairs)
        if not G.has_edge(nodes[i], nodes[j])
    )


def endpoint_edge_candidates(
    G: nx.MultiGraph, nodes: list[int], pos: np.ndarray, max_radius: float
) -> tuple[Candidate, ...]:
    """Each interior endpoint's nearest point on an edge it does not touch.

    One candidate per endpoint, not one per nearby edge. The question this
    answers is "does this dead end have a road to rejoin within R", and a second
    slightly-further edge is the same reconnection proposed twice.

    A KDTree over edge vertices would answer a different question: the nearest
    *vertex* of a simplified polyline can be many metres from the nearest
    *point* on it, and the projected point is what the label rule needs. So the
    lookup is an STRtree over the edge geometries, which projects exactly.

    Args:
        G: Proposal graph.
        nodes: Interior degree-1 node ids.
        pos: ``(N, 2)`` positions matching ``nodes``.
        max_radius: Largest radius the sweep will ask for.

    Returns:
        One candidate per endpoint that has a non-incident edge in range.
    """
    keys = list(G.edges(keys=True))
    if not keys or not nodes:
        return ()

    oriented = [geograph.oriented_pts(G, *k) for k in keys]
    lines = [LineString(p) for p in oriented]
    tree = STRtree(lines)

    found: list[Candidate] = []
    for node, xy in zip(nodes, pos, strict=True):
        p = Point(float(xy[0]), float(xy[1]))
        # Bounding-box query then exact distances: query_nearest cannot be told
        # to ignore the incident edge, and the incident edge is at distance 0.
        window = box(
            p.x - max_radius, p.y - max_radius, p.x + max_radius, p.y + max_radius
        )
        nearby = [j for j in tree.query(window) if node not in (keys[j][0], keys[j][1])]
        if not nearby:
            continue
        j = min(nearby, key=lambda j: lines[j].distance(p))
        gap = float(lines[j].distance(p))
        if gap > max_radius:
            continue
        arc = float(lines[j].project(p))
        projected = lines[j].interpolate(arc)
        u, v, key = keys[j]
        found.append(
            Candidate(
                kind="endpoint_edge",
                node_a=node,
                a=(float(xy[0]), float(xy[1])),
                b=(float(projected.x), float(projected.y)),
                gap=gap,
                edge=(u, v, key),
                arc=arc,
            )
        )
    return tuple(found)


def proposal_distance(
    G: nx.MultiGraph, lengths: dict[int, float], candidate: Candidate
) -> float:
    """Shortest route the proposal already offers between a candidate's two ends.

    For an ``endpoint_edge`` candidate the far end lies inside an edge rather
    than at a node, and any route into it must arrive from one of that edge's
    two ends, so the distance is the better of the two entries. That is exact
    and costs no extra graph surgery.

    Args:
        G: Proposal graph.
        lengths: Shortest-path lengths from ``candidate.node_a`` to every node
            it can reach, as returned by Dijkstra.
        candidate: The pair to measure.

    Returns:
        Route length in pixels, or ``inf`` when the two ends are in different
        components.
    """
    if candidate.node_b is not None:
        return lengths.get(candidate.node_b, float("inf"))

    assert candidate.edge is not None and candidate.arc is not None
    u, v, key = candidate.edge
    total = float(G.edges[u, v, key]["length"])
    from_u = lengths.get(u, float("inf")) + candidate.arc
    from_v = lengths.get(v, float("inf")) + max(total - candidate.arc, 0.0)
    return min(from_u, from_v)


def label_candidate(
    truth: nx.MultiGraph,
    candidate: Candidate,
    prop_dist: float,
    tau: float,
    eps: float,
    label_snap: float,
) -> tuple[Label, float]:
    """Ask the ground truth whether a candidate is a severed road.

    Both ends are injected into the truth graph at once, so the far end lands on
    the truth edge nearest it rather than on the nearest truth *node*, which
    could be tens of metres away and would put bogus length into every short
    route.

    Args:
        truth: Ground-truth graph for this chip.
        candidate: The pair to label.
        prop_dist: What the proposal graph already offers between the two ends.
        tau: A truth route longer than ``tau`` times the straight-line gap is
            not the road this candidate would be reconstructing.
        eps: The proposal must be worse than the truth route by more than this
            fraction before the gap counts as unclosed.
        label_snap: Furthest a point may be from a truth edge and still land.
            The labeller's own radius, not ``metrics.apls``'s ``max_snap``.

    Returns:
        ``(label, truth_dist)``. ``truth_dist`` is ``inf`` when the two ends are
        in different truth components and ``nan`` when the label is
        ``undeterminable``.
    """
    points = np.array([candidate.a, candidate.b], dtype=float)
    injected, landed = geograph.inject_points(truth, points, label_snap)
    if landed[0] is None or landed[1] is None:
        return "undeterminable", float("nan")

    try:
        truth_dist = float(
            nx.shortest_path_length(injected, landed[0], landed[1], weight="length")
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return "negative", float("inf")

    direct = truth_dist <= tau * candidate.gap
    unclosed = prop_dist > (1.0 + eps) * truth_dist
    return ("positive" if direct and unclosed else "negative"), truth_dist


def stub_purity(pts: np.ndarray, mask: np.ndarray, spacing: float) -> float:
    """Fraction of a polyline lying on truth road.

    Sampling the polyline's own vertices would undersample badly: after
    Douglas-Peucker a straight 40 px edge is two vertices, and reading two
    pixels says nothing about the 38 between them. So the polyline is resampled
    at roughly ``spacing`` px first.

    Args:
        pts: ``(N, 2)`` polyline in ``(x, y)`` pixel coordinates.
        mask: ``(H, W)`` boolean ground-truth road mask for the same chip.
        spacing: Nominal distance between samples, in pixels.

    Returns:
        Share of samples landing on a True mask pixel, in ``[0, 1]``.
    """
    pts = np.asarray(pts, dtype=float)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    pts = pts[np.concatenate([[True], steps > 0.0])]

    if len(pts) < 2:
        xy = pts.reshape(-1, 2)
    else:
        cum = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        )
        n_samples = max(int(np.ceil(cum[-1] / spacing)) + 1, 2)
        arc = np.linspace(0.0, cum[-1], n_samples)
        xy = np.column_stack(
            [np.interp(arc, cum, pts[:, 0]), np.interp(arc, cum, pts[:, 1])]
        )

    rows = np.clip(np.rint(xy[:, 1]).astype(int), 0, mask.shape[0] - 1)
    cols = np.clip(np.rint(xy[:, 0]).astype(int), 0, mask.shape[1] - 1)
    return float(mask[rows, cols].mean())


def endpoint_purities(
    G: nx.MultiGraph, mask: np.ndarray, nodes: Sequence[int], spacing: float
) -> dict[int, float]:
    """Stub purity of every listed endpoint's single incident edge.

    Args:
        G: Proposal graph.
        mask: ``(H, W)`` boolean ground-truth road mask.
        nodes: Degree-1 node ids to measure.
        spacing: Nominal distance between polyline samples, in pixels.

    Returns:
        Purity keyed by node id, omitting any node with no incident edge.
    """
    purities: dict[int, float] = {}
    for node in nodes:
        incident = list(G.edges(node, keys=True))
        if not incident:
            continue
        purities[node] = stub_purity(
            geograph.oriented_pts(G, *incident[0]), mask, spacing
        )
    return purities


def positive_kind(prop_dist: float) -> PositiveKind:
    """Whether a positive repairs connectivity or shortcuts an existing route."""
    return "component_join" if np.isinf(prop_dist) else "detour_shortcut"


def candidate_purity(row: LabelledCandidate) -> float:
    """Lower stub purity of a candidate's two ends.

    A dead-end pair is only a severed road if both halves are real road, so the
    weaker stub is the one that decides. An ``endpoint_edge`` candidate has one
    stub and one point inside an existing edge, so its own stub decides alone.
    """
    return row.purity_a if row.purity_b is None else min(row.purity_a, row.purity_b)


def truth_mask(sample: data.TileSample, dilate: float) -> np.ndarray:
    """The chip's label mask, optionally widened before purity is read off it.

    ``raster.rasterize`` dilates the truth centrelines to about 5 px, so a
    correctly-found road traced 3 px off-centre already reads as impure. The
    knob exists to test whether that width is what the purity split is
    measuring; the committed default leaves the mask alone.

    Args:
        sample: Chip whose ``mask`` to use.
        dilate: Extra dilation radius in pixels; ``0`` returns the mask as-is.

    Returns:
        ``(H, W)`` boolean mask.
    """
    mask = np.asarray(sample.mask, dtype=bool)
    if dilate <= 0:
        return mask
    return ndimage.binary_dilation(
        mask, ndimage.generate_binary_structure(2, 2), int(dilate)
    )


def census_chip(
    model,
    source: data.TileSource,
    sample_id: str,
    split: Split,
    threshold: float,
    radii: tuple[float, ...],
    border_margin: float,
    match_radius: float,
    tau: float,
    eps: float,
    label_snaps: tuple[float, ...],
    purity_spacing: float,
    purity_dilate: float,
    device: str,
) -> tuple[ChipCensus, tuple[LabelledCandidate, ...]]:
    """Run the whole census on one chip.

    Args:
        model: Frozen checkpoint.
        source: Where imagery and labels come from.
        sample_id: Chip to measure.
        split: Which side of the checkpoint's split this chip is on.
        threshold: Probability above which a pixel counts as road.
        radii: Proposal radii, ascending; candidates are generated once at the
            largest and filtered down by gap.
        border_margin: Distance from the chip edge inside which an endpoint is
            an artifact of chipping.
        match_radius: How far a pruned-graph endpoint may have moved and still
            count as the same endpoint.
        tau: Directness bound on the truth route.
        eps: How much worse the proposal route must be to count as a gap.
        label_snaps: Snap radii to label at, ascending. Every candidate is
            labelled once per value.
        purity_spacing: Polyline resampling step for stub purity, in pixels.
        purity_dilate: Extra dilation on the truth mask before purity is read.
        device: Device string for inference.

    Returns:
        The chip's record, and the labelled candidates it was derived from.
    """
    sample = source.load(sample_id)
    logits = train.predict_tile_logits(model, sample, device=device)
    raw = skeleton.graph_from_mask(predict_mask(logits, threshold))

    preprune = cleanup.snap_junctions(cleanup.simplify_edges(raw))
    cleaned = cleanup.clean(raw)

    pre_nodes, pre_pos = endpoints(preprune)
    pre_interior = interior_flags(pre_pos, sample.mask.shape, border_margin)
    _, clean_pos = endpoints(cleaned)
    clean_interior = interior_flags(clean_pos, sample.mask.shape, border_margin)

    interior_nodes = [n for n, keep in zip(pre_nodes, pre_interior, strict=True) if keep]
    interior_pos = pre_pos[pre_interior]

    max_radius = max(radii)
    candidates = endpoint_endpoint_candidates(
        preprune, interior_nodes, interior_pos, max_radius
    ) + endpoint_edge_candidates(preprune, interior_nodes, interior_pos, max_radius)

    # One Dijkstra per source endpoint, reused across that endpoint's candidates
    # and across every radius and snap radius.
    sources = {c.node_a for c in candidates}
    lengths = {
        n: nx.single_source_dijkstra_path_length(preprune, n, weight="length")
        for n in sources
    }
    purities = endpoint_purities(
        preprune,
        truth_mask(sample, purity_dilate),
        sorted(sources | {c.node_b for c in candidates if c.node_b is not None}),
        purity_spacing,
    )

    rows: list[LabelledCandidate] = []
    for candidate in candidates:
        prop_dist = proposal_distance(preprune, lengths[candidate.node_a], candidate)
        labelled = {
            snap: label_candidate(sample.truth, candidate, prop_dist, tau, eps, snap)
            for snap in label_snaps
        }
        rows.append(
            LabelledCandidate(
                sample_id=sample_id,
                split=split,
                candidate=candidate,
                prop_dist=prop_dist,
                labels={snap: label for snap, (label, _) in labelled.items()},
                truth_dists={snap: dist for snap, (_, dist) in labelled.items()},
                purity_a=purities[candidate.node_a],
                purity_b=(
                    None if candidate.node_b is None else purities[candidate.node_b]
                ),
            )
        )

    record = ChipCensus(
        sample_id=sample_id,
        split=split,
        height=int(sample.mask.shape[0]),
        width=int(sample.mask.shape[1]),
        truth_edges=sample.truth.number_of_edges(),
        preprune_border=int((~pre_interior).sum()),
        preprune_interior=int(pre_interior.sum()),
        cleaned_border=int((~clean_interior).sum()),
        cleaned_interior=int(clean_interior.sum()),
        destroyed_interior=destroyed_count(interior_pos, clean_pos, match_radius),
        radii=tuple(
            _radius_census(rows, radius, kind, snap)
            for kind in KINDS
            for radius in radii
            for snap in label_snaps
        ),
        endpoint_radii=tuple(
            _radius_endpoints(rows, radius, snap)
            for radius in radii
            for snap in label_snaps
        ),
        transitions=tuple(
            TransitionCensus(
                radius=radius,
                from_snap=max(label_snaps),
                to_snap=snap,
                **_transition_counts(rows, radius, max(label_snaps), snap),
            )
            for radius in radii
            for snap in label_snaps
            if snap != max(label_snaps)
        ),
        snap_diagnostics=tuple(
            SnapDiagnostics(
                label_snap=snap,
                n_determinable=sum(r.labels[snap] != "undeterminable" for r in rows),
                zero_truth_dist=sum(r.truth_dists[snap] == 0.0 for r in rows),
                coincident_truth_dist=sum(0.0 < r.truth_dists[snap] <= 1.0 for r in rows),
            )
            for snap in label_snaps
        ),
    )
    return record, tuple(rows)


def _radius_census(
    rows: Sequence[LabelledCandidate], radius: float, kind: Kind, label_snap: float
) -> RadiusCensus:
    """Count labels among the candidates of one kind whose gap fits a radius."""
    inside = [r for r in rows if r.candidate.kind == kind and r.candidate.gap <= radius]
    labels = [r.labels[label_snap] for r in inside]
    positives = [r.prop_dist for r in inside if r.labels[label_snap] == "positive"]
    kinds = [positive_kind(d) for d in positives]
    return RadiusCensus(
        radius=radius,
        kind=kind,
        label_snap=label_snap,
        n_candidates=len(inside),
        n_positive=len(positives),
        n_negative=labels.count("negative"),
        n_undeterminable=labels.count("undeterminable"),
        n_component_join=kinds.count("component_join"),
        n_detour_shortcut=kinds.count("detour_shortcut"),
    )


def _radius_endpoints(
    rows: Sequence[LabelledCandidate], radius: float, label_snap: float
) -> RadiusEndpoints:
    """Count the source endpoints one radius reaches, pooled over both kinds."""
    inside = [r for r in rows if r.candidate.gap <= radius]
    return RadiusEndpoints(
        radius=radius,
        label_snap=label_snap,
        n_with_candidate=len({r.candidate.node_a for r in inside}),
        n_with_positive=len(
            {r.candidate.node_a for r in inside if r.labels[label_snap] == "positive"}
        ),
    )


def _transition_counts(
    rows: Sequence[LabelledCandidate], radius: float, from_snap: float, to_snap: float
) -> dict[str, int]:
    """How positives at one snap radius read at another."""
    was = [
        r.labels[to_snap]
        for r in rows
        if r.candidate.gap <= radius and r.labels[from_snap] == "positive"
    ]
    return {
        "n_from_positive": len(was),
        "n_positive": was.count("positive"),
        "n_negative": was.count("negative"),
        "n_undeterminable": was.count("undeterminable"),
    }


def _mean(values: list[float] | list[int]) -> float:
    return float(np.mean(values)) if values else 0.0


def _median(values: list[float] | list[int]) -> float:
    return float(np.median(values)) if values else 0.0


def _quantile(values: Sequence[float], q: float) -> float | None:
    """Percentile of a possibly-empty sample; ``None`` rather than ``nan``."""
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else None


def _fraction(numerator: int, denominator: int) -> float | None:
    """Share, or ``None`` when nothing was counted rather than a misleading 0."""
    return numerator / denominator if denominator else None


def summarize_endpoints(records: tuple[ChipCensus, ...], split: Split) -> EndpointSummary:
    """Aggregate endpoint counts over one split.

    Args:
        records: Every chip's record; filtered to this split here.
        split: Which split to summarize.

    Returns:
        Means and medians per chip, plus what the shipped cleanup destroys.
    """
    rows = [r for r in records if r.split == split]
    destroyed = [r.destroyed_interior for r in rows]
    interior = [r.preprune_interior for r in rows]
    total_interior = sum(interior)

    return EndpointSummary(
        split=split,
        n_chips=len(rows),
        preprune_interior_mean=_mean(interior),
        preprune_interior_median=_median(interior),
        preprune_border_mean=_mean([r.preprune_border for r in rows]),
        preprune_border_median=_median([r.preprune_border for r in rows]),
        cleaned_interior_mean=_mean([r.cleaned_interior for r in rows]),
        cleaned_interior_median=_median([r.cleaned_interior for r in rows]),
        cleaned_border_mean=_mean([r.cleaned_border for r in rows]),
        cleaned_border_median=_median([r.cleaned_border for r in rows]),
        destroyed_interior_total=sum(destroyed),
        destroyed_interior_mean=_mean(destroyed),
        destroyed_interior_median=_median(destroyed),
        destroyed_interior_fraction=(
            sum(destroyed) / total_interior if total_interior else 0.0
        ),
    )


def summarize_candidates(
    records: tuple[ChipCensus, ...],
    split: Split,
    kind: Kind,
    radius: float,
    label_snap: float,
) -> CandidateSummary:
    """Aggregate candidates and labels over one split, kind, radius and snap.

    Args:
        records: Every chip's record; filtered here.
        split: Which split to summarize.
        kind: Which candidate geometry to summarize.
        radius: Which radius to summarize.
        label_snap: Which snap radius to summarize.

    Returns:
        Pooled totals plus the per-chip distributions.
    """
    rows = [
        rc
        for r in records
        if r.split == split
        for rc in r.radii
        if rc.kind == kind and rc.radius == radius and rc.label_snap == label_snap
    ]
    positives = sum(rc.n_positive for rc in rows)
    negatives = sum(rc.n_negative for rc in rows)
    rated = [
        rc.n_positive / (rc.n_positive + rc.n_negative)
        for rc in rows
        if rc.n_positive + rc.n_negative > 0
    ]

    return CandidateSummary(
        split=split,
        kind=kind,
        radius=radius,
        label_snap=label_snap,
        n_chips=len(rows),
        n_candidates=sum(rc.n_candidates for rc in rows),
        n_positive=positives,
        n_negative=negatives,
        n_undeterminable=sum(rc.n_undeterminable for rc in rows),
        n_component_join=sum(rc.n_component_join for rc in rows),
        n_detour_shortcut=sum(rc.n_detour_shortcut for rc in rows),
        positive_rate=positives / (positives + negatives)
        if positives + negatives
        else 0.0,
        candidates_per_chip_mean=_mean([rc.n_candidates for rc in rows]),
        candidates_per_chip_median=_median([rc.n_candidates for rc in rows]),
        positives_per_chip_mean=_mean([rc.n_positive for rc in rows]),
        positives_per_chip_median=_median([rc.n_positive for rc in rows]),
        chip_positive_rate_mean=_mean(rated),
        chip_positive_rate_median=_median(rated),
        n_chips_rated=len(rated),
    )


def summarize_coverage(
    records: tuple[ChipCensus, ...], split: Split, radius: float, label_snap: float
) -> EndpointCoverage:
    """Aggregate endpoint reach over one split, radius and snap radius.

    Args:
        records: Every chip's record; filtered here.
        split: Which split to summarize.
        radius: Which radius to summarize.
        label_snap: Which snap radius to summarize.

    Returns:
        Pooled totals plus the per-chip positive-endpoint distribution.
    """
    rows = [r for r in records if r.split == split]
    reach = [
        next(
            e
            for e in r.endpoint_radii
            if e.radius == radius and e.label_snap == label_snap
        )
        for r in rows
    ]
    interior = sum(r.preprune_interior for r in rows)
    with_candidate = sum(e.n_with_candidate for e in reach)
    with_positive = sum(e.n_with_positive for e in reach)

    return EndpointCoverage(
        split=split,
        radius=radius,
        label_snap=label_snap,
        n_chips=len(rows),
        interior_endpoints=interior,
        with_candidate=with_candidate,
        with_positive=with_positive,
        with_candidate_fraction=with_candidate / interior if interior else 0.0,
        with_positive_fraction=with_positive / interior if interior else 0.0,
        with_positive_per_chip_mean=_mean([e.n_with_positive for e in reach]),
        with_positive_per_chip_median=_median([e.n_with_positive for e in reach]),
    )


def summarize_transitions(
    records: tuple[ChipCensus, ...],
    split: Split,
    radius: float,
    from_snap: float,
    to_snap: float,
) -> TransitionSummary:
    """Aggregate one split's positive-label transitions between two snap radii.

    Args:
        records: Every chip's record; filtered here.
        split: Which split to summarize.
        radius: Which radius to summarize.
        from_snap: Snap radius the positives were selected at.
        to_snap: Snap radius they are re-read at.

    Returns:
        Pooled counts, with "became negative" and "became undeterminable" kept
        apart because only the first is a disagreement about the road.
    """
    rows = [
        t
        for r in records
        if r.split == split
        for t in r.transitions
        if t.radius == radius and t.from_snap == from_snap and t.to_snap == to_snap
    ]
    total = sum(t.n_from_positive for t in rows)
    still = sum(t.n_positive for t in rows)
    negative = sum(t.n_negative for t in rows)
    undeterminable = sum(t.n_undeterminable for t in rows)

    return TransitionSummary(
        split=split,
        radius=radius,
        from_snap=from_snap,
        to_snap=to_snap,
        n_from_positive=total,
        n_positive=still,
        n_negative=negative,
        n_undeterminable=undeterminable,
        stayed_positive_fraction=still / total if total else 0.0,
        became_negative_fraction=negative / total if total else 0.0,
        became_undeterminable_fraction=undeterminable / total if total else 0.0,
    )


def summarize_shortcut_ratios(
    rows: Sequence[LabelledCandidate], split: Split, radius: float, label_snap: float
) -> ShortcutRatioSummary:
    """Describe ``prop_dist / truth_dist`` over one split's shortcut positives.

    Args:
        rows: Every labelled candidate; filtered here.
        split: Which split to summarize.
        radius: Which radius to summarize.
        label_snap: Which snap radius to summarize.

    Returns:
        Counts and quantiles, with the ``truth_dist == 0`` cases counted apart
        rather than dropped silently: their ratio is infinite, so any positive
        proposal route clears the ``eps`` test.
    """
    shortcuts = [
        r
        for r in rows
        if r.split == split
        and r.candidate.gap <= radius
        and r.labels[label_snap] == "positive"
        and positive_kind(r.prop_dist) == "detour_shortcut"
    ]
    ratios = [
        r.prop_dist / r.truth_dists[label_snap]
        for r in shortcuts
        if r.truth_dists[label_snap] > 0.0
    ]
    return ShortcutRatioSummary(
        split=split,
        radius=radius,
        label_snap=label_snap,
        n_shortcuts=len(shortcuts),
        n_ratio_defined=len(ratios),
        n_ratio_undefined=len(shortcuts) - len(ratios),
        ratio_median=_quantile(ratios, 50),
        ratio_p90=_quantile(ratios, 90),
        fraction_below_1_5=_fraction(sum(x < 1.5 for x in ratios), len(ratios)),
    )


def summarize_endpoint_purity(
    rows: Sequence[LabelledCandidate], split: Split
) -> EndpointPuritySummary:
    """Describe stub purity over one split's candidate endpoints.

    Args:
        rows: Every labelled candidate; filtered here.
        split: Which split to summarize.

    Returns:
        Quantiles over one value per endpoint, deduplicated by
        ``(sample_id, node)`` so a busy endpoint does not weigh more.
    """
    by_endpoint: dict[tuple[str, int], float] = {}
    for r in rows:
        if r.split != split:
            continue
        by_endpoint[(r.sample_id, r.candidate.node_a)] = r.purity_a
        if r.candidate.node_b is not None and r.purity_b is not None:
            by_endpoint[(r.sample_id, r.candidate.node_b)] = r.purity_b

    values = list(by_endpoint.values())
    return EndpointPuritySummary(
        split=split,
        n_endpoints=len(values),
        purity_mean=float(np.mean(values)) if values else None,
        purity_min=_quantile(values, 0),
        purity_q1=_quantile(values, 25),
        purity_median=_quantile(values, 50),
        purity_q3=_quantile(values, 75),
        purity_max=_quantile(values, 100),
        fraction_at_zero=_fraction(sum(v == 0.0 for v in values), len(values)),
        fraction_at_one=_fraction(sum(v == 1.0 for v in values), len(values)),
    )


def _in_bucket(row: LabelledCandidate, bucket: Bucket, label_snap: float) -> bool:
    """Whether one candidate belongs to a purity-reporting subset."""
    positive = row.labels[label_snap] == "positive"
    coincident = 0.0 < row.truth_dists[label_snap] <= 1.0
    match bucket:
        case "all":
            return True
        case "positive":
            return positive
        case "coincident":
            return coincident
        case "coincident_positive":
            return coincident and positive


def summarize_purity_bucket(
    rows: Sequence[LabelledCandidate],
    split: Split,
    radius: float,
    label_snap: float,
    bucket: Bucket,
    threshold: float,
) -> PurityBucketSummary:
    """Cut one candidate subset by stub purity.

    Args:
        rows: Every labelled candidate; filtered here.
        split: Which split to summarize.
        radius: Which radius to summarize.
        label_snap: Which snap radius to summarize.
        bucket: Which subset; see :data:`Bucket`.
        threshold: Purity at or above which a candidate reads as a genuinely
            severed road rather than an invented stub.

    Returns:
        The high/low split and the subset's median purity.
    """
    values = [
        candidate_purity(r)
        for r in rows
        if r.split == split
        and r.candidate.gap <= radius
        and _in_bucket(r, bucket, label_snap)
    ]
    high = sum(v >= threshold for v in values)
    return PurityBucketSummary(
        split=split,
        radius=radius,
        label_snap=label_snap,
        bucket=bucket,
        purity_threshold=threshold,
        n=len(values),
        n_high_purity=high,
        n_low_purity=len(values) - high,
        high_purity_fraction=_fraction(high, len(values)),
        purity_median=_quantile(values, 50),
    )


def _diagnostics(
    records: tuple[ChipCensus, ...], split: Split, max_radius: float, label_snap: float
) -> dict[str, float]:
    """Where the label rule is weakest, counted so the numbers can be discounted.

    Args:
        records: Every chip's record; filtered to this split here.
        split: Which split to summarize.
        max_radius: The largest swept radius, which is what the per-chip
            truth-distance counters were pooled over.
        label_snap: Which snap radius to report.

    Returns:
        Counts of chips with no ground truth at all, and of candidates whose
        truth route was zero or under a metre at this snap radius.
    """
    rows = [r for r in records if r.split == split]
    diagnostics = [
        d for r in rows for d in r.snap_diagnostics if d.label_snap == label_snap
    ]
    return {
        "label_snap": label_snap,
        "chips_without_truth_edges": sum(r.truth_edges == 0 for r in rows),
        "candidates_at_max_radius": sum(
            rc.n_candidates
            for r in rows
            for rc in r.radii
            if rc.radius == max_radius and rc.label_snap == label_snap
        ),
        "determinable_at_max_radius": sum(d.n_determinable for d in diagnostics),
        "zero_truth_dist": sum(d.zero_truth_dist for d in diagnostics),
        "coincident_truth_dist": sum(d.coincident_truth_dist for d in diagnostics),
    }


def census(
    model,
    source: data.TileSource,
    ids_by_split: dict[Split, list[str]],
    threshold: float,
    radii: tuple[float, ...],
    border_margin: float,
    match_radius: float,
    tau: float,
    eps: float,
    label_snaps: tuple[float, ...],
    purity_spacing: float,
    purity_dilate: float,
    device: str,
) -> tuple[tuple[ChipCensus, ...], tuple[LabelledCandidate, ...]]:
    """Measure every chip in both splits.

    Args:
        model: Frozen checkpoint.
        source: Where imagery and labels come from.
        ids_by_split: Chip ids to measure, keyed by split.
        threshold: Probability above which a pixel counts as road.
        radii: Proposal radii, ascending.
        border_margin: Chip-edge margin in pixels.
        match_radius: Endpoint survival match radius in pixels.
        tau: Directness bound on the truth route.
        eps: How much worse the proposal route must be to count as a gap.
        label_snaps: Snap radii to label at, ascending.
        purity_spacing: Polyline resampling step for stub purity, in pixels.
        purity_dilate: Extra dilation on the truth mask before purity is read.
        device: Device string for inference.

    Returns:
        One record per chip, train split first, and every labelled candidate
        behind them. The candidates are the distribution-level input and are
        not written to the artifact.
    """
    total = sum(len(v) for v in ids_by_split.values())
    records: list[ChipCensus] = []
    rows: list[LabelledCandidate] = []
    for split, sample_ids in ids_by_split.items():
        for sample_id in sample_ids:
            record, chip_rows = census_chip(
                model,
                source,
                sample_id,
                split,
                threshold,
                radii,
                border_margin,
                match_radius,
                tau,
                eps,
                label_snaps,
                purity_spacing,
                purity_dilate,
                device,
            )
            records.append(record)
            rows.extend(chip_rows)
            if len(records) % 25 == 0 or len(records) == total:
                logger.info(
                    f"[{len(records)}/{total}] {sample_id} ({split}): "
                    f"{record.preprune_interior} interior endpoints, "
                    f"{record.destroyed_interior} destroyed by clean(), "
                    f"{len(rows)} candidates so far"
                )
    return tuple(records), tuple(rows)


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/vegas_best.pt"))
    parser.add_argument(
        "--run-json",
        type=Path,
        default=Path("outputs/vegas_best.json"),
        help="run artifact supplying the train/val split the checkpoint was fit on",
    )
    parser.add_argument("--aoi-root", type=Path, default=Path("data/AOI_2_Vegas"))
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="the tuned operating point from outputs/threshold_sweep.json",
    )
    parser.add_argument("--radii", type=float, nargs="+", default=list(DEFAULT_RADII))
    parser.add_argument(
        "--border-margin",
        type=float,
        default=10.0,
        help="endpoints this close to the chip edge are chipping artifacts",
    )
    parser.add_argument(
        "--match-radius",
        type=float,
        default=8.0,
        help=(
            "how far an endpoint may move between the unpruned and cleaned "
            "graphs and still count as the same endpoint; defaults to the "
            "junction snap tolerance, which is what moves it"
        ),
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.5,
        help="a truth route longer than tau x the straight-line gap is a different road",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.2,
        help="the proposal route must be worse than the truth route by more than this",
    )
    parser.add_argument(
        "--label-snap",
        type=float,
        nargs="+",
        default=list(DEFAULT_LABEL_SNAPS),
        help=(
            "snap radii the labeller runs at, in pixels; every candidate is "
            "labelled once per value and the transitions between them are the "
            "headline. The labeller's own parameter, not metrics.apls's "
            "max_snap; the largest value is the reference the rest are read "
            "against. A cheap run passes one value"
        ),
    )
    parser.add_argument(
        "--purity-threshold",
        type=float,
        default=0.5,
        help="stub purity at or above which a candidate reads as a real severed road",
    )
    parser.add_argument(
        "--purity-spacing",
        type=float,
        default=1.0,
        help="polyline resampling step for stub purity, in pixels",
    )
    parser.add_argument(
        "--purity-dilate",
        type=float,
        default=0.0,
        help=(
            "extra dilation on the truth mask before purity is read; the "
            "rasterized truth is already about 5 px wide, and this exists to "
            "test whether that width is what the purity split measures"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="chips per split; 0 runs every chip, which is what a reported number needs",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("outputs/endpoint_census.json"))
    args = parser.parse_args()

    run = json.loads(args.run_json.read_text())
    limit = args.limit or None
    ids_by_split: dict[Split, list[str]] = {
        "train": list(run["config"]["train_ids"])[:limit],
        "val": list(run["config"]["val_ids"])[:limit],
    }
    radii = tuple(sorted(float(r) for r in args.radii))
    label_snaps = tuple(sorted({float(s) for s in args.label_snap}))
    reference_snap = max(label_snaps)

    source, _ = train.build_source(args.aoi_root, "spacenet", args.resolution)
    model = train.load_checkpoint(args.checkpoint, device=args.device)
    logger.info(
        f"{len(ids_by_split['train'])} train + {len(ids_by_split['val'])} val chips "
        f"at threshold {args.threshold}, radii {radii}, label snaps {label_snaps}"
    )

    records, rows = census(
        model,
        source,
        ids_by_split,
        args.threshold,
        radii,
        args.border_margin,
        args.match_radius,
        args.tau,
        args.eps,
        label_snaps,
        args.purity_spacing,
        args.purity_dilate,
        args.device,
    )

    splits: tuple[Split, ...] = ("train", "val")
    endpoint_summaries = tuple(summarize_endpoints(records, s) for s in splits)
    candidate_summaries = tuple(
        summarize_candidates(records, s, kind, radius, snap)
        for s in splits
        for kind in KINDS
        for radius in radii
        for snap in label_snaps
    )
    coverage_summaries = tuple(
        summarize_coverage(records, s, radius, snap)
        for s in splits
        for radius in radii
        for snap in label_snaps
    )
    transition_summaries = tuple(
        summarize_transitions(records, s, radius, reference_snap, snap)
        for s in splits
        for radius in radii
        for snap in label_snaps
        if snap != reference_snap
    )
    shortcut_summaries = tuple(
        summarize_shortcut_ratios(rows, s, radius, snap)
        for s in splits
        for radius in radii
        for snap in label_snaps
    )
    purity_summaries = tuple(summarize_endpoint_purity(rows, s) for s in splits)
    purity_buckets = tuple(
        summarize_purity_bucket(rows, s, radius, snap, bucket, args.purity_threshold)
        for s in splits
        for radius in radii
        for snap in label_snaps
        for bucket in BUCKETS
    )

    for s in endpoint_summaries:
        logger.info(
            f"{s.split}: interior/chip mean {s.preprune_interior_mean:.2f} "
            f"median {s.preprune_interior_median:.0f}, border/chip mean "
            f"{s.preprune_border_mean:.2f}, clean() destroys "
            f"{s.destroyed_interior_total} interior endpoints "
            f"({s.destroyed_interior_fraction:.1%})"
        )

    logger.info(
        f"{'split':>5} {'kind':>18} {'R':>4} {'snap':>5} {'cand':>7} {'pos':>7} "
        f"{'neg':>7} {'undet':>7} {'rate':>7} {'join':>7} {'short':>7}"
    )
    for c in candidate_summaries:
        logger.info(
            f"{c.split:>5} {c.kind:>18} {c.radius:>4.0f} {c.label_snap:>5.0f} "
            f"{c.n_candidates:>7} {c.n_positive:>7} {c.n_negative:>7} "
            f"{c.n_undeterminable:>7} {c.positive_rate:>7.4f} "
            f"{c.n_component_join:>7} {c.n_detour_shortcut:>7}"
        )

    logger.info(
        f"{'split':>5} {'R':>4} {'from':>5} {'to':>5} {'pos@from':>9} "
        f"{'stays':>9} {'->neg':>9} {'->undet':>9}"
    )
    for t in transition_summaries:
        logger.info(
            f"{t.split:>5} {t.radius:>4.0f} {t.from_snap:>5.0f} {t.to_snap:>5.0f} "
            f"{t.n_from_positive:>9} {t.stayed_positive_fraction:>9.4f} "
            f"{t.became_negative_fraction:>9.4f} "
            f"{t.became_undeterminable_fraction:>9.4f}"
        )

    for p in purity_summaries:
        logger.info(
            f"{p.split}: stub purity over {p.n_endpoints} candidate endpoints, "
            f"q1 {p.purity_q1}, median {p.purity_median}, q3 {p.purity_q3}, "
            f"zero {p.fraction_at_zero}, one {p.fraction_at_one}"
        )

    for b in purity_buckets:
        if b.radius == max(radii) and b.label_snap == reference_snap:
            logger.info(
                f"{b.split:>5} R={b.radius:>3.0f} snap={b.label_snap:>3.0f} "
                f"{b.bucket:>20}: n={b.n:>6} high={b.n_high_purity:>6} "
                f"low={b.n_low_purity:>6} frac_high={b.high_purity_fraction}"
            )

    for e in coverage_summaries:
        if e.label_snap == reference_snap:
            logger.info(
                f"{e.split:>5} R={e.radius:>3.0f}: "
                f"{e.with_positive}/{e.interior_endpoints} interior endpoints have a "
                f"positive ({e.with_positive_fraction:.1%}), "
                f"{e.with_candidate_fraction:.1%} have any candidate"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "run_json": str(args.run_json),
                "params": {
                    "threshold": args.threshold,
                    "radii": list(radii),
                    "border_margin": args.border_margin,
                    "match_radius": args.match_radius,
                    "tau": args.tau,
                    "eps": args.eps,
                    "label_snaps": list(label_snaps),
                    "reference_label_snap": reference_snap,
                    "purity_threshold": args.purity_threshold,
                    "purity_spacing": args.purity_spacing,
                    "purity_dilate": args.purity_dilate,
                    "resolution": args.resolution,
                    "limit": args.limit,
                    "device": args.device,
                    "preprune_pipeline": "simplify_edges -> snap_junctions",
                    "cleaned_pipeline": "cleanup.clean",
                },
                "n_chips": {k: len(v) for k, v in ids_by_split.items()},
                "label_rule_diagnostics": {
                    s: [
                        _diagnostics(records, s, max(radii), snap) for snap in label_snaps
                    ]
                    for s in splits
                },
                "endpoint_summaries": [asdict(s) for s in endpoint_summaries],
                "candidate_summaries": [asdict(c) for c in candidate_summaries],
                "coverage_summaries": [asdict(e) for e in coverage_summaries],
                "transition_summaries": [asdict(t) for t in transition_summaries],
                "shortcut_ratio_summaries": [asdict(s) for s in shortcut_summaries],
                "endpoint_purity_summaries": [asdict(p) for p in purity_summaries],
                "purity_bucket_summaries": [asdict(b) for b in purity_buckets],
                "per_chip": [asdict(r) for r in records],
            },
            indent=2,
        )
    )
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()

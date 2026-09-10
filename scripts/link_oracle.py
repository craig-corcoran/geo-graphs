"""Score an oracle gap-closer, to bound what a link predictor can buy.

A learned link predictor is only worth building if the edges it would add are
worth adding at all. This adds exactly the edges the ground truth says are
missing and scores the result, so the ceiling of the whole line of work is
measured before any model exists. An oracle worth +0.005 ends it; one worth
+0.06 justifies it.

**Every arm is scored under four metrics, because the ceiling is a property of
the metric as much as of the oracle.** APLS scores routes: an edge counts in
proportion to how many shortest paths cross it, so a missing cul-de-sac is worth
close to nothing and a severed arterial is worth a great deal. Length coverage
and junction agreement weight presence instead, which is what a digital twin
someone recognises their own street in actually asks for. The four:

| metric | weights by | column |
|---|---|---|
| ``metrics.apls``, reference sampling | betweenness | ``apls`` |
| ``metrics.apls``, uniform sampling | betweenness, sampled denser | ``apls_uniform`` |
| ``metrics.buffer_length_prf`` | road length | ``buffer_f1_X`` |
| ``metrics.junction_prf`` | junction count | ``junction_f1_X`` |

``apls`` is the only one comparable to a leaderboard number. ``apls_uniform`` is
a *different estimator* of the same quantity, drifting up to 0.06 from the
reference rule, so it is reported in its own column and never against the first.
Road lengths and junction counts ride along in the same map. They are not
scores; they are there so a move in a score can be attributed to what the arm
added rather than guessed at from precision and recall.

Read the multi-metric result knowing it was added after APLS returned an
unwelcome answer on the same holdout. Two things defend it. The objective
genuinely moved, from routing to a map someone recognises their own street in.
And the demand for a local metric predates the answer: ``BACKLOG.md`` has asked
for TOPO since the repository's first commit, 21 commits before the oracle was
scored. What this suite is not is the plan all along.

Builds no model and retrains nothing. The checkpoint is the frozen prior stage,
so the whole measurement costs one inference pass plus graph work and scoring.

**Candidate generation and labelling are imported from**
``scripts/endpoint_census.py`` rather than restated here, so the candidates the
oracle connects are by construction the ones the census counted.

The arms exist to keep two changes apart. ``cleanup.clean`` is ``simplify ->
prune -> snap -> prune``; the predictor has to run before the first prune,
because that prune deletes exactly the short stub a severed road leaves behind.
Dropping it changes what ``snap_junctions`` sees and so changes the emitted
graph on its own, with no edge added. ``reorder`` scores that change alone, and
every oracle arm is read against it as well as against the shipped decoder.

| arm | decoder | what it adds |
|---|---|---|
| ``baseline`` | ``cleanup.clean`` | nothing; the shipped number |
| ``reorder`` | ``simplify -> snap -> prune`` | nothing; the reorder alone |
| ``join`` | ``reorder`` + oracle | ``component_join`` positives |
| ``shortcut`` | ``reorder`` + oracle | ``detour_shortcut`` positives |
| ``both`` | ``reorder`` + oracle | both positive classes |
| ``both_high_purity`` | ``reorder`` + oracle | both, above a stub-purity cut |

Both APLS directions are reported separately for every arm. The shortcut class
is expected to move them in opposite directions -- a shortcut that is wrong
costs ``prop_to_gt`` directly, while a missed component join only fails to earn
``gt_to_prop`` -- and the harmonic mean would hide exactly that.

``label_snap`` is swept because the gap between its values *is* the label-noise
measurement. The census established that snap-10 positives are a strict subset
of snap-25 positives: ``inject_points`` picks the globally nearest truth edge,
so shrinking the radius can only make a candidate undeterminable, never flip it
to negative. It also established that the snap-25 positive class is the dirtier
one. So a snap-25 oracle adds spurious edges a snap-10 oracle declines to add,
and the ``prop_to_gt`` gap between the two reads as how much of the label set is
snapping slack.

Every expensive threshold- and arm-independent quantity is computed once per
chip and reused across all 16 scoring passes: the model's logits, the traced
skeleton, the pre-prune proposal graph, the chip's ceiling graph, and the whole
labelled candidate set. Only edge insertion, pruning and scoring repeat per arm.

``--marginal-arm`` additionally re-scores one arm edge by edge, adding each
oracle edge *alone* and measuring what it is worth on its own. That converts
"average precision over candidates" from a proxy into a direct estimate of the
metric gain a link predictor selecting that edge would earn.

The connector is a straight line between the two candidate points. That
*understates* the ceiling: a real severed road that curves is reconstructed by a
chord, and the length error a chord introduces is exactly what APLS measures. A
connector following the probability ridge would score at least as well.
"""

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np
from endpoint_census import (
    POSITIVE_KINDS,
    Candidate,
    LabelledCandidate,
    PositiveKind,
    candidate_purity,
    endpoint_edge_candidates,
    endpoint_endpoint_candidates,
    endpoint_purities,
    endpoints,
    interior_flags,
    label_candidate,
    positive_kind,
    proposal_distance,
    truth_mask,
)
from loguru import logger
from scipy import stats

from geo_graphs import cleanup, data, geograph, metrics, skeleton, train
from geo_graphs.model import predict_mask

#: Proposal radii, in pixels (1 px = 1 m), the oracle arms are scored at.
#:
#: The census put endpoint coverage at 0.687 at ``R = 25`` and 0.949 at
#: ``R = 60``, so these bracket the frontier: the radius where coverage stops
#: climbing steeply, and the one where almost nothing is out of reach.
DEFAULT_RADII = (25.0, 60.0)

#: Snap radii, in pixels, the labeller is run at.
#:
#: Not ``metrics.apls``'s ``max_snap``. The two values are the label-noise
#: measurement, not a tuning sweep; see the module docstring.
DEFAULT_LABEL_SNAPS = (10.0, 25.0)

#: Which decoder an arm scores.
#:
#: ``clean`` is ``cleanup.clean`` exactly as shipped. ``reorder`` is
#: ``simplify -> snap -> [oracle edges] -> prune``, which is where a predictor
#: would have to sit.
Decoder = Literal["clean", "reorder"]

#: Buffer half-widths, in pixels (1 px = 1 m), length coverage is scored at.
#:
#: Swept rather than fixed because displacement below the buffer is invisible to
#: the metric by construction, so the buffer *is* the geometric tolerance the
#: number is quoted at.
DEFAULT_BUFFERS = (5.0, 10.0, 20.0)

#: Radii, in pixels, junctions are matched within.
DEFAULT_JUNCTION_RADII = (5.0, 10.0, 20.0)

#: The arm re-scored edge by edge when ``--marginal-arm`` is left at its default.
DEFAULT_MARGINAL_ARM = "both@R60_snap10"

#: Metrics scored on the reference sampling rule, comparable to the leaderboard.
APLS_METRICS = ("apls", "gt_to_prop", "prop_to_gt")

#: The same three under ``sampling="uniform"``, which is a *different estimator*
#: and is never compared against the three above.
APLS_UNIFORM_METRICS = ("apls_uniform", "gt_to_prop_uniform", "prop_to_gt_uniform")


def metric_names(
    buffers: tuple[float, ...], junction_radii: tuple[float, ...]
) -> tuple[str, ...]:
    """Every metric key a chip-arm score carries, in reporting order.

    Args:
        buffers: Buffer half-widths length coverage was scored at.
        junction_radii: Radii junctions were matched within.

    Returns:
        Metric keys, the two APLS estimators first.
    """
    coverage = (
        "road_length_truth",
        "road_length_proposal",
        *(
            f"buffer_{field}_{b:.0f}"
            for b in buffers
            for field in ("precision", "recall", "f1")
        ),
    )
    junction = tuple(
        f"junction_{field}_{r:.0f}"
        for r in junction_radii
        for field in (
            "precision",
            "recall",
            "f1",
            "degree_agreement",
            "mean_offset",
            "n_truth",
            "n_proposal",
            "n_matched",
        )
    )
    return APLS_METRICS + APLS_UNIFORM_METRICS + coverage + junction


def score_metrics(
    truth: nx.MultiGraph,
    proposal: nx.MultiGraph,
    buffers: tuple[float, ...],
    junction_radii: tuple[float, ...],
    junction_min_degree: int,
) -> tuple[dict[str, float | None], metrics.APLSResult]:
    """Score one proposal graph under every metric.

    Args:
        truth: Ground-truth graph for the chip.
        proposal: The arm's decoded graph.
        buffers: Buffer half-widths for length coverage.
        junction_radii: Radii for junction matching.
        junction_min_degree: Degree at or above which a node is a junction.

    Returns:
        The metric map keyed by :func:`metric_names`, and the reference-sampled
        APLS result, which also carries the control point counts. A metric is
        ``None`` where it is undefined rather than zero: junction degree
        agreement on a chip where no junction matched at all. The map also
        carries road lengths and junction counts, which are not scores; they
        are there so a change in a score can be attributed rather than guessed.
    """
    reference = metrics.apls(truth, proposal)
    uniform = metrics.apls(truth, proposal, sampling="uniform")
    values: dict[str, float | None] = {
        "apls": reference.score,
        "gt_to_prop": reference.gt_to_prop,
        "prop_to_gt": reference.prop_to_gt,
        "apls_uniform": uniform.score,
        "gt_to_prop_uniform": uniform.gt_to_prop,
        "prop_to_gt_uniform": uniform.prop_to_gt,
    }
    for b in buffers:
        coverage = metrics.buffer_length_prf(truth, proposal, b)
        # Buffer-independent, so the last buffer's copy wins and they agree.
        values["road_length_truth"] = coverage.truth_length
        values["road_length_proposal"] = coverage.proposal_length
        values[f"buffer_precision_{b:.0f}"] = coverage.precision
        values[f"buffer_recall_{b:.0f}"] = coverage.recall
        values[f"buffer_f1_{b:.0f}"] = coverage.f1
    for r in junction_radii:
        junction = metrics.junction_prf(truth, proposal, r, junction_min_degree)
        values[f"junction_precision_{r:.0f}"] = junction.precision
        values[f"junction_recall_{r:.0f}"] = junction.recall
        values[f"junction_f1_{r:.0f}"] = junction.f1
        values[f"junction_degree_agreement_{r:.0f}"] = junction.degree_agreement
        values[f"junction_mean_offset_{r:.0f}"] = junction.mean_offset
        # Counts, not scores. Carried through the same map so the paired
        # statistics answer "how many junctions did the arm invent" directly
        # rather than leaving it to be inferred from precision and recall.
        values[f"junction_n_truth_{r:.0f}"] = float(junction.n_truth)
        values[f"junction_n_proposal_{r:.0f}"] = float(junction.n_proposal)
        values[f"junction_n_matched_{r:.0f}"] = float(junction.n_matched)
    return values, reference


@dataclass(frozen=True, slots=True)
class Arm:
    """One scoring configuration.

    Attributes:
        name: Which arm of the table this belongs to.
        decoder: Which decoder to run; see :data:`Decoder`.
        radius: Proposal radius in pixels, or ``None`` for the two arms that add
            no edges and so cannot depend on it.
        label_snap: Snap radius the labeller ran at, or ``None`` likewise.
        positive_kinds: Which positive classes the oracle adds. Empty for the
            two arms that add nothing.
        min_purity: Stub purity at or above which a selected candidate is kept,
            or ``None`` to keep every positive.
    """

    name: str
    decoder: Decoder
    radius: float | None
    label_snap: float | None
    positive_kinds: tuple[PositiveKind, ...]
    min_purity: float | None


@dataclass(frozen=True, slots=True)
class ChipArmScore:
    """One chip scored under one arm, under every metric.

    Attributes:
        sample_id: Which chip this scored.
        arm: Arm key, from :func:`arm_key`.
        n_selected: Positives the oracle chose to connect.
        n_edges_added: How many of those became an edge. Below ``n_selected``
            when a connector would have duplicated an edge already present or
            when ``inject_points`` declined to split.
        n_control_gt: Control points scoring ``gt_to_prop``; fixed across arms,
            since they are sampled from the truth graph.
        n_control_prop: Control points scoring ``prop_to_gt``. Moves with the
            arm, because splitting a target edge adds a node the proposal then
            samples.
        values: Every metric keyed by :func:`metric_names`. ``None`` marks a
            metric undefined on this chip rather than zero.
    """

    sample_id: str
    arm: str
    n_selected: int
    n_edges_added: int
    n_control_gt: int
    n_control_prop: int
    values: Mapping[str, float | None]


@dataclass(frozen=True, slots=True)
class ArmSummary:
    """Aggregate quality for one arm, under every metric.

    Attributes:
        arm: Arm key.
        name: Which arm of the table this belongs to.
        decoder: Which decoder ran.
        radius: Proposal radius, or ``None``.
        label_snap: Labeller snap radius, or ``None``.
        positive_kinds: Which positive classes the oracle added.
        min_purity: Stub purity cut, or ``None``.
        n_chips: Chips contributing.
        selected_total: Positives selected across the holdout.
        edges_added_total: Oracle edges actually added across the holdout.
        edges_added_mean: The same per chip.
        means: Per-metric mean over chips, skipping chips where the metric is
            undefined. ``None`` when it was undefined everywhere.
        medians: Per-metric median over the same chips, which one bad chip
            cannot drag.
    """

    arm: str
    name: str
    decoder: Decoder
    radius: float | None
    label_snap: float | None
    positive_kinds: tuple[str, ...]
    min_purity: float | None
    n_chips: int
    selected_total: int
    edges_added_total: int
    edges_added_mean: float
    means: Mapping[str, float | None]
    medians: Mapping[str, float | None]


@dataclass(frozen=True, slots=True)
class PairedStat:
    """One arm against a reference arm, paired over chips.

    Paired rather than pooled because the arms score the same chips: the
    between-chip variance is far larger than the effect, so an unpaired
    comparison would drown it.

    Attributes:
        arm: Arm key under test.
        reference: Arm key it is measured against.
        metric: Which of ``apls``, ``gt_to_prop`` or ``prop_to_gt``.
        n: Chips contributing.
        mean_delta: Mean per-chip difference, arm minus reference.
        sem: Standard error of that mean.
        t: ``mean_delta / sem``, or ``None`` when every chip moved identically
            and the statistic is undefined rather than infinite.
        ci_low: Lower end of the 95% interval on ``mean_delta``, or ``None``.
        ci_high: Upper end, or ``None``.
        n_better: Chips the arm improved.
        n_worse: Chips it damaged.
        n_tied: Chips it left exactly unchanged. Counted apart because for the
            oracle arms a tie usually means no candidate was selected there at
            all, which is a different fact from a change that cancelled out.
    """

    arm: str
    reference: str
    metric: str
    n: int
    mean_delta: float
    sem: float
    t: float | None
    ci_low: float | None
    ci_high: float | None
    n_better: int
    n_worse: int
    n_tied: int


def arm_key(arm: Arm) -> str:
    """Stable identifier for an arm, used as the key everywhere it is reported."""
    if arm.radius is None or arm.label_snap is None:
        return arm.name
    return f"{arm.name}@R{arm.radius:.0f}_snap{arm.label_snap:.0f}"


def build_arms(
    radii: tuple[float, ...],
    label_snaps: tuple[float, ...],
    purity_threshold: float,
    purity_snap: float,
) -> tuple[Arm, ...]:
    """Enumerate the scoring grid.

    Args:
        radii: Proposal radii the oracle arms are scored at.
        label_snaps: Snap radii the labeller runs at.
        purity_threshold: Stub purity cut for the ``both_high_purity`` arm.
        purity_snap: The one snap radius that arm is scored at. Running it
            across the whole grid would double the cost to answer a question
            about the filter rather than about the snap radius.

    Returns:
        Every arm to score, the two edge-free ones first.
    """
    edge_free = (
        Arm("baseline", "clean", None, None, (), None),
        Arm("reorder", "reorder", None, None, (), None),
    )
    classes: tuple[tuple[str, tuple[PositiveKind, ...]], ...] = (
        ("join", ("component_join",)),
        ("shortcut", ("detour_shortcut",)),
        ("both", POSITIVE_KINDS),
    )
    grid = tuple(
        Arm(name, "reorder", radius, snap, kinds, None)
        for name, kinds in classes
        for radius in radii
        for snap in label_snaps
    )
    purity = tuple(
        Arm(
            "both_high_purity",
            "reorder",
            radius,
            purity_snap,
            POSITIVE_KINDS,
            purity_threshold,
        )
        for radius in radii
    )
    return edge_free + grid + purity


def select(rows: Sequence[LabelledCandidate], arm: Arm) -> tuple[LabelledCandidate, ...]:
    """The candidates one arm's oracle chooses to connect.

    Args:
        rows: Every labelled candidate on the chip, generated at the largest
            swept radius and filtered here by gap.
        arm: The arm doing the choosing.

    Returns:
        The selected candidates, empty for an arm that adds no edges.
    """
    if not arm.positive_kinds or arm.radius is None or arm.label_snap is None:
        return ()
    return tuple(
        r
        for r in rows
        if r.candidate.gap <= arm.radius
        and r.labels[arm.label_snap] == "positive"
        and positive_kind(r.prop_dist) in arm.positive_kinds
        and (arm.min_purity is None or candidate_purity(r) >= arm.min_purity)
    )


def add_connectors(
    G: nx.MultiGraph, rows: Sequence[LabelledCandidate], inject_tolerance: float
) -> tuple[nx.MultiGraph, int]:
    """Draw a straight connector for each selected candidate.

    An ``endpoint_edge`` candidate's far end is a point in the interior of an
    existing edge, so that edge has to be split before anything can attach to
    it. ``geograph.inject_points`` performs exactly that split and reports the
    node each point became, and it handles several points landing on one edge,
    so every such target is injected in a single call before any connector is
    drawn.

    Args:
        G: Pre-prune proposal graph; not modified.
        rows: Candidates the oracle selected.
        inject_tolerance: Furthest an edge may be from a target point and still
            receive it. The target already lies on its edge, so this only
            absorbs floating-point error from the projection.

    Returns:
        The augmented graph, and how many connectors were actually drawn. That
        count falls below ``len(rows)`` when a connector would have duplicated
        an edge already present, when two candidates propose the same pair, or
        when ``inject_points`` declined to split.
    """
    interior = [r for r in rows if r.candidate.node_b is None]
    if interior:
        points = np.array([r.candidate.b for r in interior], dtype=float)
        out, landed = geograph.inject_points(G, points, inject_tolerance)
    else:
        out, landed = G.copy(), []

    pairs = [
        (r.candidate.node_a, r.candidate.node_b)
        for r in rows
        if r.candidate.node_b is not None
    ] + [(r.candidate.node_a, node) for r, node in zip(interior, landed, strict=True)]

    added = 0
    for a, b in pairs:
        if b is None or a == b or out.has_edge(a, b):
            continue
        pts = np.vstack([out.nodes[a]["pos"], out.nodes[b]["pos"]]).astype(float)
        length = geograph.polyline_length(pts)
        if length <= 0.0:
            continue
        out.add_edge(a, b, pts=pts, length=length)
        added += 1
    return out, added


def decode(
    raw: nx.MultiGraph,
    preprune: nx.MultiGraph,
    rows: Sequence[LabelledCandidate],
    arm: Arm,
    simplify_tolerance: float,
    spur_length: float,
    snap_tolerance: float,
    inject_tolerance: float,
) -> tuple[nx.MultiGraph, int, int]:
    """Build one arm's proposal graph.

    The prune runs *after* the oracle edges are added, so a stub the oracle
    declines to connect is still pruned and nothing survives that the shipped
    pipeline would have deleted.

    Args:
        raw: Traced skeleton, before any cleanup.
        preprune: ``simplify_edges -> snap_junctions`` of the same skeleton,
            which is the graph the candidates were generated on.
        rows: Every labelled candidate on the chip.
        arm: Which arm to build.
        simplify_tolerance: Douglas-Peucker tolerance in pixels.
        spur_length: Shortest dead-end edge worth keeping.
        snap_tolerance: Junction merge radius in pixels.
        inject_tolerance: Snap radius for injecting ``endpoint_edge`` targets.

    Returns:
        ``(graph, n_selected, n_edges_added)``.
    """
    if arm.decoder == "clean":
        return cleanup.clean(raw, simplify_tolerance, spur_length, snap_tolerance), 0, 0

    selected = select(rows, arm)
    linked, added = add_connectors(preprune, selected, inject_tolerance)
    return cleanup.prune_spurs(linked, spur_length), len(selected), added


def chip_candidates(
    preprune: nx.MultiGraph,
    sample: data.TileSample,
    sample_id: str,
    max_radius: float,
    border_margin: float,
    tau: float,
    eps: float,
    label_snaps: tuple[float, ...],
    purity_spacing: float,
    purity_dilate: float,
) -> tuple[LabelledCandidate, ...]:
    """Generate and label every candidate on one chip.

    Arm-independent, so this runs once per chip and every arm filters the
    result. Both candidate kinds are pooled: ``endpoint_edge`` is what supplies
    the endpoint coverage, and ``endpoint_endpoint`` supplies alternative
    targets for the same endpoints, which for a *ceiling* is reach the oracle
    should be allowed to use.

    Args:
        preprune: ``simplify_edges -> snap_junctions`` proposal graph.
        sample: Chip carrying the truth graph and label mask.
        sample_id: Recorded on each row so results stay traceable.
        max_radius: Largest swept radius; smaller ones filter by gap.
        border_margin: Endpoints this close to the chip edge are chipping
            artifacts rather than gaps.
        tau: Directness bound on the truth route.
        eps: How much worse the proposal route must be to count as a gap.
        label_snaps: Snap radii to label at. Every candidate is labelled once
            per value.
        purity_spacing: Polyline resampling step for stub purity, in pixels.
        purity_dilate: Extra dilation on the truth mask before purity is read.

    Returns:
        One labelled candidate per proposed reconnection.
    """
    nodes, pos = endpoints(preprune)
    keep = interior_flags(pos, sample.mask.shape, border_margin)
    interior_nodes = [n for n, k in zip(nodes, keep, strict=True) if k]
    interior_pos = pos[keep]

    candidates: tuple[Candidate, ...] = endpoint_endpoint_candidates(
        preprune, interior_nodes, interior_pos, max_radius
    ) + endpoint_edge_candidates(preprune, interior_nodes, interior_pos, max_radius)
    if not candidates:
        return ()

    # One Dijkstra per source endpoint, reused across that endpoint's
    # candidates and across every radius, snap radius and arm.
    lengths = {
        n: nx.single_source_dijkstra_path_length(preprune, n, weight="length")
        for n in {c.node_a for c in candidates}
    }
    purities = endpoint_purities(
        preprune,
        truth_mask(sample, purity_dilate),
        sorted(
            {c.node_a for c in candidates}
            | {c.node_b for c in candidates if c.node_b is not None}
        ),
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
                split="val",
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
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class MarginalEdge:
    """One oracle edge's worth, measured with that edge and nothing else added.

    The reference is the arm's own edge-free decoder rather than the shipped
    ``cleanup.clean``: the reorder moves the score on its own, and charging that
    to whichever edge happened to be scored first would be an artifact.

    Attributes:
        sample_id: Chip the edge was added on.
        arm: Arm whose selection this edge came from.
        kind: ``endpoint_endpoint`` or ``endpoint_edge``.
        positive_kind: ``component_join`` or ``detour_shortcut``.
        gap: Straight-line length of the connector, in pixels.
        purity: Lower stub purity of the candidate's two ends.
        n_edges_added: 1 normally, 0 when the connector was declined, which
            makes every delta on the row exactly zero.
        deltas: Per-metric change from adding this edge alone.
    """

    sample_id: str
    arm: str
    kind: str
    positive_kind: str
    gap: float
    purity: float
    n_edges_added: int
    deltas: Mapping[str, float | None]


def marginal_edges(
    truth: nx.MultiGraph,
    preprune: nx.MultiGraph,
    rows: Sequence[LabelledCandidate],
    arm: Arm,
    baseline: Mapping[str, float | None],
    spur_length: float,
    inject_tolerance: float,
    buffers: tuple[float, ...],
    junction_radii: tuple[float, ...],
    junction_min_degree: int,
) -> tuple[MarginalEdge, ...]:
    """Score each of one arm's oracle edges on its own.

    Args:
        truth: Ground-truth graph for the chip.
        preprune: ``simplify_edges -> snap_junctions`` proposal graph.
        rows: Every labelled candidate on the chip.
        arm: The arm whose selection is taken apart.
        baseline: Metric map of the same decoder with no edges added.
        spur_length: Shortest dead-end edge worth keeping.
        inject_tolerance: Snap radius for injecting ``endpoint_edge`` targets.
        buffers: Buffer half-widths for length coverage.
        junction_radii: Radii for junction matching.
        junction_min_degree: Degree at or above which a node is a junction.

    Returns:
        One row per selected candidate, carrying its per-metric delta.
    """
    out: list[MarginalEdge] = []
    for row in select(rows, arm):
        linked, added = add_connectors(preprune, (row,), inject_tolerance)
        values, _ = score_metrics(
            truth,
            cleanup.prune_spurs(linked, spur_length),
            buffers,
            junction_radii,
            junction_min_degree,
        )
        out.append(
            MarginalEdge(
                sample_id=row.sample_id,
                arm=arm_key(arm),
                kind=row.candidate.kind,
                positive_kind=positive_kind(row.prop_dist),
                gap=row.candidate.gap,
                purity=candidate_purity(row),
                n_edges_added=added,
                deltas={
                    name: (
                        None
                        if value is None or baseline.get(name) is None
                        else value - (baseline[name] or 0.0)
                    )
                    for name, value in values.items()
                },
            )
        )
    return tuple(out)


def score_chip(
    model,
    source: data.TileSource,
    sample_id: str,
    arms: tuple[Arm, ...],
    threshold: float,
    border_margin: float,
    tau: float,
    eps: float,
    label_snaps: tuple[float, ...],
    purity_spacing: float,
    purity_dilate: float,
    simplify_tolerance: float,
    spur_length: float,
    snap_tolerance: float,
    inject_tolerance: float,
    buffers: tuple[float, ...],
    junction_radii: tuple[float, ...],
    junction_min_degree: int,
    marginal_arm: Arm | None,
    device: str,
) -> tuple[tuple[ChipArmScore, ...], tuple[MarginalEdge, ...], float]:
    """Score one chip under every arm and every metric.

    Args:
        model: Frozen checkpoint.
        source: Where imagery and labels come from.
        sample_id: Chip to score.
        arms: Every arm to score, which all share this chip's cached work.
        threshold: Probability above which a pixel counts as road.
        border_margin: Chip-edge margin in pixels.
        tau: Directness bound on the truth route.
        eps: How much worse the proposal route must be to count as a gap.
        label_snaps: Snap radii to label at.
        purity_spacing: Polyline resampling step for stub purity, in pixels.
        purity_dilate: Extra dilation on the truth mask before purity is read.
        simplify_tolerance: Douglas-Peucker tolerance in pixels.
        spur_length: Shortest dead-end edge worth keeping.
        snap_tolerance: Junction merge radius in pixels.
        inject_tolerance: Snap radius for injecting ``endpoint_edge`` targets.
        buffers: Buffer half-widths for length coverage.
        junction_radii: Radii for junction matching.
        junction_min_degree: Degree at or above which a node is a junction.
        marginal_arm: Arm to additionally re-score edge by edge, or ``None`` to
            skip that pass entirely.
        device: Device string for inference.

    Returns:
        One score per arm, the per-edge marginals, and the chip's ceiling APLS.
        An empty tuple of scores means the chip was skipped: either it carries
        no ground-truth roads, or a perfect mask scores zero on it and no arm
        could be judged there.
    """
    sample = source.load(sample_id)
    if sample.truth.number_of_edges() == 0:
        return (), (), 0.0

    ceiling = cleanup.clean(
        skeleton.graph_from_mask(sample.mask),
        simplify_tolerance,
        spur_length,
        snap_tolerance,
    )
    ceiling_apls = metrics.apls(sample.truth, ceiling).score
    if ceiling_apls == 0.0:
        return (), (), 0.0

    logits = train.predict_tile_logits(model, sample, device=device)
    raw = skeleton.graph_from_mask(predict_mask(logits, threshold))
    preprune = cleanup.snap_junctions(
        cleanup.simplify_edges(raw, simplify_tolerance), snap_tolerance
    )
    rows = chip_candidates(
        preprune,
        sample,
        sample_id,
        max(a.radius for a in arms if a.radius is not None),
        border_margin,
        tau,
        eps,
        label_snaps,
        purity_spacing,
        purity_dilate,
    )

    scores: list[ChipArmScore] = []
    edge_free: dict[str, float | None] = {}
    for arm in arms:
        graph, n_selected, n_added = decode(
            raw,
            preprune,
            rows,
            arm,
            simplify_tolerance,
            spur_length,
            snap_tolerance,
            inject_tolerance,
        )
        values, reference = score_metrics(
            sample.truth, graph, buffers, junction_radii, junction_min_degree
        )
        if arm.decoder == "reorder" and not arm.positive_kinds:
            edge_free = values
        scores.append(
            ChipArmScore(
                sample_id=sample_id,
                arm=arm_key(arm),
                n_selected=n_selected,
                n_edges_added=n_added,
                n_control_gt=reference.n_control_gt,
                n_control_prop=reference.n_control_prop,
                values=values,
            )
        )

    marginals = (
        marginal_edges(
            sample.truth,
            preprune,
            rows,
            marginal_arm,
            edge_free,
            spur_length,
            inject_tolerance,
            buffers,
            junction_radii,
            junction_min_degree,
        )
        if marginal_arm is not None
        else ()
    )
    return tuple(scores), marginals, ceiling_apls


def summarize(
    scores: Sequence[ChipArmScore], arm: Arm, names: tuple[str, ...]
) -> ArmSummary:
    """Aggregate one arm across the holdout.

    Args:
        scores: Every chip-arm score; filtered to this arm here.
        arm: The arm to summarize.
        names: Metric keys to aggregate, from :func:`metric_names`.

    Returns:
        Per-metric means and medians, plus what the oracle added.
    """
    key = arm_key(arm)
    rows = [s for s in scores if s.arm == key]

    def defined(name: str) -> np.ndarray:
        return np.array(
            [r.values[name] for r in rows if r.values.get(name) is not None], dtype=float
        )

    means = {n: (float(v.mean()) if len(v := defined(n)) else None) for n in names}
    medians = {n: (float(np.median(v)) if len(v := defined(n)) else None) for n in names}

    return ArmSummary(
        arm=key,
        name=arm.name,
        decoder=arm.decoder,
        radius=arm.radius,
        label_snap=arm.label_snap,
        positive_kinds=tuple(arm.positive_kinds),
        min_purity=arm.min_purity,
        n_chips=len(rows),
        selected_total=sum(r.n_selected for r in rows),
        edges_added_total=sum(r.n_edges_added for r in rows),
        edges_added_mean=(
            float(np.mean([r.n_edges_added for r in rows])) if rows else 0.0
        ),
        means=means,
        medians=medians,
    )


def paired_deltas(
    scores: Sequence[ChipArmScore], arm: str, reference: str, metric: str
) -> np.ndarray:
    """Per-chip difference on one metric, arm minus reference.

    Chips where either side left the metric undefined are dropped rather than
    read as zero: a chip with no matched junction has no degree agreement to
    difference, and calling that "unchanged" would count it as a tie.

    Args:
        scores: Every chip-arm score.
        arm: Arm key under test.
        reference: Arm key to measure against.
        metric: Metric key, from :func:`metric_names`.

    Returns:
        One delta per chip where both sides defined the metric.
    """
    lookup = {(s.arm, s.sample_id): s for s in scores}
    sample_ids = [s.sample_id for s in scores if s.arm == reference]
    pairs = (
        (lookup[(arm, i)].values.get(metric), lookup[(reference, i)].values.get(metric))
        for i in sample_ids
        if (arm, i) in lookup
    )
    return np.array(
        [a - b for a, b in pairs if a is not None and b is not None], dtype=float
    )


@dataclass(frozen=True, slots=True)
class GainDistribution:
    """How one arm's gain is spread over chips, rather than averaged over them.

    The mean answers "how much", this answers "how concentrated". A metric that
    weights by betweenness pays out on the few chips holding a severed arterial
    and nothing on the rest; one that weights by length should pay out more
    evenly, and whether it actually does is the measurement.

    Attributes:
        arm: Arm key under test.
        reference: Arm key it is measured against.
        metric: Metric key the gain is measured on.
        n: Chips contributing.
        n_zero: Chips whose metric did not move at all.
        n_gain: Chips that improved.
        n_loss: Chips that got worse.
        total_delta: Summed per-chip delta, the net gain.
        total_gain: Summed positive deltas alone.
        total_loss: Summed negative deltas alone, as a negative number.
        top5_share_of_total: Gain carried by the five best chips as a share of
            ``total_delta``. Above 1.0 when losses elsewhere eat into a net
            gain the top five alone exceed.
        top5_share_of_gain: The same numerator over ``total_gain``, which
            unlike the line above cannot exceed 1.0.
        gini: Gini coefficient of the per-chip gain, clipped at zero. 0.0 means
            every chip gained equally, 1.0 that one chip carries everything.
            Scale-free, so it compares across metrics of different sizes.
    """

    arm: str
    reference: str
    metric: str
    n: int
    n_zero: int
    n_gain: int
    n_loss: int
    total_delta: float
    total_gain: float
    total_loss: float
    top5_share_of_total: float | None
    top5_share_of_gain: float | None
    gini: float | None


def gini_coefficient(values: np.ndarray) -> float:
    """Concentration of a non-negative vector, 0.0 even to 1.0 all in one place."""
    ordered = np.sort(values)
    n = len(ordered)
    total = ordered.sum()
    if n == 0 or total <= 0.0:
        return 0.0
    index = np.arange(1, n + 1, dtype=float)
    return float((2.0 * (index * ordered).sum()) / (n * total) - (n + 1.0) / n)


def gain_distribution(
    scores: Sequence[ChipArmScore],
    arm: str,
    reference: str,
    metric: str,
    top_k: int = 5,
) -> GainDistribution:
    """Measure how one arm's gain is spread across chips.

    Args:
        scores: Every chip-arm score.
        arm: Arm key under test.
        reference: Arm key to measure against.
        metric: Metric key, from :func:`metric_names`.
        top_k: How many of the best chips the concentration shares cover.

    Returns:
        Counts, totals and two concentration measures. The shares and the Gini
        are ``None`` when no chip gained anything, which is a different fact
        from a gain spread perfectly evenly.
    """
    deltas = paired_deltas(scores, arm, reference, metric)
    gains = np.clip(deltas, 0.0, None)
    total_delta = float(deltas.sum())
    total_gain = float(gains.sum())
    top = float(np.sort(gains)[-top_k:].sum())

    return GainDistribution(
        arm=arm,
        reference=reference,
        metric=metric,
        n=len(deltas),
        n_zero=int((deltas == 0).sum()),
        n_gain=int((deltas > 0).sum()),
        n_loss=int((deltas < 0).sum()),
        total_delta=total_delta,
        total_gain=total_gain,
        total_loss=float(np.clip(deltas, None, 0.0).sum()),
        top5_share_of_total=(top / total_delta if total_delta > 0.0 else None),
        top5_share_of_gain=(top / total_gain if total_gain > 0.0 else None),
        gini=(gini_coefficient(gains) if total_gain > 0.0 else None),
    )


def paired_stat(
    scores: Sequence[ChipArmScore], arm: str, reference: str, metric: str
) -> PairedStat:
    """Compare one arm against a reference, chip by chip.

    Args:
        scores: Every chip-arm score.
        arm: Arm key under test.
        reference: Arm key to measure against.
        metric: Metric key, from :func:`metric_names`.

    Returns:
        The mean per-chip difference with its standard error, t statistic, 95%
        interval, and the better/worse/tied tile counts. ``t`` and the interval
        are ``None`` when every chip moved by the same amount, which makes the
        statistic undefined rather than infinite. Chips where either side left
        the metric undefined are dropped, so ``n`` can fall below the holdout.
    """
    deltas = paired_deltas(scores, arm, reference, metric)
    n = len(deltas)
    mean_delta = float(deltas.mean()) if n else 0.0
    sem = float(stats.sem(deltas)) if n > 1 else 0.0
    defined = n > 1 and sem > 0.0
    half_width = float(stats.t.ppf(0.975, n - 1)) * sem if defined else 0.0

    return PairedStat(
        arm=arm,
        reference=reference,
        metric=metric,
        n=n,
        mean_delta=mean_delta,
        sem=sem,
        t=mean_delta / sem if defined else None,
        ci_low=mean_delta - half_width if defined else None,
        ci_high=mean_delta + half_width if defined else None,
        n_better=int((deltas > 0).sum()),
        n_worse=int((deltas < 0).sum()),
        n_tied=int((deltas == 0).sum()),
    )


@dataclass(frozen=True, slots=True)
class MarginalSummary:
    """What one arm's edges are worth individually, on one metric.

    Attributes:
        arm: Arm whose selection was taken apart.
        metric: Metric key the deltas are measured on.
        group: ``all``, or one positive class on its own.
        n_edges: Selected candidates contributing.
        n_positive: Edges that improved the metric alone.
        n_zero: Edges that changed nothing, including the ones whose connector
            was declined.
        n_negative: Edges that damaged the metric alone.
        mean_delta: Mean per-edge contribution, the quantity a link predictor
            should be selecting on.
        median_delta: Median of the same.
        total_delta: Summed contribution. Compare against the arm's own joint
            delta: the two differ exactly to the extent edges interact, and a
            joint delta above the sum means edges only pay off together.
        top5_share_of_gain: Share of the summed positive contribution carried
            by the five best edges, or ``None`` when nothing gained. Near 1.0
            means a predictor would have to find those five specifically.
        gini: Concentration of the per-edge gain, or ``None`` likewise.
    """

    arm: str
    metric: str
    group: str
    n_edges: int
    n_positive: int
    n_zero: int
    n_negative: int
    mean_delta: float
    median_delta: float
    total_delta: float
    top5_share_of_gain: float | None
    gini: float | None


def summarize_marginals(
    edges: Sequence[MarginalEdge], arm: str, metric: str, group: str, top_k: int = 5
) -> MarginalSummary:
    """Aggregate per-edge contributions on one metric.

    Args:
        edges: Every marginal row; filtered to ``group`` here.
        arm: Arm key the rows came from.
        metric: Metric key, from :func:`metric_names`.
        group: ``all``, or a positive class to restrict to.
        top_k: How many of the best edges the concentration share covers.

    Returns:
        Counts, central tendency, and how concentrated the value is.
    """
    selected = [e for e in edges if group in ("all", e.positive_kind)]
    deltas = np.array(
        [e.deltas[metric] for e in selected if e.deltas.get(metric) is not None],
        dtype=float,
    )
    gains = np.clip(deltas, 0.0, None)
    total_gain = float(gains.sum())
    top = float(np.sort(gains)[-top_k:].sum())

    return MarginalSummary(
        arm=arm,
        metric=metric,
        group=group,
        n_edges=len(deltas),
        n_positive=int((deltas > 0).sum()),
        n_zero=int((deltas == 0).sum()),
        n_negative=int((deltas < 0).sum()),
        mean_delta=float(deltas.mean()) if len(deltas) else 0.0,
        median_delta=float(np.median(deltas)) if len(deltas) else 0.0,
        total_delta=float(deltas.sum()),
        top5_share_of_gain=(top / total_gain if total_gain > 0.0 else None),
        gini=(gini_coefficient(gains) if total_gain > 0.0 else None),
    )


def flatten(summary: ArmSummary) -> dict:
    """One arm summary as a flat JSON row, metric means beside the metadata."""
    row = {k: v for k, v in asdict(summary).items() if k not in ("means", "medians")}
    return (
        row
        | dict(summary.means)
        | {f"{name}_median": value for name, value in summary.medians.items()}
    )


def headline_metrics(
    buffers: tuple[float, ...], junction_radii: tuple[float, ...]
) -> tuple[str, ...]:
    """The metrics the console table prints, one per family."""
    return (
        "apls",
        "apls_uniform",
        f"buffer_f1_{buffers[len(buffers) // 2]:.0f}",
        f"junction_f1_{junction_radii[len(junction_radii) // 2]:.0f}",
    )


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/vegas_best.pt"))
    parser.add_argument(
        "--run-json",
        type=Path,
        default=Path("outputs/vegas_best.json"),
        help="run artifact supplying the val split the checkpoint was fit against",
    )
    parser.add_argument("--aoi-root", type=Path, default=Path("data/AOI_2_Vegas"))
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="the tuned operating point from outputs/threshold_sweep.json",
    )
    parser.add_argument(
        "--skip-tiles",
        type=int,
        default=40,
        help=(
            "validation tiles to skip before scoring; the first 40 were spent "
            "selecting the mask threshold, so the reporting holdout is what "
            "follows them"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="chips to score after the skip; 0 runs all of them, which is what "
        "a reported number needs",
    )
    parser.add_argument("--radii", type=float, nargs="+", default=list(DEFAULT_RADII))
    parser.add_argument(
        "--label-snap",
        type=float,
        nargs="+",
        default=list(DEFAULT_LABEL_SNAPS),
        help=(
            "snap radii the labeller runs at, in pixels; the gap between the "
            "arms at two values is the label-noise measurement, not a tuning "
            "sweep. The labeller's own parameter, not metrics.apls's max_snap"
        ),
    )
    parser.add_argument(
        "--border-margin",
        type=float,
        default=10.0,
        help="endpoints this close to the chip edge are chipping artifacts",
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
        "--purity-threshold",
        type=float,
        default=0.5,
        help="stub purity at or above which the high-purity arm keeps a positive",
    )
    parser.add_argument(
        "--purity-snap",
        type=float,
        default=10.0,
        help="the one label snap the high-purity arm is scored at",
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
        help="extra dilation on the truth mask before purity is read",
    )
    parser.add_argument("--simplify-tolerance", type=float, default=2.0)
    parser.add_argument("--spur-length", type=float, default=20.0)
    parser.add_argument("--snap-tolerance", type=float, default=8.0)
    parser.add_argument(
        "--inject-tolerance",
        type=float,
        default=1.0,
        help=(
            "snap radius for injecting an endpoint_edge target into its own "
            "edge; the point already lies on that edge, so this only absorbs "
            "projection round-off"
        ),
    )
    parser.add_argument(
        "--buffers",
        type=float,
        nargs="+",
        default=list(DEFAULT_BUFFERS),
        help=(
            "buffer half-widths for length coverage, in pixels; displacement "
            "below the buffer is invisible to that metric, so the sweep is the "
            "geometric tolerance the number is quoted at"
        ),
    )
    parser.add_argument(
        "--junction-radii",
        type=float,
        nargs="+",
        default=list(DEFAULT_JUNCTION_RADII),
        help="radii, in pixels, truth and proposal junctions are matched within",
    )
    parser.add_argument(
        "--junction-min-degree",
        type=int,
        default=metrics.JUNCTION_MIN_DEGREE,
        help="degree at or above which a node counts as a junction",
    )
    parser.add_argument(
        "--marginal-arm",
        default=DEFAULT_MARGINAL_ARM,
        help=(
            "arm key to additionally re-score edge by edge, measuring what each "
            "oracle edge is worth alone; empty string skips that pass"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("outputs/link_oracle.json"))
    args = parser.parse_args()

    run = json.loads(args.run_json.read_text())
    limit = args.limit or None
    requested = list(run["config"]["val_ids"])[args.skip_tiles :][:limit]
    radii = tuple(sorted(float(r) for r in args.radii))
    label_snaps = tuple(sorted({float(s) for s in args.label_snap}))
    buffers = tuple(sorted({float(b) for b in args.buffers}))
    junction_radii = tuple(sorted({float(r) for r in args.junction_radii}))
    arms = build_arms(radii, label_snaps, args.purity_threshold, args.purity_snap)
    names = metric_names(buffers, junction_radii)

    by_key = {arm_key(a): a for a in arms}
    if args.marginal_arm and args.marginal_arm not in by_key:
        parser.error(
            f"--marginal-arm {args.marginal_arm!r} is not a scored arm; "
            f"choose one of {sorted(by_key)}"
        )
    marginal_arm = by_key.get(args.marginal_arm) if args.marginal_arm else None

    source, _ = train.build_source(args.aoi_root, "spacenet", args.resolution)
    model = train.load_checkpoint(args.checkpoint, device=args.device)
    logger.info(
        f"{len(requested)} chips x {len(arms)} arms x {len(names)} metrics at "
        f"threshold {args.threshold}, radii {radii}, label snaps {label_snaps}, "
        f"buffers {buffers}, junction radii {junction_radii}"
    )
    if marginal_arm is not None:
        logger.info(f"per-edge marginals for {arm_key(marginal_arm)}")

    started = time.perf_counter()
    scores: list[ChipArmScore] = []
    marginals: list[MarginalEdge] = []
    scored_ids: list[str] = []
    skipped: list[str] = []
    ceilings: dict[str, float] = {}
    for n, sample_id in enumerate(requested, start=1):
        chip_scores, chip_marginals, ceiling_apls = score_chip(
            model,
            source,
            sample_id,
            arms,
            args.threshold,
            args.border_margin,
            args.tau,
            args.eps,
            label_snaps,
            args.purity_spacing,
            args.purity_dilate,
            args.simplify_tolerance,
            args.spur_length,
            args.snap_tolerance,
            args.inject_tolerance,
            buffers,
            junction_radii,
            args.junction_min_degree,
            marginal_arm,
            args.device,
        )
        if not chip_scores:
            # A perfect mask scores zero here, so no arm can be judged on this
            # chip. Degenerate, not a hard case.
            logger.warning(f"{sample_id}: zero ceiling, skipping")
            skipped.append(sample_id)
            continue

        scores.extend(chip_scores)
        marginals.extend(chip_marginals)
        scored_ids.append(sample_id)
        ceilings[sample_id] = ceiling_apls
        by_arm = {s.arm: s for s in chip_scores}
        best = max(chip_scores, key=lambda s: s.values["apls"] or 0.0)
        coverage = f"buffer_f1_{buffers[len(buffers) // 2]:.0f}"
        logger.info(
            f"[{n}/{len(requested)}] {sample_id}: ceiling {ceiling_apls:.4f}, "
            f"baseline {by_arm['baseline'].values['apls']:.4f}"
            f"/{by_arm['baseline'].values[coverage]:.4f}, "
            f"reorder {by_arm['reorder'].values['apls']:.4f}, "
            f"best {best.arm} {best.values['apls']:.4f} (+{best.n_edges_added} edges)"
        )

    elapsed = time.perf_counter() - started
    summaries = tuple(summarize(scores, a, names) for a in arms)
    comparisons = tuple(
        (arm_key(a), reference, metric)
        for a in arms
        for reference in ("baseline", "reorder")
        for metric in names
        if arm_key(a) != reference
    )
    paired = tuple(paired_stat(scores, *c) for c in comparisons)
    gains = tuple(gain_distribution(scores, *c) for c in comparisons)
    marginal_summaries = tuple(
        summarize_marginals(marginals, arm_key(marginal_arm), metric, group)
        for metric in names
        for group in ("all", "component_join", "detour_shortcut")
        if marginal_arm is not None
    )
    delta = {(p.arm, p.reference, p.metric): p for p in paired}
    spread = {(g.arm, g.reference, g.metric): g for g in gains}

    for metric in headline_metrics(buffers, junction_radii):
        logger.info(f"--- {metric} ---")
        logger.info(
            f"{'arm':>26} {'mean':>7} {'med':>7} {'edges':>6} {'Δvs0':>8} "
            f"{'Δvs1':>8} {'t vs1':>7} {'b/w/=':>12} {'top5':>6} {'gini':>6}"
        )
        for s in summaries:
            vs_baseline = delta.get((s.arm, "baseline", metric))
            vs_reorder = delta.get((s.arm, "reorder", metric))
            concentration = spread.get((s.arm, "reorder", metric))
            d0 = vs_baseline.mean_delta if vs_baseline else 0.0
            d1 = vs_reorder.mean_delta if vs_reorder else 0.0
            t = (
                "n/a"
                if vs_reorder is None or vs_reorder.t is None
                else f"{vs_reorder.t:+.2f}"
            )
            tiles = (
                ""
                if vs_reorder is None
                else f"{vs_reorder.n_better}/{vs_reorder.n_worse}/{vs_reorder.n_tied}"
            )
            top5 = (
                ""
                if concentration is None or concentration.top5_share_of_gain is None
                else f"{concentration.top5_share_of_gain:.2f}"
            )
            gini = (
                ""
                if concentration is None or concentration.gini is None
                else f"{concentration.gini:.2f}"
            )
            logger.info(
                f"{s.arm:>26} {s.means[metric] or 0.0:>7.4f} "
                f"{s.medians[metric] or 0.0:>7.4f} {s.edges_added_total:>6} "
                f"{d0:>+8.4f} {d1:>+8.4f} {t:>7} {tiles:>12} {top5:>6} {gini:>6}"
            )

    for summary in marginal_summaries:
        if summary.group != "all" or summary.metric not in headline_metrics(
            buffers, junction_radii
        ):
            continue
        logger.info(
            f"marginal {summary.metric:>16}: n={summary.n_edges} "
            f"mean {summary.mean_delta:+.5f} median {summary.median_delta:+.5f} "
            f"sum {summary.total_delta:+.4f} "
            f"(+{summary.n_positive}/-{summary.n_negative}/={summary.n_zero})"
        )

    mean_ceiling = float(np.mean(list(ceilings.values()))) if ceilings else 0.0
    logger.info(
        f"scored {len(scored_ids)} chips, skipped {len(skipped)} for zero ceiling; "
        f"mean ceiling {mean_ceiling:.4f}"
    )
    logger.info(f"wall time {elapsed / 60:.1f} min")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "run_json": str(args.run_json),
                "params": {
                    "threshold": args.threshold,
                    "radii": list(radii),
                    "label_snaps": list(label_snaps),
                    "border_margin": args.border_margin,
                    "tau": args.tau,
                    "eps": args.eps,
                    "purity_threshold": args.purity_threshold,
                    "purity_snap": args.purity_snap,
                    "purity_spacing": args.purity_spacing,
                    "purity_dilate": args.purity_dilate,
                    "simplify_tolerance": args.simplify_tolerance,
                    "spur_length": args.spur_length,
                    "snap_tolerance": args.snap_tolerance,
                    "inject_tolerance": args.inject_tolerance,
                    "skip_tiles": args.skip_tiles,
                    "limit": args.limit,
                    "resolution": args.resolution,
                    "device": args.device,
                    "apls_spacing": 50.0,
                    "apls_max_snap": 25.0,
                    "apls_sampling": "reference",
                    "buffers": list(buffers),
                    "junction_radii": list(junction_radii),
                    "junction_min_degree": args.junction_min_degree,
                    "marginal_arm": args.marginal_arm or None,
                    "marginal_reference": "the arm's own edge-free decoder (reorder)",
                    "metric_names": list(names),
                    "baseline_pipeline": "cleanup.clean",
                    "oracle_pipeline": (
                        "simplify_edges -> snap_junctions -> oracle edges -> prune_spurs"
                    ),
                    "connector": "straight line",
                    "candidate_kinds": ["endpoint_endpoint", "endpoint_edge"],
                },
                "elapsed_seconds": elapsed,
                "n_requested": len(requested),
                "sample_ids": scored_ids,
                "skipped_zero_ceiling": skipped,
                "ceiling_apls": ceilings,
                "ceiling_apls_mean": mean_ceiling,
                "arms": [asdict(a) | {"arm": arm_key(a)} for a in arms],
                "summaries": [flatten(s) for s in summaries],
                "paired_stats": [asdict(p) for p in paired],
                "gain_distributions": [asdict(g) for g in gains],
                "marginal_summaries": [asdict(m) for m in marginal_summaries],
                "marginal_edges": [
                    {k: v for k, v in asdict(m).items() if k != "deltas"} | dict(m.deltas)
                    for m in marginals
                ],
                "per_chip": [
                    {k: v for k, v in asdict(s).items() if k != "values"} | dict(s.values)
                    for s in scores
                ],
            },
            indent=2,
        )
    )
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()

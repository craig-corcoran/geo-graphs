"""Score an oracle gap-closer against APLS, to bound what a link predictor can buy.

A learned link predictor is only worth building if the edges it would add are
worth adding at all. This adds exactly the edges the ground truth says are
missing and scores the result, so the ceiling of the whole line of work is
measured before any model exists. An oracle worth +0.005 ends it; one worth
+0.06 justifies it.

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
labelled candidate set. Only edge insertion, pruning and APLS repeat per arm.

The connector is a straight line between the two candidate points. That
*understates* the ceiling: a real severed road that curves is reconstructed by a
chord, and the length error a chord introduces is exactly what APLS measures. A
connector following the probability ridge would score at least as well.
"""

import argparse
import json
import time
from collections.abc import Sequence
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
    """One chip scored under one arm.

    Attributes:
        sample_id: Which chip this scored.
        arm: Arm key, from :func:`arm_key`.
        apls: Harmonic mean of the two directions.
        gt_to_prop: Falls when real roads are missing.
        prop_to_gt: Falls when roads are invented.
        n_selected: Positives the oracle chose to connect.
        n_edges_added: How many of those became an edge. Below ``n_selected``
            when a connector would have duplicated an edge already present or
            when ``inject_points`` declined to split.
        n_control_gt: Control points scoring ``gt_to_prop``; fixed across arms,
            since they are sampled from the truth graph.
        n_control_prop: Control points scoring ``prop_to_gt``. Moves with the
            arm, because splitting a target edge adds a node the proposal then
            samples.
    """

    sample_id: str
    arm: str
    apls: float
    gt_to_prop: float
    prop_to_gt: float
    n_selected: int
    n_edges_added: int
    n_control_gt: int
    n_control_prop: int


@dataclass(frozen=True, slots=True)
class ArmSummary:
    """Aggregate quality for one arm.

    Attributes:
        arm: Arm key.
        name: Which arm of the table this belongs to.
        decoder: Which decoder ran.
        radius: Proposal radius, or ``None``.
        label_snap: Labeller snap radius, or ``None``.
        positive_kinds: Which positive classes the oracle added.
        min_purity: Stub purity cut, or ``None``.
        n_chips: Chips contributing.
        apls: Mean combined APLS.
        apls_median: Median of the same, which one bad chip cannot drag.
        gt_to_prop: Mean of the missed-road direction.
        gt_to_prop_median: Median of the same.
        prop_to_gt: Mean of the invented-road direction.
        prop_to_gt_median: Median of the same.
        selected_total: Positives selected across the holdout.
        edges_added_total: Oracle edges actually added across the holdout.
        edges_added_mean: The same per chip.
    """

    arm: str
    name: str
    decoder: Decoder
    radius: float | None
    label_snap: float | None
    positive_kinds: tuple[str, ...]
    min_purity: float | None
    n_chips: int
    apls: float
    apls_median: float
    gt_to_prop: float
    gt_to_prop_median: float
    prop_to_gt: float
    prop_to_gt_median: float
    selected_total: int
    edges_added_total: int
    edges_added_mean: float


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
    device: str,
) -> tuple[tuple[ChipArmScore, ...], float]:
    """Score one chip under every arm.

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
        device: Device string for inference.

    Returns:
        One score per arm and the chip's ceiling APLS. An empty tuple of scores
        means the chip was skipped: either it carries no ground-truth roads, or
        a perfect mask scores zero on it and no arm could be judged there.
    """
    sample = source.load(sample_id)
    if sample.truth.number_of_edges() == 0:
        return (), 0.0

    ceiling = cleanup.clean(
        skeleton.graph_from_mask(sample.mask),
        simplify_tolerance,
        spur_length,
        snap_tolerance,
    )
    ceiling_apls = metrics.apls(sample.truth, ceiling).score
    if ceiling_apls == 0.0:
        return (), 0.0

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
        result = metrics.apls(sample.truth, graph)
        scores.append(
            ChipArmScore(
                sample_id=sample_id,
                arm=arm_key(arm),
                apls=result.score,
                gt_to_prop=result.gt_to_prop,
                prop_to_gt=result.prop_to_gt,
                n_selected=n_selected,
                n_edges_added=n_added,
                n_control_gt=result.n_control_gt,
                n_control_prop=result.n_control_prop,
            )
        )
    return tuple(scores), ceiling_apls


def summarize(scores: Sequence[ChipArmScore], arm: Arm) -> ArmSummary:
    """Aggregate one arm across the holdout.

    Args:
        scores: Every chip-arm score; filtered to this arm here.
        arm: The arm to summarize.

    Returns:
        Means and medians on all three metrics, plus what the oracle added.
    """
    key = arm_key(arm)
    rows = [s for s in scores if s.arm == key]

    def mean(attribute: str) -> float:
        return float(np.mean([getattr(r, attribute) for r in rows])) if rows else 0.0

    def median(attribute: str) -> float:
        return float(np.median([getattr(r, attribute) for r in rows])) if rows else 0.0

    return ArmSummary(
        arm=key,
        name=arm.name,
        decoder=arm.decoder,
        radius=arm.radius,
        label_snap=arm.label_snap,
        positive_kinds=tuple(arm.positive_kinds),
        min_purity=arm.min_purity,
        n_chips=len(rows),
        apls=mean("apls"),
        apls_median=median("apls"),
        gt_to_prop=mean("gt_to_prop"),
        gt_to_prop_median=median("gt_to_prop"),
        prop_to_gt=mean("prop_to_gt"),
        prop_to_gt_median=median("prop_to_gt"),
        selected_total=sum(r.n_selected for r in rows),
        edges_added_total=sum(r.n_edges_added for r in rows),
        edges_added_mean=mean("n_edges_added"),
    )


def paired_stat(
    scores: Sequence[ChipArmScore], arm: str, reference: str, metric: str
) -> PairedStat:
    """Compare one arm against a reference, chip by chip.

    Args:
        scores: Every chip-arm score.
        arm: Arm key under test.
        reference: Arm key to measure against.
        metric: ``apls``, ``gt_to_prop`` or ``prop_to_gt``.

    Returns:
        The mean per-chip difference with its standard error, t statistic, 95%
        interval, and the better/worse/tied tile counts. ``t`` and the interval
        are ``None`` when every chip moved by the same amount, which makes the
        statistic undefined rather than infinite.
    """
    lookup = {(s.arm, s.sample_id): s for s in scores}
    sample_ids = [s.sample_id for s in scores if s.arm == reference]
    deltas = np.array(
        [
            getattr(lookup[(arm, i)], metric) - getattr(lookup[(reference, i)], metric)
            for i in sample_ids
            if (arm, i) in lookup
        ],
        dtype=float,
    )

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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("outputs/link_oracle.json"))
    args = parser.parse_args()

    run = json.loads(args.run_json.read_text())
    limit = args.limit or None
    requested = list(run["config"]["val_ids"])[args.skip_tiles :][:limit]
    radii = tuple(sorted(float(r) for r in args.radii))
    label_snaps = tuple(sorted({float(s) for s in args.label_snap}))
    arms = build_arms(radii, label_snaps, args.purity_threshold, args.purity_snap)

    source, _ = train.build_source(args.aoi_root, "spacenet", args.resolution)
    model = train.load_checkpoint(args.checkpoint, device=args.device)
    logger.info(
        f"{len(requested)} chips x {len(arms)} arms at threshold {args.threshold}, "
        f"radii {radii}, label snaps {label_snaps}"
    )

    started = time.perf_counter()
    scores: list[ChipArmScore] = []
    scored_ids: list[str] = []
    skipped: list[str] = []
    ceilings: dict[str, float] = {}
    for n, sample_id in enumerate(requested, start=1):
        chip_scores, ceiling_apls = score_chip(
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
            args.device,
        )
        if not chip_scores:
            # A perfect mask scores zero here, so no arm can be judged on this
            # chip. Degenerate, not a hard case.
            logger.warning(f"{sample_id}: zero ceiling, skipping")
            skipped.append(sample_id)
            continue

        scores.extend(chip_scores)
        scored_ids.append(sample_id)
        ceilings[sample_id] = ceiling_apls
        by_arm = {s.arm: s for s in chip_scores}
        best = max(chip_scores, key=lambda s: s.apls)
        logger.info(
            f"[{n}/{len(requested)}] {sample_id}: ceiling {ceiling_apls:.4f}, "
            f"baseline {by_arm['baseline'].apls:.4f}, "
            f"reorder {by_arm['reorder'].apls:.4f}, "
            f"best {best.arm} {best.apls:.4f} (+{best.n_edges_added} edges)"
        )

    elapsed = time.perf_counter() - started
    summaries = tuple(summarize(scores, a) for a in arms)
    metric_names = ("apls", "gt_to_prop", "prop_to_gt")
    paired = tuple(
        paired_stat(scores, arm_key(a), reference, metric)
        for a in arms
        for reference in ("baseline", "reorder")
        for metric in metric_names
        if arm_key(a) != reference
    )
    delta = {(p.arm, p.reference, p.metric): p for p in paired}

    logger.info(
        f"{'arm':>26} {'APLS':>7} {'med':>7} {'gt→pr':>7} {'pr→gt':>7} "
        f"{'edges':>6} {'Δvs0':>8} {'Δvs1':>8} {'t vs1':>7} {'b/w':>9}"
    )
    for s in summaries:
        vs_baseline = delta.get((s.arm, "baseline", "apls"))
        vs_reorder = delta.get((s.arm, "reorder", "apls"))
        d0 = vs_baseline.mean_delta if vs_baseline else 0.0
        d1 = vs_reorder.mean_delta if vs_reorder else 0.0
        t = (
            "n/a"
            if vs_reorder is None or vs_reorder.t is None
            else f"{vs_reorder.t:+.2f}"
        )
        tiles = (
            "" if vs_reorder is None else f"{vs_reorder.n_better}/{vs_reorder.n_worse}"
        )
        logger.info(
            f"{s.arm:>26} {s.apls:>7.4f} {s.apls_median:>7.4f} "
            f"{s.gt_to_prop:>7.4f} {s.prop_to_gt:>7.4f} "
            f"{s.edges_added_total:>6} {d0:>+8.4f} {d1:>+8.4f} {t:>7} {tiles:>9}"
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
                "summaries": [asdict(s) for s in summaries],
                "paired_stats": [asdict(p) for p in paired],
                "per_chip": [asdict(s) for s in scores],
            },
            indent=2,
        )
    )
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()

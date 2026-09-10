"""Price APLS's snapping slack by scoring the same graphs at four snap radii.

``metrics.apls`` moves a control point up to ``max_snap`` metres onto the other
graph before measuring anything, and at the committed 25.0 that is about a
sixteenth of a 396x324 chip. A road drawn anywhere inside that radius measures
the same length and scores the same 1. This sweep asks what the reported score
would be if the slack were smaller.

**This is a diagnostic and its numbers are not the reported score.** The
committed ``max_snap`` of 25.0 is what reproduces the ``CosmiQ/apls`` reference
to 3e-5 and what makes 0.7976 comparable to the published SpaceNet Vegas
column. Every radius below it is on the same footing as ``sampling="uniform"``:
separately labelled, never quoted against a leaderboard. 25.0 is swept as the
reference row so the others can be read against it.

Two arenas are scored at every radius, because the model's error and the
decoder's error are different questions:

* ``proposal`` -- the shipped decoder's output, ``clean(trace(predict(image)))``
* ``ceiling`` -- the same decoder run on the *label* mask, so no model error is
  in the number. What it loses as the radius falls is geometric damage done by
  ``skeletonize -> simplify -> snap``, which APLS at 25 m has been forgiving.

**The score falls for two reasons and they are kept apart.** A control point
with no edge within ``max_snap`` has no ``l_b`` at all, so its pairs score 0
without any length ever being compared: an unmatched-point penalty, not a
displacement penalty. A pair can also score 0 because both points landed and
the target graph cannot route between them. Whatever deficit is left over is
length disagreement. ``metrics.APLSResult`` now carries the landing and pair
counts, and those three terms sum exactly to ``1 - score``.

Note what that decomposition implies before the numbers arrive.
``inject_points`` takes the *nearest* edge inside the radius, so shrinking the
radius can only withdraw a landing, never redirect one to a nearer edge, and a
pair whose two endpoints both still land keeps the ``l_b`` it had. The pair set
itself is fixed by the source graph and ``spacing``. So every directional score
should be monotone non-increasing in ``max_snap`` and the whole drop should
land in the unlanded term. Both are checked against the data rather than
assumed; a violation of either would mean this reasoning is wrong.

``metrics.buffer_length_prf`` is scored at the same four radii as an
independent measure. It reads displacement directly, off geometry, with no
snapping and no routing involved. If the two agree about how far off the line
these roads are, that is corroboration from a different mechanism; if they
disagree, the disagreement is the finding.

Retrains nothing. The checkpoint is the frozen prior stage, and the logits are
what the sweep cannot change, so they are computed once per chip and reused
across both arenas and all four radii.
"""

import argparse
import itertools
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np
from loguru import logger

from geo_graphs import cleanup, data, metrics, skeleton, train
from geo_graphs.model import predict_mask

#: Snap radii, in pixels (1 px = 1 m), APLS is scored at.
#:
#: 25.0 is ``metrics.apls``'s committed default and is present as the reference
#: row every other radius is read against, not as a candidate.
DEFAULT_MAX_SNAPS = (5.0, 10.0, 15.0, 25.0)

#: Buffer half-widths, in pixels, the independent coverage measure is scored at.
#:
#: Deliberately the same four values as :data:`DEFAULT_MAX_SNAPS`, since the
#: point of the cross-check is to read one against the other at equal tolerance.
DEFAULT_BUFFERS = (5.0, 10.0, 15.0, 25.0)

#: The radius the sweep is reported against, and the only one comparable to a
#: published number.
REFERENCE_MAX_SNAP = 25.0

#: Which graph is being scored against the truth.
#:
#: ``proposal`` carries model error and decoder error together; ``ceiling``
#: runs the same decoder on the label mask and so carries decoder error alone.
Arena = Literal["proposal", "ceiling"]

ARENAS: tuple[Arena, ...] = ("proposal", "ceiling")

#: Direction keys, in reporting order.
DIRECTIONS = ("gt_to_prop", "prop_to_gt")


@dataclass(frozen=True, slots=True)
class DirectionCounts:
    """One direction of one chip's APLS at one snap radius.

    Attributes:
        score: Mean route score over ``n_pairs`` pairs.
        n_control: Control points sampled from this direction's source graph.
        n_landed: Of those, how many found an edge of the target within the
            snap radius.
        n_source_nodes: Nodes of the source graph before densification. The
            reference sampling rule places no interior control point on a
            straight edge, so on a street grid most control points *are* the
            graph's own nodes: junctions and dead ends. The landing rate is
            therefore weighted toward junctions rather than toward road length,
            which is what makes it and buffer recall different measurements.
        n_pairs: Control point pairs entering the mean.
        n_pairs_unlanded: Pairs scoring 0 because an endpoint did not land.
        n_pairs_no_path: Pairs scoring 0 because the target cannot route
            between two points that both landed.
    """

    score: float
    n_control: int
    n_landed: int
    n_source_nodes: int
    n_pairs: int
    n_pairs_unlanded: int
    n_pairs_no_path: int


def landing_rate(counts: DirectionCounts) -> float | None:
    """Share of control points that found an edge, or ``None`` when there were none."""
    return counts.n_landed / counts.n_control if counts.n_control else None


def loss_shares(
    counts: DirectionCounts,
) -> tuple[float | None, float | None, float | None]:
    """Split ``1 - score`` into unlanded, unroutable, and length disagreement.

    The three terms are shares of the pair count and sum to ``1 - score``
    exactly, since a pair in either of the first two categories contributes 0
    to the mean.

    Args:
        counts: One direction's score and pair counts.

    Returns:
        ``(unlanded, no_path, length)``, each ``None`` when the direction
        scored no pairs at all and the split is undefined rather than zero.
    """
    if counts.n_pairs == 0:
        return None, None, None
    unlanded = counts.n_pairs_unlanded / counts.n_pairs
    no_path = counts.n_pairs_no_path / counts.n_pairs
    return unlanded, no_path, (1.0 - counts.score) - unlanded - no_path


@dataclass(frozen=True, slots=True)
class ChipScore:
    """One chip, one arena, one snap radius.

    Attributes:
        sample_id: Which chip this scored.
        arena: Which graph was scored; see :data:`Arena`.
        max_snap: The swept radius, in pixels.
        apls: Harmonic mean of the two directional scores.
        gt_to_prop: Truth routes measured on the arena's graph, with its counts.
        prop_to_gt: The arena's routes measured on the truth, with its counts.
    """

    sample_id: str
    arena: str
    max_snap: float
    apls: float
    gt_to_prop: DirectionCounts
    prop_to_gt: DirectionCounts


@dataclass(frozen=True, slots=True)
class ChipBuffer:
    """One chip, one arena, one buffer half-width.

    Attributes:
        sample_id: Which chip this scored.
        arena: Which graph was scored.
        buffer: Corridor half-width in pixels.
        precision: Share of arena road length within ``buffer`` of truth road.
        recall: Share of truth road length within ``buffer`` of arena road.
        f1: Harmonic mean of the two.
    """

    sample_id: str
    arena: str
    buffer: float
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class DirectionSummary:
    """One direction of one arena at one snap radius, across the holdout.

    Attributes:
        direction: ``gt_to_prop`` or ``prop_to_gt``.
        score: Mean directional score over chips.
        landing_rate: Landed control points over all control points, pooled
            across chips rather than averaged per chip.
        n_landed: Landed control points, summed.
        n_control: Control points, summed. The pair the rate came from.
        landing_rate_mean: Per-chip landing rate averaged over chips, which is
            how the score itself aggregates.
        node_share: Source graph nodes over control points, pooled. Near 1.0
            means the landing rate is a statement about junctions and dead ends
            rather than about road length; see :class:`DirectionCounts`.
        n_pairs: Scored pairs, summed across chips.
        n_pairs_mean: The same per chip.
        unlanded_loss: Mean per-chip share of the pair count lost to control
            points that found no edge. This is the displacement term: the only
            way a smaller radius can cost anything.
        no_path_loss: Mean per-chip share lost to pairs the target cannot route
            between, both endpoints having landed.
        length_loss: Mean per-chip share lost to length disagreement among the
            pairs that did reach a comparison. The three terms sum to
            ``1 - score``.
        n_chips_no_pairs: Chips whose direction scored no pairs at all, and so
            contributed to the score but not to the three loss terms.
    """

    direction: str
    score: float
    landing_rate: float | None
    n_landed: int
    n_control: int
    landing_rate_mean: float | None
    node_share: float | None
    n_pairs: int
    n_pairs_mean: float
    unlanded_loss: float | None
    no_path_loss: float | None
    length_loss: float | None
    n_chips_no_pairs: int


@dataclass(frozen=True, slots=True)
class SnapSummary:
    """One arena at one snap radius, across the holdout.

    Attributes:
        arena: Which graph was scored.
        max_snap: The swept radius.
        n_chips: Chips contributing.
        apls: Mean APLS over chips.
        apls_median: Median APLS, which one bad chip cannot drag.
        fraction_of_reference: ``apls`` over the same arena's APLS at
            :data:`REFERENCE_MAX_SNAP`. The share of the reported score that
            survives at this radius.
        directions: One summary per direction, in :data:`DIRECTIONS` order.
    """

    arena: str
    max_snap: float
    n_chips: int
    apls: float
    apls_median: float
    fraction_of_reference: float
    directions: tuple[DirectionSummary, ...]


@dataclass(frozen=True, slots=True)
class BufferSummary:
    """One arena at one buffer half-width, across the holdout.

    Attributes:
        arena: Which graph was scored.
        buffer: Corridor half-width in pixels.
        n_chips: Chips contributing.
        precision: Mean precision over chips.
        recall: Mean recall.
        f1: Mean F1.
    """

    arena: str
    buffer: float
    n_chips: int
    precision: float
    recall: float
    f1: float


def counts_of(
    result: metrics.APLSResult,
    direction: str,
    truth: nx.MultiGraph,
    proposal: nx.MultiGraph,
) -> DirectionCounts:
    """Pull one direction's score and counts off an APLS result.

    Args:
        result: A scored comparison.
        direction: ``gt_to_prop`` or ``prop_to_gt``.
        truth: Ground-truth graph, which sources the ``gt_to_prop`` direction.
        proposal: The arena's graph, which sources the other.

    Returns:
        That direction's score with the counts that decompose it.
    """
    if direction == "gt_to_prop":
        return DirectionCounts(
            score=result.gt_to_prop,
            n_control=result.n_control_gt,
            n_landed=result.n_landed_gt,
            n_source_nodes=truth.number_of_nodes(),
            n_pairs=result.n_pairs_gt,
            n_pairs_unlanded=result.n_pairs_unlanded_gt,
            n_pairs_no_path=result.n_pairs_no_path_gt,
        )
    return DirectionCounts(
        score=result.prop_to_gt,
        n_control=result.n_control_prop,
        n_landed=result.n_landed_prop,
        n_source_nodes=proposal.number_of_nodes(),
        n_pairs=result.n_pairs_prop,
        n_pairs_unlanded=result.n_pairs_unlanded_prop,
        n_pairs_no_path=result.n_pairs_no_path_prop,
    )


def score_arena(
    truth: nx.MultiGraph,
    proposal: nx.MultiGraph,
    sample_id: str,
    arena: Arena,
    max_snaps: tuple[float, ...],
    buffers: tuple[float, ...],
) -> tuple[tuple[ChipScore, ...], tuple[ChipBuffer, ...]]:
    """Score one arena's graph at every snap radius and every buffer.

    Args:
        truth: Ground-truth graph for the chip.
        proposal: The arena's decoded graph.
        sample_id: Recorded on every row so results stay traceable.
        arena: Which arena this graph is.
        max_snaps: Snap radii to score APLS at.
        buffers: Buffer half-widths to score coverage at.

    Returns:
        One APLS row per radius, and one coverage row per buffer. Coverage is
        snap-independent by construction, which is what makes it an independent
        check rather than a restatement.
    """
    scores = tuple(
        ChipScore(
            sample_id=sample_id,
            arena=arena,
            max_snap=r,
            apls=result.score,
            gt_to_prop=counts_of(result, "gt_to_prop", truth, proposal),
            prop_to_gt=counts_of(result, "prop_to_gt", truth, proposal),
        )
        for r in max_snaps
        for result in (metrics.apls(truth, proposal, max_snap=r),)
    )
    coverage = tuple(
        ChipBuffer(
            sample_id=sample_id,
            arena=arena,
            buffer=b,
            precision=result.precision,
            recall=result.recall,
            f1=result.f1,
        )
        for b in buffers
        for result in (metrics.buffer_length_prf(truth, proposal, b),)
    )
    return scores, coverage


def score_chip(
    model,
    source: data.TileSource,
    sample_id: str,
    threshold: float,
    max_snaps: tuple[float, ...],
    buffers: tuple[float, ...],
    simplify_tolerance: float,
    spur_length: float,
    snap_tolerance: float,
    device: str,
) -> tuple[tuple[ChipScore, ...], tuple[ChipBuffer, ...], float]:
    """Score one chip's two arenas at every radius.

    The ceiling arm traces the *label* mask through the same decoder, so the
    only error in it is the decoder's own geometry. It is built first, because
    a chip whose perfect mask already scores zero cannot judge any radius and
    is dropped before inference is paid for.

    Args:
        model: Frozen checkpoint.
        source: Where imagery and labels come from.
        sample_id: Chip to score.
        threshold: Probability above which a pixel counts as road.
        max_snaps: Snap radii to score APLS at.
        buffers: Buffer half-widths to score coverage at.
        simplify_tolerance: Douglas-Peucker tolerance in pixels.
        spur_length: Shortest dead-end edge worth keeping.
        snap_tolerance: Junction merge radius in pixels.
        device: Device string for inference.

    Returns:
        Every APLS row, every coverage row, and the chip's ceiling APLS at
        :data:`REFERENCE_MAX_SNAP`. Empty tuples mean the chip was skipped:
        either it carries no ground-truth roads, or a perfect mask scores zero
        on it.
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
    ceiling_apls = metrics.apls(sample.truth, ceiling, max_snap=REFERENCE_MAX_SNAP).score
    if ceiling_apls == 0.0:
        return (), (), 0.0

    logits = train.predict_tile_logits(model, sample, device=device)
    proposal = cleanup.clean(
        skeleton.graph_from_mask(predict_mask(logits, threshold)),
        simplify_tolerance,
        spur_length,
        snap_tolerance,
    )

    graphs: dict[Arena, nx.MultiGraph] = {"proposal": proposal, "ceiling": ceiling}
    scored = [
        score_arena(sample.truth, graphs[arena], sample_id, arena, max_snaps, buffers)
        for arena in ARENAS
    ]
    return (
        tuple(row for rows, _ in scored for row in rows),
        tuple(row for _, rows in scored for row in rows),
        ceiling_apls,
    )


def summarize_direction(rows: Sequence[ChipScore], direction: str) -> DirectionSummary:
    """Aggregate one direction across the chips of one arena and radius.

    Args:
        rows: Chip scores already filtered to one arena and one radius.
        direction: ``gt_to_prop`` or ``prop_to_gt``.

    Returns:
        The mean score, the landing rate both pooled and per chip, and the
        three-way split of what the score is missing.
    """
    counts = [getattr(r, direction) for r in rows]
    shares = [loss_shares(c) for c in counts]
    n_control = sum(c.n_control for c in counts)
    rates = [rate for c in counts if (rate := landing_rate(c)) is not None]

    def mean_share(index: int) -> float | None:
        defined = [value for s in shares if (value := s[index]) is not None]
        return float(np.mean(defined)) if defined else None

    return DirectionSummary(
        direction=direction,
        score=float(np.mean([c.score for c in counts])) if counts else 0.0,
        landing_rate=(sum(c.n_landed for c in counts) / n_control if n_control else None),
        n_landed=sum(c.n_landed for c in counts),
        n_control=n_control,
        landing_rate_mean=float(np.mean(rates)) if rates else None,
        node_share=(
            sum(c.n_source_nodes for c in counts) / n_control if n_control else None
        ),
        n_pairs=sum(c.n_pairs for c in counts),
        n_pairs_mean=float(np.mean([c.n_pairs for c in counts])) if counts else 0.0,
        unlanded_loss=mean_share(0),
        no_path_loss=mean_share(1),
        length_loss=mean_share(2),
        n_chips_no_pairs=sum(1 for c in counts if c.n_pairs == 0),
    )


def summarize(
    scores: Sequence[ChipScore], arena: Arena, max_snap: float, reference_apls: float
) -> SnapSummary:
    """Aggregate one arena at one radius.

    Args:
        scores: Every chip score; filtered to this arena and radius here.
        arena: Which arena to summarize.
        max_snap: Which radius to summarize.
        reference_apls: The same arena's mean APLS at
            :data:`REFERENCE_MAX_SNAP`, which the fraction is taken against.

    Returns:
        Means across chips, the median, and both directional summaries.
    """
    rows = [s for s in scores if s.arena == arena and s.max_snap == max_snap]
    values = [r.apls for r in rows]
    mean_apls = float(np.mean(values)) if values else 0.0

    return SnapSummary(
        arena=arena,
        max_snap=max_snap,
        n_chips=len(rows),
        apls=mean_apls,
        apls_median=float(np.median(values)) if values else 0.0,
        fraction_of_reference=mean_apls / reference_apls if reference_apls else 0.0,
        directions=tuple(summarize_direction(rows, d) for d in DIRECTIONS),
    )


def summarize_buffer(
    rows: Sequence[ChipBuffer], arena: Arena, buffer: float
) -> BufferSummary:
    """Aggregate one arena's coverage at one buffer half-width."""
    selected = [r for r in rows if r.arena == arena and r.buffer == buffer]

    def mean(attribute: str) -> float:
        values = [getattr(r, attribute) for r in selected]
        return float(np.mean(values)) if values else 0.0

    return BufferSummary(
        arena=arena,
        buffer=buffer,
        n_chips=len(selected),
        precision=mean("precision"),
        recall=mean("recall"),
        f1=mean("f1"),
    )


def pareto_front(summaries: Sequence[SnapSummary]) -> tuple[float, ...]:
    """Radii not dominated on both directional scores at once.

    Taken over the two directions rather than over the harmonic mean that
    collapses them, mirroring ``scripts/threshold_sweep.py``. The radius is not
    a tuning axis, so a frontier of one value is the expected result and is
    itself the finding: nothing is traded, the smaller radius is simply worse in
    both directions.

    Args:
        summaries: One summary per swept radius, for a single arena.

    Returns:
        The non-dominated radii, ascending.
    """
    directions = {s.max_snap: tuple(d.score for d in s.directions) for s in summaries}
    return tuple(
        sorted(
            a
            for a in directions
            if not any(
                all(x >= y for x, y in zip(directions[b], directions[a], strict=True))
                and directions[b] != directions[a]
                for b in directions
            )
        )
    )


def monotone_violations(summaries: Sequence[SnapSummary]) -> tuple[dict, ...]:
    """Radii where a directional score rose as the snap radius fell.

    ``inject_points`` takes the nearest edge inside the radius, so shrinking it
    can only withdraw a landing. Every directional score should therefore be
    monotone non-decreasing in ``max_snap``, and anything here falsifies that.

    Args:
        summaries: One summary per swept radius, for a single arena.

    Returns:
        One row per violation, empty when the sweep is monotone.
    """
    ordered = sorted(summaries, key=lambda s: s.max_snap)
    return tuple(
        {
            "arena": b.arena,
            "direction": lo.direction,
            "max_snap_low": a.max_snap,
            "max_snap_high": b.max_snap,
            "score_low": lo.score,
            "score_high": hi.score,
        }
        for a, b in itertools.pairwise(ordered)
        for lo, hi in zip(a.directions, b.directions, strict=True)
        if lo.score > hi.score
    )


def flatten_score(row: ChipScore) -> dict:
    """One chip score as a flat JSON row, both directions prefixed."""
    flat = {
        "sample_id": row.sample_id,
        "arena": row.arena,
        "max_snap": row.max_snap,
        "apls": row.apls,
    }
    for direction in DIRECTIONS:
        counts: DirectionCounts = getattr(row, direction)
        prefix = "gt" if direction == "gt_to_prop" else "prop"
        unlanded, no_path, length = loss_shares(counts)
        flat |= {
            direction: counts.score,
            f"n_control_{prefix}": counts.n_control,
            f"n_landed_{prefix}": counts.n_landed,
            f"landing_rate_{prefix}": landing_rate(counts),
            f"n_source_nodes_{prefix}": counts.n_source_nodes,
            f"n_pairs_{prefix}": counts.n_pairs,
            f"n_pairs_unlanded_{prefix}": counts.n_pairs_unlanded,
            f"n_pairs_no_path_{prefix}": counts.n_pairs_no_path,
            f"unlanded_loss_{prefix}": unlanded,
            f"no_path_loss_{prefix}": no_path,
            f"length_loss_{prefix}": length,
        }
    return flat


def log_tables(
    summaries: Sequence[SnapSummary],
    coverage: Sequence[BufferSummary],
    frontiers: dict[str, tuple[float, ...]],
) -> None:
    """Print the per-arena APLS table and the coverage cross-check beside it."""
    for arena in ARENAS:
        rows = [s for s in summaries if s.arena == arena]
        logger.info(f"--- {arena} ---")
        logger.info(
            f"{'snap':>5} {'APLS':>7} {'med':>7} {'frac':>6} | "
            f"{'dir':>10} {'score':>7} {'land%':>7} {'landed/total':>14} "
            f"{'pairs':>8} {'unland':>7} {'nopath':>7} {'length':>7}"
        )
        for s in rows:
            for d in s.directions:
                head = (
                    f"{s.max_snap:>5.0f} {s.apls:>7.4f} {s.apls_median:>7.4f} "
                    f"{s.fraction_of_reference:>6.3f}"
                    if d.direction == DIRECTIONS[0]
                    else " " * 27
                )
                rate = "n/a" if d.landing_rate is None else f"{d.landing_rate:>7.4f}"
                logger.info(
                    f"{head} | {d.direction:>10} {d.score:>7.4f} {rate:>7} "
                    f"{d.n_landed:>6}/{d.n_control:<7} {d.n_pairs:>8} "
                    f"{d.unlanded_loss or 0.0:>7.4f} {d.no_path_loss or 0.0:>7.4f} "
                    f"{d.length_loss or 0.0:>7.4f}"
                )
        logger.info(f"pareto frontier (gt→prop vs prop→gt): {frontiers[arena]}")
        # Radius-independent, so it is stated once rather than per row. It is
        # what stops the landing rate being read as a length-weighted measure.
        shares = {d.direction: d.node_share for s in rows for d in s.directions}
        logger.info(
            "control points that are source-graph nodes: "
            + ", ".join(f"{k} {v:.3f}" for k, v in shares.items() if v is not None)
        )

    logger.info("--- buffer_length_prf cross-check (no snapping, no routing) ---")
    logger.info(f"{'arena':>10} {'buffer':>7} {'prec':>7} {'recall':>7} {'F1':>7}")
    for c in coverage:
        logger.info(
            f"{c.arena:>10} {c.buffer:>7.0f} {c.precision:>7.4f} "
            f"{c.recall:>7.4f} {c.f1:>7.4f}"
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
        help=(
            "chips to score after the skip; 0 runs all of them, which is what a "
            "reported number needs. A smaller value smokes the plumbing and its "
            "numbers are not results"
        ),
    )
    parser.add_argument(
        "--max-snap",
        type=float,
        nargs="+",
        default=list(DEFAULT_MAX_SNAPS),
        help=(
            "snap radii, in pixels, to score APLS at. Diagnostic only: 25.0 is "
            "the committed default that reproduces the reference and is the row "
            "the others are read against"
        ),
    )
    parser.add_argument(
        "--buffers",
        type=float,
        nargs="+",
        default=list(DEFAULT_BUFFERS),
        help=(
            "buffer half-widths, in pixels, for the independent coverage "
            "cross-check; matched to --max-snap so the two read against each "
            "other at equal tolerance"
        ),
    )
    parser.add_argument("--simplify-tolerance", type=float, default=2.0)
    parser.add_argument("--spur-length", type=float, default=20.0)
    parser.add_argument("--snap-tolerance", type=float, default=8.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("outputs/snap_sweep.json"))
    args = parser.parse_args()

    run = json.loads(args.run_json.read_text())
    requested = list(run["config"]["val_ids"])[args.skip_tiles :][: args.limit or None]
    max_snaps = tuple(sorted({float(r) for r in args.max_snap}))
    buffers = tuple(sorted({float(b) for b in args.buffers}))

    source, _ = train.build_source(args.aoi_root, "spacenet", args.resolution)
    model = train.load_checkpoint(args.checkpoint, device=args.device)
    logger.info(
        f"{len(requested)} chips x {len(ARENAS)} arenas x {len(max_snaps)} snap "
        f"radii at threshold {args.threshold}; buffers {buffers}"
    )

    started = time.perf_counter()
    scores: list[ChipScore] = []
    coverage: list[ChipBuffer] = []
    scored_ids: list[str] = []
    skipped: list[str] = []
    ceilings: dict[str, float] = {}
    for n, sample_id in enumerate(requested, start=1):
        chip_scores, chip_coverage, ceiling_apls = score_chip(
            model,
            source,
            sample_id,
            args.threshold,
            max_snaps,
            buffers,
            args.simplify_tolerance,
            args.spur_length,
            args.snap_tolerance,
            args.device,
        )
        if not chip_scores:
            # A perfect mask scores zero here, so no radius can be judged on
            # this chip. Degenerate, not a hard case.
            logger.warning(f"{sample_id}: zero ceiling, skipping")
            skipped.append(sample_id)
            continue

        scores.extend(chip_scores)
        coverage.extend(chip_coverage)
        scored_ids.append(sample_id)
        ceilings[sample_id] = ceiling_apls
        at = {(s.arena, s.max_snap): s.apls for s in chip_scores}
        logger.info(
            f"[{n}/{len(requested)}] {sample_id}: proposal "
            f"{at[('proposal', max_snaps[-1])]:.4f} -> "
            f"{at[('proposal', max_snaps[0])]:.4f}, ceiling "
            f"{at[('ceiling', max_snaps[-1])]:.4f} -> "
            f"{at[('ceiling', max_snaps[0])]:.4f} "
            f"(snap {max_snaps[-1]:.0f} -> {max_snaps[0]:.0f})"
        )

    elapsed = time.perf_counter() - started
    reference = {
        arena: float(
            np.mean(
                [
                    s.apls
                    for s in scores
                    if s.arena == arena and s.max_snap == REFERENCE_MAX_SNAP
                ]
                or [0.0]
            )
        )
        for arena in ARENAS
    }
    summaries = tuple(
        summarize(scores, arena, r, reference[arena])
        for arena in ARENAS
        for r in max_snaps
    )
    buffer_summaries = tuple(
        summarize_buffer(coverage, arena, b) for arena in ARENAS for b in buffers
    )
    frontiers = {
        arena: pareto_front([s for s in summaries if s.arena == arena])
        for arena in ARENAS
    }
    violations = tuple(
        row
        for arena in ARENAS
        for row in monotone_violations([s for s in summaries if s.arena == arena])
    )

    log_tables(summaries, buffer_summaries, frontiers)
    logger.info(
        f"monotone in max_snap: {'no' if violations else 'yes'} "
        f"({len(violations)} violations)"
    )
    mean_ceiling = float(np.mean(list(ceilings.values()))) if ceilings else 0.0
    logger.info(
        f"scored {len(scored_ids)} chips, skipped {len(skipped)} for zero ceiling; "
        f"mean ceiling {mean_ceiling:.4f} at max_snap {REFERENCE_MAX_SNAP:.0f}"
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
                    "max_snaps": list(max_snaps),
                    "reference_max_snap": REFERENCE_MAX_SNAP,
                    "buffers": list(buffers),
                    "arenas": list(ARENAS),
                    "skip_tiles": args.skip_tiles,
                    "limit": args.limit,
                    "resolution": args.resolution,
                    "device": args.device,
                    "simplify_tolerance": args.simplify_tolerance,
                    "spur_length": args.spur_length,
                    "snap_tolerance": args.snap_tolerance,
                    "apls_spacing": 50.0,
                    "apls_sampling": "reference",
                    "apls_min_path_length": 10.0,
                    "decoder": "cleanup.clean",
                    "ceiling_input": "label mask",
                    "diagnostic": (
                        "max_snap below 25.0 is not comparable to a published "
                        "number; 25.0 is the reference row"
                    ),
                },
                "elapsed_seconds": elapsed,
                "n_requested": len(requested),
                "sample_ids": scored_ids,
                "skipped_zero_ceiling": skipped,
                "ceiling_apls": ceilings,
                "ceiling_apls_mean": mean_ceiling,
                "reference_apls": reference,
                "pareto_frontiers": {a: list(f) for a, f in frontiers.items()},
                "monotone_violations": list(violations),
                "summaries": [asdict(s) for s in summaries],
                "buffer_summaries": [asdict(s) for s in buffer_summaries],
                "per_chip": [flatten_score(s) for s in scores],
                "per_chip_buffer": [asdict(s) for s in coverage],
            },
            indent=2,
        )
    )
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()

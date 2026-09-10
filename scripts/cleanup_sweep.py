"""Sweep ``cleanup.clean``'s three constants against every metric, not just APLS.

``cleanup.clean`` is ``simplify_edges(2.0) -> prune_spurs(20.0) ->
snap_junctions(8.0) -> prune_spurs(20.0)``. Those three constants have only ever
been evaluated against APLS, and APLS is nearly blind to what they do. On the
155-chip reporting holdout the decoder's own loss -- a *perfect* mask, traced
back and scored against truth -- is 0.0280 of APLS but 0.1186 of junction F1 at
10 px. So ``skeletonize -> simplify -> snap(8 px)`` destroys 18% of truth
junctions before the model is charged for anything, and the constants were tuned
against a metric that cannot see it.

Mechanistically plausible on 396x324 px chips at 1 px = 1 m: an 8 m snap radius
fuses genuinely distinct junctions, and a 20 m spur rule deletes real short
stubs. **The question this answers is how much of that 0.1186 is recoverable by
tuning alone.**

**Both the model's proposal and the perfect-mask ceiling are scored at every
grid point.** The ceiling is not a fixed backdrop: it was measured at the
current constants, and different constants give a different ceiling. Reporting
only the model score would leave "the constants are near-optimal and the loss is
intrinsic to skeletonization" indistinguishable from "the constants are wrong
and the model happens not to benefit".

Every metric in ``link_oracle.py``'s map is reported, and junction *precision*
and *recall* are reported separately rather than only through F1: a looser snap
should raise precision and cost recall, so the F1 alone would hide the trade
that the sweep exists to find.

Retrains nothing. The checkpoint is the frozen prior stage, so a grid point
costs a decode plus scoring rather than a training run. Per chip the logits are
computed once, both raw skeletons are traced once -- the model's, and the
perfect-mask ceiling's -- and every grid point reuses them. The decode itself
shares its pipeline prefix across the grid: ``simplify_edges`` runs once per
simplify tolerance and the first ``prune_spurs`` once per (simplify, spur) pair,
rather than once per grid point.

Smoke runs turn the grid and the chip count down at the call site: pass shorter
``--simplify-tolerance`` / ``--spur-length`` / ``--snap-tolerance`` lists and a
small ``--limit``. The committed defaults are the full factorial on the whole
holdout, which is what a reported number needs.
"""

import argparse
import json
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from link_oracle import ChipArmScore, metric_names, paired_stat, score_metrics
from loguru import logger

from geo_graphs import cleanup, data, metrics, skeleton, train
from geo_graphs.model import predict_mask

#: Douglas-Peucker tolerances, in pixels (1 px = 1 m), swept by default.
#:
#: Brackets the shipped 2.0 by a factor of four each way. Below 0.5 the
#: simplification is doing nothing to a skeleton whose vertices are already one
#: pixel apart; above 4.0 it starts cutting corners off real bends.
DEFAULT_SIMPLIFY_TOLERANCES = (0.5, 1.0, 2.0, 4.0)

#: Shortest dead-end edge worth keeping, in pixels, swept by default.
#:
#: The census put 47.0% of interior endpoints destroyed by the shipped 20.0, so
#: the interesting direction is down; 40.0 is carried to show what the trade
#: looks like on the other side.
DEFAULT_SPUR_LENGTHS = (5.0, 10.0, 20.0, 40.0)

#: Junction merge radius, in pixels, swept by default.
#:
#: The suspected cause of the junction-recall loss, so this axis brackets the
#: shipped 8.0 most tightly at the low end: 2.0 merges only the two-or-three
#: pixel clusters skeletonization actually produces.
DEFAULT_SNAP_TOLERANCES = (2.0, 4.0, 8.0, 16.0)

#: The constants ``cleanup.clean`` ships with, in the order the axes are swept.
SHIPPED = (2.0, 20.0, 8.0)

#: Buffer half-widths, in pixels, length coverage is scored at.
#:
#: 20 is dropped from ``link_oracle``'s three: it rewards indiscriminate edge
#: addition, so it cannot referee a sweep whose whole subject is how much
#: geometry to delete.
DEFAULT_BUFFERS = (5.0, 10.0)

#: Radii, in pixels, junctions are matched within.
DEFAULT_JUNCTION_RADII = (5.0, 10.0, 20.0)

#: Which graph a score came from: the model's proposal, or the perfect mask.
ARENAS = ("model", "ceiling")

#: The three swept fields of :class:`CleanupConfig`, in sweep order.
AXES = ("simplify_tolerance", "spur_length", "snap_tolerance")

#: Metrics the console tables print and the Pareto frontier is taken over.
#:
#: One per family, at the middle matching radius. Junction precision and recall
#: ride along beside F1 because the expected effect of the snap radius is a
#: trade between them that F1 alone would average away.
HEADLINE_METRICS = (
    "apls",
    "buffer_f1_5",
    "buffer_f1_10",
    "junction_f1_10",
    "junction_precision_10",
    "junction_recall_10",
)

#: Column headings for :data:`HEADLINE_METRICS`, in the same order.
HEADLINE_LABELS = ("APLS", "bufF1@5", "bufF1@10", "jF1@10", "jP@10", "jR@10")

#: The headline columns as one preformatted table header.
HEADLINE_HEADER = " ".join(f"{label:>9}" for label in HEADLINE_LABELS)

#: The three axes the Pareto frontier is taken over.
PARETO_METRICS = ("apls", "buffer_f1_10", "junction_f1_10")


@dataclass(frozen=True, slots=True)
class CleanupConfig:
    """One point of the cleanup grid.

    Attributes:
        simplify_tolerance: Douglas-Peucker tolerance in pixels.
        spur_length: Shortest dead-end edge worth keeping, in pixels. Used by
            both prune passes, as ``cleanup.clean`` does.
        snap_tolerance: Junction merge radius in pixels.
    """

    simplify_tolerance: float
    spur_length: float
    snap_tolerance: float


@dataclass(frozen=True, slots=True)
class ConfigSummary:
    """Aggregate quality of one grid point in one arena, under every metric.

    Attributes:
        arm: Arena-qualified key, from :func:`arm_key`.
        arena: ``model`` or ``ceiling``.
        config_key: The grid point, from :func:`config_key`.
        simplify_tolerance: Douglas-Peucker tolerance in pixels.
        spur_length: Shortest dead-end edge worth keeping, in pixels.
        snap_tolerance: Junction merge radius in pixels.
        is_shipped: Whether this is the setting ``cleanup.clean`` ships with.
        n_chips: Chips contributing.
        means: Per-metric mean over chips, skipping chips where the metric is
            undefined. ``None`` when it was undefined everywhere.
        medians: Per-metric median over the same chips, which one bad chip
            cannot drag.
    """

    arm: str
    arena: str
    config_key: str
    simplify_tolerance: float
    spur_length: float
    snap_tolerance: float
    is_shipped: bool
    n_chips: int
    means: Mapping[str, float | None]
    medians: Mapping[str, float | None]


def config_key(config: CleanupConfig) -> str:
    """Stable identifier for one grid point, used as the key everywhere."""
    return (
        f"s{config.simplify_tolerance:g}"
        f"_p{config.spur_length:g}"
        f"_n{config.snap_tolerance:g}"
    )


def arm_key(arena: str, config: CleanupConfig) -> str:
    """Arena-qualified grid point key, so both arenas share one score list."""
    return f"{arena}/{config_key(config)}"


def build_grid(
    simplify_tolerances: tuple[float, ...],
    spur_lengths: tuple[float, ...],
    snap_tolerances: tuple[float, ...],
) -> tuple[CleanupConfig, ...]:
    """Enumerate the factorial grid, in the order :func:`clean_variants` walks it.

    Args:
        simplify_tolerances: Douglas-Peucker tolerances to sweep.
        spur_lengths: Spur-length cutoffs to sweep.
        snap_tolerances: Junction merge radii to sweep.

    Returns:
        Every combination, simplify varying slowest.
    """
    return tuple(
        CleanupConfig(s, p, n)
        for s in simplify_tolerances
        for p in spur_lengths
        for n in snap_tolerances
    )


def clean_variants(
    raw: nx.MultiGraph,
    simplify_tolerances: tuple[float, ...],
    spur_lengths: tuple[float, ...],
    snap_tolerances: tuple[float, ...],
) -> Iterator[tuple[CleanupConfig, nx.MultiGraph]]:
    """Run ``cleanup.clean`` at every grid point, sharing the pipeline prefix.

    ``clean`` is ``simplify -> prune -> snap -> prune``, and its first two stages
    do not depend on the snap radius. Nesting the loops in pipeline order runs
    ``simplify_edges`` once per simplify tolerance and the first ``prune_spurs``
    once per (simplify, spur) pair instead of once per grid point. The emitted
    graph is identical to ``cleanup.clean(raw, s, p, n)``; only the sharing is
    new.

    Args:
        raw: Traced skeleton, before any cleanup. Not modified.
        simplify_tolerances: Douglas-Peucker tolerances to sweep.
        spur_lengths: Spur-length cutoffs to sweep.
        snap_tolerances: Junction merge radii to sweep.

    Yields:
        Each grid point and its cleaned graph, simplify varying slowest. One
        graph is live at a time, so the grid costs no memory in its own size.
    """
    for s in simplify_tolerances:
        simplified = cleanup.simplify_edges(raw, s)
        for p in spur_lengths:
            pruned = cleanup.prune_spurs(simplified, p)
            for n in snap_tolerances:
                snapped = cleanup.snap_junctions(pruned, n)
                yield CleanupConfig(s, p, n), cleanup.prune_spurs(snapped, p)


def score_chip(
    model,
    source: data.TileSource,
    sample_id: str,
    grid: tuple[CleanupConfig, ...],
    shipped: CleanupConfig,
    threshold: float,
    simplify_tolerances: tuple[float, ...],
    spur_lengths: tuple[float, ...],
    snap_tolerances: tuple[float, ...],
    buffers: tuple[float, ...],
    junction_radii: tuple[float, ...],
    junction_min_degree: int,
    device: str,
) -> tuple[tuple[ChipArmScore, ...], float]:
    """Score one chip at every grid point, in both arenas.

    Both expensive quantities are grid-independent and computed once here: the
    model's logits, and the two raw skeletons the grid decodes. Only the decode
    and the scoring repeat.

    The chip is skipped on the *shipped* setting's ceiling rather than on each
    grid point's own, so every grid point is summarized over the same chips and
    the paired statistics stay paired. It also makes the shipped row's chip set
    identical to ``link_oracle.py``'s, which is what lets the two be compared.

    Args:
        model: Frozen checkpoint.
        source: Where imagery and labels come from.
        sample_id: Chip to score.
        grid: Every grid point, used only to size the log line.
        shipped: The setting ``cleanup.clean`` ships with, which decides the
            skip.
        threshold: Probability above which a pixel counts as road.
        simplify_tolerances: Douglas-Peucker tolerances to sweep.
        spur_lengths: Spur-length cutoffs to sweep.
        snap_tolerances: Junction merge radii to sweep.
        buffers: Buffer half-widths for length coverage.
        junction_radii: Radii for junction matching.
        junction_min_degree: Degree at or above which a node is a junction.
        device: Device string for inference.

    Returns:
        One score per (arena, grid point), and the chip's ceiling APLS at the
        shipped setting. An empty tuple of scores means the chip was skipped:
        either it carries no ground-truth roads, or a perfect mask scores zero
        APLS on it and no grid point could be judged there.
    """
    sample = source.load(sample_id)
    if sample.truth.number_of_edges() == 0:
        return (), 0.0

    ceiling_raw = skeleton.graph_from_mask(sample.mask)
    shipped_ceiling = cleanup.clean(
        ceiling_raw,
        shipped.simplify_tolerance,
        shipped.spur_length,
        shipped.snap_tolerance,
    )
    ceiling_apls = metrics.apls(sample.truth, shipped_ceiling).score
    if ceiling_apls == 0.0:
        return (), 0.0

    logits = train.predict_tile_logits(model, sample, device=device)
    raws = {
        "model": skeleton.graph_from_mask(predict_mask(logits, threshold)),
        "ceiling": ceiling_raw,
    }

    scores: list[ChipArmScore] = []
    for arena in ARENAS:
        for config, graph in clean_variants(
            raws[arena], simplify_tolerances, spur_lengths, snap_tolerances
        ):
            values, reference = score_metrics(
                sample.truth, graph, buffers, junction_radii, junction_min_degree
            )
            scores.append(
                ChipArmScore(
                    sample_id=sample_id,
                    arm=arm_key(arena, config),
                    # No edge is ever added here; the fields exist because the
                    # carrier is shared with the oracle sweep's paired helper.
                    n_selected=0,
                    n_edges_added=0,
                    n_control_gt=reference.n_control_gt,
                    n_control_prop=reference.n_control_prop,
                    values=values,
                )
            )
    logger.debug(f"{sample_id}: {len(scores)} scores over {len(grid)} grid points")
    return tuple(scores), ceiling_apls


def summarize(
    scores: Sequence[ChipArmScore],
    arena: str,
    config: CleanupConfig,
    shipped: CleanupConfig,
    names: tuple[str, ...],
) -> ConfigSummary:
    """Aggregate one grid point in one arena across the holdout.

    Args:
        scores: The chip scores for this arm alone, already filtered.
        arena: ``model`` or ``ceiling``.
        config: The grid point being summarized.
        shipped: The setting ``cleanup.clean`` ships with.
        names: Metric keys to aggregate, from ``link_oracle.metric_names``.

    Returns:
        Per-metric means and medians over the chips that defined the metric.
    """

    def defined(name: str) -> np.ndarray:
        return np.array(
            [s.values[name] for s in scores if s.values.get(name) is not None],
            dtype=float,
        )

    return ConfigSummary(
        arm=arm_key(arena, config),
        arena=arena,
        config_key=config_key(config),
        simplify_tolerance=config.simplify_tolerance,
        spur_length=config.spur_length,
        snap_tolerance=config.snap_tolerance,
        is_shipped=config == shipped,
        n_chips=len(scores),
        means={n: (float(v.mean()) if len(v := defined(n)) else None) for n in names},
        medians={
            n: (float(np.median(v)) if len(v := defined(n)) else None) for n in names
        },
    )


def pareto_front(
    summaries: Sequence[ConfigSummary], names: tuple[str, ...]
) -> tuple[str, ...]:
    """Grid points no other grid point beats on every named metric at once.

    Taken over the raw metrics rather than any weighted combination of them: a
    weighting is exactly the value judgement this sweep exists to hand back
    rather than make.

    Args:
        summaries: One summary per grid point, all from the same arena.
        names: Metrics to take the frontier over, all maximized.

    Returns:
        The non-dominated grid point keys, in grid order.
    """
    vectors: dict[str, tuple[float, ...]] = {
        s.config_key: tuple(float(s.means[n] or 0.0) for n in names) for s in summaries
    }
    return tuple(
        key
        for key, value in vectors.items()
        if not any(
            all(o >= v for o, v in zip(other, value, strict=True)) and other != value
            for other in vectors.values()
        )
    )


def axis_profile(
    grid: tuple[CleanupConfig, ...], shipped: CleanupConfig, axis: str
) -> tuple[CleanupConfig, ...]:
    """The 1-D slice through the shipped setting along one axis.

    Free, because the factorial already contains it. Reads the question "does
    this constant matter" off the grid without a second run.

    Args:
        grid: Every swept grid point.
        shipped: The setting ``cleanup.clean`` ships with.
        axis: Field of :class:`CleanupConfig` to vary; the other two are pinned.

    Returns:
        The grid points differing from ``shipped`` on ``axis`` alone, ascending.
    """
    pinned = [a for a in AXES if a != axis]
    return tuple(
        sorted(
            (
                c
                for c in grid
                if all(getattr(c, a) == getattr(shipped, a) for a in pinned)
            ),
            key=lambda c: getattr(c, axis),
        )
    )


def flatten(summary: ConfigSummary) -> dict:
    """One grid point summary as a flat JSON row, metric means beside metadata."""
    row = {k: v for k, v in asdict(summary).items() if k not in ("means", "medians")}
    return (
        row
        | dict(summary.means)
        | {f"{name}_median": value for name, value in summary.medians.items()}
    )


def log_table(
    summaries: Sequence[ConfigSummary],
    shipped_key: str,
    frontier: tuple[str, ...],
    deltas: Mapping[tuple[str, str], float],
    title: str,
) -> None:
    """Print one arena's grid, marking the shipped setting and the frontier.

    Args:
        summaries: Every grid point in one arena, in grid order.
        shipped_key: Config key of the setting ``cleanup.clean`` ships with.
        frontier: Non-dominated config keys in this arena.
        deltas: Mean paired delta against the shipped setting, keyed by
            ``(arm, metric)``.
        title: Table heading.
    """
    logger.info(f"--- {title} ---")
    logger.info(f"{'config':>18} {HEADLINE_HEADER} {'ΔAPLS':>8} {'ΔjF1@10':>8}  ")
    for s in summaries:
        marks = ("*" if s.config_key in frontier else " ") + (
            "S" if s.config_key == shipped_key else " "
        )
        cells = " ".join(f"{s.means[m] or 0.0:>9.4f}" for m in HEADLINE_METRICS)
        d_apls = deltas.get((s.arm, "apls"), 0.0)
        d_junction = deltas.get((s.arm, "junction_f1_10"), 0.0)
        logger.info(
            f"{s.config_key:>18} {cells} {d_apls:>+8.4f} {d_junction:>+8.4f} {marks}"
        )


def log_profiles(
    grid: tuple[CleanupConfig, ...],
    shipped: CleanupConfig,
    by_arm: Mapping[str, ConfigSummary],
) -> None:
    """Print the three 1-D slices through the shipped setting, both arenas.

    Args:
        grid: Every swept grid point.
        shipped: The setting ``cleanup.clean`` ships with.
        by_arm: Every summary, keyed by :func:`arm_key`.
    """
    for axis in AXES:
        logger.info(f"--- 1-D profile: {axis} (others at shipped) ---")
        logger.info(f"{axis:>20} {'arena':>8} {HEADLINE_HEADER}")
        for config in axis_profile(grid, shipped, axis):
            for arena in ARENAS:
                summary = by_arm[arm_key(arena, config)]
                cells = " ".join(
                    f"{summary.means[m] or 0.0:>9.4f}" for m in HEADLINE_METRICS
                )
                mark = " <- shipped" if config == shipped else ""
                logger.info(f"{getattr(config, axis):>20g} {arena:>8} {cells}{mark}")


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
            "reported number needs. The smoke knob, alongside shorter axis lists"
        ),
    )
    parser.add_argument(
        "--simplify-tolerance",
        type=float,
        nargs="+",
        default=list(DEFAULT_SIMPLIFY_TOLERANCES),
        help=(
            "Douglas-Peucker tolerances in pixels; pass a shorter list for a "
            "coarse smoke grid. Must contain the shipped value"
        ),
    )
    parser.add_argument(
        "--spur-length",
        type=float,
        nargs="+",
        default=list(DEFAULT_SPUR_LENGTHS),
        help="spur-length cutoffs in pixels, used by both prune passes",
    )
    parser.add_argument(
        "--snap-tolerance",
        type=float,
        nargs="+",
        default=list(DEFAULT_SNAP_TOLERANCES),
        help="junction merge radii in pixels",
    )
    parser.add_argument(
        "--shipped",
        type=float,
        nargs=3,
        default=list(SHIPPED),
        metavar=("SIMPLIFY", "SPUR", "SNAP"),
        help=(
            "the constants cleanup.clean ships with, which every grid point is "
            "measured against; must appear in the swept grid"
        ),
    )
    parser.add_argument(
        "--buffers",
        type=float,
        nargs="+",
        default=list(DEFAULT_BUFFERS),
        help="buffer half-widths for length coverage, in pixels",
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("outputs/cleanup_sweep.json"))
    args = parser.parse_args()

    simplify_tolerances = tuple(sorted({float(v) for v in args.simplify_tolerance}))
    spur_lengths = tuple(sorted({float(v) for v in args.spur_length}))
    snap_tolerances = tuple(sorted({float(v) for v in args.snap_tolerance}))
    buffers = tuple(sorted({float(b) for b in args.buffers}))
    junction_radii = tuple(sorted({float(r) for r in args.junction_radii}))

    grid = build_grid(simplify_tolerances, spur_lengths, snap_tolerances)
    shipped = CleanupConfig(*(float(v) for v in args.shipped))
    if shipped not in grid:
        parser.error(
            f"the shipped setting {config_key(shipped)} is not in the swept grid; "
            "it is the reference every grid point is measured against, so it has "
            "to be scored too"
        )
    names = metric_names(buffers, junction_radii)
    missing = [m for m in (*HEADLINE_METRICS, *PARETO_METRICS) if m not in names]
    if missing:
        parser.error(
            f"--buffers/--junction-radii drop metrics the report needs: {missing}"
        )

    run = json.loads(args.run_json.read_text())
    requested = list(run["config"]["val_ids"])[args.skip_tiles :][: args.limit or None]

    source, _ = train.build_source(args.aoi_root, "spacenet", args.resolution)
    model = train.load_checkpoint(args.checkpoint, device=args.device)
    logger.info(
        f"{len(requested)} chips x {len(grid)} grid points x {len(ARENAS)} arenas "
        f"x {len(names)} metrics at threshold {args.threshold}; "
        f"simplify {simplify_tolerances}, spur {spur_lengths}, snap {snap_tolerances}; "
        f"shipped {config_key(shipped)}"
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
            grid,
            shipped,
            args.threshold,
            simplify_tolerances,
            spur_lengths,
            snap_tolerances,
            buffers,
            junction_radii,
            args.junction_min_degree,
            args.device,
        )
        if not chip_scores:
            # A perfect mask scores zero APLS here, so no grid point can be
            # judged on this chip. Degenerate, not a hard case.
            logger.warning(f"{sample_id}: zero ceiling, skipping")
            skipped.append(sample_id)
            continue

        scores.extend(chip_scores)
        scored_ids.append(sample_id)
        ceilings[sample_id] = ceiling_apls
        by_key = {s.arm: s for s in chip_scores}
        shipped_model = by_key[arm_key("model", shipped)]
        shipped_ceiling = by_key[arm_key("ceiling", shipped)]
        best = max(
            (s for s in chip_scores if s.arm.startswith("ceiling/")),
            key=lambda s: s.values["junction_f1_10"] or 0.0,
        )
        logger.info(
            f"[{n}/{len(requested)}] {sample_id}: shipped model "
            f"{shipped_model.values['apls']:.4f} APLS / "
            f"{shipped_model.values['junction_f1_10']:.4f} jF1, ceiling "
            f"{shipped_ceiling.values['apls']:.4f}/"
            f"{shipped_ceiling.values['junction_f1_10']:.4f}, best ceiling jF1 "
            f"{best.arm.split('/')[1]} {best.values['junction_f1_10']:.4f}"
        )

    elapsed = time.perf_counter() - started
    by_arm_scores: dict[str, list[ChipArmScore]] = {}
    for score in scores:
        by_arm_scores.setdefault(score.arm, []).append(score)

    summaries = tuple(
        summarize(by_arm_scores.get(arm_key(arena, c), []), arena, c, shipped, names)
        for arena in ARENAS
        for c in grid
    )
    by_arm = {s.arm: s for s in summaries}

    # Each grid point against the shipped setting *in its own arena*: the
    # ceiling moves with the constants, so charging a ceiling shift against the
    # model's shipped row would mix the two effects the sweep exists to separate.
    paired = tuple(
        paired_stat(
            by_arm_scores.get(arm_key(arena, c), [])
            + by_arm_scores.get(arm_key(arena, shipped), []),
            arm_key(arena, c),
            arm_key(arena, shipped),
            metric,
        )
        for arena in ARENAS
        for c in grid
        if c != shipped
        for metric in names
    )
    deltas = {(p.arm, p.metric): p.mean_delta for p in paired}

    # Ceiling minus model at the same grid point: what the *model* still owes,
    # as against what the decoder throws away before the model is charged.
    headroom = tuple(
        {
            "config_key": config_key(c),
            "simplify_tolerance": c.simplify_tolerance,
            "spur_length": c.spur_length,
            "snap_tolerance": c.snap_tolerance,
            "is_shipped": c == shipped,
        }
        | {
            name: (
                None
                if by_arm[arm_key("ceiling", c)].means[name] is None
                or by_arm[arm_key("model", c)].means[name] is None
                else (by_arm[arm_key("ceiling", c)].means[name] or 0.0)
                - (by_arm[arm_key("model", c)].means[name] or 0.0)
            )
            for name in names
        }
        for c in grid
    )

    frontier = {
        arena: pareto_front(
            [s for s in summaries if s.arena == arena], tuple(PARETO_METRICS)
        )
        for arena in ARENAS
    }
    profiles = {
        axis: [config_key(c) for c in axis_profile(grid, shipped, axis)] for axis in AXES
    }

    for arena in ARENAS:
        log_table(
            [s for s in summaries if s.arena == arena],
            config_key(shipped),
            frontier[arena],
            deltas,
            f"{arena}: mean over {len(scored_ids)} chips "
            f"(* on the {'/'.join(PARETO_METRICS)} frontier, S = shipped)",
        )
        logger.info(f"{arena} frontier: {frontier[arena]}")

    log_profiles(grid, shipped, by_arm)

    mean_ceiling = float(np.mean(list(ceilings.values()))) if ceilings else 0.0
    logger.info(
        f"scored {len(scored_ids)} chips, skipped {len(skipped)} for zero ceiling; "
        f"mean shipped-setting ceiling APLS {mean_ceiling:.4f}"
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
                    "simplify_tolerances": list(simplify_tolerances),
                    "spur_lengths": list(spur_lengths),
                    "snap_tolerances": list(snap_tolerances),
                    "shipped": list(args.shipped),
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
                    "metric_names": list(names),
                    "pareto_metrics": list(PARETO_METRICS),
                    "arenas": list(ARENAS),
                    "model_pipeline": (
                        "predict_mask -> graph_from_mask -> cleanup.clean(grid point)"
                    ),
                    "ceiling_pipeline": (
                        "label mask -> graph_from_mask -> cleanup.clean(grid point)"
                    ),
                },
                "elapsed_seconds": elapsed,
                "n_requested": len(requested),
                "sample_ids": scored_ids,
                "skipped_zero_ceiling": skipped,
                "ceiling_apls_shipped": ceilings,
                "ceiling_apls_shipped_mean": mean_ceiling,
                "shipped_config_key": config_key(shipped),
                "configs": [
                    asdict(c) | {"config_key": config_key(c), "is_shipped": c == shipped}
                    for c in grid
                ],
                "pareto_frontier": {a: list(f) for a, f in frontier.items()},
                "profiles": profiles,
                "summaries": [flatten(s) for s in summaries],
                "headroom": list(headroom),
                "paired_stats": [asdict(p) for p in paired],
                "per_chip": [
                    {
                        k: v
                        for k, v in asdict(s).items()
                        if k not in ("values", "n_selected", "n_edges_added")
                    }
                    | dict(s.values)
                    for s in scores
                ],
            },
            indent=2,
        )
    )
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()

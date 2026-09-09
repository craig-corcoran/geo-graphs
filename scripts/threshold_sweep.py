"""Sweep the mask threshold against APLS on a frozen checkpoint.

`predict_mask`'s threshold is a real hyperparameter and it is currently untuned.
Raising it thins the mask, so fewer roads are invented and more are severed:
it trades `prop_to_gt` against `gt_to_prop` directly. Pixel IoU cannot see that
trade, which is why this sweeps on APLS.

Retrains nothing. The checkpoint is the frozen prior stage, so a sweep costs
inference plus scoring rather than a training run.

Both expensive quantities are threshold-independent and so are computed once per
tile and reused across the sweep: the model's logits, and the tile's ceiling
graph (skeletonized from the *label* mask). Only mask -> skeleton -> cleanup ->
APLS repeats per threshold.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from geo_graphs import cleanup, data, metrics, skeleton, train
from geo_graphs.model import predict_mask

#: Thresholds scored by default, dense below 0.15 and coarse above it. The
#: asymmetry is empirical, not aesthetic: APLS peaks around 0.02-0.10 and is
#: flat across that plateau, while everything above 0.15 declines monotonically.
#: A range starting at 0.2 misses the optimum entirely, which is what the first
#: sweep did.
DEFAULT_THRESHOLDS = (
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.07,
    0.10,
    0.12,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.65,
    0.80,
)


@dataclass(frozen=True, slots=True)
class TileScore:
    """One tile scored at one threshold.

    Attributes:
        sample_id: Which tile this scored.
        threshold: Probability above which a pixel counted as road.
        mask_iou: Pixel agreement between predicted and label mask.
        apls: Harmonic mean of the two directions, after cleanup.
        gt_to_prop: Falls when real roads are missing.
        prop_to_gt: Falls when roads are invented.
        ceiling_apls: APLS the same pipeline reaches from a perfect mask here.
        fraction_of_ceiling: ``apls / ceiling_apls``.
    """

    sample_id: str
    threshold: float
    mask_iou: float
    apls: float
    gt_to_prop: float
    prop_to_gt: float
    ceiling_apls: float
    fraction_of_ceiling: float


@dataclass(frozen=True, slots=True)
class ThresholdSummary:
    """Aggregate quality at one threshold.

    Attributes:
        threshold: The swept value.
        n_scored: Tiles contributing to the means.
        mask_iou: Mean pixel agreement.
        apls: Mean APLS after cleanup.
        apls_median: Median APLS, which one bad chip cannot drag.
        gt_to_prop: Mean of the missed-road direction.
        prop_to_gt: Mean of the invented-road direction.
        fraction_of_ceiling: Mean per-tile share of the achievable score.
    """

    threshold: float
    n_scored: int
    mask_iou: float
    apls: float
    apls_median: float
    gt_to_prop: float
    prop_to_gt: float
    fraction_of_ceiling: float


def score_at_threshold(
    logits: np.ndarray,
    sample: data.TileSample,
    ceiling_apls: float,
    threshold: float,
    sample_id: str,
) -> TileScore:
    """Score one tile's cached logits at one threshold.

    Args:
        logits: Cached model output for this tile, at the tile's own size.
        sample: Tile carrying the label mask and ground-truth graph.
        ceiling_apls: This tile's precomputed ceiling, which the threshold
            cannot move.
        threshold: Probability above which a pixel counts as road.
        sample_id: Recorded on the score so results stay traceable.

    Returns:
        Per-stage quality for this tile at this threshold.
    """
    predicted = predict_mask(logits, threshold)
    cleaned = cleanup.clean(skeleton.graph_from_mask(predicted))
    result = metrics.apls(sample.truth, cleaned)

    return TileScore(
        sample_id=sample_id,
        threshold=threshold,
        mask_iou=metrics.iou(predicted, sample.mask),
        apls=result.score,
        gt_to_prop=result.gt_to_prop,
        prop_to_gt=result.prop_to_gt,
        ceiling_apls=ceiling_apls,
        fraction_of_ceiling=result.score / ceiling_apls if ceiling_apls else 0.0,
    )


def sweep(
    model,
    source: data.TileSource,
    sample_ids: list[str],
    thresholds: tuple[float, ...],
    device: str = "cpu",
) -> tuple[TileScore, ...]:
    """Score every tile at every threshold, reusing what the threshold cannot change.

    Args:
        model: Trained network, from a checkpoint.
        source: Where imagery and labels come from.
        sample_ids: Tiles to score.
        thresholds: Values to sweep.
        device: Device string for inference.

    Returns:
        One score per (tile, threshold) pair, tiles in the order given.
    """
    scores: list[TileScore] = []
    for n, sample_id in enumerate(sample_ids, start=1):
        sample = source.load(sample_id)
        if sample.truth.number_of_edges() == 0:
            logger.debug(f"{sample_id}: no ground-truth roads, skipping")
            continue

        logits = train.predict_tile_logits(model, sample, device=device)
        ceiling = cleanup.clean(skeleton.graph_from_mask(sample.mask))
        ceiling_apls = metrics.apls(sample.truth, ceiling).score
        if ceiling_apls == 0.0:
            # A perfect mask scores zero here, so no threshold can be judged on
            # this chip. Degenerate, not a hard case.
            logger.warning(f"{sample_id}: zero ceiling, skipping")
            continue

        scores.extend(
            score_at_threshold(logits, sample, ceiling_apls, t, sample_id)
            for t in thresholds
        )
        logger.info(f"[{n}/{len(sample_ids)}] {sample_id}: ceiling {ceiling_apls:.4f}")

    return tuple(scores)


def summarize(scores: tuple[TileScore, ...], threshold: float) -> ThresholdSummary:
    """Aggregate every tile's score at one threshold.

    Args:
        scores: All scores from the sweep; filtered to this threshold here.
        threshold: The value to summarize.

    Returns:
        Means across tiles, plus the median APLS.
    """
    rows = [s for s in scores if s.threshold == threshold]

    def mean(attribute: str) -> float:
        return float(np.mean([getattr(r, attribute) for r in rows]))

    return ThresholdSummary(
        threshold=threshold,
        n_scored=len(rows),
        mask_iou=mean("mask_iou"),
        apls=mean("apls"),
        apls_median=float(np.median([r.apls for r in rows])),
        gt_to_prop=mean("gt_to_prop"),
        prop_to_gt=mean("prop_to_gt"),
        fraction_of_ceiling=mean("fraction_of_ceiling"),
    )


def pareto_front(summaries: tuple[ThresholdSummary, ...]) -> tuple[float, ...]:
    """Thresholds not dominated on both APLS directions at once.

    The threshold's whole effect is a trade between missing roads and inventing
    them, so the frontier is taken over those two axes rather than over the
    harmonic mean that already collapses them.

    Args:
        summaries: One summary per swept threshold.

    Returns:
        The non-dominated thresholds, ascending.
    """
    return tuple(
        sorted(
            a.threshold
            for a in summaries
            if not any(
                b.gt_to_prop >= a.gt_to_prop
                and b.prop_to_gt >= a.prop_to_gt
                and (b.gt_to_prop, b.prop_to_gt) != (a.gt_to_prop, a.prop_to_gt)
                for b in summaries
            )
        )
    )


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/vegas_best.pt"))
    parser.add_argument(
        "--run-json",
        type=Path,
        default=Path("outputs/vegas_best.json"),
        help="run artifact supplying the val split, so the same tiles are scored",
    )
    parser.add_argument("--aoi-root", type=Path, default=Path("data/AOI_2_Vegas"))
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument(
        "--n-tiles",
        type=int,
        default=40,
        help="validation tiles to score; the run's own eval used 40",
    )
    parser.add_argument(
        "--skip-tiles",
        type=int,
        default=0,
        help=(
            "validation tiles to skip before scoring. Selecting a threshold on "
            "one slice and reporting it on a disjoint one is what keeps the "
            "reported number free of the choice made to produce it."
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("outputs/threshold_sweep.json"))
    args = parser.parse_args()

    run = json.loads(args.run_json.read_text())
    sample_ids = run["config"]["val_ids"][
        args.skip_tiles : args.skip_tiles + args.n_tiles
    ]
    thresholds = tuple(sorted(args.thresholds))

    source, _ = train.build_source(args.aoi_root, "spacenet", args.resolution)
    model = train.load_checkpoint(args.checkpoint, device=args.device)
    logger.info(f"{len(sample_ids)} tiles x {len(thresholds)} thresholds")

    scores = sweep(model, source, sample_ids, thresholds, device=args.device)
    summaries = tuple(summarize(scores, t) for t in thresholds)
    frontier = pareto_front(summaries)

    header = f"{'thr':>5} {'IoU':>7} {'APLS':>7} {'med':>7} {'gt→pr':>7} {'pr→gt':>7} {'frac':>7}"
    logger.info(header)
    for s in summaries:
        mark = " *" if s.threshold in frontier else "  "
        logger.info(
            f"{s.threshold:>5.2f} {s.mask_iou:>7.4f} {s.apls:>7.4f} "
            f"{s.apls_median:>7.4f} {s.gt_to_prop:>7.4f} {s.prop_to_gt:>7.4f} "
            f"{s.fraction_of_ceiling:>7.4f}{mark}"
        )
    logger.info(f"pareto frontier (gt→prop vs prop→gt): {frontier}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "sample_ids": sample_ids,
                "thresholds": list(thresholds),
                "pareto_frontier": list(frontier),
                "summaries": [asdict(s) for s in summaries],
                "per_tile": [asdict(s) for s in scores],
            },
            indent=2,
        )
    )
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Score the OSM -> mask -> graph round trip on a single tile.

Run against a *perfect* mask rendered from ground truth, this measures the
pipeline's own ceiling: whatever it loses here, no model scored through it can
recover. Keeping that number visible stops post-processing changes from being
credited to the model.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger

from . import cleanup, geograph, metrics, raster, skeleton, tiles
from .osm import ground_truth_graph


@dataclass(frozen=True, slots=True)
class RoundTripReport:
    """Measurements from one tile's round trip.

    Attributes:
        lat: Tile centre latitude.
        lon: Tile centre longitude.
        size_m: Tile side length in metres.
        truth_nodes: Node count of the ground-truth graph.
        truth_edges: Edge count of the ground-truth graph.
        truth_length_m: Total ground-truth road length.
        recovered_nodes: Node count after skeletonization and cleanup.
        recovered_edges: Edge count after skeletonization and cleanup.
        recovered_length_m: Total recovered road length.
        mask_iou: IoU between the truth mask and the recovered graph re-rendered.
        apls: Combined APLS score.
        apls_gt_to_prop: Directional score punishing missed roads.
        apls_prop_to_gt: Directional score punishing invented roads.
    """

    lat: float
    lon: float
    size_m: float
    truth_nodes: int
    truth_edges: int
    truth_length_m: float
    recovered_nodes: int
    recovered_edges: int
    recovered_length_m: float
    mask_iou: float
    apls: float
    apls_gt_to_prop: float
    apls_prop_to_gt: float


def run(
    lat: float,
    lon: float,
    size_m: float = 1024.0,
    resolution: float = 1.0,
    sampling: geograph.Sampling = "reference",
) -> RoundTripReport:
    """Render ground truth to a mask, trace it back, and score the result.

    The defaults are the expensive, correct settings: 1 m/px matches the
    SpaceNet benchmarks, and ``"reference"`` sampling is what agrees with the
    published metric. Turn them down at the call site for a smoke run, and treat
    that run as proof of plumbing only -- its numbers are not results.

    Args:
        lat: Tile centre latitude in degrees.
        lon: Tile centre longitude in degrees.
        size_m: Tile side length in metres.
        resolution: Metres per pixel. Cost scales with its inverse square.
        sampling: Control point rule; see :data:`geograph.Sampling`.

    Returns:
        The measurements for this tile.
    """
    tile = tiles.tile_from_center(lat, lon, size_m=size_m, resolution=resolution)
    truth = ground_truth_graph(tile)

    truth_mask = raster.rasterize(truth, tile)
    recovered = cleanup.clean(skeleton.graph_from_mask(truth_mask))
    score = metrics.apls(truth, recovered, sampling=sampling)

    return RoundTripReport(
        lat=lat,
        lon=lon,
        size_m=size_m,
        truth_nodes=truth.number_of_nodes(),
        truth_edges=truth.number_of_edges(),
        truth_length_m=geograph.total_length(truth),
        recovered_nodes=recovered.number_of_nodes(),
        recovered_edges=recovered.number_of_edges(),
        recovered_length_m=geograph.total_length(recovered),
        mask_iou=metrics.iou(truth_mask, raster.rasterize(recovered, tile)),
        apls=score.score,
        apls_gt_to_prop=score.gt_to_prop,
        apls_prop_to_gt=score.prop_to_gt,
    )


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", type=float, default=36.1699)
    parser.add_argument("--lon", type=float, default=-115.1398)
    parser.add_argument("--size", type=float, default=1024.0, help="tile side in metres")
    parser.add_argument("--resolution", type=float, default=1.0, help="metres per pixel")
    parser.add_argument(
        "--sampling", choices=["reference", "uniform"], default="reference"
    )
    parser.add_argument("--out", type=Path, help="write the report as JSON here")
    args = parser.parse_args()

    report = run(args.lat, args.lon, args.size, args.resolution, args.sampling)

    logger.info(f"tile        {args.size:.0f} m at {args.lat}, {args.lon}")
    logger.info(
        f"truth       {report.truth_nodes:4d} nodes {report.truth_edges:4d} edges "
        f"{report.truth_length_m:8.0f} m"
    )
    logger.info(
        f"recovered   {report.recovered_nodes:4d} nodes "
        f"{report.recovered_edges:4d} edges "
        f"{report.recovered_length_m:8.0f} m"
    )
    logger.info(f"mask IoU    {report.mask_iou:.4f}")
    logger.info(
        f"APLS        {report.apls:.4f}  "
        f"(gt->prop {report.apls_gt_to_prop:.4f}, prop->gt {report.apls_prop_to_gt:.4f})"
    )

    if args.out:
        args.out.write_text(json.dumps(asdict(report), indent=2))
        logger.info(f"wrote       {args.out}")


if __name__ == "__main__":
    main()

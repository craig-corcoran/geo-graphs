"""Precompute everything the showcase page displays.

The page is static: no server, no torch, no imagery download. Every number and
every geometry it shows is computed here once and embedded, so the result is a
single file that opens anywhere.

Reads the checkpoint and the run's JSON artifact rather than retraining, per the
house rule that acceptance numbers come from the pipeline and the presentation
layer reads them.
"""

import argparse
import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from loguru import logger
from PIL import Image
from shapely.geometry import LineString

from geo_graphs import cleanup, geograph, metrics, raster, skeleton, spacenet, train

#: Gap radius in pixels for the divergence demo. Two or three pixels is the
#: scale of a tree shadow or a vehicle occluding a lane.
GAP_RADIUS = 2

#: Damage levels the slider steps through.
GAP_COUNTS = (0, 5, 10, 20, 40, 60, 80, 120)

#: Random placements averaged per level, since where a gap lands matters more
#: than how many there are.
GAP_SEEDS = range(4)


@dataclass(frozen=True, slots=True)
class Paths:
    """Inputs and output for the build.

    Attributes:
        aoi_root: Extracted SpaceNet AOI.
        checkpoint: Trained weights.
        run_json: The run artifact holding per-tile scores.
        out: Where to write the site data.
    """

    aoi_root: Path
    checkpoint: Path
    run_json: Path
    out: Path


def encode_jpeg(image: np.ndarray, quality: int = 78) -> str:
    """Encode a float image in ``[0, 1]`` as a base64 JPEG data URI."""
    buffer = io.BytesIO()
    Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8)).save(
        buffer, format="JPEG", quality=quality, optimize=True
    )
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def edges_as_paths(G: nx.MultiGraph, component_of: dict | None = None) -> list[dict]:
    """Flatten a graph into polylines the page can draw directly.

    Coordinates are rounded to one decimal: the page renders at a few hundred
    pixels, so more precision is bytes with no visible effect.

    Args:
        G: Graph in tile pixel coordinates.
        component_of: Optional node-to-component index, used to colour a
            fragmented network so breaks are visible rather than implied.

    Returns:
        One dict per edge with its points and component index.
    """
    paths = []
    for u, v, key in G.edges(keys=True):
        pts = geograph.oriented_pts(G, u, v, key)
        paths.append(
            {
                "pts": [[round(float(x), 1), round(float(y), 1)] for x, y in pts],
                "c": int(component_of.get(u, 0)) if component_of else 0,
            }
        )
    return paths


def component_index(G: nx.MultiGraph) -> dict:
    """Map each node to its connected component, largest component first."""
    groups = sorted(nx.connected_components(G), key=len, reverse=True)
    return {node: i for i, group in enumerate(groups) for node in group}


def punch_gaps(mask: np.ndarray, n_gaps: int, seed: int) -> np.ndarray:
    """Erase small squares at random road pixels."""
    rng = np.random.default_rng(seed)
    out = mask.copy()
    rows, cols = np.nonzero(mask)
    if not len(rows) or n_gaps == 0:
        return out
    for i in rng.choice(len(rows), min(n_gaps, len(rows)), replace=False):
        y, x = int(rows[i]), int(cols[i])
        out[
            max(0, y - GAP_RADIUS) : y + GAP_RADIUS + 1,
            max(0, x - GAP_RADIUS) : x + GAP_RADIUS + 1,
        ] = False
    return out


def build_divergence(sample) -> dict:
    """Sweep gap damage on a perfect mask, recording both metrics.

    This is the page's central claim made measurable: the pixel score barely
    moves while the topology score collapses, because a two-pixel break severs a
    route without costing many pixels.
    """
    perfect = sample.mask
    levels = []
    for n_gaps in GAP_COUNTS:
        ious, aplss, graphs = [], [], []
        for seed in GAP_SEEDS:
            damaged = punch_gaps(perfect, n_gaps, seed)
            proposal = cleanup.clean(skeleton.graph_from_mask(damaged))
            ious.append(metrics.iou(perfect, damaged))
            aplss.append(metrics.apls(sample.truth, proposal).score)
            graphs.append(proposal)

        shown = graphs[0]
        levels.append(
            {
                "gaps": n_gaps,
                "iou": round(float(np.mean(ious)), 4),
                "apls": round(float(np.mean(aplss)), 4),
                "components": nx.number_connected_components(shown),
                "edges": edges_as_paths(shown, component_index(shown)),
            }
        )
        logger.info(
            f"  {n_gaps:3d} gaps: IoU {levels[-1]['iou']:.4f}  "
            f"APLS {levels[-1]['apls']:.4f}  {levels[-1]['components']} components"
        )
    return {
        "chip": sample.tile.crs.name,
        "width": int(sample.mask.shape[1]),
        "height": int(sample.mask.shape[0]),
        "image": encode_jpeg(sample.image),
        "levels": levels,
    }


def non_planar_crossings(G: nx.MultiGraph) -> int:
    """Edge pairs that cross geometrically while sharing no node."""
    lines = {k: LineString(geograph.oriented_pts(G, *k)) for k in G.edges(keys=True)}
    return sum(
        1
        for a, la in lines.items()
        for b, lb in lines.items()
        if a < b and not set(a[:2]) & set(b[:2]) and la.crosses(lb)
    )


def build_gallery(source, model, reports, n_chips: int) -> list[dict]:
    """Render a spread of held-out chips, worst through best.

    Deliberately a spread rather than the best few: the interesting chips are
    the ones the model struggles with, and hiding them would make the page a
    worse account of the work.
    """
    ranked = sorted(reports, key=lambda r: r["apls_cleaned"])
    picks = [ranked[round(i * (len(ranked) - 1) / (n_chips - 1))] for i in range(n_chips)]

    gallery = []
    for report in picks:
        sample = source.load(report["sample_id"])
        logits = train.predict_tile_logits(model, sample)
        predicted = logits > 0.0
        proposal = cleanup.clean(skeleton.graph_from_mask(predicted))

        gallery.append(
            {
                "id": report["sample_id"],
                "image": encode_jpeg(sample.image),
                "width": int(sample.mask.shape[1]),
                "height": int(sample.mask.shape[0]),
                "apls": round(report["apls_cleaned"], 4),
                "ceiling": round(report["ceiling_apls"], 4),
                "fraction": round(report["fraction_of_ceiling"], 4),
                "iou": round(report["mask_iou"], 4),
                "crossings": non_planar_crossings(sample.truth),
                "truth": edges_as_paths(sample.truth),
                "pred": edges_as_paths(proposal),
            }
        )
        logger.info(f"  {report['sample_id']}: APLS {report['apls_cleaned']:.4f}")
    return gallery


def build(paths: Paths, n_chips: int, divergence_chip: str | None) -> dict:
    """Assemble the whole payload."""
    run = json.loads(paths.run_json.read_text())
    source = spacenet.SpaceNetTileSource(paths.aoi_root)
    model = train.load_checkpoint(paths.checkpoint)
    reports = run["eval"]["per_tile"]

    logger.info("divergence sweep")
    chip = divergence_chip or max(reports, key=lambda r: r["ceiling_apls"])["sample_id"]
    divergence = build_divergence(source.load(chip))
    divergence["chip"] = chip

    logger.info(f"gallery ({n_chips} chips)")
    gallery = build_gallery(source, model, reports, n_chips)

    aplss = np.array([r["apls_cleaned"] for r in reports])
    return {
        "summary": {
            "n_train_chips": len(run["config"]["train_ids"]),
            "n_val_chips": len(run["config"]["val_ids"]),
            "n_scored": run["eval"]["n_scored"],
            "apls_mean": round(run["eval"]["apls_cleaned"], 4),
            "apls_median": round(run["eval"]["apls_median"], 4),
            "ceiling": round(run["eval"]["ceiling_apls"], 4),
            "fraction_of_ceiling": round(run["eval"]["fraction_of_ceiling"], 4),
            "mask_iou": round(run["eval"]["mask_iou"], 4),
            "best_epoch": run.get("best_epoch"),
            "epochs_run": len(run["history"]),
            "stopped_early": run.get("stopped_early"),
            "crop_size": run["config"]["crop_size"],
            "resolution_m": 1.0,
        },
        "distribution": [round(float(v), 4) for v in np.sort(aplss)],
        "history": [
            {
                "epoch": h["epoch"],
                "train": round(h["train_loss"], 4),
                "val": round(h["val_loss"], 4),
                "iou": round(h["val_iou"], 4),
            }
            for h in run["history"]
        ],
        "divergence": divergence,
        "gallery": gallery,
    }


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi-root", type=Path, default=Path("data/AOI_2_Vegas"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/vegas_best.pt"))
    parser.add_argument("--run-json", type=Path, default=Path("outputs/vegas_best.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/site_data.json"))
    parser.add_argument(
        "--template", type=Path, default=Path("site/showcase.template.html")
    )
    parser.add_argument("--html", type=Path, default=Path("outputs/showcase.html"))
    parser.add_argument("--chips", type=int, default=10)
    parser.add_argument("--divergence-chip", default=None)
    args = parser.parse_args()

    payload = build(
        Paths(args.aoi_root, args.checkpoint, args.run_json, args.out),
        args.chips,
        args.divergence_chip,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload, separators=(",", ":"))
    args.out.write_text(blob)
    logger.info(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")

    if args.template.exists():
        # </script> anywhere inside the payload would close the host tag early.
        safe = blob.replace("</", "<\\/")
        args.html.write_text(args.template.read_text().replace("__SITE_DATA__", safe))
        logger.info(f"wrote {args.html} ({args.html.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()

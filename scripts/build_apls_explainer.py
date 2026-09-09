"""Precompute the geometry an APLS explainer page needs.

APLS is hard to reason about from the formula because the thing being scored is
a *route between two points*, not an edge or a pixel. This exports, for a few
real tiles, the actual paths a handful of control point pairs take through both
graphs, so the page can draw them side by side.

Mirrors `metrics._directional` deliberately: same densify, same injection, same
snapping radius, same pair score. It recomputes rather than imports only because
`_directional` returns aggregated numbers and throws the paths away, and the
paths are the whole point here.

Emits both the JSON payload and the explainer page, by substituting the payload
into `site/apls_explainer.template.html`. `--page-only` rebuilds just the page
from an existing payload, so editing the template costs no inference.

Reads a frozen checkpoint. No training, no network.
"""

import argparse
import base64
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from loguru import logger
from PIL import Image

from geo_graphs import cleanup, geograph, metrics, skeleton, train
from geo_graphs.model import predict_mask

#: Tiles to export, chosen from the threshold sweep for what each one shows.
#: Two of them fail in opposite directions, which is the contrast the page is
#: built around.
TILES = {
    "img553": "missed roads: much of the real network is absent from the prediction",
    "img948": "invented roads: the real roads are found, plus many that are not there",
    "img63": "a typical tile, both directions balanced",
    "img540": "near-perfect, for a sense of what a good score looks like",
}

#: Pairs exported per tile per direction. Enough to show the range of outcomes
#: without shipping every pair, which runs to tens of thousands.
PAIRS_PER_DIRECTION = 14

#: Longest side of the embedded background image, in pixels.
IMAGE_MAX_SIDE = 700

#: Control point spacing along every edge, in metres. Matches `metrics.apls`.
SPACING = 50.0

#: Furthest a control point may move to reach the target graph, in metres.
MAX_SNAP = 25.0


@dataclass(frozen=True, slots=True)
class Pair:
    """One scored control point pair, with the route it takes through each graph.

    Attributes:
        a1: Index into the direction's ``control`` list, one endpoint.
        a2: The other endpoint.
        path_a: Polyline through the source graph, as ``[[x, y], ...]``.
        path_b: Polyline through the target graph, empty when no route exists.
        length_a: Route length in the source graph.
        length_b: Route length in the target graph, ``None`` when unreachable.
        score: The pair's contribution, ``1 - min(1, |la - lb| / la)``.
    """

    a1: int
    a2: int
    path_a: list[list[float]]
    path_b: list[list[float]]
    length_a: float
    length_b: float | None
    score: float


def _positions(G: nx.MultiGraph, nodes: list) -> list[list[float]]:
    """Node positions as ``[[x, y], ...]``, rounded for a smaller payload."""
    return [[round(float(c), 1) for c in G.nodes[n]["pos"]] for n in nodes]


def _sample_pairs(
    DA: nx.MultiGraph,
    DB: nx.MultiGraph,
    control: list,
    match: dict,
    min_path_length: float,
    n_pairs: int,
    rng: np.random.Generator,
) -> tuple[Pair, ...]:
    """Score candidate pairs and keep a spread across the outcome range.

    Picks across score bands rather than uniformly, so a tile whose pairs mostly
    score 1.0 still shows the failures that cost it the rest.

    Args:
        DA: Densified source graph, supplying control points and reference routes.
        DB: Target graph with control points injected.
        control: Control point nodes in ``DA``.
        match: Map from control node to its landed node in ``DB``.
        min_path_length: Routes shorter than this are skipped, as in the metric.
        n_pairs: How many pairs to keep.
        rng: Source of randomness for candidate selection.

    Returns:
        Scored pairs, spread across the score range.
    """
    candidates = [
        (i, j)
        for i in range(len(control))
        for j in range(i + 1, len(control))
    ]
    rng.shuffle(candidates)

    scored: list[Pair] = []
    for i, j in candidates[:4000]:
        a1, a2 = control[i], control[j]
        try:
            la, nodes_a = nx.single_source_dijkstra(DA, a1, a2, weight="length")
        except nx.NetworkXNoPath:
            continue
        if not np.isfinite(la) or la < min_path_length:
            continue

        b1, b2 = match.get(a1), match.get(a2)
        lb: float | None = None
        nodes_b: list = []
        if b1 is not None and b2 is not None:
            try:
                lb, nodes_b = nx.single_source_dijkstra(DB, b1, b2, weight="length")
            except nx.NetworkXNoPath:
                lb = None

        score = 0.0 if lb is None else 1.0 - min(1.0, abs(la - lb) / la)
        scored.append(
            Pair(
                a1=i,
                a2=j,
                path_a=_positions(DA, nodes_a),
                path_b=_positions(DB, nodes_b),
                length_a=round(float(la), 1),
                length_b=None if lb is None else round(float(lb), 1),
                score=round(float(score), 4),
            )
        )
        if len(scored) >= 600:
            break

    bands = ((-0.01, 0.01), (0.01, 0.5), (0.5, 0.9), (0.9, 0.999), (0.999, 1.01))
    per_band = max(n_pairs // len(bands), 1)
    kept: list[Pair] = []
    for low, high in bands:
        members = [p for p in scored if low < p.score <= high]
        kept.extend(members[:per_band])

    remaining = [p for p in scored if p not in kept]
    kept.extend(remaining[: max(n_pairs - len(kept), 0)])
    return tuple(sorted(kept, key=lambda p: p.score))


def build_direction(
    A: nx.MultiGraph,
    B: nx.MultiGraph,
    spacing: float,
    max_snap: float,
    min_path_length: float,
    n_pairs: int,
    rng: np.random.Generator,
) -> dict:
    """Export control points and sampled pairs for one scoring direction.

    Args:
        A: Graph supplying control points and reference path lengths.
        B: Graph the control points are injected into.
        spacing: Distance between control points along each edge of A.
        max_snap: Furthest a control point may move to reach B.
        min_path_length: Shortest route worth scoring.
        n_pairs: Pairs to keep for this direction.
        rng: Source of randomness.

    Returns:
        A JSON-ready dict of control points, their landing sites and the pairs.
    """
    DA = geograph.densify(A, spacing, "reference")
    control = list(DA.nodes)
    xy = np.array([DA.nodes[n]["pos"] for n in control], dtype=float)
    DB, landed = geograph.inject_points(B, xy, max_snap)
    match = {c: n for c, n in zip(control, landed, strict=True) if n is not None}

    pairs = _sample_pairs(DA, DB, control, match, min_path_length, n_pairs, rng)
    return {
        "control": _positions(DA, control),
        "landed": [
            None if c not in match else _positions(DB, [match[c]])[0] for c in control
        ],
        "n_control": len(control),
        "n_matched": len(match),
        "pairs": [asdict(p) for p in pairs],
    }


def _encode_image(image: np.ndarray) -> str:
    """Downscale a tile to a base64 JPEG data URI for embedding."""
    array = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
    pil = Image.fromarray(array)
    pil.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def _graph_edges(G: nx.MultiGraph) -> list[list[list[float]]]:
    """Every edge's polyline, rounded, for drawing the graph."""
    return [
        [[round(float(x), 1), round(float(y), 1)] for x, y in geograph.oriented_pts(G, u, v, k)]
        for u, v, k in G.edges(keys=True)
    ]


def build(
    aoi_root: Path,
    checkpoint: Path,
    resolution: float,
    threshold: float,
    n_pairs: int,
    device: str,
) -> dict:
    """Assemble the whole payload: one entry per tile in :data:`TILES`.

    Args:
        aoi_root: Root of the extracted SpaceNet AOI.
        checkpoint: Frozen model checkpoint to predict with.
        resolution: Ground resolution the tiles are resampled to, in m/px.
        threshold: Probability above which a pixel is road.
        n_pairs: Control point pairs to export per tile per direction.
        device: Torch device for inference.

    Returns:
        A JSON-ready dict of the scoring constants and the per-tile geometry.
    """
    source, _ = train.build_source(aoi_root, "spacenet", resolution)
    model = train.load_checkpoint(checkpoint, device=device)
    rng = np.random.default_rng(0)

    tiles = []
    for sample_id, caption in TILES.items():
        sample = source.load(sample_id)
        logits = train.predict_tile_logits(model, sample, device=device)
        proposal = cleanup.clean(
            skeleton.graph_from_mask(predict_mask(logits, threshold))
        )
        result = metrics.apls(sample.truth, proposal)
        logger.info(f"{sample_id}: {result}")

        height, width = sample.mask.shape
        tiles.append(
            {
                "id": sample_id,
                "caption": caption,
                "width": int(width),
                "height": int(height),
                "image": _encode_image(sample.image),
                "truth": _graph_edges(sample.truth),
                "proposal": _graph_edges(proposal),
                "apls": round(result.score, 4),
                "gt_to_prop": round(result.gt_to_prop, 4),
                "prop_to_gt": round(result.prop_to_gt, 4),
                "directions": {
                    "gt_to_prop": build_direction(
                        sample.truth, proposal, SPACING, MAX_SNAP, 10.0, n_pairs, rng
                    ),
                    "prop_to_gt": build_direction(
                        proposal, sample.truth, SPACING, MAX_SNAP, 10.0, n_pairs, rng
                    ),
                },
            }
        )

    return {
        "threshold": threshold,
        "spacing": SPACING,
        "max_snap": MAX_SNAP,
        "tiles": tiles,
    }


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/vegas_best.pt"))
    parser.add_argument("--aoi-root", type=Path, default=Path("data/AOI_2_Vegas"))
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--pairs", type=int, default=PAIRS_PER_DIRECTION)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("outputs/apls_explainer.json"))
    parser.add_argument(
        "--template", type=Path, default=Path("site/apls_explainer.template.html")
    )
    parser.add_argument("--html", type=Path, default=Path("outputs/apls_explainer.html"))
    parser.add_argument(
        "--page-only",
        action="store_true",
        help="Rebuild the page from an existing --out JSON, skipping inference.",
    )
    args = parser.parse_args()

    if args.page_only:
        blob = args.out.read_text()
    else:
        payload = build(
            args.aoi_root,
            args.checkpoint,
            args.resolution,
            args.threshold,
            args.pairs,
            args.device,
        )
        blob = json.dumps(payload, separators=(",", ":"))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(blob)
        logger.info(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")

    if args.template.exists():
        # </script> anywhere inside the payload would close the host tag early.
        safe = blob.replace("</", "<\\/")
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(args.template.read_text().replace("__APLS_DATA__", safe))
        logger.info(f"wrote {args.html} ({args.html.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()

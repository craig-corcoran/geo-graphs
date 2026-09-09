"""Rendering graphs to binary road masks."""

from collections.abc import Iterable

import networkx as nx
import numpy as np
from skimage.draw import line as bresenham
from skimage.morphology import dilation, disk

from . import geograph, tiles
from .tiles import Tile


def draw_polylines(
    shape: tuple[int, int],
    polylines: Iterable[np.ndarray],
    half_width_px: int = 2,
) -> np.ndarray:
    """Render polylines onto a pixel grid, dilated to the width of a surface.

    Split out from :func:`rasterize` because the same drawing is needed for
    things that are not the label mask — a single edge lifted to overpass
    height, for one — and those callers have a grid but no :class:`Tile`.

    Args:
        shape: ``(height, width)`` of the output grid.
        polylines: ``(N, 2)`` arrays of ``(x, y)`` pixel coordinates.
        half_width_px: Dilation radius. At 1 m/px, 2 gives roads ~5 m wide.
            Zero draws bare centerlines.

    Returns:
        Boolean array of shape ``shape``, True on the drawn surface.
    """
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)

    for pts in polylines:
        pts = np.asarray(pts, dtype=float)
        if len(pts) < 2:
            continue
        cols = np.clip(np.round(pts[:, 0]).astype(int), 0, width - 1)
        rows = np.clip(np.round(pts[:, 1]).astype(int), 0, height - 1)
        for i in range(len(pts) - 1):
            # Bresenham is not symmetric: drawing A->B and B->A can differ by a
            # pixel on ties. Edge orientation in an undirected MultiGraph is
            # arbitrary, so draw from a canonical endpoint to keep the mask a
            # function of the geometry rather than of node numbering.
            ends = sorted([(rows[i], cols[i]), (rows[i + 1], cols[i + 1])])
            rr, cc = bresenham(*ends[0], *ends[1])
            mask[rr, cc] = True

    if half_width_px > 0:
        mask = dilation(mask, disk(half_width_px))
    return mask


def rasterize(G: nx.MultiGraph, tile: Tile, half_width_px: int = 2) -> np.ndarray:
    """Render edge centerlines, then dilate to the width of a road surface.

    Matches how SpaceNet builds its own masks: the vector ground truth is a
    centerline and the mask is that centerline buffered by a fixed radius.

    Args:
        G: Graph in tile pixel coordinates.
        tile: Tile defining the output grid.
        half_width_px: Dilation radius. At 1 m/px, 2 gives roads ~5 m wide.

    Returns:
        Boolean array of shape ``(height, width)``, True on road.
    """
    return draw_polylines(
        tiles.shape(tile),
        (geograph.oriented_pts(G, u, v, k) for u, v, k in G.edges(keys=True)),
        half_width_px=half_width_px,
    )

"""Rendering graphs to binary road masks."""

import networkx as nx
import numpy as np
from skimage.draw import line as bresenham
from skimage.morphology import dilation, disk

from . import geograph, tiles
from .tiles import Tile


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
    height, width = tiles.shape(tile)
    mask = np.zeros((height, width), dtype=bool)

    for u, v, k in G.edges(keys=True):
        pts = geograph.oriented_pts(G, u, v, k)
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

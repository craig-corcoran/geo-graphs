"""Ground-truth road graphs from OpenStreetMap, in tile pixel coordinates."""

import networkx as nx
import numpy as np
import osmnx as ox
from shapely.geometry import LineString, MultiLineString, box

from . import geograph, tiles
from .tiles import Tile


def _edge_polylines(G: nx.MultiGraph | nx.MultiDiGraph) -> list[np.ndarray]:
    """Extract each edge's geometry, falling back to a straight run."""
    out = []
    for u, v, data in G.edges(data=True):
        geom = data.get("geometry")
        if geom is None:
            geom = LineString(
                [(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]
            )
        out.append(np.asarray(geom.coords, dtype=float))
    return out


def _clip(pts: np.ndarray, region) -> list[np.ndarray]:
    """Cut a polyline to a region, which may split it into several pieces."""
    clipped = LineString(pts).intersection(region)
    if clipped.is_empty:
        return []
    parts = clipped.geoms if isinstance(clipped, MultiLineString) else [clipped]
    return [
        np.asarray(p.coords, dtype=float)
        for p in parts
        if isinstance(p, LineString) and len(p.coords) >= 2
    ]


def ground_truth_graph(
    tile: Tile, network_type: str = "drive", pad_m: float = 250.0
) -> nx.MultiGraph:
    """Fetch OSM roads covering a tile and return them in pixel coordinates.

    Args:
        tile: Tile to fetch roads for.
        network_type: osmnx network filter, e.g. ``"drive"`` or ``"all"``.
        pad_m: Margin added to the query extent so roads crossing the tile edge
            arrive whole and are cut at the boundary here, rather than ending
            wherever OSM's own bbox filter truncated them.

    Returns:
        An undirected graph in the conventions described in :mod:`geograph`.
    """
    raw = ox.graph_from_bbox(
        tiles.lonlat_bounds(tile, pad_m=pad_m), network_type=network_type
    )
    raw = ox.project_graph(raw, to_crs=tile.crs)
    # OSM arrives directed; a two-way street is two edges and would otherwise
    # be rasterized and scored twice.
    raw = ox.convert.to_undirected(raw)

    region = box(0, 0, tile.width, tile.height)
    edges = []
    for world_pts in _edge_polylines(raw):
        edges.extend(_clip(tiles.world_to_px(tile, world_pts), region))

    G = geograph.build(edges, tol=1e-3)
    G.graph["resolution"] = tile.resolution
    return G

import networkx as nx
import numpy as np
import pytest

from geo_graphs import geograph, raster, tiles
from geo_graphs.tiles import Tile


def _tile(size: int = 100) -> Tile:
    """A tile whose CRS is irrelevant: rasterize only reads its grid shape."""
    return Tile(
        crs=tiles.utm_crs_for(0.0, 0.0),
        x_min=0.0,
        y_max=float(size),
        height=size,
        width=size,
        resolution=1.0,
    )


def _horizontal(row: float = 50.0, x0: float = 10.0, x1: float = 90.0):
    return geograph.build([np.array([[x0, row], [x1, row]])])


def test_empty_graph_gives_empty_mask():
    mask = raster.rasterize(nx.MultiGraph(), _tile())
    assert mask.dtype == bool
    assert not mask.any()


def test_shape_matches_the_tile():
    tile = Tile(
        crs=tiles.utm_crs_for(0.0, 0.0),
        x_min=0.0,
        y_max=40.0,
        height=40,
        width=70,
        resolution=1.0,
    )
    assert raster.rasterize(_horizontal(row=20.0), tile).shape == tiles.shape(tile)


def test_zero_width_draws_a_single_pixel_centerline():
    mask = raster.rasterize(_horizontal(), _tile(), half_width_px=0)
    assert mask[50, 10:91].all()
    assert mask.sum() == 81
    assert not mask[49].any() and not mask[51].any()


def test_dilation_thickens_symmetrically():
    mask = raster.rasterize(_horizontal(), _tile(), half_width_px=2)
    rows = np.flatnonzero(mask[:, 50])
    assert rows.tolist() == [48, 49, 50, 51, 52]


@pytest.mark.parametrize("half_width", [0, 1, 2, 4])
def test_wider_roads_never_cover_fewer_pixels(half_width):
    tile, G = _tile(), _horizontal()
    baseline = raster.rasterize(G, tile, half_width_px=0)
    widened = raster.rasterize(G, tile, half_width_px=half_width)
    assert widened.sum() >= baseline.sum()
    assert (baseline & ~widened).sum() == 0


def test_geometry_outside_the_tile_is_clipped_without_raising():
    G = geograph.build([np.array([[-500.0, 50.0], [500.0, 50.0]])])
    mask = raster.rasterize(G, _tile(), half_width_px=0)
    assert mask[50].all()


def test_orientation_does_not_change_the_mask():
    """A polyline and its reverse describe the same road.

    An undirected MultiGraph may report an edge either way round, so a
    rasterizer that trusted the stored point order could differ run to run.
    """
    pts = np.array([[10.0, 10.0], [40.0, 62.0], [80.0, 30.0]])
    forward = raster.rasterize(geograph.build([pts]), _tile())
    reverse = raster.rasterize(geograph.build([pts[::-1]]), _tile())
    assert np.array_equal(forward, reverse)

"""Offline checks on the tile grid: no network, pyproj works from local data."""

import numpy as np
import pytest
import utm
from pyproj import CRS, Transformer

from geo_graphs import tiles
from geo_graphs.tiles import WGS84, Tile

VEGAS = (36.1699, -115.1398)
SYDNEY = (-33.8688, 151.2093)


def vegas_tile(size_m: float = 1024.0, resolution: float = 1.0) -> Tile:
    return tiles.tile_from_center(*VEGAS, size_m=size_m, resolution=resolution)


def to_lonlat(tile: Tile, xy: np.ndarray) -> tuple[float, float]:
    lon, lat = Transformer.from_crs(tile.crs, WGS84, always_xy=True).transform(
        float(xy[0]), float(xy[1])
    )
    return (lon, lat)


def test_shape_and_bounds_agree_with_fields():
    tile = Tile(crs=CRS.from_epsg(32611), x_min=100.0, y_max=900.0, height=30, width=40)
    assert tiles.shape(tile) == (30, 40)
    assert tiles.bounds_world(tile) == (100.0, 870.0, 140.0, 900.0)


def test_bounds_world_scales_with_resolution():
    tile = Tile(
        crs=CRS.from_epsg(32611),
        x_min=100.0,
        y_max=900.0,
        height=30,
        width=40,
        resolution=2.5,
    )
    assert tiles.bounds_world(tile) == (100.0, 825.0, 200.0, 900.0)


def test_corners_map_to_grid_corners():
    tile = vegas_tile()
    x_min, y_min, x_max, y_max = tiles.bounds_world(tile)
    corners = np.array([[x_min, y_max], [x_max, y_min]])
    expected = np.array([[0.0, 0.0], [tile.width, tile.height]])
    assert tiles.world_to_px(tile, corners) == pytest.approx(expected, abs=1e-6)


def test_row_increases_southward_and_col_increases_eastward():
    """Rasters count rows downward, so +y (north) must map to a *smaller* row."""
    tile = vegas_tile()
    origin = np.array([tile.x_min, tile.y_max])
    east = tiles.world_to_px(tile, origin + np.array([100.0, 0.0]))
    north = tiles.world_to_px(tile, origin + np.array([0.0, 100.0]))

    assert east[0] > 0.0 and east[1] == pytest.approx(0.0)
    assert north[1] < 0.0 and north[0] == pytest.approx(0.0)


@pytest.mark.parametrize("resolution", [0.5, 1.0, 2.0])
def test_world_to_px_and_px_to_world_are_inverses(resolution):
    tile = vegas_tile(resolution=resolution)
    x_min, y_min, x_max, y_max = tiles.bounds_world(tile)
    rng = np.random.default_rng(0)
    xy = rng.uniform([x_min, y_min], [x_max, y_max], size=(64, 2))

    assert tiles.px_to_world(tile, tiles.world_to_px(tile, xy)) == pytest.approx(xy)

    cr = rng.uniform([0.0, 0.0], [tile.width, tile.height], size=(64, 2))
    assert tiles.world_to_px(tile, tiles.px_to_world(tile, cr)) == pytest.approx(cr)


def test_unit_resolution_makes_one_pixel_one_metre():
    """Why resolution 1.0 is the default: pixel distances are already metres."""
    tile = vegas_tile(resolution=1.0)
    a = np.array([tile.x_min + 10.0, tile.y_max - 20.0])
    b = np.array([tile.x_min + 310.0, tile.y_max - 420.0])

    px = tiles.world_to_px(tile, np.stack([a, b]))
    assert np.linalg.norm(px[1] - px[0]) == pytest.approx(np.linalg.norm(b - a))
    assert np.linalg.norm(b - a) == pytest.approx(500.0)


def test_coarser_resolution_shrinks_pixel_distances():
    tile = vegas_tile(resolution=4.0)
    a = np.array([tile.x_min, tile.y_max])
    b = a + np.array([400.0, -300.0])

    px = tiles.world_to_px(tile, np.stack([a, b]))
    assert np.linalg.norm(px[1] - px[0]) == pytest.approx(500.0 / 4.0)


def test_transforms_accept_single_points_and_stacked_arrays():
    tile = vegas_tile()
    pts = np.array([[tile.x_min, tile.y_max], [tile.x_min + 7.0, tile.y_max - 3.0]])

    batch = tiles.world_to_px(tile, pts)
    singles = np.stack([tiles.world_to_px(tile, p) for p in pts])

    assert tiles.world_to_px(tile, pts[0]).shape == (2,)
    assert batch.shape == (2, 2)
    assert batch == pytest.approx(singles)


def test_transforms_broadcast_over_leading_axes():
    """Only the trailing axis is the coordinate pair; anything ahead of it rides along."""
    tile = vegas_tile()
    rng = np.random.default_rng(1)
    cr = rng.uniform(0.0, 512.0, size=(3, 5, 2))

    world = tiles.px_to_world(tile, cr)
    assert world.shape == (3, 5, 2)
    assert tiles.world_to_px(tile, world) == pytest.approx(cr)
    assert world[1, 2] == pytest.approx(tiles.px_to_world(tile, cr[1, 2]))


def test_transforms_accept_integer_input():
    """Integer pixel indices are the natural way to index a raster.

    The output array is allocated from the *converted* input, so an int array in
    must not silently truncate the world coordinates coming out.
    """
    tile = vegas_tile()
    assert tiles.px_to_world(tile, np.array([0, 0])) == pytest.approx(
        [tile.x_min, tile.y_max]
    )


def test_tile_from_center_is_square_and_sized_in_metres():
    tile = vegas_tile(size_m=2048.0, resolution=4.0)
    x_min, y_min, x_max, y_max = tiles.bounds_world(tile)

    assert tiles.shape(tile) == (512, 512)
    assert (x_max - x_min, y_max - y_min) == pytest.approx((2048.0, 2048.0))


def test_tile_from_center_is_centred_on_the_requested_point():
    for lat, lon in (VEGAS, SYDNEY):
        tile = tiles.tile_from_center(lat, lon, size_m=1024.0)
        half = np.array([tile.width / 2.0, tile.height / 2.0])
        centre = tiles.px_to_world(tile, half)
        got_lon, got_lat = to_lonlat(tile, centre)
        assert (got_lat, got_lon) == pytest.approx((lat, lon), abs=1e-9)


def test_utm_crs_for_picks_zone_and_hemisphere():
    assert tiles.utm_crs_for(*VEGAS).to_epsg() == 32611
    assert tiles.utm_crs_for(*SYDNEY).to_epsg() == 32756


def test_utm_crs_for_switches_zone_at_the_boundary():
    """Zone 11 starts at 120W: the boundary belongs to the zone east of it."""
    assert tiles.utm_crs_for(0.0, -120.0).to_epsg() == 32611
    assert tiles.utm_crs_for(0.0, -120.000001).to_epsg() == 32610
    assert tiles.utm_crs_for(0.0, -114.000001).to_epsg() == 32611
    assert tiles.utm_crs_for(0.0, -114.0).to_epsg() == 32612


@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        VEGAS,
        SYDNEY,
        (51.5074, -0.1278),
        (-22.9068, -43.1729),
        (35.6762, 139.6503),
        (0.0, 0.0),
        (-0.0001, 179.9),
        (12.0, -179.9),
    ],
)
def test_utm_crs_for_matches_the_utm_reference(lat, lon):
    """Sample points dodge the Norway/Svalbard zone widenings.

    ``utm`` implements those MGRS exceptions and the plain arithmetic here does
    not, so a point in them would report a disagreement that is not a defect:
    the neighbouring zone is still a valid local metric projection, which is all
    a tile needs.
    """
    expected = (32600 if lat >= 0 else 32700) + utm.latlon_to_zone_number(lat, lon)
    assert tiles.utm_crs_for(lat, lon).to_epsg() == expected


def test_utm_crs_is_metric_and_north_up():
    tile = vegas_tile()
    lon_e, lat_e = to_lonlat(tile, np.array([tile.x_min + 1000.0, tile.y_max]))
    _, lat_n = to_lonlat(tile, np.array([tile.x_min, tile.y_max + 1000.0]))
    lon_w, lat_w = to_lonlat(tile, np.array([tile.x_min, tile.y_max]))

    assert lon_e > lon_w
    assert lat_n > lat_w
    assert lat_e == pytest.approx(lat_w, abs=0.01)


def test_lonlat_bounds_contains_the_tile_centre():
    tile = vegas_tile()
    west, south, east, north = tiles.lonlat_bounds(tile)
    lat, lon = VEGAS
    assert west < lon < east
    assert south < lat < north


def test_lonlat_bounds_padding_strictly_grows_the_extent():
    for lat, lon in (VEGAS, SYDNEY):
        tile = tiles.tile_from_center(lat, lon, size_m=1024.0)
        west, south, east, north = tiles.lonlat_bounds(tile)
        p_west, p_south, p_east, p_north = tiles.lonlat_bounds(tile, pad_m=250.0)

        assert p_west < west and p_east > east
        assert p_south < south and p_north > north


def test_lonlat_bounds_padding_is_roughly_the_requested_metres():
    tile = vegas_tile()
    _, south, _, north = tiles.lonlat_bounds(tile)
    _, p_south, _, p_north = tiles.lonlat_bounds(tile, pad_m=1000.0)

    degrees_per_metre = (north - south) / 1024.0
    assert p_north - north == pytest.approx(1000.0 * degrees_per_metre, rel=0.05)
    assert south - p_south == pytest.approx(1000.0 * degrees_per_metre, rel=0.05)

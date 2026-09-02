"""Ground patches and the raster grid laid over them.

Pixel coordinates are ``(col, row)`` with row increasing southward, matching
raster convention. Holding resolution at 1.0 makes one pixel one metre, so
graph lengths and APLS distances are already in metres with no conversion.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer

WGS84 = CRS.from_epsg(4326)


@dataclass(frozen=True, slots=True)
class Tile:
    """A patch of ground with a raster grid over it.

    Attributes:
        crs: Projected CRS the tile's world coordinates live in.
        x_min: Western edge, in ``crs`` units.
        y_max: Northern edge, in ``crs`` units.
        height: Grid height in pixels.
        width: Grid width in pixels.
        resolution: Metres per pixel.
    """

    crs: CRS
    x_min: float
    y_max: float
    height: int
    width: int
    resolution: float = 1.0


def utm_crs_for(lat: float, lon: float) -> CRS:
    """Pick the UTM zone containing a point.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.

    Returns:
        The WGS84 UTM CRS for that zone and hemisphere.
    """
    zone = int((lon + 180.0) / 6.0) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def tile_from_center(
    lat: float, lon: float, size_m: float = 2048.0, resolution: float = 1.0
) -> Tile:
    """Build a square tile centred on a latitude/longitude.

    Args:
        lat: Centre latitude in degrees.
        lon: Centre longitude in degrees.
        size_m: Side length in metres.
        resolution: Metres per pixel.

    Returns:
        A tile in the local UTM projection.
    """
    crs = utm_crs_for(lat, lon)
    cx, cy = Transformer.from_crs(WGS84, crs, always_xy=True).transform(lon, lat)
    n_px = round(size_m / resolution)
    return Tile(
        crs=crs,
        x_min=cx - size_m / 2.0,
        y_max=cy + size_m / 2.0,
        height=n_px,
        width=n_px,
        resolution=resolution,
    )


def shape(tile: Tile) -> tuple[int, int]:
    """Return the grid shape as ``(height, width)``."""
    return (tile.height, tile.width)


def bounds_world(tile: Tile) -> tuple[float, float, float, float]:
    """Return ``(x_min, y_min, x_max, y_max)`` in the tile's own CRS."""
    return (
        tile.x_min,
        tile.y_max - tile.height * tile.resolution,
        tile.x_min + tile.width * tile.resolution,
        tile.y_max,
    )


def world_to_px(tile: Tile, xy: np.ndarray) -> np.ndarray:
    """Convert projected coordinates to pixel coordinates.

    Args:
        tile: Tile defining the grid.
        xy: Array of ``(x, y)`` in the tile's CRS; trailing axis is the pair.

    Returns:
        Array of the same shape holding ``(col, row)``.
    """
    xy = np.asarray(xy, dtype=float)
    out = np.empty_like(xy)
    out[..., 0] = (xy[..., 0] - tile.x_min) / tile.resolution
    out[..., 1] = (tile.y_max - xy[..., 1]) / tile.resolution
    return out


def px_to_world(tile: Tile, cr: np.ndarray) -> np.ndarray:
    """Convert pixel coordinates back to projected coordinates.

    Args:
        tile: Tile defining the grid.
        cr: Array of ``(col, row)``; trailing axis is the pair.

    Returns:
        Array of the same shape holding ``(x, y)`` in the tile's CRS.
    """
    cr = np.asarray(cr, dtype=float)
    out = np.empty_like(cr)
    out[..., 0] = tile.x_min + cr[..., 0] * tile.resolution
    out[..., 1] = tile.y_max - cr[..., 1] * tile.resolution
    return out


def lonlat_bounds(tile: Tile, pad_m: float = 0.0) -> tuple[float, float, float, float]:
    """Return the tile's extent in EPSG:4326.

    Args:
        tile: Tile to convert.
        pad_m: Metres to expand the extent by on every side, used to fetch
            features that cross the tile edge so they can be clipped locally
            rather than truncated by the upstream query.

    Returns:
        ``(west, south, east, north)`` in degrees.
    """
    x_min, y_min, x_max, y_max = bounds_world(tile)
    to_wgs = Transformer.from_crs(tile.crs, WGS84, always_xy=True)
    xs = [x_min - pad_m, x_max + pad_m] * 2
    ys = [y_min - pad_m, y_min - pad_m, y_max + pad_m, y_max + pad_m]
    lons, lats = to_wgs.transform(xs, ys)
    return (min(lons), min(lats), max(lons), max(lats))


def tile_from_transform(
    crs: CRS, transform: Sequence[float], width: int, height: int
) -> Tile:
    """Build a tile from a raster's own georeferencing.

    Imagery carries its CRS and affine transform with it, so a tile covering a
    real GeoTIFF is derived from the file rather than from a centre point.

    Args:
        crs: The raster's coordinate reference system.
        transform: Affine coefficients ``(a, b, c, d, e, f)`` mapping pixel to
            world as ``x = a*col + b*row + c`` and ``y = d*col + e*row + f``.
        width: Raster width in pixels.
        height: Raster height in pixels.

    Returns:
        A tile over the same ground as the raster.

    Raises:
        ValueError: If the CRS is geographic, or the raster is rotated, or its
            pixels are not square. Each would break the assumption that pixel
            distance is ground distance, which the graph metrics rely on.
    """
    if crs.is_geographic:
        # A degree of longitude is shorter than a degree of latitude everywhere
        # but the equator, so a raster that is square in degrees is not square
        # on the ground. SpaceNet ships EPSG:4326 imagery that looks uniform in
        # its own units and is 0.24 m by 0.30 m in reality. Reproject first.
        raise ValueError(
            f"{crs} is geographic; reproject to a metric CRS before building a tile"
        )

    a, b, c, d, e, f = (float(v) for v in transform)
    if b or d:
        raise ValueError(f"rotated rasters are unsupported; got shear ({b}, {d})")
    if not np.isclose(a, -e):
        raise ValueError(f"non-square pixels: x resolution {a}, y resolution {-e}")

    return Tile(crs=crs, x_min=c, y_max=f, height=height, width=width, resolution=a)

"""SpaceNet Roads as a tile source.

The imagery ships as EPSG:4326 GeoTIFFs — geographic, not projected. At the
Las Vegas AOI a pixel is about 0.243 m east-west and 0.300 m north-south, so it
is square in degrees and distinctly not square on the ground. Everything
downstream measures distance in pixels and calls it metres, so each chip is
reprojected to its local UTM zone at a fixed metric resolution on load.

Resolution defaults to 1 m/px. That is the convention the published SpaceNet
numbers use, and it is what our own ceiling measurements were taken at, so
scores stay comparable. It does discard roughly three-fold linear detail
against the native 0.3 m imagery — an honest cost, but the resolution sweep in
the experiment log found the pipeline ceiling moves only between 0.861 and
0.901 across 0.5 to 2 m/px, so resolution is not what limits Stage 1.
Non-planarity is.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import LineString, shape
from shapely.ops import unary_union

from . import geograph, raster, tiles
from .data import TileSample, TileSource
from .tiles import Tile

WGS84 = "EPSG:4326"

#: Image products shipped per AOI. RGB-PanSharpen is the three-band visible
#: product; the others carry more spectral bands than a plain U-Net wants.
DEFAULT_PRODUCT = "RGB-PanSharpen"

#: Shortest noded segment worth keeping, in pixels. Below this a piece is a
#: rounding artifact of the intersection, not a road.
_MIN_SEGMENT_PX = 1e-6


@dataclass(frozen=True, slots=True)
class Chip:
    """One SpaceNet image and its matching road labels.

    Attributes:
        image_id: SpaceNet identifier, e.g. ``AOI_2_Vegas_img1454``.
        image_path: GeoTIFF path.
        labels_path: Road geojson path.
    """

    image_id: str
    image_path: Path
    labels_path: Path


def find_chips(aoi_root: Path, product: str = DEFAULT_PRODUCT) -> tuple[Chip, ...]:
    """Pair every image in an AOI directory with its label file.

    Args:
        aoi_root: An ``AOI_*_Roads_Sample`` or extracted AOI directory.
        product: Image product subdirectory to read.

    Returns:
        Chips with both files present, ordered by id. Images whose labels are
        missing are skipped rather than failing the whole listing, because the
        public test split ships imagery without labels.

    Raises:
        FileNotFoundError: If the product directory does not exist.
    """
    images = aoi_root / product
    if not images.is_dir():
        raise FileNotFoundError(f"no {product} directory under {aoi_root}")

    labels = aoi_root / "geojson" / "spacenetroads"
    chips = []
    for path in sorted(images.glob(f"{product}_*.tif")):
        image_id = path.stem.removeprefix(f"{product}_")
        label_path = labels / f"spacenetroads_{image_id}.geojson"
        if label_path.exists():
            chips.append(Chip(image_id=image_id, image_path=path, labels_path=label_path))
    return tuple(chips)


def _reproject_to_utm(path: Path, resolution: float) -> tuple[np.ndarray, Tile]:
    """Load a GeoTIFF into its local UTM zone at a metric resolution.

    Returns:
        ``(image, tile)`` where image is ``(H, W, C)`` float32 scaled to
        ``[0, 1]`` and tile carries the reprojected georeferencing.
    """
    with rasterio.open(path) as src:
        to_wgs = Transformer.from_crs(src.crs, WGS84, always_xy=True)
        lon, lat = to_wgs.transform(src.bounds.left, src.bounds.top)
        dst_crs = tiles.utm_crs_for(lat, lon)

        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=resolution
        )
        bands = np.zeros((src.count, height, width), dtype=np.float32)
        for index in range(src.count):
            reproject(
                source=rasterio.band(src, index + 1),
                destination=bands[index],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )

    tile = tiles.tile_from_transform(dst_crs, transform[:6], width, height)
    image = np.moveaxis(bands, 0, -1)
    # 11-bit WorldView data stored as uint16; scale per chip rather than by a
    # fixed constant, since exposure varies between captures.
    high = float(np.percentile(image, 99.5)) or 1.0
    return np.clip(image / high, 0.0, 1.0).astype(np.float32), tile


def labels_to_graph(labels_path: Path, tile: Tile) -> nx.MultiGraph:
    """Read road labels and express them in tile pixel coordinates.

    Args:
        labels_path: SpaceNet road geojson, in EPSG:4326.
        tile: Tile defining the target projection and grid.

    Returns:
        A graph following the :mod:`geograph` conventions. Empty when the chip
        has no roads, which SpaceNet does ship.
    """
    features = json.loads(labels_path.read_text()).get("features") or []
    to_tile = Transformer.from_crs(CRS.from_user_input(WGS84), tile.crs, always_xy=True)

    lines = []
    for feature in features:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        for line in getattr(shape(geometry), "geoms", [shape(geometry)]):
            coords = np.asarray(line.coords, dtype=float)
            if len(coords) < 2:
                continue
            xs, ys = to_tile.transform(coords[:, 0], coords[:, 1])
            lines.append(LineString(tiles.world_to_px(tile, np.column_stack([xs, ys]))))

    graph = geograph.build(_node_network(lines), tol=1e-3)
    graph.graph["resolution"] = tile.resolution
    return graph


def _node_network(lines: list[LineString]) -> list[np.ndarray]:
    """Split lines at every point where they meet, so junctions become nodes.

    SpaceNet ships each road as one LineString running straight through its
    intersections. A side street's endpoint lands on a main road's *interior*,
    which is a junction on the ground but shares no vertex in the file, so
    building a graph from the features as given leaves a pile of disconnected
    stubs — one Vegas chip gives 33 edges in 30 components.

    OSM does this noding for us, which is why the OSM path never needed it.

    Args:
        lines: Road centerlines in tile pixel coordinates.

    Returns:
        Polylines split at every intersection, ready for :func:`geograph.build`.
    """
    if not lines:
        return []
    noded = unary_union(lines)
    parts = getattr(noded, "geoms", [noded])
    return [
        np.asarray(part.coords, dtype=float)
        for part in parts
        # unary_union can emit a degenerate zero-length piece at an
        # intersection. Kept, it becomes a self-loop that adds two to the
        # junction's degree and no geometry at all.
        if isinstance(part, LineString)
        and len(part.coords) >= 2
        and part.length > _MIN_SEGMENT_PX
    ]


class SpaceNetTileSource:
    """Real imagery and labels from an extracted SpaceNet Roads AOI."""

    def __init__(
        self,
        aoi_root: Path | str,
        resolution: float = 1.0,
        product: str = DEFAULT_PRODUCT,
        half_width_px: int = 2,
    ) -> None:
        """
        Args:
            aoi_root: Directory holding the image product and geojson folders.
            resolution: Metres per pixel to reproject to.
            product: Image product subdirectory.
            half_width_px: Road half-width used to rasterize the label mask.
        """
        self.aoi_root = Path(aoi_root)
        self.resolution = resolution
        self.product = product
        self.half_width_px = half_width_px
        self._chips = {chip.image_id: chip for chip in find_chips(self.aoi_root, product)}

    def ids(self) -> tuple[str, ...]:
        """Return every chip id that has both imagery and labels."""
        return tuple(self._chips)

    def load(self, sample_id: str) -> TileSample:
        """Load one chip, reprojected and with its labels rasterized.

        Args:
            sample_id: A chip id from :meth:`ids`.

        Returns:
            The loaded sample.

        Raises:
            KeyError: If the chip is not in this AOI.
        """
        if sample_id not in self._chips:
            raise KeyError(f"unknown chip {sample_id!r} under {self.aoi_root}")

        chip = self._chips[sample_id]
        image, tile = _reproject_to_utm(chip.image_path, self.resolution)
        truth = labels_to_graph(chip.labels_path, tile)
        mask = raster.rasterize(truth, tile, half_width_px=self.half_width_px)
        return TileSample(tile=tile, image=image, mask=mask, truth=truth)


_: type[TileSource] = SpaceNetTileSource


def register(registry: dict, key: str = "spacenet") -> None:
    """Add this source to a tile-source registry.

    Kept explicit rather than importing for side effects, so a caller that
    never touches SpaceNet does not pay for rasterio.

    Args:
        registry: The registry to mutate, normally ``data.TILE_SOURCE_REGISTRY``.
        key: Config key to register under.
    """
    registry[key] = lambda **kwargs: SpaceNetTileSource(**kwargs)


def aoi_roots(sample_root: Path | str) -> tuple[Path, ...]:
    """List AOI directories inside an extracted sample or train archive.

    Args:
        sample_root: Directory such as ``data/SpaceNet_Roads_Sample``.

    Returns:
        Every immediate subdirectory that holds a road geojson folder.
    """
    root = Path(sample_root)
    return tuple(
        sorted(p for p in root.iterdir() if (p / "geojson" / "spacenetroads").is_dir())
    )

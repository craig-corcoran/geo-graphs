"""Coverage for the SpaceNet loader.

Most tests build a GeoTIFF and a geojson on the fly, so they run offline and
without the 0.7 GB sample archive. The handful that need the real thing skip
when it is absent.
"""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import rasterio
from pyproj import CRS
from rasterio.transform import from_origin

from geo_graphs import data, geograph, spacenet, tiles

SAMPLE_ROOT = Path("data/SpaceNet_Roads_Sample")
needs_sample = pytest.mark.skipif(
    not SAMPLE_ROOT.is_dir(), reason="SpaceNet sample archive not downloaded"
)

# A patch of Las Vegas, in the geographic CRS SpaceNet actually ships.
WEST, NORTH = -115.167, 36.125
DEGREES_PER_PX = 2.7e-06


def write_chip(root: Path, image_id: str, size: int = 64, with_labels: bool = True):
    """Write a GeoTIFF and matching geojson in SpaceNet's layout."""
    images = root / spacenet.DEFAULT_PRODUCT
    images.mkdir(parents=True, exist_ok=True)
    path = images / f"{spacenet.DEFAULT_PRODUCT}_{image_id}.tif"

    transform = from_origin(WEST, NORTH, DEGREES_PER_PX, DEGREES_PER_PX)
    pixels = np.full((3, size, size), 400, dtype=np.uint16)
    pixels[:, size // 2 - 2 : size // 2 + 2, :] = 1200  # a bright horizontal road

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=3,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(pixels)

    if not with_labels:
        return path

    labels = root / "geojson" / "spacenetroads"
    labels.mkdir(parents=True, exist_ok=True)
    mid_lat = NORTH - (size / 2) * DEGREES_PER_PX
    (labels / f"spacenetroads_{image_id}.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [WEST + 2 * DEGREES_PER_PX, mid_lat],
                                [WEST + (size - 2) * DEGREES_PER_PX, mid_lat],
                            ],
                        },
                    }
                ],
            }
        )
    )
    return path


def test_find_chips_pairs_images_with_labels(tmp_path):
    write_chip(tmp_path, "AOI_2_Vegas_img1")
    write_chip(tmp_path, "AOI_2_Vegas_img2")

    chips = spacenet.find_chips(tmp_path)

    assert [c.image_id for c in chips] == ["AOI_2_Vegas_img1", "AOI_2_Vegas_img2"]
    assert all(c.image_path.exists() and c.labels_path.exists() for c in chips)


def test_find_chips_skips_images_without_labels(tmp_path):
    """The public test split ships imagery only; listing must not blow up."""
    write_chip(tmp_path, "AOI_2_Vegas_img1")
    write_chip(tmp_path, "AOI_2_Vegas_img2", with_labels=False)

    assert [c.image_id for c in spacenet.find_chips(tmp_path)] == ["AOI_2_Vegas_img1"]


def test_find_chips_reports_a_missing_product_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="RGB-PanSharpen"):
        spacenet.find_chips(tmp_path)


def test_load_reprojects_into_a_metric_crs(tmp_path):
    """SpaceNet ships EPSG:4326; pixels must become metres before scoring."""
    write_chip(tmp_path, "AOI_2_Vegas_img1")
    sample = spacenet.SpaceNetTileSource(tmp_path).load("AOI_2_Vegas_img1")

    assert not sample.tile.crs.is_geographic
    assert sample.tile.resolution == 1.0
    assert sample.image.shape[:2] == sample.mask.shape


def test_resolution_is_a_real_parameter(tmp_path):
    write_chip(tmp_path, "AOI_2_Vegas_img1")
    fine = spacenet.SpaceNetTileSource(tmp_path, resolution=0.5).load("AOI_2_Vegas_img1")
    coarse = spacenet.SpaceNetTileSource(tmp_path, resolution=2.0).load(
        "AOI_2_Vegas_img1"
    )

    assert fine.tile.resolution == 0.5
    assert fine.mask.shape[0] > coarse.mask.shape[0]


def test_labels_land_on_the_road_in_the_imagery(tmp_path):
    """The alignment check: a mask offset by a reprojection slip would fail."""
    write_chip(tmp_path, "AOI_2_Vegas_img1", size=128)
    sample = spacenet.SpaceNetTileSource(tmp_path).load("AOI_2_Vegas_img1")

    assert sample.mask.any()
    grey = sample.image.mean(axis=-1)
    assert grey[sample.mask].mean() > grey[~sample.mask].mean()


def test_labels_become_a_graph_in_pixel_coordinates(tmp_path):
    write_chip(tmp_path, "AOI_2_Vegas_img1", size=128)
    sample = spacenet.SpaceNetTileSource(tmp_path).load("AOI_2_Vegas_img1")

    assert sample.truth.number_of_edges() == 1
    assert geograph.total_length(sample.truth) > 10
    for u, v, k in sample.truth.edges(keys=True):
        pts = geograph.oriented_pts(sample.truth, u, v, k)
        assert pts[:, 0].min() >= -1
        assert pts[:, 1].min() >= -1


def test_load_rejects_an_unknown_chip(tmp_path):
    write_chip(tmp_path, "AOI_2_Vegas_img1")
    with pytest.raises(KeyError, match="unknown chip"):
        spacenet.SpaceNetTileSource(tmp_path).load("AOI_2_Vegas_img999")


def test_source_satisfies_the_protocol(tmp_path):
    write_chip(tmp_path, "AOI_2_Vegas_img1")
    assert isinstance(spacenet.SpaceNetTileSource(tmp_path), data.TileSource)


def test_register_adds_a_factory(tmp_path):
    write_chip(tmp_path, "AOI_2_Vegas_img1")
    registry: dict = {}
    spacenet.register(registry)

    source = registry["spacenet"](aoi_root=tmp_path)
    assert source.ids() == ("AOI_2_Vegas_img1",)


def test_geographic_tiles_are_refused_outright():
    """Guards the unit trap: square in degrees is not square on the ground."""
    with pytest.raises(ValueError, match="geographic"):
        tiles.tile_from_transform(
            CRS.from_epsg(4326), (2.7e-06, 0, -115.0, 0, -2.7e-06, 36.0), 100, 100
        )


@needs_sample
def test_real_vegas_chips_all_load():
    vegas = next(r for r in spacenet.aoi_roots(SAMPLE_ROOT) if "Vegas" in r.name)
    source = spacenet.SpaceNetTileSource(vegas)

    assert len(source.ids()) > 0
    for chip_id in source.ids():
        sample = source.load(chip_id)
        assert sample.image.shape[:2] == sample.mask.shape
        assert sample.tile.resolution == 1.0


@needs_sample
def test_real_labels_align_with_real_imagery():
    """Vegas asphalt is dark against bright desert, so the sign is known."""
    vegas = next(r for r in spacenet.aoi_roots(SAMPLE_ROOT) if "Vegas" in r.name)
    source = spacenet.SpaceNetTileSource(vegas)
    sample = source.load(source.ids()[0])

    grey = sample.image.mean(axis=-1)
    assert grey[sample.mask].mean() < grey[~sample.mask].mean() - 0.05


def test_labels_are_noded_at_crossings(tmp_path):
    """SpaceNet runs roads straight through intersections without a shared node.

    Two streets crossing are two LineStrings in the file with no vertex in
    common, which is a junction on the ground and a pair of disconnected edges
    in a naive graph. Left unnoded, one real Vegas chip gives 33 edges across 30
    components and an achievable APLS of 0.10; noded it is 3 components and
    0.98. OSM does this for us, which is why the OSM path never needed it.
    """
    images = tmp_path / spacenet.DEFAULT_PRODUCT
    images.mkdir(parents=True)
    with rasterio.open(
        images / f"{spacenet.DEFAULT_PRODUCT}_AOI_2_Vegas_imgX.tif",
        "w",
        driver="GTiff",
        height=128,
        width=128,
        count=3,
        dtype="uint16",
        crs="EPSG:4326",
        transform=from_origin(WEST, NORTH, DEGREES_PER_PX, DEGREES_PER_PX),
    ) as dst:
        dst.write(np.full((3, 128, 128), 400, dtype=np.uint16))

    d = DEGREES_PER_PX
    horizontal = [[WEST + 10 * d, NORTH - 64 * d], [WEST + 118 * d, NORTH - 64 * d]]
    vertical = [[WEST + 64 * d, NORTH - 10 * d], [WEST + 64 * d, NORTH - 118 * d]]
    labels = tmp_path / "geojson" / "spacenetroads"
    labels.mkdir(parents=True)
    (labels / "spacenetroads_AOI_2_Vegas_imgX.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {"type": "LineString", "coordinates": c},
                    }
                    for c in (horizontal, vertical)
                ],
            }
        )
    )

    truth = spacenet.SpaceNetTileSource(tmp_path).load("AOI_2_Vegas_imgX").truth

    assert nx.number_connected_components(truth) == 1
    assert truth.number_of_edges() == 4  # both roads split at the crossing
    assert sorted(deg for _, deg in truth.degree()) == [1, 1, 1, 1, 4]
    assert nx.number_of_selfloops(truth) == 0


def test_noding_drops_degenerate_slivers(tmp_path):
    """A zero-length piece at an intersection would become a self-loop.

    It adds two to the junction's degree and contributes no geometry, so the
    graph would claim a junction busier than the road actually is.
    """
    from shapely.geometry import LineString

    line = LineString([(0.0, 0.0), (10.0, 0.0)])
    sliver = LineString([(5.0, 0.0), (5.0, 0.0)])
    pieces = spacenet._node_network([line, sliver])

    assert all(LineString(p).length > 0 for p in pieces)

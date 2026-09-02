"""End-to-end checks on a real OSM tile. Opt in with ``pytest -m network``.

Downtown Las Vegas is deliberate: it is inside SpaceNet's AOI 2, so anything
tuned against this tile stays meaningful once real imagery is in play.
"""

import numpy as np
import pytest

from geo_graphs import cleanup, geograph, metrics, raster, skeleton, tiles
from geo_graphs.osm import ground_truth_graph
from geo_graphs.tiles import Tile
from tests.reference.oracle import reference_apls

pytestmark = pytest.mark.network

VEGAS = (36.1699, -115.1398)


@pytest.fixture(scope="module")
def tile() -> Tile:
    return tiles.tile_from_center(*VEGAS, size_m=1024)


@pytest.fixture(scope="module")
def truth(tile) -> "object":
    return ground_truth_graph(tile)


@pytest.fixture(scope="module")
def recovered(tile, truth):
    """Ground truth rendered to a perfect mask and traced back out again."""
    return cleanup.clean(skeleton.graph_from_mask(raster.rasterize(truth, tile)))


def test_tile_is_metric(tile):
    """One pixel is one metre, so graph lengths need no unit conversion."""
    assert tile.resolution == 1.0
    corner = tiles.px_to_world(tile, np.array([0.0, 0.0]))
    assert corner == pytest.approx([tile.x_min, tile.y_max])


def test_ground_truth_is_undirected(truth):
    """OSM arrives directed; two-way streets must not be counted twice."""
    assert truth.number_of_edges() < 2 * truth.number_of_nodes()
    assert geograph.total_length(truth) > 10_000


def test_round_trip_recovers_most_of_the_network(truth, recovered):
    assert geograph.total_length(recovered) == pytest.approx(
        geograph.total_length(truth), rel=0.10
    )
    assert metrics.apls(truth, recovered).score > 0.80


def test_agrees_with_reference_on_real_data(truth, recovered):
    """Tighter than the synthetic bound: real tiles are where it matters."""
    ours = metrics.apls(truth, recovered).score
    theirs = reference_apls(truth, recovered)[0]
    assert ours == pytest.approx(theirs, abs=0.03)


def test_perfect_mask_does_not_round_trip_perfectly(truth, recovered):
    """The harness has a ceiling below 1.0, and it must stay visible.

    Rasterizing at 1 m/px and skeletonizing loses real structure: parallel
    streets closer than the road width merge into one centerline. Any model
    scored through this pipeline inherits that ceiling, so a future change that
    silently hides it would make model numbers look better than they are.
    """
    assert metrics.apls(truth, recovered).score < 0.95

import numpy as np
import pytest

from geo_graphs import data, tiles


def road_mask(height: int = 128, width: int = 128) -> np.ndarray:
    m = np.zeros((height, width), bool)
    m[60:66, 5:120] = True
    m[5:120, 60:66] = True
    return m


def sample_with(mask: np.ndarray) -> data.TileSample:
    rng = np.random.default_rng(0)
    tile = tiles.tile_from_center(36.1699, -115.1398, size_m=float(mask.shape[0]))
    import networkx as nx

    return data.TileSample(
        tile=tile,
        image=data.synthesize_image(mask, rng),
        mask=mask,
        truth=nx.MultiGraph(),
    )


def test_synthetic_image_has_expected_shape_and_range():
    mask = road_mask()
    image = data.synthesize_image(mask, np.random.default_rng(0))

    assert image.shape == (*mask.shape, 3)
    assert image.dtype == np.float32
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_synthetic_image_is_reproducible_from_its_seed():
    mask = road_mask()
    a = data.synthesize_image(mask, np.random.default_rng(7))
    b = data.synthesize_image(mask, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_synthetic_image_is_brighter_on_road():
    """The signal has to exist, or the loop would be exercising nothing."""
    mask = road_mask()
    image = data.synthesize_image(mask, np.random.default_rng(0), n_distractors=0)
    grey = image.mean(axis=-1)
    assert grey[mask].mean() > grey[~mask].mean() + 0.1


def test_synthetic_image_is_not_a_copy_of_the_mask():
    """A trivially separable image would hide bugs rather than expose them.

    If a single global threshold recovered the mask exactly, the fake would be
    teaching the model nothing and testing the loop barely more.
    """
    mask = road_mask()
    grey = data.synthesize_image(mask, np.random.default_rng(0)).mean(axis=-1)

    best = max(
        (grey > t)[mask].mean() - (grey > t)[~mask].mean()
        for t in np.linspace(grey.min(), grey.max(), 64)
    )
    assert best < 0.99


def test_synthetic_image_channels_are_not_identical():
    """Guards against a channel-order or broadcast slip going unnoticed."""
    image = data.synthesize_image(road_mask(), np.random.default_rng(0))
    assert not np.allclose(image[..., 0], image[..., 2])


def test_crop_specs_stay_in_bounds():
    mask = road_mask()
    specs = data.crop_specs(mask, size=32, count=25, rng=np.random.default_rng(0))

    assert len(specs) == 25
    assert all(0 <= s.row <= mask.shape[0] - s.size for s in specs)
    assert all(0 <= s.col <= mask.shape[1] - s.size for s in specs)


def test_crop_specs_prefer_crops_containing_road():
    mask = road_mask()
    rng = np.random.default_rng(0)
    biased = data.crop_specs(mask, 32, 40, rng, min_road_fraction=0.02)
    uniform = data.crop_specs(mask, 32, 40, rng, min_road_fraction=0.0)

    def road_share(specs):
        return np.mean(
            [mask[s.row : s.row + s.size, s.col : s.col + s.size].mean() for s in specs]
        )

    assert road_share(biased) > road_share(uniform)


def test_crop_specs_still_return_the_full_count_on_a_sparse_tile():
    """Rejection sampling must degrade to accepting, not to returning fewer."""
    sparse = np.zeros((128, 128), bool)
    sparse[0, 0] = True
    specs = data.crop_specs(
        sparse, size=32, count=10, rng=np.random.default_rng(0), min_road_fraction=0.5
    )
    assert len(specs) == 10


def test_crop_specs_are_reproducible():
    mask = road_mask()
    a = data.crop_specs(mask, 32, 10, np.random.default_rng(3))
    b = data.crop_specs(mask, 32, 10, np.random.default_rng(3))
    assert a == b


def test_crop_specs_reject_a_size_that_does_not_fit():
    with pytest.raises(ValueError, match="exceeds mask shape"):
        data.crop_specs(
            road_mask(64, 64), size=128, count=1, rng=np.random.default_rng(0)
        )


def test_take_crop_returns_aligned_image_and_mask():
    mask = road_mask()
    sample = sample_with(mask)
    spec = data.CropSpec(row=50, col=40, size=32)

    image_crop, mask_crop = data.take_crop(sample, spec)

    assert image_crop.shape == (32, 32, 3)
    assert mask_crop.shape == (32, 32)
    assert np.array_equal(mask_crop, mask[50:82, 40:72])


def test_registry_yields_a_fresh_instance_each_lookup():
    factory = data.TILE_SOURCE_REGISTRY["synthetic"]
    assert factory() is not factory()


def test_registry_entries_satisfy_the_protocol():
    assert all(
        isinstance(factory(), data.TileSource)
        for factory in data.TILE_SOURCE_REGISTRY.values()
    )


def test_load_tile_rejects_an_unknown_source():
    with pytest.raises(KeyError, match="unknown tile source"):
        data.load_tile(36.1699, -115.1398, source="spacenet")


@pytest.mark.network
def test_load_tile_produces_aligned_image_and_labels():
    sample = data.load_tile(36.1699, -115.1398, size_m=512.0)

    assert sample.image.shape[:2] == sample.mask.shape
    assert sample.mask.any()
    assert sample.truth.number_of_edges() > 0


@pytest.mark.network
def test_the_same_tile_yields_the_same_image():
    a = data.load_tile(36.1699, -115.1398, size_m=512.0)
    b = data.load_tile(36.1699, -115.1398, size_m=512.0)
    assert np.array_equal(a.image, b.image)

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest
from scipy import ndimage

from geo_graphs import data, geograph, tiles


def road_mask(height: int = 128, width: int = 128) -> np.ndarray:
    m = np.zeros((height, width), bool)
    m[60:66, 5:120] = True
    m[5:120, 60:66] = True
    return m


def road_cross(size: int = 256) -> tuple[np.ndarray, nx.MultiGraph]:
    """A wide cross and the graph it was drawn from, for the lidar synthesizer.

    Larger than :func:`road_mask` because the correlated fields have feature
    scales of 20-24 px, and a tile only a few features across gives statistics
    too coarse to assert on.
    """
    mask = np.zeros((size, size), bool)
    middle = size // 2
    mask[middle - 3 : middle + 3, :] = True
    mask[:, middle - 3 : middle + 3] = True

    edge, mid = float(size - 1), float(middle)
    truth = geograph.build(
        [
            np.array([[0.0, mid], [edge, mid]]),
            np.array([[mid, 0.0], [mid, edge]]),
        ]
    )
    return mask, truth


def best_separation(channel: np.ndarray, mask: np.ndarray) -> float:
    """Best road-vs-background separation any single threshold on it achieves.

    Youden's J, taken over both polarities, so a channel that is dark on road
    scores the same as one that is bright on it. 1 is a perfect predictor.
    """
    thresholds = np.linspace(channel.min(), channel.max(), 128)
    return max(
        max(
            (channel > t)[mask].mean() - (channel > t)[~mask].mean(),
            (channel < t)[mask].mean() - (channel < t)[~mask].mean(),
        )
        for t in thresholds
    )


def sample_with(mask: np.ndarray) -> data.TileSample:
    rng = np.random.default_rng(0)
    tile = tiles.tile_from_center(36.1699, -115.1398, size_m=float(mask.shape[0]))
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


def test_canopy_patches_cover_the_requested_share():
    canopy = data.canopy_patches((256, 256), np.random.default_rng(0), fraction=0.2)
    assert canopy.mean() == pytest.approx(0.2, abs=0.01)


def test_canopy_patches_are_contiguous_blobs_not_speckle():
    """Per-pixel dropout would occlude nothing an image model cannot infill."""
    canopy = data.canopy_patches((256, 256), np.random.default_rng(0), fraction=0.15)
    _, n_blobs = ndimage.label(canopy)  # pyright: ignore[reportGeneralTypeIssues]
    assert 0 < n_blobs < 60


def test_canopy_of_zero_fraction_is_bare_ground():
    assert not data.canopy_patches((64, 64), np.random.default_rng(0), 0.0).any()


def test_synthetic_height_has_expected_shapes_and_ranges():
    mask, truth = road_cross()
    ndsm, intensity = data.synthesize_height(mask, truth, np.random.default_rng(0))

    assert ndsm.shape == mask.shape
    assert intensity.shape == mask.shape
    assert ndsm.dtype == np.float32
    assert intensity.dtype == np.float32
    assert ndsm.min() >= 0.0
    assert intensity.min() >= 0.0 and intensity.max() <= 1.0


def test_synthetic_height_is_reproducible_from_its_seed():
    mask, truth = road_cross()
    a = data.synthesize_height(mask, truth, np.random.default_rng(7))
    b = data.synthesize_height(mask, truth, np.random.default_rng(7))
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_canopy_occludes_the_road_in_the_image():
    """Fusion can only gain what the imagery has lost; here is the loss.

    Measured on the canopy interior rather than the whole patch: the blur that
    softens a blob's outline also leaves the road partly visible at its rim,
    and that rim is not where the claim lives.
    """
    mask, _ = road_cross()
    rng = np.random.default_rng(0)
    canopy = data.canopy_patches(mask.shape, rng, fraction=0.25)
    grey = data.synthesize_image(mask, rng, occlusion=canopy).mean(axis=-1)
    interior = ndimage.binary_erosion(canopy, iterations=3)

    open_ground = grey[mask & ~canopy].mean() - grey[~mask & ~canopy].mean()
    under_canopy = grey[mask & interior].mean() - grey[~mask & interior].mean()
    assert open_ground > 0.25
    assert under_canopy < 0.15 * open_ground


def test_the_ground_return_survives_the_canopy_that_hid_the_road():
    """The mechanism the whole experiment rests on: lidar sees under trees."""
    mask, truth = road_cross()
    rng = np.random.default_rng(0)
    canopy = data.canopy_patches(mask.shape, rng, fraction=0.25)
    _, intensity = data.synthesize_height(mask, truth, rng, canopy=canopy)
    interior = ndimage.binary_erosion(canopy, iterations=3)

    assert intensity[mask & interior].mean() < intensity[~mask & interior].mean() - 0.15


def test_lidar_alone_is_not_a_near_perfect_predictor_of_the_label():
    """The sanity guard, to be checked before any coverage curve is read.

    If height were a clean function of the mask the model would read the answer
    off it, every coverage sweep would come out trivially monotone, and nothing
    measured on this data would mean anything.
    """
    mask, truth = road_cross()
    rng = np.random.default_rng(0)
    canopy = data.canopy_patches(mask.shape, rng, fraction=0.15)
    channels = data.synthesize_height(mask, truth, rng, canopy=canopy)

    assert all(best_separation(channel, mask) < 0.8 for channel in channels)
    # And it must not be useless either, or fusion has nothing to fuse.
    assert max(best_separation(channel, mask) for channel in channels) > 0.2


def test_overpass_lifts_one_deck_where_edges_cross_without_meeting():
    """A grade separation is the one place a 2D mask is provably wrong."""
    size = 128
    mask = np.zeros((size, size), bool)
    mask[60:68, :] = True
    mask[:, 60:68] = True
    crossing = geograph.build(
        [
            np.array([[0.0, 64.0], [127.0, 64.0]]),
            np.array([[64.0, 0.0], [64.0, 127.0]]),
        ]
    )
    # The same geometry, noded at the crossing: edges that meet are at grade.
    noded = geograph.build(
        [
            np.array([[0.0, 64.0], [64.0, 64.0]]),
            np.array([[64.0, 64.0], [127.0, 64.0]]),
            np.array([[64.0, 0.0], [64.0, 64.0]]),
            np.array([[64.0, 64.0], [64.0, 127.0]]),
        ]
    )

    lifted, _ = data.synthesize_height(mask, crossing, np.random.default_rng(0))
    flat, _ = data.synthesize_height(mask, noded, np.random.default_rng(0))
    assert lifted[60:68, 60:68].max() > flat[60:68, 60:68].max() + 3.0


@pytest.mark.parametrize("mode", ["full", "none", "strips", "blocks"])
@pytest.mark.parametrize("target", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_coverage_hits_its_target_fraction(mode, target):
    """Endpoints are exact by construction; the patterns quantize to the grid."""
    config = data.CoverageSampler(mode=mode)
    covered = [
        data.sample_coverage(
            config, (256, 256), np.random.default_rng(seed), fraction=target
        ).mean()
        for seed in range(8)
    ]

    expected = {"full": 1.0, "none": 0.0}.get(mode, target)
    assert all(share == pytest.approx(expected, abs=0.05) for share in covered)


def test_the_endpoints_are_exactly_one_and_zero():
    """0.999 is not the imagery-only case, and these two are what get reported."""
    full = data.sample_coverage(
        data.CoverageSampler(mode="full"), (64, 64), np.random.default_rng(0)
    )
    none = data.sample_coverage(
        data.CoverageSampler(mode="none"), (64, 64), np.random.default_rng(0)
    )
    assert full.all() and full.mean() == 1.0
    assert not none.any() and none.mean() == 0.0


def test_coverage_masks_are_boolean_at_crop_resolution():
    config = data.CoverageSampler(mode="strips")
    covered = data.sample_coverage(config, (48, 80), np.random.default_rng(0))
    assert covered.shape == (48, 80)
    assert covered.dtype == np.bool_


def test_strips_are_bands_rather_than_speckle():
    """Per-pixel dropout would train an inpainter, which is not the deployment."""
    config = data.CoverageSampler(mode="strips")
    covered = data.sample_coverage(
        config, (256, 256), np.random.default_rng(0), fraction=0.4
    )
    _, n_bands = ndimage.label(covered)  # pyright: ignore[reportGeneralTypeIssues]
    assert 0 < n_bands < 20


def test_the_curriculum_puts_explicit_mass_on_both_endpoints():
    """A uniform draw would make the two reported numbers vanishingly rare."""
    config = data.CoverageSampler(mode="strips", endpoint_mass=0.4)
    rng = np.random.default_rng(0)
    drawn = np.array([data.coverage_fraction(config, rng) for _ in range(4000)])

    assert (drawn == 0.0).mean() == pytest.approx(0.2, abs=0.03)
    assert (drawn == 1.0).mean() == pytest.approx(0.2, abs=0.03)
    assert ((drawn > 0.0) & (drawn < 1.0)).mean() == pytest.approx(0.6, abs=0.04)


def test_the_endpoint_modes_ignore_the_curriculum():
    rng = np.random.default_rng(0)
    assert data.coverage_fraction(data.CoverageSampler(mode="full"), rng) == 1.0
    assert data.coverage_fraction(data.CoverageSampler(mode="none"), rng) == 0.0


def test_coverage_sampler_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="unknown coverage mode"):
        data.CoverageSampler(mode="swathe")  # pyright: ignore[reportArgumentType]


def test_coverage_sampler_rejects_inverted_fraction_bounds():
    with pytest.raises(ValueError, match="min_fraction"):
        data.CoverageSampler(mode="strips", min_fraction=0.8, max_fraction=0.2)


def aux_crop(size: int = 16, valid: np.ndarray | None = None) -> data.Crop:
    """A crop whose lidar values are all distinct, so any leak is visible."""
    values = np.arange(size * size * 2, dtype=np.float32).reshape(size, size, 2)
    return data.Crop(
        image=np.zeros((size, size, 3), np.float32),
        mask=np.zeros((size, size), bool),
        aux=values,
        aux_valid=np.ones((size, size), bool) if valid is None else valid,
    )


def test_aux_stack_appends_the_availability_indicator():
    stack = data.aux_stack(aux_crop())
    assert stack.shape == (16, 16, 3)
    assert stack.dtype == np.float32
    assert (stack[..., -1] == 1.0).all()


def test_aux_stack_writes_the_fill_wherever_nothing_was_observed():
    valid = np.ones((16, 16), bool)
    valid[:8] = False
    stack = data.aux_stack(aux_crop(valid=valid))

    assert (stack[:8, :, :2] == data.AUX_FILL).all()
    assert (stack[:8, :, -1] == 0.0).all()
    assert (stack[8:, :, -1] == 1.0).all()


def test_coverage_and_sensor_validity_intersect_rather_than_override():
    """Two different reasons for a hole, and both have to hold for a pixel."""
    valid = np.zeros((16, 16), bool)
    valid[:, :8] = True
    coverage = np.zeros((16, 16), bool)
    coverage[:8, :] = True
    stack = data.aux_stack(replace(aux_crop(valid=valid), coverage=coverage))

    observed = stack[..., -1] > 0.5
    assert observed[:8, :8].all()
    assert not observed[8:, :].any()
    assert not observed[:, 8:].any()


def test_aux_stack_rescales_each_channel_by_its_own_divisor():
    stack = data.aux_stack(aux_crop(), scales=(2.0, 4.0))
    raw = np.arange(16 * 16 * 2, dtype=np.float32).reshape(16, 16, 2)
    assert np.allclose(stack[..., 0], raw[..., 0] / 2.0)
    assert np.allclose(stack[..., 1], raw[..., 1] / 4.0)


def test_aux_stack_rejects_a_scale_list_of_the_wrong_length():
    with pytest.raises(ValueError, match="aux channels"):
        data.aux_stack(aux_crop(), scales=(1.0,))


def test_aux_stack_refuses_a_crop_with_no_lidar():
    """A source with no lidar is a different case from one that measured none."""
    crop = data.Crop(image=np.zeros((8, 8, 3), np.float32), mask=np.zeros((8, 8), bool))
    with pytest.raises(ValueError, match="no lidar"):
        data.aux_stack(crop)


def test_an_all_false_validity_mask_is_not_the_same_as_no_lidar():
    """The overloaded-boolean rule, at tensor scale.

    A source that flew and measured nothing here still produces a stack, and
    that stack still tells the model *that* nothing was measured. A source with
    no lidar at all produces none, and the network is built without the
    channels rather than fed a plane of zeros it must learn to distrust.
    """
    measured_nothing = aux_crop(valid=np.zeros((16, 16), bool))
    stack = data.aux_stack(measured_nothing)
    assert stack.shape[-1] == 3
    assert (stack[..., -1] == 0.0).all()

    no_lidar = data.Crop(image=measured_nothing.image, mask=measured_nothing.mask)
    assert no_lidar.aux is None
    with pytest.raises(ValueError, match="no lidar"):
        data.aux_stack(no_lidar)


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

    crop = data.take_crop(sample, spec)

    assert crop.image.shape == (32, 32, 3)
    assert crop.mask.shape == (32, 32)
    assert np.array_equal(crop.mask, mask[50:82, 40:72])
    assert crop.aux is None and crop.aux_valid is None


def test_take_crop_carries_the_lidar_stack_through_the_same_window():
    mask = road_mask()
    base = sample_with(mask)
    aux = np.stack(
        [np.arange(mask.size, dtype=np.float32).reshape(mask.shape)] * 2, axis=-1
    )
    sample = replace(base, aux=aux, aux_valid=np.ones(mask.shape, bool))
    spec = data.CropSpec(row=50, col=40, size=32)

    crop = data.take_crop(sample, spec)

    assert crop.aux is not None and crop.aux_valid is not None
    assert crop.aux.shape == (32, 32, 2)
    assert crop.aux_valid.shape == (32, 32)
    assert np.array_equal(crop.aux[..., 0], aux[50:82, 40:72, 0])


def corner_mask(size: int = 32) -> np.ndarray:
    """A mask no dihedral transform maps onto itself.

    A symmetric probe would let a mismatched image/mask pair pass, because both
    orderings of a symmetric shape look the same afterwards.
    """
    m = np.zeros((size, size), bool)
    m[2:6, 2:20] = True
    m[10:14, 2:8] = True
    return m


def ramp(size: int = 4) -> np.ndarray:
    """A crop whose every pixel differs, so any permutation of it shows."""
    return np.arange(size * size, dtype=np.float32).reshape(size, size)


def test_dihedral_index_zero_is_the_identity():
    crop = ramp()
    assert np.array_equal(data.dihedral(crop, 0), crop)


def test_dihedral_has_eight_distinct_elements():
    """Fewer than eight would mean the group is not being covered."""
    crop = ramp()
    turned = [data.dihedral(crop, i) for i in range(data.DIHEDRAL_ORDER)]
    assert len({t.tobytes() for t in turned}) == data.DIHEDRAL_ORDER


def test_dihedral_is_closed_under_composition():
    """Two transforms compose to a third in the group — that is what makes it one."""
    crop = ramp()
    orbit = {data.dihedral(crop, i).tobytes() for i in range(data.DIHEDRAL_ORDER)}
    assert all(
        data.dihedral(data.dihedral(crop, i), j).tobytes() in orbit
        for i in range(data.DIHEDRAL_ORDER)
        for j in range(data.DIHEDRAL_ORDER)
    )


def test_dihedral_preserves_road_pixel_count_exactly():
    """A grid symmetry permutes pixels; an interpolated rotation would not."""
    mask = road_mask()
    assert all(
        data.dihedral(mask, i).sum() == mask.sum() for i in range(data.DIHEDRAL_ORDER)
    )


def test_dihedral_preserves_dtype_and_shape_on_a_square_crop():
    mask = road_mask()
    turned = data.dihedral(mask, 3)
    assert turned.dtype == mask.dtype
    assert turned.shape == mask.shape


def test_dihedral_leaves_the_channel_axis_alone():
    """Rotating an (H, W, C) image must not rotate or reorder the channels."""
    image = data.synthesize_image(corner_mask(), np.random.default_rng(0))

    for index in range(data.DIHEDRAL_ORDER):
        turned = data.dihedral(image, index)
        assert turned.shape == image.shape
        assert all(
            np.array_equal(turned[..., c], data.dihedral(image[..., c], index))
            for c in range(image.shape[-1])
        )


def test_dihedral_returns_a_copy_not_a_view():
    """A view would let a later write reach back into the tile it was cut from."""
    mask = road_mask()
    turned = data.dihedral(mask, 3)
    assert not np.shares_memory(turned, mask)
    assert turned.flags["C_CONTIGUOUS"]


def test_dihedral_rejects_an_index_outside_the_group():
    with pytest.raises(ValueError, match="outside"):
        data.dihedral(road_mask(), data.DIHEDRAL_ORDER)


def test_dihedral_rejects_an_array_with_too_few_axes():
    with pytest.raises(ValueError, match="two axes"):
        data.dihedral(np.zeros(8, np.float32), 0)


def test_augment_crop_moves_image_and_mask_in_lockstep():
    """Different transforms on the two would mislabel every augmented crop.

    The image carries the mask in its channels, so any disagreement between the
    image transform and the mask transform shows up as a mismatch afterwards.
    """
    mask = corner_mask()
    image = np.repeat(mask.astype(np.float32)[..., None], 3, axis=-1)
    crop = data.Crop(image=image, mask=mask)

    moved = [data.augment_crop(crop, i) for i in range(data.DIHEDRAL_ORDER)]

    # The probe only proves anything if the transforms differ on it.
    assert len({c.mask.tobytes() for c in moved}) == data.DIHEDRAL_ORDER
    assert all(np.array_equal(c.image[..., 0].astype(bool), c.mask) for c in moved)


def test_augment_crop_preserves_the_road_pixel_count():
    mask = corner_mask()
    crop = data.Crop(
        image=data.synthesize_image(mask, np.random.default_rng(0)), mask=mask
    )
    assert all(
        data.augment_crop(crop, i).mask.sum() == mask.sum()
        for i in range(data.DIHEDRAL_ORDER)
    )


def test_augment_crop_keeps_the_whole_channel_stack_registered():
    """Imagery, lidar, label and coverage mask must land on one group element.

    Every array carries the same corner probe, so a stack that came apart shows
    up as one of them disagreeing with the others rather than as a quietly
    worse model later.
    """
    mask = corner_mask()
    crop = data.Crop(
        image=np.repeat(mask.astype(np.float32)[..., None], 3, axis=-1),
        mask=mask,
        aux=np.repeat(mask.astype(np.float32)[..., None], 2, axis=-1),
        aux_valid=mask,
        coverage=mask,
    )

    moved = [data.augment_crop(crop, i) for i in range(data.DIHEDRAL_ORDER)]

    assert len({c.mask.tobytes() for c in moved}) == data.DIHEDRAL_ORDER
    for c in moved:
        assert c.aux is not None and c.aux_valid is not None
        assert c.coverage is not None
        assert np.array_equal(c.image[..., 0].astype(bool), c.mask)
        assert np.array_equal(c.aux[..., 0].astype(bool), c.mask)
        assert np.array_equal(c.aux[..., 1].astype(bool), c.mask)
        assert np.array_equal(c.aux_valid, c.mask)
        assert np.array_equal(c.coverage, c.mask)


def test_crop_rejects_a_mask_that_does_not_match_its_image():
    with pytest.raises(ValueError, match="disagree on shape"):
        data.Crop(image=np.zeros((8, 8, 3), np.float32), mask=np.zeros((4, 4), bool))


def test_crop_rejects_aux_without_its_validity_mask():
    with pytest.raises(ValueError, match="supplied together"):
        data.Crop(
            image=np.zeros((8, 8, 3), np.float32),
            mask=np.zeros((8, 8), bool),
            aux=np.zeros((8, 8, 2), np.float32),
        )


def test_crop_rejects_coverage_with_nothing_to_cover():
    """A coverage mask on an imagery-only crop is a wiring mistake, not a no-op."""
    with pytest.raises(ValueError, match="masks nothing"):
        data.Crop(
            image=np.zeros((8, 8, 3), np.float32),
            mask=np.zeros((8, 8), bool),
            coverage=np.ones((8, 8), bool),
        )


def test_registry_yields_a_fresh_instance_each_lookup():
    factory = data.TILE_SOURCE_REGISTRY["synthetic"]
    assert factory() is not factory()


def test_registry_entries_satisfy_the_protocol():
    assert all(
        isinstance(factory(), data.TileSource)
        for factory in data.TILE_SOURCE_REGISTRY.values()
    )


def test_load_sample_rejects_an_unknown_source():
    with pytest.raises(KeyError, match="unknown tile source"):
        data.load_sample("tile_0", source="nope")


def test_synthetic_source_enumerates_its_tiles():
    source = data.SyntheticTileSource()
    assert source.ids() == tuple(f"tile_{i}" for i in range(len(source.centers)))


def test_synthetic_source_rejects_an_unknown_id():
    with pytest.raises(KeyError, match="unknown sample"):
        data.SyntheticTileSource().load("tile_999")


@pytest.mark.network
def test_load_sample_produces_aligned_image_and_labels():
    sample = data.load_sample("tile_0", size_m=512.0)

    assert sample.image.shape[:2] == sample.mask.shape
    assert sample.mask.any()
    assert sample.truth.number_of_edges() > 0


@pytest.mark.network
def test_the_same_tile_yields_the_same_image():
    a = data.load_sample("tile_0", size_m=512.0)
    b = data.load_sample("tile_0", size_m=512.0)
    assert np.array_equal(a.image, b.image)

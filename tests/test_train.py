from dataclasses import replace

import networkx as nx
import numpy as np
import pytest
import torch

from geo_graphs import data, geograph, tiles, train
from geo_graphs.model import FUSION_REGISTRY, UNet

# Small enough to train in a test, large enough to survive three downsamples.
TINY = dict(crop_size=32, widths=(8, 16), device="cpu", lr=1e-2, dice_weight=0.5)


def synthetic_sample(size: int = 128, lidar: bool = False) -> data.TileSample:
    """A tile with a road cross, built without touching the network.

    The truth graph is noded at the centre — four edges meeting at one degree-4
    junction — so it matches what the mask actually depicts rather than two
    lines that merely overlap.

    Args:
        size: Tile side in pixels.
        lidar: Attach a synthetic aux stack, so the fusion path can be
            exercised offline.
    """
    mask = np.zeros((size, size), bool)
    mask[size // 2 - 3 : size // 2 + 3, :] = True
    mask[:, size // 2 - 3 : size // 2 + 3] = True

    mid = float(size // 2)
    edge = float(size - 1)
    truth = geograph.build(
        [
            np.array([[0.0, mid], [mid, mid]]),
            np.array([[mid, mid], [edge, mid]]),
            np.array([[mid, 0.0], [mid, mid]]),
            np.array([[mid, mid], [mid, edge]]),
        ]
    )

    tile = tiles.tile_from_center(36.1699, -115.1398, size_m=float(size))
    rng = np.random.default_rng(0)
    canopy = data.canopy_patches(mask.shape, rng, fraction=0.15)
    image = data.synthesize_image(mask, rng, occlusion=canopy)
    sample = data.TileSample(tile=tile, image=image, mask=mask, truth=truth)
    if not lidar:
        return sample

    ndsm, intensity = data.synthesize_height(mask, truth, rng, canopy=canopy)
    return replace(
        sample,
        aux=np.stack([ndsm, intensity], axis=-1),
        aux_valid=np.ones(mask.shape, bool),
    )


class _FixedSource:
    """A TileSource over samples already in memory, so tests need no imagery."""

    def __init__(self, samples):
        self.samples = list(samples)

    def ids(self) -> tuple[str, ...]:
        return tuple(f"s{i}" for i in range(len(self.samples)))

    def load(self, sample_id: str) -> data.TileSample:
        return self.samples[int(sample_id.removeprefix("s"))]


_: type[data.TileSource] = _FixedSource


def tiny_dataset(
    n_crops: int = 8, size: int = 32, augment: bool = False, seed: int = 0
) -> train.CropDataset:
    sample = synthetic_sample()
    specs = data.crop_specs(sample.mask, size, n_crops, np.random.default_rng(0))
    return train.CropDataset(
        [sample],
        [(0, spec) for spec in specs],
        augment=augment,
        rng=np.random.default_rng(seed),
    )


def dihedral_chw(chw: np.ndarray, index: int) -> np.ndarray:
    """Transform a ``(C, H, W)`` tensor's spatial axes, leaving channels put."""
    return data.dihedral(chw.transpose(1, 2, 0), index).transpose(2, 0, 1)


def test_crop_dataset_yields_channel_first_float_tensors():
    dataset = tiny_dataset()
    image, mask = dataset[0]

    assert image.shape == (3, 32, 32)
    assert mask.shape == (1, 32, 32)
    assert image.dtype is torch.float32
    assert mask.dtype is torch.float32
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_crop_dataset_length_matches_its_specs():
    assert len(tiny_dataset(n_crops=5)) == 5


def test_augmentation_off_reproduces_the_raw_crop():
    """The default path must hand back exactly the tensors it did before."""
    dataset = tiny_dataset(4)
    sample_index, spec = dataset.specs[0]
    crop = data.take_crop(dataset.samples[sample_index], spec)

    got_image, got_mask = dataset[0]
    assert np.array_equal(got_image.numpy(), crop.image.transpose(2, 0, 1))
    assert np.array_equal(got_mask.numpy()[0].astype(bool), crop.mask)


def test_augmentation_off_repeats_itself_across_epochs():
    dataset = tiny_dataset(4)
    first = [dataset[i][0].clone() for i in range(len(dataset))]
    assert all(torch.equal(first[i], dataset[i][0]) for i in range(len(dataset)))


def test_augmentation_varies_the_same_crop_across_epochs():
    """A transform fixed per index is a 1x dataset dressed up as an 8x one."""
    dataset = tiny_dataset(8, augment=True)
    first = [dataset[i][0].clone() for i in range(len(dataset))]
    assert any(not torch.equal(first[i], dataset[i][0]) for i in range(len(dataset)))


def test_augmented_crops_are_dihedral_transforms_of_the_originals():
    """Image and mask must land on the *same* group element, not merely on one.

    Matching them separately would let a rotated image train against an
    unrotated label, which looks like a model problem rather than a data one.
    """
    plain, augmented = tiny_dataset(8), tiny_dataset(8, augment=True)

    for i in range(len(plain)):
        image, mask = (t.numpy() for t in plain[i])
        got_image, got_mask = (t.numpy() for t in augmented[i])

        matches = [
            k
            for k in range(data.DIHEDRAL_ORDER)
            if np.array_equal(dihedral_chw(image, k), got_image)
        ]
        assert matches, "augmented image is not a dihedral transform of the crop"
        assert any(np.array_equal(dihedral_chw(mask, k), got_mask) for k in matches)


def test_augmentation_preserves_the_road_pixel_count_of_every_crop():
    """Reflection and rotation on a grid are measure-preserving; warps are not."""
    plain, augmented = tiny_dataset(8), tiny_dataset(8, augment=True)
    assert [float(plain[i][1].sum()) for i in range(len(plain))] == [
        float(augmented[i][1].sum()) for i in range(len(augmented))
    ]


def test_augmentation_is_reproducible_from_its_seed():
    a, b = tiny_dataset(8, augment=True, seed=5), tiny_dataset(8, augment=True, seed=5)
    assert all(torch.equal(a[i][0], b[i][0]) for i in range(len(a)))


def test_a_different_seed_draws_a_different_sequence():
    a, b = tiny_dataset(8, augment=True, seed=0), tiny_dataset(8, augment=True, seed=1)
    assert any(not torch.equal(a[i][0], b[i][0]) for i in range(len(a)))


def test_augmentation_without_a_generator_is_refused():
    """An implicit default would make the run unreplayable from its seed."""
    sample = synthetic_sample()
    specs = data.crop_specs(sample.mask, 32, 2, np.random.default_rng(0))
    with pytest.raises(ValueError, match="rng"):
        train.CropDataset([sample], [(0, s) for s in specs], augment=True)


def build_split_datasets(augment: bool) -> tuple[train.CropDataset, train.CropDataset]:
    source = _FixedSource([synthetic_sample(size=128), synthetic_sample(size=128)])
    config = train.TrainConfig(
        **TINY, train_ids=("s0",), val_ids=("s1",), augment=augment, seed=0
    )
    return (
        train.build_dataset(source, config.train_ids, 4, config, 0),
        train.build_dataset(source, config.val_ids, 4, config, 1000),
    )


def test_build_dataset_augments_the_training_split_only():
    """Augmented validation crops make the epoch-to-epoch number incomparable."""
    train_set, val_set = build_split_datasets(augment=True)
    assert train_set.augment
    assert not val_set.augment


def test_build_dataset_leaves_augmentation_off_when_the_config_does():
    train_set, val_set = build_split_datasets(augment=False)
    assert not train_set.augment
    assert not val_set.augment


def lidar_dataset(
    n_crops: int = 8,
    size: int = 32,
    coverage: str = "full",
    augment: bool = False,
    seed: int = 0,
) -> train.CropDataset:
    sample = synthetic_sample(lidar=True)
    specs = data.crop_specs(sample.mask, size, n_crops, np.random.default_rng(0))
    return train.CropDataset(
        [sample],
        [(0, spec) for spec in specs],
        augment=augment,
        rng=np.random.default_rng(seed),
        coverage=data.CoverageSampler(mode=coverage),  # pyright: ignore[reportArgumentType]
        coverage_seed=seed,
    )


def test_a_dataset_without_lidar_reports_no_aux_channels():
    assert tiny_dataset().aux_channels == 0
    assert len(tiny_dataset()[0]) == 2


def test_a_lidar_dataset_yields_the_stack_plus_its_indicator():
    dataset = lidar_dataset()
    assert dataset.aux_channels == len(data.AUX_CHANNEL_NAMES) + 1

    image, mask, aux = dataset[0]
    assert image.shape == (3, 32, 32)
    assert mask.shape == (1, 32, 32)
    assert aux.shape == (dataset.aux_channels, 32, 32)
    assert aux.dtype is torch.float32


def test_full_coverage_marks_every_pixel_observed():
    _, _, aux = lidar_dataset(coverage="full")[0]
    assert (aux[-1] == 1.0).all()


def test_no_coverage_marks_every_pixel_unobserved_and_fills_the_data():
    """The zero-coverage endpoint, and the imagery-only deployment case."""
    _, _, aux = lidar_dataset(coverage="none")[0]
    assert (aux[-1] == 0.0).all()
    assert (aux[:-1] == data.AUX_FILL).all()


def test_coverage_is_fixed_per_crop_across_epochs():
    """A resampled validation coverage compares two datasets, not two epochs."""
    dataset = lidar_dataset(coverage="strips")
    first = [dataset[i][2].clone() for i in range(len(dataset))]
    assert all(torch.equal(first[i], dataset[i][2]) for i in range(len(dataset)))


def test_augmentation_moves_the_aux_stack_with_the_image():
    """Lidar rotated away from its imagery is worse than no lidar at all."""
    plain, augmented = lidar_dataset(8), lidar_dataset(8, augment=True)

    for i in range(len(plain)):
        image, _, aux = (t.numpy() for t in plain[i])
        got_image, _, got_aux = (t.numpy() for t in augmented[i])

        matches = [
            k
            for k in range(data.DIHEDRAL_ORDER)
            if np.array_equal(dihedral_chw(image, k), got_image)
        ]
        assert matches, "augmented image is not a dihedral transform of the crop"
        assert any(np.array_equal(dihedral_chw(aux, k), got_aux) for k in matches)


def test_train_builds_a_fused_network_when_the_source_supplies_lidar():
    config = train.TrainConfig(**TINY, epochs=1, batch_size=4, seed=0)
    result = train.train(config, datasets=(lidar_dataset(8), lidar_dataset(4)))

    assert result.model.aux_channels == len(data.AUX_CHANNEL_NAMES) + 1
    assert isinstance(result.model.fusion, type(FUSION_REGISTRY["early"]()))


@pytest.mark.parametrize("coverage", ["full", "none"])
def test_a_run_launches_at_either_endpoint(coverage):
    """Tier one's whole exit condition: both endpoints run end to end."""
    config = train.TrainConfig(
        **TINY,
        epochs=2,
        batch_size=4,
        seed=0,
        coverage=data.CoverageSampler(mode=coverage),
    )
    result = train.train(
        config,
        datasets=(
            lidar_dataset(8, coverage=coverage),
            lidar_dataset(4, coverage=coverage),
        ),
    )

    assert len(result.history) == 2
    assert all(np.isfinite(m.train_loss) for m in result.history)


def test_train_refuses_splits_that_disagree_about_lidar():
    config = train.TrainConfig(**TINY, epochs=1, batch_size=4, seed=0)
    with pytest.raises(ValueError, match="aux channels"):
        train.train(config, datasets=(lidar_dataset(4), tiny_dataset(4)))


def test_a_fused_checkpoint_round_trips(tmp_path):
    """The fusion key and stack width have to survive, or the weights will not."""
    config = train.TrainConfig(**TINY, epochs=1, batch_size=4, seed=0)
    path = tmp_path / "fused.pt"
    result = train.train(
        config, checkpoint=path, datasets=(lidar_dataset(4), lidar_dataset(4))
    )

    restored = train.load_checkpoint(path)
    assert restored.aux_channels == result.model.aux_channels

    image = torch.randn(1, 3, 32, 32)
    aux = torch.randn(1, restored.aux_channels, 32, 32)
    with torch.no_grad():
        assert torch.allclose(result.model.eval()(image, aux), restored(image, aux))


def test_predict_tile_logits_feeds_the_aux_stack_through():
    sample = synthetic_sample(size=128, lidar=True)
    model = UNet(
        in_channels=3,
        widths=(8, 16),
        aux_channels=len(data.AUX_CHANNEL_NAMES) + 1,
        fusion=FUSION_REGISTRY["early"](),
    )

    logits = train.predict_tile_logits(model, sample)

    assert logits.shape == sample.mask.shape
    assert np.isfinite(logits).all()


def test_scoring_coverage_changes_what_a_fused_model_sees():
    """Held-out coverage is a condition of the measurement, so it must bite."""
    sample = synthetic_sample(size=128, lidar=True)
    model = UNet(
        in_channels=3,
        widths=(8, 16),
        aux_channels=len(data.AUX_CHANNEL_NAMES) + 1,
        fusion=FUSION_REGISTRY["early"](),
    ).eval()

    full = train.predict_tile_logits(model, sample, coverage=data.FULL_COVERAGE)
    none = train.predict_tile_logits(
        model, sample, coverage=data.CoverageSampler(mode="none")
    )
    assert not np.allclose(full, none)


def test_predict_tile_logits_refuses_a_sample_with_no_lidar():
    model = UNet(
        in_channels=3,
        widths=(8, 16),
        aux_channels=len(data.AUX_CHANNEL_NAMES) + 1,
        fusion=FUSION_REGISTRY["early"](),
    )
    with pytest.raises(ValueError, match="carries none"):
        train.predict_tile_logits(model, synthetic_sample(size=128))


def test_resolve_device_honours_an_explicit_request():
    assert train.resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_returns_something_usable():
    torch.zeros(1, device=train.resolve_device("auto"))


def test_overfit_one_batch_drives_the_loss_down():
    """The single most useful diagnostic: can it memorize a handful of crops?

    If this fails the fault is the model or the loss, not the data or the
    schedule — which is exactly what makes it worth asserting.
    """
    config = train.TrainConfig(**TINY, seed=0)
    losses = train.overfit_one_batch(
        config, steps=250, batch_size=4, dataset=tiny_dataset(n_crops=4)
    )

    assert len(losses) == 250
    assert losses[-1] < 0.05
    assert losses[-1] < losses[0] / 10


def test_train_returns_history_for_every_epoch():
    config = train.TrainConfig(**TINY, epochs=2, batch_size=4, seed=0)
    result = train.train(config, datasets=(tiny_dataset(8), tiny_dataset(4)))

    assert [m.epoch for m in result.history] == [0, 1]
    assert all(0.0 <= m.val_iou <= 1.0 for m in result.history)
    assert all(np.isfinite(m.train_loss) for m in result.history)


def test_training_reduces_the_loss():
    config = train.TrainConfig(**TINY, epochs=6, batch_size=4, seed=0)
    history = train.train(config, datasets=(tiny_dataset(8), tiny_dataset(4))).history
    assert history[-1].train_loss < history[0].train_loss


def test_checkpoint_round_trips(tmp_path):
    config = train.TrainConfig(**TINY, epochs=1, batch_size=4, seed=0)
    path = tmp_path / "model.pt"
    result = train.train(
        config, checkpoint=path, datasets=(tiny_dataset(4), tiny_dataset(4))
    )

    restored = train.load_checkpoint(path)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(result.model.eval()(x), restored(x))


def test_checkpoint_survives_a_run_that_never_finishes(tmp_path, monkeypatch):
    """A killed run should still leave the best epoch so far on disk.

    These runs are long enough that losing one to an OOM is a real cost, and
    the end-of-run write alone leaves nothing behind.
    """
    config = train.TrainConfig(**TINY, epochs=10, batch_size=4, seed=0)
    path = tmp_path / "model.pt"

    real_epoch, calls = train._run_epoch, {"n": 0}

    def die_partway(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 6:
            raise KeyboardInterrupt("pretend the OOM killer arrived")
        return real_epoch(*args, **kwargs)

    monkeypatch.setattr(train, "_run_epoch", die_partway)
    with pytest.raises(KeyboardInterrupt):
        train.train(config, checkpoint=path, datasets=(tiny_dataset(4), tiny_dataset(4)))

    assert path.exists(), "no checkpoint survived the interrupted run"
    assert train.load_checkpoint(path) is not None


def test_predict_tile_logits_covers_the_whole_tile():
    sample = synthetic_sample(size=128)
    model = UNet(in_channels=3, widths=(8, 16))

    logits = train.predict_tile_logits(model, sample)

    assert logits.shape == sample.mask.shape
    assert np.isfinite(logits).all()


def test_predict_tile_logits_handles_an_awkward_tile_size():
    """Real chips are not multiples of the downsampling factor.

    SpaceNet tiles land on sizes like 396x323, so inference pads up and crops
    back rather than refusing. The output must still match the input exactly.
    """
    sample = synthetic_sample(size=100)
    model = UNet(in_channels=3, widths=(8, 16, 32, 64))

    logits = train.predict_tile_logits(model, sample)

    assert logits.shape == sample.mask.shape
    assert np.isfinite(logits).all()


def test_predict_tile_logits_is_unaffected_by_padding_on_exact_sizes():
    sample = synthetic_sample(size=128)
    model = UNet(in_channels=3, widths=(8, 16)).eval()
    assert train.predict_tile_logits(model, sample).shape == (128, 128)


def test_evaluate_tile_separates_the_stages():
    """A single end-to-end score cannot tell a bad mask from a bad cleanup."""
    sample = synthetic_sample(size=128)
    model = UNet(in_channels=3, widths=(8, 16))

    report = train.evaluate_tile(model, sample)

    assert 0.0 <= report.mask_iou <= 1.0
    for score in (report.apls_raw, report.apls_cleaned, report.ceiling_apls):
        assert 0.0 <= score <= 1.0


def test_evaluate_tile_reports_ceiling_relative_score():
    """Model quality is meaningless without the ceiling for that same tile."""
    sample = synthetic_sample(size=128)
    report = train.evaluate_tile(UNet(in_channels=3, widths=(8, 16)), sample)

    if report.ceiling_apls:
        assert report.fraction_of_ceiling == pytest.approx(
            report.apls_cleaned / report.ceiling_apls
        )


def test_evaluate_tiles_records_which_tile_each_report_came_from():
    """Without the id a per-tile list cannot be traced back to a chip."""
    source = _FixedSource([synthetic_sample(size=128) for _ in range(3)])
    report = train.evaluate_tiles(
        UNet(in_channels=3, widths=(8, 16)), source, source.ids()
    )
    assert [r.sample_id for r in report.per_tile] == list(source.ids())


def test_evaluate_tiles_aggregates_across_a_set():
    samples = [synthetic_sample(size=128) for _ in range(3)]
    source = _FixedSource(samples)
    model = UNet(in_channels=3, widths=(8, 16))

    report = train.evaluate_tiles(model, source, source.ids())

    assert report.n_scored == 3
    assert len(report.per_tile) == 3
    assert report.mask_iou == pytest.approx(
        float(np.mean([r.mask_iou for r in report.per_tile]))
    )


def test_evaluate_tiles_skips_tiles_with_no_roads():
    """SpaceNet ships chips with no labelled road; they score degenerately."""
    empty = synthetic_sample(size=128)
    blank = data.TileSample(
        tile=empty.tile, image=empty.image, mask=empty.mask, truth=nx.MultiGraph()
    )
    source = _FixedSource([synthetic_sample(size=128), blank])

    report = train.evaluate_tiles(
        UNet(in_channels=3, widths=(8, 16)), source, source.ids()
    )

    assert report.n_scored == 1
    assert report.n_skipped == 1


def test_evaluate_tiles_on_an_empty_set_is_not_an_error():
    source = _FixedSource([])
    report = train.evaluate_tiles(UNet(in_channels=3, widths=(8, 16)), source, ())
    assert report.n_scored == 0
    assert report.per_tile == ()


def placed_source(n: int = 6, pitch: float = 2000.0) -> _FixedSource:
    """Samples spread along easting, so a spatial split has somewhere to cut."""
    sample = synthetic_sample(size=64)
    return _FixedSource(
        replace(sample, tile=replace(sample.tile, x_min=sample.tile.x_min + i * pitch))
        for i in range(n)
    )


def test_load_checkpoint_places_the_model_on_the_requested_device(tmp_path):
    """The device argument has to move the module, not only map its storage.

    A module built on the CPU and left there fails inside the first convolution
    when it meets a tensor on an accelerator, which reads as a dtype bug rather
    than a placement one.
    """
    path = tmp_path / "model.pt"
    model = train.UNet(in_channels=3, widths=(4, 8))
    torch.save({"widths": [4, 8], "state_dict": model.state_dict()}, path)

    loaded = train.load_checkpoint(path, device="cpu")
    assert not loaded.training
    assert all(p.device.type == "cpu" for p in loaded.parameters())

    if torch.backends.mps.is_available():
        on_mps = train.load_checkpoint(path, device="mps")
        assert all(p.device.type == "mps" for p in on_mps.parameters())


def test_assign_split_dispatches_through_the_registry():
    source = placed_source()
    assignment = train.assign_split(
        source, "random", val_fraction=0.34, seed=0, block_m=1.0, buffer_m=0.0
    )

    assert len(assignment.val) == 2
    assert set(assignment.train) | set(assignment.val) == set(source.ids())
    assert assignment.dropped == ()


def test_assign_split_loads_samples_only_for_a_spatial_split():
    """`random` ignores position, so it must not pay to find out where anything is."""
    source = placed_source()
    loaded: list[str] = []
    original = source.load
    source.load = lambda i: (loaded.append(i), original(i))[1]

    train.assign_split(source, "random", 0.34, 0, 1.0, 0.0)
    assert loaded == []

    train.assign_split(source, "blocked", 0.34, 0, 1000.0, 0.0)
    assert sorted(loaded) == sorted(source.ids())


def test_assign_split_drops_a_buffer_only_when_asked():
    source = placed_source()
    blocked = train.assign_split(source, "blocked", 0.34, 0, 1000.0, 3000.0)
    buffered = train.assign_split(source, "buffered", 0.34, 0, 1000.0, 3000.0)

    assert blocked.dropped == ()
    assert buffered.dropped != ()
    assert buffered.val == blocked.val


def _decreasing_then_rising(config, monkeypatch, losses):
    """Drive train() with a scripted validation curve."""
    calls = iter(losses)

    # Tolerant of extra loss-blend parameters: the scripted curve is the point,
    # and a fake pinned to the exact signature breaks on every loss term added.
    def fake_epoch(model, loader, device, dice_weight=0.0, optimizer=None, **_):
        if optimizer is not None:
            return 0.1, 0.5
        return next(calls), 0.5

    monkeypatch.setattr(train, "_run_epoch", fake_epoch)
    return train.train(config, datasets=(tiny_dataset(4), tiny_dataset(4)))


def test_returns_the_best_epoch_not_the_last(monkeypatch):
    """Validation loss bottoms before the epoch budget does.

    Returning the final weights hands back a measurably worse model and makes
    every later comparison lie about what changed.
    """
    config = train.TrainConfig(**TINY, epochs=5, patience=None, seed=0)
    result = _decreasing_then_rising(config, monkeypatch, [0.5, 0.3, 0.2, 0.4, 0.6])
    assert result.best_epoch == 2
    assert not result.stopped_early


def test_early_stopping_fires_after_patience_epochs(monkeypatch):
    config = train.TrainConfig(**TINY, epochs=20, patience=2, seed=0)
    result = _decreasing_then_rising(config, monkeypatch, [0.5, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert result.best_epoch == 1
    assert result.stopped_early
    assert len(result.history) == 4  # improved, then two stale epochs


def test_patience_none_runs_every_epoch(monkeypatch):
    config = train.TrainConfig(**TINY, epochs=4, patience=None, seed=0)
    result = _decreasing_then_rising(config, monkeypatch, [0.5, 0.6, 0.7, 0.8])
    assert len(result.history) == 4
    assert not result.stopped_early


def test_selecting_on_iou_prefers_higher(monkeypatch):
    """Loss is minimized, IoU maximized; the direction must follow the choice."""
    ious = iter([0.2, 0.9, 0.4])

    def fake_epoch(model, loader, device, dice_weight=0.0, optimizer=None, **_):
        return (0.1, 0.5) if optimizer is not None else (0.1, next(ious))

    monkeypatch.setattr(train, "_run_epoch", fake_epoch)
    config = train.TrainConfig(**TINY, epochs=3, patience=None, select_on="val_iou")
    assert (
        train.train(config, datasets=(tiny_dataset(4), tiny_dataset(4))).best_epoch == 1
    )


def test_selecting_on_apls_without_tiles_is_an_error():
    config = train.TrainConfig(**TINY, epochs=1, select_on="val_apls", apls_eval_tiles=0)
    with pytest.raises(ValueError, match="apls_eval_tiles"):
        train.train(config, datasets=(tiny_dataset(4), tiny_dataset(4)))


def test_restored_weights_are_the_ones_checkpointed(tmp_path, monkeypatch):
    config = train.TrainConfig(**TINY, epochs=3, patience=None, seed=0)
    path = tmp_path / "best.pt"
    calls = iter([0.5, 0.2, 0.9])

    def fake_epoch(model, loader, device, dice_weight=0.0, optimizer=None, **_):
        return (0.1, 0.5) if optimizer is not None else (next(calls), 0.5)

    monkeypatch.setattr(train, "_run_epoch", fake_epoch)
    result = train.train(
        config, checkpoint=path, datasets=(tiny_dataset(4), tiny_dataset(4))
    )

    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(result.model.eval()(x), train.load_checkpoint(path)(x))

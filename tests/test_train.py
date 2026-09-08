import networkx as nx
import numpy as np
import pytest
import torch

from geo_graphs import data, geograph, tiles, train
from geo_graphs.model import UNet

# Small enough to train in a test, large enough to survive three downsamples.
TINY = dict(crop_size=32, widths=(8, 16), device="cpu", lr=1e-2, dice_weight=0.5)


def synthetic_sample(size: int = 128) -> data.TileSample:
    """A tile with a road cross, built without touching the network.

    The truth graph is noded at the centre — four edges meeting at one degree-4
    junction — so it matches what the mask actually depicts rather than two
    lines that merely overlap.
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
    image = data.synthesize_image(mask, np.random.default_rng(0))
    return data.TileSample(tile=tile, image=image, mask=mask, truth=truth)


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
    image, mask = data.take_crop(dataset.samples[sample_index], spec)

    got_image, got_mask = dataset[0]
    assert np.array_equal(got_image.numpy(), image.transpose(2, 0, 1))
    assert np.array_equal(got_mask.numpy()[0].astype(bool), mask)


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


def test_split_ids_holds_out_the_requested_fraction():
    ids = tuple(f"img{i}" for i in range(100))
    train_ids, val_ids = train.split_ids(ids, val_fraction=0.2, seed=0)

    assert len(val_ids) == 20
    assert len(train_ids) == 80
    assert not set(train_ids) & set(val_ids)
    assert set(train_ids) | set(val_ids) == set(ids)


def test_split_ids_shuffles_rather_than_taking_a_contiguous_tail():
    """SpaceNet chip numbers run along the ground.

    A contiguous tail would be one neighbourhood held out, not a sample of the
    city, and would flatter or punish the model depending on what is there.
    """
    ids = tuple(f"img{i}" for i in range(100))
    _, val_ids = train.split_ids(ids, val_fraction=0.2, seed=0)
    assert val_ids != ids[:20]


def test_split_ids_is_reproducible_and_seed_dependent():
    ids = tuple(f"img{i}" for i in range(100))
    assert train.split_ids(ids, 0.2, seed=0) == train.split_ids(ids, 0.2, seed=0)
    assert train.split_ids(ids, 0.2, seed=1) != train.split_ids(ids, 0.2, seed=0)


def test_split_ids_always_holds_out_at_least_one():
    train_ids, val_ids = train.split_ids(("a", "b", "c"), val_fraction=0.01, seed=0)
    assert len(val_ids) == 1
    assert len(train_ids) == 2


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

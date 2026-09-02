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


def tiny_dataset(n_crops: int = 8, size: int = 32) -> train.CropDataset:
    sample = synthetic_sample()
    specs = data.crop_specs(sample.mask, size, n_crops, np.random.default_rng(0))
    return train.CropDataset([sample], [(0, spec) for spec in specs])


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

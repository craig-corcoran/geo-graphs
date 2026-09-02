"""Training loop for the road segmentation model.

Written against synthetic imagery from :mod:`data`, so the loop, the loss and
the evaluation path can be debugged before any imagery is downloaded. Swapping
in a real tile source changes one config key and nothing else.

**Numbers from a synthetic run are not results.** The image is derived from the
label, so the model fits it easily; what a run proves is that the plumbing works
and the shapes line up.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from . import cleanup, data, metrics, skeleton
from .model import UNet, predict_mask, segmentation_loss


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Everything that defines a training run.

    Train and validation samples are deliberately different patches of ground.
    Crops drawn from one tile overlap, so a split within a tile would leak and
    report a flattering validation score.

    Attributes:
        train_ids: Sample ids the source should supply for training.
        val_ids: Sample ids held out for validation.
        tile_size_m: Tile side length in metres.
        crop_size: Model input size; must divide by ``2 ** depth``.
        n_train_crops: Crops sampled from the training tile.
        n_val_crops: Crops sampled from the validation tile.
        batch_size: Crops per optimizer step.
        epochs: Passes over the training crops.
        lr: AdamW learning rate.
        dice_weight: Blend between cross-entropy and soft Dice.
        widths: U-Net channel widths per level.
        source: Registry key naming the tile source.
        seed: Seeds crop sampling and weight initialization.
        device: ``"auto"``, or an explicit torch device string.
    """

    train_ids: tuple[str, ...] = ("tile_0", "tile_2")
    val_ids: tuple[str, ...] = ("tile_1",)
    tile_size_m: float = 1024.0
    crop_size: int = 256
    n_train_crops: int = 256
    n_val_crops: int = 64
    batch_size: int = 8
    epochs: int = 20
    lr: float = 1e-3
    dice_weight: float = 0.5
    widths: Sequence[int] = (32, 64, 128, 256)
    source: str = "synthetic"
    seed: int = 0
    device: str = "auto"


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """One epoch's losses and pixel agreement.

    Attributes:
        epoch: Zero-based epoch index.
        train_loss: Mean training loss.
        val_loss: Mean validation loss.
        val_iou: Pixel IoU on validation crops, at a 0.5 threshold.
    """

    epoch: int
    train_loss: float
    val_loss: float
    val_iou: float


@dataclass(frozen=True, slots=True)
class TrainResult:
    """A finished run.

    Attributes:
        model: The trained network, on CPU.
        history: Per-epoch metrics in order.
        checkpoint: Where weights were written, if anywhere.
    """

    model: UNet
    history: tuple[EpochMetrics, ...] = field(default_factory=tuple)
    checkpoint: Path | None = None


class CropDataset(Dataset):
    """Materializes crop windows drawn from one or more tiles.

    Spans samples because real imagery arrives as many small chips rather than
    one large tile: SpaceNet Roads ships roughly 300 m squares, so a useful
    training set is a few hundred chips rather than a few big scenes.

    Thin by design: it holds the windows chosen by :func:`data.crop_specs` and
    cuts them when asked, so the sampling logic stays testable without torch.
    """

    def __init__(
        self,
        samples: Sequence[data.TileSample],
        specs: Sequence[tuple[int, data.CropSpec]],
    ) -> None:
        """
        Args:
            samples: Tiles the crops are cut from.
            specs: ``(sample index, window)`` pairs.
        """
        self.samples = tuple(samples)
        self.specs = tuple(specs)

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample_index, spec = self.specs[index]
        image, mask = data.take_crop(self.samples[sample_index], spec)
        return (
            torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            torch.from_numpy(mask.astype(np.float32))[None],
        )


def resolve_device(requested: str = "auto") -> torch.device:
    """Pick a device, preferring accelerators when asked for ``"auto"``.

    Args:
        requested: ``"auto"`` or an explicit torch device string.

    Returns:
        The device to train on.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataset(
    source: data.TileSource,
    sample_ids: Sequence[str],
    n_crops: int,
    config: TrainConfig,
    seed_offset: int,
) -> CropDataset:
    """Load samples and spread crop windows across them.

    Args:
        source: Where imagery and labels come from.
        sample_ids: Ids to load.
        n_crops: Total crops to sample, divided between the loaded samples.
        config: Run configuration.
        seed_offset: Added to the run seed, so train and validation draw
            different windows.

    Returns:
        A dataset over those samples.

    Raises:
        ValueError: If no sample is large enough to hold one crop.
    """
    rng = np.random.default_rng(config.seed + seed_offset)
    samples, specs = [], []

    usable = []
    for sample_id in sample_ids:
        sample = source.load(sample_id)
        if min(sample.mask.shape) < config.crop_size:
            logger.warning(
                f"{sample_id}: {sample.mask.shape} smaller than "
                f"{config.crop_size}px crop, skipping"
            )
            continue
        usable.append(sample)

    if not usable:
        raise ValueError(
            f"no sample in {list(sample_ids)} fits a {config.crop_size}px crop"
        )

    per_sample = max(n_crops // len(usable), 1)
    for index, sample in enumerate(usable):
        samples.append(sample)
        specs.extend(
            (index, spec)
            for spec in data.crop_specs(
                sample.mask, size=config.crop_size, count=per_sample, rng=rng
            )
        )
    return CropDataset(samples, specs)


def _run_epoch(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
    dice_weight: float,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    """Run one pass. Trains when an optimizer is given, otherwise evaluates.

    Returns:
        ``(mean_loss, mean_iou)``.
    """
    training = optimizer is not None
    model.train(training)

    losses, intersections, unions = [], 0.0, 0.0
    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = segmentation_loss(logits, masks, dice_weight=dice_weight)

            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            losses.append(float(loss.detach()))
            predicted = predict_mask(logits)
            target = masks > 0.5
            intersections += float((predicted & target).sum())
            unions += float((predicted | target).sum())

    iou = intersections / unions if unions else 1.0
    return float(np.mean(losses)) if losses else 0.0, iou


def train(
    config: TrainConfig,
    source: data.TileSource | None = None,
    checkpoint: Path | None = None,
    datasets: tuple[CropDataset, CropDataset] | None = None,
) -> TrainResult:
    """Train a U-Net and return it along with its per-epoch history.

    Args:
        config: Run configuration.
        source: Where imagery comes from. Defaults to the registry entry named
            by ``config.source``.
        checkpoint: Optional path to write the trained weights to.
        datasets: Pre-built ``(train, val)`` datasets. Supplying them skips
            tile loading entirely, so a sweep over model settings pays the
            ingestion cost once rather than once per run.

    Returns:
        The trained model, its history, and where it was saved.
    """
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)

    if datasets is None:
        if source is None:
            source = data.TILE_SOURCE_REGISTRY[config.source]()
        train_set = build_dataset(
            source, config.train_ids, config.n_train_crops, config, 0
        )
        val_set = build_dataset(source, config.val_ids, config.n_val_crops, config, 1000)
    else:
        train_set, val_set = datasets
    logger.info(
        f"train {len(train_set)} crops, val {len(val_set)} crops, "
        f"{config.crop_size}px, device {device}"
    )

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size)

    model = UNet(in_channels=3, widths=config.widths).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    history = []
    for epoch in range(config.epochs):
        train_loss, _ = _run_epoch(
            model, train_loader, device, config.dice_weight, optimizer
        )
        val_loss, val_iou = _run_epoch(
            model, val_loader, device, config.dice_weight, None
        )
        history.append(
            EpochMetrics(
                epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_iou=val_iou
            )
        )
        logger.info(
            f"epoch {epoch:3d}  train {train_loss:.4f}  "
            f"val {val_loss:.4f}  val IoU {val_iou:.4f}"
        )

    model = model.to("cpu")
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "widths": list(config.widths)}, checkpoint
        )
        logger.info(f"wrote {checkpoint}")

    return TrainResult(model=model, history=tuple(history), checkpoint=checkpoint)


def overfit_one_batch(
    config: TrainConfig,
    steps: int = 400,
    batch_size: int = 4,
    dataset: CropDataset | None = None,
) -> tuple[float, ...]:
    """Drive the loss toward zero on a handful of crops.

    The single most useful diagnostic available: if the model cannot memorize
    four examples, the fault is in the model or the loss, not in the data or the
    schedule. It is also cheap enough to run as a test.

    Args:
        config: Run configuration; only the model and data settings are used.
        steps: Optimizer steps to take.
        batch_size: Number of crops to memorize.
        dataset: Pre-built dataset, to skip tile loading.

    Returns:
        The loss after each step.
    """
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)

    if dataset is None:
        source = data.TILE_SOURCE_REGISTRY[config.source]()
        dataset = build_dataset(source, config.train_ids[:1], batch_size, config, 0)
    images = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device)
    masks = torch.stack([dataset[i][1] for i in range(len(dataset))]).to(device)

    model = UNet(in_channels=3, widths=config.widths).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    model.train()

    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = segmentation_loss(model(images), masks, dice_weight=config.dice_weight)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return tuple(losses)


def load_checkpoint(path: Path, device: str = "cpu") -> UNet:
    """Rebuild a model from saved weights.

    Args:
        path: Checkpoint written by :func:`train`.
        device: Device string to map storage onto.

    Returns:
        The model in eval mode.
    """
    payload = torch.load(path, map_location=device, weights_only=True)
    model = UNet(in_channels=3, widths=payload["widths"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def predict_tile_logits(
    model: UNet, sample: data.TileSample, device: str = "cpu"
) -> np.ndarray:
    """Run the model over a whole tile in one pass.

    Real imagery does not arrive in convenient sizes — SpaceNet chips vary by a
    pixel or two around 396x324 — so the tile is reflection-padded up to the
    model's downsampling factor and the result cropped back. Reflection rather
    than zeros, because a black border invents a hard edge the model would
    happily segment as a road.

    Args:
        model: Trained network.
        sample: Tile to segment.
        device: Device string.

    Returns:
        ``(H, W)`` float32 logits, at the tile's own size.
    """
    torch_device = torch.device(device)
    model = model.to(torch_device).eval()

    image = np.ascontiguousarray(sample.image.transpose(2, 0, 1))
    batch = torch.from_numpy(image)[None].to(torch_device)

    height, width = batch.shape[-2:]
    factor = 2**model.depth
    pad_h = (-height) % factor
    pad_w = (-width) % factor
    if pad_h or pad_w:
        batch = nn.functional.pad(batch, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        logits = model(batch)
    return logits[0, 0, :height, :width].cpu().numpy()


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Quality at each stage of the pipeline, on one tile.

    Separated by stage on purpose. A single end-to-end score cannot tell a bad
    mask from a bad cleanup, so a regression in one looks like a regression in
    the other. Comparing ``apls_raw`` against ``apls_cleaned`` isolates the
    post-processing; comparing ``apls_cleaned`` against ``ceiling_apls``
    isolates the model.

    Attributes:
        mask_iou: Pixel agreement between predicted and label mask.
        apls_raw: APLS of the traced graph, before cleanup.
        apls_cleaned: APLS of the final graph.
        ceiling_apls: APLS the same pipeline reaches from a *perfect* mask on
            this tile. Tile-dependent, so it is measured rather than assumed.
        fraction_of_ceiling: ``apls_cleaned / ceiling_apls``; how much of the
            achievable score the model actually captured.
    """

    mask_iou: float
    apls_raw: float
    apls_cleaned: float
    ceiling_apls: float
    fraction_of_ceiling: float


def evaluate_tile(
    model: UNet, sample: data.TileSample, threshold: float = 0.5, device: str = "cpu"
) -> EvalReport:
    """Score a model over a whole tile, stage by stage.

    Args:
        model: Trained network.
        sample: Tile with imagery, label mask and ground-truth graph.
        threshold: Probability above which a pixel counts as road.
        device: Device string for inference.

    Returns:
        Per-stage quality for this tile.
    """
    logits = predict_tile_logits(model, sample, device=device)
    predicted = predict_mask(logits, threshold)

    raw = skeleton.graph_from_mask(predicted)
    cleaned = cleanup.clean(raw)
    ceiling = cleanup.clean(skeleton.graph_from_mask(sample.mask))

    apls_cleaned = metrics.apls(sample.truth, cleaned).score
    ceiling_apls = metrics.apls(sample.truth, ceiling).score

    return EvalReport(
        mask_iou=metrics.iou(predicted, sample.mask),
        apls_raw=metrics.apls(sample.truth, raw).score,
        apls_cleaned=apls_cleaned,
        ceiling_apls=ceiling_apls,
        fraction_of_ceiling=apls_cleaned / ceiling_apls if ceiling_apls else 0.0,
    )


def main() -> None:
    """Command line entry point."""
    import argparse
    import json
    from dataclasses import asdict

    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--crop-size", type=int, default=defaults.crop_size)
    parser.add_argument("--tile-size", type=float, default=defaults.tile_size_m)
    parser.add_argument("--train-crops", type=int, default=defaults.n_train_crops)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--source", default=defaults.source)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, help="write metrics as JSON here")
    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        crop_size=args.crop_size,
        tile_size_m=args.tile_size,
        n_train_crops=args.train_crops,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        source=args.source,
    )
    result = train(config, checkpoint=args.checkpoint)

    source = data.TILE_SOURCE_REGISTRY[config.source]()
    report = evaluate_tile(result.model, source.load(config.val_ids[0]))

    logger.info(f"mask IoU            {report.mask_iou:.4f}")
    logger.info(f"APLS raw skeleton   {report.apls_raw:.4f}")
    logger.info(f"APLS cleaned        {report.apls_cleaned:.4f}")
    logger.info(
        f"APLS ceiling        {report.ceiling_apls:.4f}  (perfect mask, this tile)"
    )
    logger.info(f"fraction of ceiling {report.fraction_of_ceiling:.3f}")

    if config.source == "synthetic":
        logger.warning(
            "synthetic imagery: these numbers prove plumbing, not quality -- "
            "the image is derived from the label. Do not report them."
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "config": asdict(config),
                    "history": [asdict(m) for m in result.history],
                    "eval": asdict(report),
                },
                indent=2,
                default=str,
            )
        )
        logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()

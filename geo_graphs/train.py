"""Training loop for the road segmentation model.

Written against synthetic imagery from :mod:`data`, so the loop, the loss and
the evaluation path can be debugged before any imagery is downloaded. Swapping
in a real tile source changes one config key and nothing else.

**Numbers from a synthetic run are not results.** The image is derived from the
label, so the model fits it easily; what a run proves is that the plumbing works
and the shapes line up.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from loguru import logger
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from . import cleanup, data, metrics, skeleton, split
from .model import FUSION_REGISTRY, UNet, predict_mask, segmentation_loss

#: Which validation number decides the best epoch.
#:
#: ``val_loss`` and ``val_iou`` are pixel measures and cost nothing extra.
#: ``val_apls`` is the metric the project actually cares about, and selecting
#: on it costs a graph extraction per epoch — but selecting a checkpoint on a
#: pixel score is precisely the mistake this project exists to demonstrate, so
#: the option is here to be measured rather than assumed away.
SelectOn = Literal["val_loss", "val_iou", "val_apls"]

#: Direction of improvement for each criterion.
_HIGHER_IS_BETTER: dict[str, bool] = {
    "val_loss": False,
    "val_iou": True,
    "val_apls": True,
}


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
        select_on: Which validation number picks the returned weights.
        patience: Epochs without improvement before stopping early. ``None``
            runs every epoch.
        apls_eval_tiles: Validation tiles scored end to end each epoch. Zero
            skips it, which is required unless ``select_on`` is ``val_apls``.
        augment: Apply the dihedral symmetry group to training crops. Overhead
            imagery has no canonical orientation, so the eight rotations and
            reflections are exact symmetries rather than approximations.
        cldice_weight: Blend factor for the centreline Dice term. Zero is the
            pixel-only loss.
        fusion: Registry key naming how the lidar stack reaches the network.
            Used only when the source supplies one; an imagery-only source
            builds an imagery-only network whatever this says.
        coverage: How much lidar each crop is allowed to see, and in what
            shape. The endpoints ``full`` and ``none`` are the two runs the
            first stage of this work compares.
        split: Registry key naming how ``train_ids`` and ``val_ids`` were
            divided. Recorded rather than used: the ids are already resolved by
            the time a config exists, and a run artifact that does not say how
            they were drawn cannot be told apart from one drawn differently.
        split_block_m: Block side length the spatial splitters used, in metres.
        split_buffer_m: Training margin the buffered splitter dropped, in
            metres.
        split_digest: Content hash of a frozen split's ids, or empty when the
            split was drawn rather than read. Two runs whose ids differ cannot
            be compared, and this is what says so without diffing the lists.
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
    select_on: SelectOn = "val_loss"
    patience: int | None = 5
    apls_eval_tiles: int = 0
    augment: bool = False
    cldice_weight: float = 0.0
    fusion: str = "early"
    coverage: data.CoverageSampler = data.FULL_COVERAGE
    split: str = "random"
    split_block_m: float = 1280.0
    split_buffer_m: float = 500.0
    split_digest: str = ""


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """One epoch's losses and pixel agreement.

    Attributes:
        epoch: Zero-based epoch index.
        train_loss: Mean training loss.
        val_loss: Mean validation loss.
        val_iou: Pixel IoU on validation crops, at a 0.5 threshold.
        val_apls: APLS over a few validation tiles, or ``None`` when not
            computed. Costs a graph extraction per tile, so it is opt-in.
    """

    epoch: int
    train_loss: float
    val_loss: float
    val_iou: float
    val_apls: float | None = None


@dataclass(frozen=True, slots=True)
class TrainResult:
    """A finished run.

    Attributes:
        model: The trained network, on CPU, holding the *best* epoch's weights
            rather than the last.
        history: Per-epoch metrics in order.
        checkpoint: Where weights were written, if anywhere.
        best_epoch: Epoch the returned weights came from.
        stopped_early: Whether patience ran out before the epoch budget did.
    """

    model: UNet
    history: tuple[EpochMetrics, ...] = field(default_factory=tuple)
    checkpoint: Path | None = None
    best_epoch: int = -1
    stopped_early: bool = False


#: Second word of the augmentation seed, so those draws form a stream
#: independent of the crop-window sampling that shares the same run seed.
_AUGMENT_STREAM = 1

#: Second word of the coverage seed. Same reasoning as :data:`_AUGMENT_STREAM`:
#: turning coverage sampling on must not shift which windows get drawn.
_COVERAGE_STREAM = 2


class CropDataset(Dataset):
    """Materializes crop windows drawn from one or more tiles.

    Spans samples because real imagery arrives as many small chips rather than
    one large tile: SpaceNet Roads ships roughly 300 m squares, so a useful
    training set is a few hundred chips rather than a few big scenes.

    Thin by design: it holds the windows chosen by :func:`data.crop_specs` and
    cuts them when asked, so the sampling logic stays testable without torch.
    Augmentation is the one thing it adds on the way out, and only when asked:
    validation crops must stay fixed or the epoch-to-epoch number is comparing
    two different datasets.
    """

    def __init__(
        self,
        samples: Sequence[data.TileSample],
        specs: Sequence[tuple[int, data.CropSpec]],
        augment: bool = False,
        rng: np.random.Generator | None = None,
        coverage: data.CoverageSampler = data.FULL_COVERAGE,
        coverage_seed: int = 0,
    ) -> None:
        """
        Args:
            samples: Tiles the crops are cut from.
            specs: ``(sample index, window)`` pairs.
            augment: Draw one of the eight dihedral transforms per crop and
                apply it to the image, its mask and its lidar together.
                Training splits only.
            rng: Source of randomness for those draws. Required when
                ``augment`` is set, so the run stays replayable from its seed.
            coverage: How much lidar each crop sees. Ignored by samples that
                carry none.
            coverage_seed: Seeds the per-crop coverage draw.

        Raises:
            ValueError: If ``augment`` is set without an ``rng``.
        """
        if augment and rng is None:
            raise ValueError("augment=True needs an explicit rng to stay reproducible")
        self.samples = tuple(samples)
        self.specs = tuple(specs)
        self.augment = augment
        self.coverage = coverage
        # Keyed on the crop index rather than advanced per visit, unlike the
        # augmentation stream: a crop's coverage is a fixed property of that
        # example on every epoch. Resampling it each visit would make a
        # validation score compare two different datasets from one epoch to
        # the next, and the curriculum already spreads across crops.
        self._coverage_seed = coverage_seed
        # Per-epoch variation and reproducibility pull against each other here.
        # Keying the transform on the crop index alone would hand every crop the
        # same rotation on every epoch -- a 1x dataset dressed up as 8x -- while
        # an unseeded draw is not replayable at all. Advancing one seeded stream
        # across __getitem__ calls gives both: a fresh transform each visit, and
        # a sequence fixed by TrainConfig.seed. The stream is the dataset's own
        # rather than torch's global one, so a change to the model cannot
        # perturb which augmentations the data sees. (It lives in the dataset
        # object, so DataLoader worker processes would each fork a copy; the
        # loaders here are single-process.)
        self._rng = rng

    def __len__(self) -> int:
        return len(self.specs)

    @property
    def aux_channels(self) -> int:
        """Channels the aux stack presents: the data channels plus validity.

        Zero when the samples carry no lidar, which is what tells :func:`train`
        to build an imagery-only network.
        """
        if not self.samples or self.samples[0].aux is None:
            return 0
        return self.samples[0].aux.shape[-1] + 1

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        sample_index, spec = self.specs[index]
        crop = data.take_crop(self.samples[sample_index], spec)
        if crop.aux is not None:
            crop = replace(
                crop,
                coverage=data.sample_coverage(
                    self.coverage,
                    crop.mask.shape,
                    np.random.default_rng((self._coverage_seed, _COVERAGE_STREAM, index)),
                ),
            )
        if self.augment and self._rng is not None:
            crop = data.augment_crop(crop, int(self._rng.integers(data.DIHEDRAL_ORDER)))

        tensors = [
            torch.from_numpy(np.ascontiguousarray(crop.image.transpose(2, 0, 1))),
            torch.from_numpy(crop.mask.astype(np.float32))[None],
        ]
        if crop.aux is not None:
            stack = data.aux_stack(crop).transpose(2, 0, 1)
            tensors.append(torch.from_numpy(np.ascontiguousarray(stack)))
        return tuple(tensors)


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

    The ids decide augmentation rather than a flag: ``config.augment`` reaches
    the dataset only when every requested id is one of ``config.train_ids``.
    Callers name the split by which ids they ask for, so keying off that keeps
    the two calls in :func:`train` from having to agree on a convention, and
    keeps a rotated crop out of validation — where it would make the score
    noisy and incomparable across epochs.

    Args:
        source: Where imagery and labels come from.
        sample_ids: Ids to load. Whether they are a subset of
            ``config.train_ids`` decides whether the crops get augmented.
        n_crops: Total crops to sample, divided between the loaded samples.
        config: Run configuration.
        seed_offset: Added to the run seed, so train and validation draw
            different windows.

    Returns:
        A dataset over those samples, augmented only if they are training ids.

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

    is_train = set(sample_ids) <= set(config.train_ids)
    return CropDataset(
        samples,
        specs,
        augment=config.augment and is_train,
        rng=np.random.default_rng((config.seed + seed_offset, _AUGMENT_STREAM)),
        coverage=config.coverage,
        coverage_seed=config.seed + seed_offset,
    )


def _run_epoch(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
    dice_weight: float,
    optimizer: torch.optim.Optimizer | None,
    cldice_weight: float = 0.0,
) -> tuple[float, float]:
    """Run one pass. Trains when an optimizer is given, otherwise evaluates.

    New parameters go after ``optimizer``, and call sites pass by keyword.
    Inserting one before it would silently rebind the optimizer to a float and
    turn training into evaluation without an error.

    Returns:
        ``(mean_loss, mean_iou)``.
    """
    training = optimizer is not None
    model.train(training)

    losses, intersections, unions = [], 0.0, 0.0
    with torch.set_grad_enabled(training):
        for batch in loader:
            images, masks = batch[0].to(device), batch[1].to(device)
            # Length rather than a sentinel tensor: a dataset with no lidar
            # yields two tensors, and an empty third would have to be shaped
            # like something the model must then learn to ignore.
            aux = batch[2].to(device) if len(batch) > 2 else None

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images, aux)
            loss = segmentation_loss(
                logits,
                masks,
                dice_weight=dice_weight,
                cldice_weight=cldice_weight,
            )

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


def _epoch_apls(
    model: UNet,
    source: data.TileSource | None,
    config: TrainConfig,
    device: torch.device,
) -> float | None:
    """Mean APLS over a few validation tiles, or ``None`` when not requested.

    Costs a full graph extraction per tile, so it is opt-in rather than always
    on. It exists because selecting a checkpoint on a pixel score is the exact
    mistake this project measures, and the alternative should be available to
    compare against rather than argued about.
    """
    if config.apls_eval_tiles <= 0 or source is None:
        return None
    report = evaluate_tiles(
        model,
        source,
        config.val_ids[: config.apls_eval_tiles],
        device=str(device),
        coverage=config.coverage,
    )
    return report.apls_cleaned if report.n_scored else None


def _write_checkpoint(
    path: Path,
    state: dict,
    widths: Sequence[int],
    aux_channels: int = 0,
    fusion: str | None = None,
) -> None:
    """Write weights plus everything needed to rebuild the network around them.

    Args:
        path: Destination; parent directories are created.
        state: A state dict, already on CPU.
        widths: Channel widths the weights were trained with.
        aux_channels: Width of the auxiliary stack the stem was built for.
        fusion: Registry key naming how that stack was fused, or ``None`` for
            an imagery-only network.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": state,
            "widths": list(widths),
            "aux_channels": aux_channels,
            "fusion": fusion,
        },
        path,
    )


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

    aux_channels = train_set.aux_channels
    if aux_channels != val_set.aux_channels:
        raise ValueError(
            f"train split has {aux_channels} aux channels and validation has "
            f"{val_set.aux_channels}; they must come from the same source"
        )
    fusion_key = config.fusion if aux_channels else None
    logger.info(
        f"train {len(train_set)} crops, val {len(val_set)} crops, "
        f"{config.crop_size}px, device {device}, "
        f"{aux_channels} aux channels, coverage {config.coverage.mode}"
        + (f", {fusion_key} fusion" if fusion_key else "")
    )

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size)

    model = UNet(
        in_channels=3,
        widths=config.widths,
        aux_channels=aux_channels,
        fusion=FUSION_REGISTRY[fusion_key]() if fusion_key else None,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    history: list[EpochMetrics] = []
    best_score: float | None = None
    best_state: dict | None = None
    best_epoch = -1
    stale_epochs = 0
    stopped_early = False
    higher_is_better = _HIGHER_IS_BETTER[config.select_on]

    for epoch in range(config.epochs):
        train_loss, _ = _run_epoch(
            model,
            train_loader,
            device,
            dice_weight=config.dice_weight,
            optimizer=optimizer,
            cldice_weight=config.cldice_weight,
        )
        val_loss, val_iou = _run_epoch(
            model,
            val_loader,
            device,
            dice_weight=config.dice_weight,
            optimizer=None,
            cldice_weight=config.cldice_weight,
        )
        val_apls = _epoch_apls(model, source, config, device)

        measured = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_iou=val_iou,
            val_apls=val_apls,
        )
        history.append(measured)

        line = (
            f"epoch {epoch:3d}  train {train_loss:.4f}  "
            f"val {val_loss:.4f}  val IoU {val_iou:.4f}"
        )
        if val_apls is not None:
            line += f"  val APLS {val_apls:.4f}"

        score = getattr(measured, config.select_on)
        if score is None:
            raise ValueError(
                f"select_on={config.select_on!r} needs apls_eval_tiles > 0 and a source"
            )

        improved = best_score is None or (
            score > best_score if higher_is_better else score < best_score
        )
        if improved:
            best_score, best_epoch, stale_epochs = score, epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            line += "  *"
            # Write as we improve rather than only at the end: a run killed at
            # epoch 29 of 40 otherwise leaves nothing on disk, and these runs
            # are long enough to lose to an OOM or a closed laptop.
            if checkpoint is not None:
                _write_checkpoint(
                    checkpoint, best_state, config.widths, aux_channels, fusion_key
                )
        else:
            stale_epochs += 1

        logger.info(line)

        if config.patience is not None and stale_epochs >= config.patience:
            stopped_early = True
            logger.info(
                f"no improvement in {config.select_on} for {stale_epochs} epochs; "
                f"stopping at epoch {epoch}"
            )
            break

    model = model.to("cpu")
    if best_state is not None:
        # Keep the best epoch, not the last. Validation loss here bottoms well
        # before the epoch budget runs out, so returning the final weights hands
        # back a measurably worse model and makes every later comparison lie.
        model.load_state_dict(best_state)
        logger.info(f"restored epoch {best_epoch} ({config.select_on} {best_score:.4f})")

    if checkpoint is not None:
        _write_checkpoint(
            checkpoint, model.state_dict(), config.widths, aux_channels, fusion_key
        )
        logger.info(f"wrote {checkpoint}")

    return TrainResult(
        model=model,
        history=tuple(history),
        checkpoint=checkpoint,
        best_epoch=best_epoch,
        stopped_early=stopped_early,
    )


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
    aux_channels = dataset.aux_channels
    aux = (
        torch.stack([dataset[i][2] for i in range(len(dataset))]).to(device)
        if aux_channels
        else None
    )

    model = UNet(
        in_channels=3,
        widths=config.widths,
        aux_channels=aux_channels,
        fusion=FUSION_REGISTRY[config.fusion]() if aux_channels else None,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    model.train()

    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = segmentation_loss(
            model(images, aux),
            masks,
            dice_weight=config.dice_weight,
            cldice_weight=config.cldice_weight,
        )
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
    # A checkpoint written before fusion existed carries neither key, and an
    # imagery-only network is exactly what it holds.
    aux_channels = payload.get("aux_channels", 0)
    fusion_key = payload.get("fusion")
    model = UNet(
        in_channels=3,
        widths=payload["widths"],
        aux_channels=aux_channels,
        fusion=FUSION_REGISTRY[fusion_key]() if fusion_key else None,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def predict_tile_logits(
    model: UNet,
    sample: data.TileSample,
    device: str = "cpu",
    coverage: data.CoverageSampler = data.FULL_COVERAGE,
    seed: int = 0,
) -> np.ndarray:
    """Run the model over a whole tile in one pass.

    Real imagery does not arrive in convenient sizes — SpaceNet chips vary by a
    pixel or two around 396x324 — so the tile is reflection-padded up to the
    model's downsampling factor and the result cropped back. Reflection rather
    than zeros, because a black border invents a hard edge the model would
    happily segment as a road. The aux stack is padded the same way, since a
    channel that fell out of registration with the imagery would be worse than
    no channel at all.

    Args:
        model: Trained network.
        sample: Tile to segment.
        device: Device string.
        coverage: How much of the tile's lidar the model is allowed to see.
            Ignored by an imagery-only network.
        seed: Seeds the coverage pattern, so a scored tile is reproducible.

    Returns:
        ``(H, W)`` float32 logits, at the tile's own size.

    Raises:
        ValueError: If the model expects lidar and the sample carries none.
    """
    torch_device = torch.device(device)
    model = model.to(torch_device).eval()

    crop = data.whole_tile(sample)
    aux_batch = None
    if model.aux_channels:
        if crop.aux is None:
            raise ValueError("model was trained with lidar but this sample carries none")
        crop = replace(
            crop,
            coverage=data.sample_coverage(
                coverage, crop.mask.shape, np.random.default_rng(seed)
            ),
        )
        stack = data.aux_stack(crop).transpose(2, 0, 1)
        aux_batch = torch.from_numpy(np.ascontiguousarray(stack))[None].to(torch_device)

    image = np.ascontiguousarray(sample.image.transpose(2, 0, 1))
    batch = torch.from_numpy(image)[None].to(torch_device)

    height, width = batch.shape[-2:]
    factor = 2**model.depth
    pad_h = (-height) % factor
    pad_w = (-width) % factor
    if pad_h or pad_w:
        batch = nn.functional.pad(batch, (0, pad_w, 0, pad_h), mode="reflect")
        if aux_batch is not None:
            aux_batch = nn.functional.pad(aux_batch, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        logits = model(batch, aux_batch)
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
        sample_id: Which tile this scored. Without it a per-tile list cannot be
            traced back to a chip, and pairing two runs relies on list order.
        mask_iou: Pixel agreement between predicted and label mask.
        apls_raw: APLS of the traced graph, before cleanup.
        apls_cleaned: APLS of the final graph.
        ceiling_apls: APLS the same pipeline reaches from a *perfect* mask on
            this tile. Tile-dependent, so it is measured rather than assumed.
        fraction_of_ceiling: ``apls_cleaned / ceiling_apls``; how much of the
            achievable score the model actually captured.
    """

    sample_id: str
    mask_iou: float
    apls_raw: float
    apls_cleaned: float
    ceiling_apls: float
    fraction_of_ceiling: float


def evaluate_tile(
    model: UNet,
    sample: data.TileSample,
    threshold: float = 0.5,
    device: str = "cpu",
    sample_id: str = "",
    coverage: data.CoverageSampler = data.FULL_COVERAGE,
) -> EvalReport:
    """Score a model over a whole tile, stage by stage.

    Args:
        model: Trained network.
        sample: Tile with imagery, label mask and ground-truth graph.
        threshold: Probability above which a pixel counts as road.
        device: Device string for inference.
        sample_id: Recorded on the report so results stay traceable.
        coverage: Lidar coverage the model is scored under. Held-out coverage
            is a condition of the measurement, not a property of the model, so
            it belongs here rather than on the checkpoint.

    Returns:
        Per-stage quality for this tile.
    """
    logits = predict_tile_logits(model, sample, device=device, coverage=coverage)
    predicted = predict_mask(logits, threshold)

    raw = skeleton.graph_from_mask(predicted)
    cleaned = cleanup.clean(raw)
    ceiling = cleanup.clean(skeleton.graph_from_mask(sample.mask))

    apls_cleaned = metrics.apls(sample.truth, cleaned).score
    ceiling_apls = metrics.apls(sample.truth, ceiling).score

    return EvalReport(
        sample_id=sample_id,
        mask_iou=metrics.iou(predicted, sample.mask),
        apls_raw=metrics.apls(sample.truth, raw).score,
        apls_cleaned=apls_cleaned,
        ceiling_apls=ceiling_apls,
        fraction_of_ceiling=apls_cleaned / ceiling_apls if ceiling_apls else 0.0,
    )


@dataclass(frozen=True, slots=True)
class AggregateReport:
    """Per-stage quality across a set of tiles.

    Reports a spread rather than a single mean. Chips are roughly 300 m square,
    small enough that one severed road moves a tile's APLS a long way, so a mean
    alone hides how variable the result is.

    Attributes:
        n_scored: Tiles that contributed to the numbers.
        n_skipped: Tiles dropped for having no ground-truth roads at all, which
            SpaceNet does ship and which score degenerately.
        mask_iou: Mean pixel agreement.
        apls_cleaned: Mean APLS of the final graph.
        apls_median: Median APLS, which a single bad chip cannot drag.
        ceiling_apls: Mean achievable APLS on these tiles.
        fraction_of_ceiling: Mean per-tile share of the achievable score.
        per_tile: Every individual report, in the order scored.
    """

    n_scored: int
    n_skipped: int
    mask_iou: float
    apls_cleaned: float
    apls_median: float
    ceiling_apls: float
    fraction_of_ceiling: float
    per_tile: tuple[EvalReport, ...]


def evaluate_tiles(
    model: UNet,
    source: data.TileSource,
    sample_ids: Sequence[str],
    threshold: float = 0.5,
    device: str = "cpu",
    coverage: data.CoverageSampler = data.FULL_COVERAGE,
) -> AggregateReport:
    """Score a model across many tiles and summarize the spread.

    Args:
        model: Trained network.
        source: Where imagery and labels come from.
        sample_ids: Tiles to score.
        threshold: Probability above which a pixel counts as road.
        device: Device string for inference.
        coverage: Lidar coverage every tile is scored under.

    Returns:
        The aggregate, with every per-tile report retained.
    """
    reports, skipped = [], 0
    for sample_id in sample_ids:
        sample = source.load(sample_id)
        if sample.truth.number_of_edges() == 0:
            skipped += 1
            continue
        reports.append(
            evaluate_tile(
                model,
                sample,
                threshold=threshold,
                device=device,
                sample_id=sample_id,
                coverage=coverage,
            )
        )

    if not reports:
        return AggregateReport(0, skipped, 0.0, 0.0, 0.0, 0.0, 0.0, ())

    def mean(attribute: str) -> float:
        return float(np.mean([getattr(r, attribute) for r in reports]))

    return AggregateReport(
        n_scored=len(reports),
        n_skipped=skipped,
        mask_iou=mean("mask_iou"),
        apls_cleaned=mean("apls_cleaned"),
        apls_median=float(np.median([r.apls_cleaned for r in reports])),
        ceiling_apls=mean("ceiling_apls"),
        fraction_of_ceiling=mean("fraction_of_ceiling"),
        per_tile=tuple(reports),
    )


def build_source(
    aoi_root: Path | None,
    source_key: str,
    resolution: float,
    lidar: bool = False,
) -> tuple[data.TileSource, str]:
    """Construct the tile source named on the command line.

    Args:
        aoi_root: Extracted SpaceNet AOI directory, or ``None`` for synthetic.
        source_key: Registry key, used only when ``aoi_root`` is ``None``.
        resolution: Metres per pixel, for real imagery.
        lidar: Ask the source for auxiliary channels. Only the synthetic source
            can fabricate them; SpaceNet ships no lidar, and asking for it
            there is a mistake worth naming rather than silently ignoring.

    Returns:
        ``(source, key)`` where key names which source was built.
    """
    if aoi_root is None:
        return data.TILE_SOURCE_REGISTRY[source_key](lidar=lidar), source_key

    from . import spacenet

    if lidar:
        logger.warning("SpaceNet carries no lidar; ignoring the lidar request")
    spacenet.register(data.TILE_SOURCE_REGISTRY)
    return (
        data.TILE_SOURCE_REGISTRY["spacenet"](aoi_root=aoi_root, resolution=resolution),
        "spacenet",
    )


def assign_split(
    source: data.TileSource,
    key: str,
    val_fraction: float,
    seed: int,
    block_m: float,
    buffer_m: float,
) -> split.Assignment:
    """Divide a source's samples between training and validation.

    The spatial splitters need to know where each sample sits, which means
    loading every one of them. That cost is paid only when one is asked for:
    ``random`` ignores position, so it never touches the imagery.

    Args:
        source: Where samples come from.
        key: A key of :data:`split.SPLIT_REGISTRY`.
        val_fraction: Share of samples to hold out.
        seed: Seeds the shuffle, whether over samples or over blocks.
        block_m: Block side length for the spatial splitters, in metres.
        buffer_m: Training margin the buffered splitter drops, in metres.

    Returns:
        The assignment, whose ``dropped`` is non-empty only for ``buffered``.
    """
    ids = source.ids()
    if key == "random":
        placement = split.Placement(tuple(ids), np.zeros(len(ids)), np.zeros(len(ids)))
    else:
        logger.info(f"{key} split: placing {len(ids)} samples")
        placement = split.placement_from_tiles(ids, [source.load(i).tile for i in ids])

    kwargs: dict[str, float] = {}
    if key in ("blocked", "buffered"):
        kwargs["block_m"] = block_m
    if key == "buffered":
        kwargs["buffer_m"] = buffer_m
    return split.SPLIT_REGISTRY[key](**kwargs).split(placement, val_fraction, seed)


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
    parser.add_argument("--val-crops", type=int, default=defaults.n_val_crops)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--source", default=defaults.source)
    parser.add_argument(
        "--aoi-root", type=Path, help="extracted SpaceNet AOI; overrides --source"
    )
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--max-eval-tiles",
        type=int,
        default=40,
        help="cap on validation tiles scored end to end; graph extraction is "
        "the slow part, and 40 already gives a stable spread",
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--select-on",
        choices=["val_loss", "val_iou", "val_apls"],
        default=defaults.select_on,
        help="which validation number picks the returned weights",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=defaults.patience,
        help="epochs without improvement before stopping; 0 disables",
    )
    parser.add_argument(
        "--apls-eval-tiles",
        type=int,
        default=defaults.apls_eval_tiles,
        help="tiles scored end to end each epoch; required for --select-on val_apls",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=defaults.augment,
        help="apply the dihedral symmetry group to training crops",
    )
    parser.add_argument(
        "--cldice-weight",
        type=float,
        default=defaults.cldice_weight,
        help="blend factor for the centreline Dice term; 0 is the pixel-only loss",
    )
    parser.add_argument(
        "--lidar",
        action="store_true",
        help="ask the source for auxiliary channels; synthetic sources only",
    )
    parser.add_argument(
        "--fusion",
        choices=sorted(FUSION_REGISTRY),
        default=defaults.fusion,
        help="how the lidar stack reaches the network",
    )
    parser.add_argument(
        "--coverage",
        choices=sorted(data.COVERAGE_PATTERNS),
        default=defaults.coverage.mode,
        help="how much lidar each crop sees; full and none are the endpoints",
    )
    parser.add_argument(
        "--split",
        choices=sorted(split.SPLIT_REGISTRY),
        default=defaults.split,
        help=(
            "how samples are divided; the spatial keys load every sample to "
            "place it, which random does not"
        ),
    )
    parser.add_argument(
        "--split-block-m",
        type=float,
        default=defaults.split_block_m,
        help="block side length for the spatial splitters, in metres",
    )
    parser.add_argument(
        "--split-buffer-m",
        type=float,
        default=defaults.split_buffer_m,
        help="training margin the buffered splitter drops, in metres",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        help=(
            "read a frozen split instead of drawing one; overrides --split and "
            "its parameters, and is how several runs are made to agree on the "
            "same ids rather than each re-deriving them"
        ),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, help="write metrics as JSON here")
    args = parser.parse_args()

    source, source_key = build_source(
        args.aoi_root, args.source, args.resolution, lidar=args.lidar
    )
    if args.split_file:
        frozen = split.read_frozen(args.split_file)
        assignment = frozen.assignment
        unknown = (set(assignment.train) | set(assignment.val)) - set(source.ids())
        if unknown:
            raise ValueError(
                f"{args.split_file} names {len(unknown)} samples this source does "
                f"not have, e.g. {sorted(unknown)[:3]}; it was frozen over "
                f"{frozen.source.get('aoi_root')}"
            )
        split_name, split_digest = frozen.name, frozen.digest
    else:
        assignment = assign_split(
            source,
            args.split,
            args.val_fraction,
            args.seed,
            args.split_block_m,
            args.split_buffer_m,
        )
        split_name, split_digest = args.split, ""
    logger.info(
        f"{source_key}: {split_name} split, {len(assignment.train)} train tiles, "
        f"{len(assignment.val)} val tiles, {len(assignment.dropped)} dropped"
    )

    config = TrainConfig(
        train_ids=assignment.train,
        val_ids=assignment.val,
        epochs=args.epochs,
        crop_size=args.crop_size,
        tile_size_m=args.tile_size,
        n_train_crops=args.train_crops,
        n_val_crops=args.val_crops,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        source=source_key,
        seed=args.seed,
        select_on=args.select_on,
        patience=args.patience or None,
        apls_eval_tiles=args.apls_eval_tiles,
        augment=args.augment,
        cldice_weight=args.cldice_weight,
        fusion=args.fusion,
        coverage=data.CoverageSampler(mode=args.coverage),
        split=split_name,
        split_block_m=args.split_block_m,
        split_buffer_m=args.split_buffer_m,
        split_digest=split_digest,
    )
    result = train(config, source=source, checkpoint=args.checkpoint)

    scored = config.val_ids[: args.max_eval_tiles]
    report = evaluate_tiles(result.model, source, scored, coverage=config.coverage)

    logger.info(
        f"best epoch {result.best_epoch} of {len(result.history)}"
        f"{' (stopped early)' if result.stopped_early else ''}"
    )
    logger.info(f"scored {report.n_scored} tiles ({report.n_skipped} had no roads)")
    logger.info(f"mask IoU            {report.mask_iou:.4f}")
    logger.info(f"APLS mean           {report.apls_cleaned:.4f}")
    logger.info(f"APLS median         {report.apls_median:.4f}")
    logger.info(f"APLS ceiling        {report.ceiling_apls:.4f}  (perfect mask)")
    logger.info(f"fraction of ceiling {report.fraction_of_ceiling:.3f}")

    if source_key == "synthetic":
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
                    "best_epoch": result.best_epoch,
                    "stopped_early": result.stopped_early,
                    "history": [asdict(m) for m in result.history],
                    "eval": {
                        **{k: v for k, v in asdict(report).items() if k != "per_tile"},
                        "per_tile": [asdict(r) for r in report.per_tile],
                    },
                },
                indent=2,
                default=str,
            )
        )
        logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Is road class readable from the imagery at all?

The attribute-inference project rests on a premise nothing has tested: that a
segmentation network trained to say *road or not* has, somewhere in its feature
maps, enough to say *which kind of road*. If it does not, no graph model built on
those features can either, and the project ends here for a cost of minutes.

The test is a linear probe. The checkpoint is frozen and never updated. Its
decoder feature maps are sampled at each labelled node with ``grid_sample``, four
scales deep so the widest receptive field is included -- road class is largely
about width and surroundings, which a single full-resolution pixel cannot see.
A classifier is fitted on those features alone and scored on the frozen spatial
split's validation side.

Three things are read together, and the middle one is the answer:

- **Majority baseline.** Always predicting ``residential``. Macro-F1 near 0.118,
  because six classes and one guess.
- **The probe.** Linear, then a small MLP. A linear probe clearing the baseline
  says the signal is present and linearly decodable; only the MLP clearing it
  says the signal is there but tangled.
- **A shuffled-label control.** The same fit against permuted labels, which has
  to come back at the baseline. It catches the plumbing bug that makes anything
  look like signal -- a feature that encodes position, a train/val mix-up.

What this does not test: whether a *graph* model beats a per-node one. That is
the ablation, and it needs both arms built. This only asks whether there is
anything for either of them to find.
"""

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from attr_resolvability import CLASSES, ChipNodes, chip_nodes, macro_f1
from loguru import logger
from torch import Tensor, nn

from geo_graphs import osm, spacenet, split, train

#: Feature maps tapped from the frozen network, coarsest first.
#:
#: The bottleneck sees eight times the ground of the final decoder, which is the
#: point: a motorway and a residential street differ by width and by what
#: surrounds them, and neither is visible in one full-resolution pixel. The head
#: is included as one channel because road probability is itself a width cue.
TAPS: tuple[str, ...] = ("bottleneck", "decoders.0", "decoders.1", "decoders.2", "head")


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """One classifier's score on the frozen split's validation side.

    Attributes:
        arm: What was fitted, and on what labels.
        macro_f1: Macro-F1 over the classes with support.
        accuracy: Node accuracy, reported alongside because the two disagree
            sharply under a 55% majority class.
        per_class_f1: F1 per class.
        n_train: Training nodes.
        n_val: Validation nodes.
        epochs: Passes over the training nodes.
    """

    arm: str
    macro_f1: float
    accuracy: float
    per_class_f1: dict[str, float]
    n_train: int
    n_val: int
    epochs: int


def tapped(model: nn.Module, taps: Sequence[str]) -> dict[str, Tensor]:
    """Register forward hooks on named submodules and return their output store.

    Hooks rather than a reimplemented forward: a second copy of ``UNet.forward``
    would drift from the first, and the drift would show up as features that
    quietly do not match what the checkpoint computes.
    """
    store: dict[str, Tensor] = {}

    def keep(name: str):
        def hook(_module, _inputs, output: Tensor) -> None:
            store[name] = output

        return hook

    for name in taps:
        module = model
        for part in name.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        module.register_forward_hook(keep(name))
    return store


def sample_features(
    store: dict[str, Tensor], taps: Sequence[str], xy: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Read every tapped map at the same ground positions.

    Args:
        store: Feature maps from :func:`tapped`, after a forward pass.
        taps: Which maps to read, in the order the columns come out.
        xy: ``(N, 2)`` positions in the padded image's pixel coordinates.
        width: Padded image width, which ``xy`` is expressed against.
        height: Padded image height.

    Returns:
        ``(N, sum of tapped channels)`` float32.
    """
    # grid_sample normalizes against each map's own size, so one grid serves
    # every scale. align_corners=False puts -1 and 1 on the outer pixel edges,
    # which is why the half-pixel shift is here.
    grid = torch.from_numpy(
        np.stack(
            [(xy[:, 0] + 0.5) / width * 2 - 1, (xy[:, 1] + 0.5) / height * 2 - 1],
            axis=-1,
        ).astype(np.float32)
    ).reshape(1, 1, -1, 2)

    columns = []
    for name in taps:
        sampled = torch.nn.functional.grid_sample(
            store[name], grid, mode="bilinear", align_corners=False
        )
        columns.append(sampled[0, :, 0, :].T)
    return torch.cat(columns, dim=1).numpy().astype(np.float32)


def chip_features(
    model: nn.Module,
    store: dict[str, Tensor],
    taps: Sequence[str],
    image: np.ndarray,
    xy: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run one chip through the frozen network and read it at the node positions."""
    tensor = torch.from_numpy(image).permute(2, 0, 1)[None].to(device)
    # Three pooling levels, so the input has to divide by eight; SpaceNet chips
    # do not. Padding is bottom-right, which leaves the node coordinates alone.
    _, _, height, width = tensor.shape
    pad_h, pad_w = (-height) % 8, (-width) % 8
    tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    with torch.no_grad():
        model(tensor)
    return sample_features(
        {k: v.cpu() for k, v in store.items()},
        taps,
        xy,
        width + pad_w,
        height + pad_h,
    )


def fit(
    features: np.ndarray,
    labels: np.ndarray,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> nn.Module:
    """Fit a classifier on frozen features, weighting classes by inverse frequency.

    Inverse-frequency weights because the score is macro-F1: an unweighted fit on
    a 55% majority class optimises the wrong average and would understate the
    signal this script exists to detect.

    Args:
        features: ``(N, D)`` standardized features.
        labels: ``(N,)`` class index.
        hidden: Hidden width, or 0 for a linear probe.
        epochs: Passes over the data.
        lr: AdamW learning rate.
        seed: Seeds initialization and batching.
        device: Where to fit.

    Returns:
        The fitted classifier, in eval mode.
    """
    torch.manual_seed(seed)
    model = (
        nn.Linear(features.shape[1], len(CLASSES))
        if hidden == 0
        else nn.Sequential(
            nn.Linear(features.shape[1], hidden),
            nn.ReLU(),
            nn.Linear(hidden, len(CLASSES)),
        )
    ).to(device)

    x = torch.from_numpy(features).to(device)
    y = torch.from_numpy(labels).to(device)
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float32)
    weight = torch.from_numpy(
        np.where(counts > 0, len(labels) / np.maximum(counts, 1) / len(CLASSES), 0.0)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=weight.float())
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(epochs):
        order = torch.randperm(len(y), generator=generator).to(device)
        total = 0.0
        for start in range(0, len(order), 4096):
            batch = order[start : start + 4096]
            optimizer.zero_grad()
            loss = loss_fn(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch)
        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(f"  epoch {epoch:>3}: train loss {total / len(order):.4f}")
    return model.eval()


def score(
    model: nn.Module,
    arm: str,
    features: np.ndarray,
    labels: np.ndarray,
    n_train: int,
    epochs: int,
    device: torch.device,
) -> ProbeResult:
    """Score a fitted classifier on held-out nodes."""
    with torch.no_grad():
        predicted = (
            model(torch.from_numpy(features).to(device)).argmax(dim=1).cpu().numpy()
        )
    c = len(CLASSES)
    confusion = np.bincount(labels * c + predicted, minlength=c * c).reshape(c, c)
    tp = np.diag(confusion)
    denom = confusion.sum(axis=1) + confusion.sum(axis=0)
    return ProbeResult(
        arm=arm,
        macro_f1=float(macro_f1(confusion)),
        accuracy=float(tp.sum() / max(len(labels), 1)),
        per_class_f1={
            name: float(2.0 * tp[i] / denom[i]) if denom[i] else 0.0
            for i, name in enumerate(CLASSES)
        },
        n_train=n_train,
        n_val=len(labels),
        epochs=epochs,
    )


def majority_result(train_labels: np.ndarray, val_labels: np.ndarray) -> ProbeResult:
    """What always predicting the training majority class scores."""
    top = int(np.bincount(train_labels, minlength=len(CLASSES)).argmax())
    c = len(CLASSES)
    confusion = np.zeros((c, c), dtype=np.int64)
    confusion[:, top] = np.bincount(val_labels, minlength=c)
    tp = np.diag(confusion)
    denom = confusion.sum(axis=1) + confusion.sum(axis=0)
    return ProbeResult(
        arm=f"majority ({CLASSES[top]})",
        macro_f1=float(macro_f1(confusion)),
        accuracy=float(tp.sum() / max(len(val_labels), 1)),
        per_class_f1={
            name: float(2.0 * tp[i] / denom[i]) if denom[i] else 0.0
            for i, name in enumerate(CLASSES)
        },
        n_train=len(train_labels),
        n_val=len(val_labels),
        epochs=0,
    )


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-file", type=Path, default=Path("splits/buffered_2560_1000_seed0.json")
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/vegas_best.pt"))
    parser.add_argument("--aoi-root", type=Path, default=Path("data/AOI_2_Vegas"))
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--extract", type=Path, default=osm.DEFAULT_EXTRACT)
    parser.add_argument("--pad-m", type=float, default=250.0)
    parser.add_argument("--spacing", type=float, default=20.0)
    parser.add_argument(
        "--taps",
        nargs="+",
        default=list(TAPS),
        help="feature maps to read, coarsest first",
    )
    parser.add_argument("--hidden", type=int, default=128, help="MLP width; 0 is linear")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--limit", type=int, default=0, help="chips per side; 0 runs every chip"
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/attr_probe.json"))
    args = parser.parse_args()

    started = time.perf_counter()
    device = train.resolve_device(args.device)
    frozen = split.read_frozen(args.split_file)
    limit = args.limit or None
    sides = {
        "train": list(frozen.assignment.train)[:limit],
        "val": list(frozen.assignment.val)[:limit],
    }
    logger.info(
        f"{frozen.name} seed {frozen.params['seed']}: {len(sides['train'])} train, "
        f"{len(sides['val'])} val chips on {device}"
    )

    model = train.load_checkpoint(args.checkpoint, device=str(device))
    store = tapped(model, args.taps)

    source, _ = train.build_source(args.aoi_root, "spacenet", args.resolution)
    chips = {c.image_id: c for c in spacenet.find_chips(args.aoi_root)}
    wanted = [(s, i) for s, ids in sides.items() for i in ids]
    extract = osm.read_extract(
        osm.covering_bounds(
            [spacenet.chip_tile(chips[i].image_path, args.resolution) for _, i in wanted],
            args.pad_m,
        ),
        args.extract,
    )
    way_source = osm.WAY_SOURCE_REGISTRY["pbf"](extract=extract)

    way_table: dict[str, int] = {}
    component_table: dict[tuple[str, int], int] = {}
    columns: dict[str, list[np.ndarray]] = {"train": [], "val": []}
    labels: dict[str, list[np.ndarray]] = {"train": [], "val": []}
    for n, (side, chip_id) in enumerate(wanted, start=1):
        sample = source.load(chip_id)
        placed: ChipNodes = chip_nodes(
            way_source.ways(sample.tile, "drive", args.pad_m),
            chip_id,
            "train",
            args.spacing,
            way_table,
            component_table,
        )
        if not len(placed.klass):
            continue
        columns[side].append(
            chip_features(model, store, args.taps, sample.image, placed.xy, device)
        )
        labels[side].append(placed.klass)
        if n % 100 == 0 or n == len(wanted):
            logger.info(f"{n}/{len(wanted)} chips featurized")

    x_train = np.concatenate(columns["train"])
    x_val = np.concatenate(columns["val"])
    y_train = np.concatenate(labels["train"])
    y_val = np.concatenate(labels["val"])
    logger.info(
        f"{len(y_train)} train and {len(y_val)} val nodes, {x_train.shape[1]} features"
    )

    # Standardized on the training side only; using validation statistics would
    # leak the very thing the spatial split was built to remove.
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    x_train = ((x_train - mean) / scale).astype(np.float32)
    x_val = ((x_val - mean) / scale).astype(np.float32)

    results = [majority_result(y_train, y_val)]
    for arm, hidden, y in (
        ("linear probe", 0, y_train),
        (f"mlp-{args.hidden}", args.hidden, y_train),
        (
            "shuffled labels (control)",
            args.hidden,
            np.random.default_rng(args.seed).permutation(y_train),
        ),
    ):
        logger.info(f"fitting {arm}")
        fitted = fit(x_train, y, hidden, args.epochs, args.lr, args.seed, device)
        results.append(
            score(fitted, arm, x_val, y_val, len(y_train), args.epochs, device)
        )

    logger.info(
        f"{'arm':>26} {'macroF1':>8} {'acc':>7} "
        + " ".join(f"{c[:9]:>10}" for c in CLASSES)
    )
    for r in results:
        logger.info(
            f"{r.arm:>26} {r.macro_f1:>8.4f} {r.accuracy:>7.4f} "
            + " ".join(f"{r.per_class_f1[c]:>10.4f}" for c in CLASSES)
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "params": {
                    "split_file": str(args.split_file),
                    "split_name": frozen.name,
                    "split_digest": frozen.digest,
                    "checkpoint": str(args.checkpoint),
                    "resolution": args.resolution,
                    "extract": str(args.extract),
                    "spacing": args.spacing,
                    "taps": list(args.taps),
                    "hidden": args.hidden,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "seed": args.seed,
                    "device": str(device),
                    "limit": args.limit,
                    "classes": list(CLASSES),
                },
                "elapsed_seconds": time.perf_counter() - started,
                "n_chips": {k: len(v) for k, v in sides.items()},
                "n_features": int(x_train.shape[1]),
                "results": [asdict(r) for r in results],
            },
            indent=2,
        )
    )
    logger.info(f"wrote {args.out} in {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()

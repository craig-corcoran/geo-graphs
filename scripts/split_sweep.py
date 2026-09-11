"""What a spatial split buys, and what it costs.

A random split over chips leaks whole roads: 876 of the 2,453 OSM ways touching
a validation chip also touch a training chip, because SpaceNet chips are
adjacent tiles and a street runs through several. For segmentation that leaks
texture. For attribute inference, where the class is a property of the way and
constant along it, it leaks the answer.

Holding out whole blocks of ground fixes that, and is not free. Larger blocks
cut fewer roads and make the validation set a smaller number of larger, more
correlated places, which widens its error bars and lets its class mix drift away
from the city's. Buffering removes the remaining contact along block edges and
pays in training chips. None of those three costs is a judgement call, so each
candidate is measured on all of them and the frontier is reported rather than a
winner.

Per candidate:

- **Leakage.** Ways touching both sides, by count and by length, and the same
  for the six-class node labels the attribute task would train on.
- **Adjacency.** Distance from each validation chip to the nearest training
  chip. Blocking alone leaves these touching along block boundaries; only a
  buffer opens a gap.
- **Composition.** The validation class mix against the whole AOI's, as the
  largest absolute deviation in class share. A blocked split can hold out a
  neighbourhood that is all residential.
- **Power.** The validation side's macro-F1 noise floor and minimum detectable
  difference, from ``attr_resolvability``. This is what a smaller or more
  spatially correlated validation set costs, in the units an ablation is read
  in.

Reads the same pinned OSM extract as the resolvability run and trains nothing.
"""

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from attr_resolvability import (
    CLASSES,
    ChipNodes,
    bootstrap_weights,
    chip_nodes,
    resolvability,
)
from loguru import logger
from scipy.spatial import KDTree

from geo_graphs import osm, spacenet, split

#: Block sides swept, in metres. A Las Vegas chip is about 320 by 390 m, so
#: 640 m is four chips and 2560 m is roughly sixty.
DEFAULT_BLOCKS: tuple[float, ...] = (640.0, 1280.0, 1920.0, 2560.0)

#: Training margins the buffered candidates drop, in metres, measured centre to
#: centre. Below a chip's own width nothing is dropped.
DEFAULT_BUFFERS: tuple[float, ...] = (500.0, 1000.0)

#: Where the reported noise floor is read: the working node spacing, a plausible
#: per-unit accuracy, and the unit one correctness draw lands on.
REPORT_SPACING = 20.0
REPORT_ACCURACY = 0.7
REPORT_REGIME = "way"


@dataclass(frozen=True, slots=True)
class Candidate:
    """One splitter configuration and how it divided the AOI.

    Attributes:
        name: Short label for the configuration.
        seed: Which draw this is. A block split holds out whole neighbourhoods,
            and with a few dozen blocks the draw matters, so every candidate is
            measured at several.
        splitter: Registry key.
        block_m: Block side, or ``None`` for the random splitter.
        buffer_m: Training margin dropped, or ``None`` when none was.
        n_train: Training chips.
        n_val: Validation chips.
        n_dropped: Chips on neither side.
        val_fraction: Achieved share of chips held out, which overshoots the
            target by up to one block.
        n_blocks: Distinct blocks the AOI was cut into, or ``None`` for random.
    """

    name: str
    seed: int
    splitter: str
    block_m: float | None
    buffer_m: float | None
    n_train: int
    n_val: int
    n_dropped: int
    val_fraction: float
    n_blocks: int | None


@dataclass(frozen=True, slots=True)
class Leakage:
    """How much of the validation side the training side has already seen.

    Attributes:
        name: The candidate.
        seed: Which draw this is.
        n_val_ways: OSM ways touching a validation chip.
        n_shared_ways: How many of those also touch a training chip.
        shared_way_fraction: Their share.
        val_length_m: Validation-side clipped length over the six classes.
        shared_length_m: How much of it lies on a shared way.
        shared_length_fraction: Their share.
        n_val_nodes: Labelled nodes on the validation side.
        shared_node_fraction: Share of them lying on a shared way, which is the
            share of the evaluation set whose answer is memorisable.
    """

    name: str
    seed: int
    n_val_ways: int
    n_shared_ways: int
    shared_way_fraction: float
    val_length_m: float
    shared_length_m: float
    shared_length_fraction: float
    n_val_nodes: int
    shared_node_fraction: float


@dataclass(frozen=True, slots=True)
class Adjacency:
    """How far a validation chip sits from the nearest training chip.

    Attributes:
        name: The candidate.
        seed: Which draw this is.
        min_m: Closest validation chip to any training chip.
        median_m: Median over validation chips.
        p10_m: Tenth percentile, which is where the contact is.
        touching_fraction: Share of validation chips whose nearest training
            chip is within one chip width, so the two share a border.
    """

    name: str
    seed: int
    min_m: float
    median_m: float
    p10_m: float
    touching_fraction: float


@dataclass(frozen=True, slots=True)
class Composition:
    """How far the validation class mix drifts from the whole AOI's.

    Attributes:
        name: The candidate.
        seed: Which draw this is.
        shares: Validation node share per class.
        aoi_shares: The same over every chip.
        max_deviation: Largest absolute difference between them.
        rarest_class: Class with the fewest validation nodes.
        rarest_nodes: Its node count.
        rarest_ways: Its distinct OSM ways, which is the effective sample the
            macro average rests on.
    """

    name: str
    seed: int
    shares: dict[str, float]
    aoi_shares: dict[str, float]
    max_deviation: float
    rarest_class: str
    rarest_nodes: int
    rarest_ways: int


def block_count(placement: split.Placement, block_m: float) -> int:
    """How many blocks the AOI is cut into at one block size."""
    return len(np.unique(split.block_index(placement, block_m)))


def _ways_touching(nodes: Sequence[ChipNodes], ids: set[str]) -> set[int]:
    """Every OSM way carrying a node on one side of a split."""
    return {int(w) for n in nodes if n.chip_id in ids for w in np.unique(n.way)}


def leakage(
    nodes: Sequence[ChipNodes],
    name: str,
    seed: int,
    train_ids: set[str],
    val_ids: set[str],
) -> Leakage:
    """How much of the validation side also appears in training."""
    val_ways = _ways_touching(nodes, val_ids)
    shared = _ways_touching(nodes, train_ids) & val_ways
    val_rows = [n for n in nodes if n.chip_id in val_ids]

    way = np.concatenate([n.way for n in val_rows]) if val_rows else np.zeros(0, int)
    metres = (
        np.concatenate([n.node_length for n in val_rows]) if val_rows else np.zeros(0)
    )
    on_shared = np.isin(way, list(shared))
    return Leakage(
        name=name,
        seed=seed,
        n_val_ways=len(val_ways),
        n_shared_ways=len(shared),
        shared_way_fraction=len(shared) / max(len(val_ways), 1),
        val_length_m=float(metres.sum()),
        shared_length_m=float(metres[on_shared].sum()),
        shared_length_fraction=float(metres[on_shared].sum() / max(metres.sum(), 1e-9)),
        n_val_nodes=len(way),
        shared_node_fraction=float(on_shared.mean()) if len(way) else 0.0,
    )


def adjacency(
    placement: split.Placement,
    name: str,
    seed: int,
    train_ids: set[str],
    val_ids: set[str],
    chip_width_m: float,
) -> Adjacency:
    """Distance from each validation chip to the nearest training chip."""
    at = {
        i: (x, y) for i, x, y in zip(placement.ids, placement.x, placement.y, strict=True)
    }
    train = np.array([at[i] for i in sorted(train_ids)])
    val = np.array([at[i] for i in sorted(val_ids)])
    distance, _ = KDTree(train).query(val, k=1)
    return Adjacency(
        name=name,
        seed=seed,
        min_m=float(distance.min()),
        median_m=float(np.median(distance)),
        p10_m=float(np.percentile(distance, 10)),
        touching_fraction=float((distance <= chip_width_m).mean()),
    )


def composition(
    nodes: Sequence[ChipNodes], name: str, seed: int, val_ids: set[str]
) -> Composition:
    """The validation class mix against the whole AOI's."""

    def shares(rows: Sequence[ChipNodes]) -> np.ndarray:
        klass = np.concatenate([r.klass for r in rows]) if rows else np.zeros(0, np.int64)
        return np.bincount(klass, minlength=len(CLASSES)) / max(len(klass), 1)

    val_rows = [n for n in nodes if n.chip_id in val_ids]
    val_shares = shares(val_rows)
    aoi_shares = shares(list(nodes))

    klass = np.concatenate([r.klass for r in val_rows])
    way = np.concatenate([r.way for r in val_rows])
    counts = np.bincount(klass, minlength=len(CLASSES))
    rarest = int(counts.argmin())
    return Composition(
        name=name,
        seed=seed,
        shares={c: float(s) for c, s in zip(CLASSES, val_shares, strict=True)},
        aoi_shares={c: float(s) for c, s in zip(CLASSES, aoi_shares, strict=True)},
        max_deviation=float(np.abs(val_shares - aoi_shares).max()),
        rarest_class=CLASSES[rarest],
        rarest_nodes=int(counts[rarest]),
        rarest_ways=len(np.unique(way[klass == rarest])),
    )


def candidates(blocks: Sequence[float], buffers: Sequence[float]) -> list[tuple]:
    """Every splitter configuration to measure, as ``(name, key, kwargs)``."""
    out: list[tuple] = [("random", "random", {})]
    out.extend((f"blocked-{b:.0f}", "blocked", {"block_m": b}) for b in blocks)
    out.extend(
        (f"buffered-{b:.0f}+{m:.0f}", "buffered", {"block_m": b, "buffer_m": m})
        for b in blocks
        for m in buffers
    )
    return out


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi-root", type=Path, default=Path("data/AOI_2_Vegas"))
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--extract", type=Path, default=osm.DEFAULT_EXTRACT)
    parser.add_argument(
        "--network-type", default="drive", choices=sorted(osm.NETWORK_FILTERS)
    )
    parser.add_argument("--pad-m", type=float, default=250.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--blocks",
        type=float,
        nargs="+",
        default=list(DEFAULT_BLOCKS),
        help="block sides to sweep, in metres",
    )
    parser.add_argument(
        "--buffers",
        type=float,
        nargs="+",
        default=list(DEFAULT_BUFFERS),
        help="training margins the buffered candidates drop, in metres",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=REPORT_SPACING,
        help="node spacing the counts and the noise floor are read at",
    )
    parser.add_argument("--accuracy", type=float, default=REPORT_ACCURACY)
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=["node", "way"],
        help="which unit carries one correctness draw when pricing the noise floor",
    )
    parser.add_argument(
        "--rhos",
        type=float,
        nargs="+",
        default=[0.0, 0.8, 0.9],
        help="between-arm correlations the detectable difference is tabulated across",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help=(
            "draws per candidate. A block split holds out whole neighbourhoods, "
            "and at 2560 m the AOI is only 35 blocks, so one draw says little "
            "about what the configuration does"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="chips to use; 0 runs every chip, which is what a reported number needs",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/split_sweep.json"))
    args = parser.parse_args()

    started = time.perf_counter()
    rng = np.random.default_rng(min(args.seeds))
    chips = spacenet.find_chips(args.aoi_root)[: args.limit or None]
    tile_by_id = {
        c.image_id: spacenet.chip_tile(c.image_path, args.resolution) for c in chips
    }
    placement = split.placement_from_tiles(list(tile_by_id), list(tile_by_id.values()))
    chip_width = float(np.median([t.width * t.resolution for t in tile_by_id.values()]))
    logger.info(f"{len(chips)} chips placed, median width {chip_width:.0f} m")

    extract_md5 = hashlib.md5(args.extract.read_bytes()).hexdigest()
    extract = osm.read_extract(
        osm.covering_bounds(list(tile_by_id.values()), args.pad_m), args.extract
    )
    way_source = osm.WAY_SOURCE_REGISTRY["pbf"](extract=extract)

    # One geometry pass serves every candidate: the nodes do not move when the
    # split does, only which side they are counted on. `split` is filled in per
    # candidate below, so this pass labels everything "train" as a placeholder.
    way_table: dict[str, int] = {}
    component_table: dict[tuple[str, int], int] = {}
    nodes: list[ChipNodes] = []
    for n, (chip_id, tile) in enumerate(tile_by_id.items(), start=1):
        nodes.append(
            chip_nodes(
                way_source.ways(tile, args.network_type, args.pad_m),
                chip_id,
                "train",
                args.spacing,
                way_table,
                component_table,
            )
        )
        if n % 200 == 0 or n == len(tile_by_id):
            logger.info(f"{n}/{len(tile_by_id)} chips, {len(way_table)} OSM ways")

    rows: list[Candidate] = []
    leaks: list[Leakage] = []
    gaps: list[Adjacency] = []
    mixes: list[Composition] = []
    powers: list[dict] = []
    for name, key, kwargs in candidates(args.blocks, args.buffers):
        for seed in args.seeds:
            assignment = split.SPLIT_REGISTRY[key](**kwargs).split(
                placement, args.val_fraction, seed
            )
            train_ids, val_ids = set(assignment.train), set(assignment.val)
            rows.append(
                Candidate(
                    name=name,
                    seed=seed,
                    splitter=key,
                    block_m=kwargs.get("block_m"),
                    buffer_m=kwargs.get("buffer_m"),
                    n_train=len(assignment.train),
                    n_val=len(assignment.val),
                    n_dropped=len(assignment.dropped),
                    val_fraction=len(assignment.val) / len(placement.ids),
                    n_blocks=(
                        None
                        if key == "random"
                        else block_count(placement, kwargs["block_m"])
                    ),
                )
            )
            leaks.append(leakage(nodes, name, seed, train_ids, val_ids))
            gaps.append(adjacency(placement, name, seed, train_ids, val_ids, chip_width))
            mixes.append(composition(nodes, name, seed, val_ids))

            # The noise floor is a property of the validation side alone, so the
            # nodes are relabelled onto it and the resolvability machinery is
            # asked the same question it was asked of the shipped split.
            relabelled = [
                ChipNodes(
                    chip_id=n.chip_id,
                    split="val" if n.chip_id in val_ids else "train",
                    spacing=n.spacing,
                    klass=n.klass,
                    way=n.way,
                    xy=n.xy,
                    node_length=n.node_length,
                    component=n.component,
                    n_other=n.n_other,
                    length_by_class=n.length_by_class,
                    length_m=n.length_m,
                )
                for n in nodes
            ]
            weights = bootstrap_weights(len(val_ids), args.bootstrap, rng)
            powers.extend(
                asdict(
                    resolvability(
                        relabelled,
                        "val",
                        args.spacing,
                        args.accuracy,
                        regime,
                        tuple(args.rhos),
                        weights,
                        args.draws,
                        args.alpha,
                        args.power,
                        rng,
                    )
                )
                | {"name": name, "seed": seed}
                for regime in args.regimes
            )
        logger.info(f"{name}: {len(args.seeds)} draws measured")

    rho_keys = [f"{rho:g}" for rho in args.rhos]

    def spread(values: Sequence[float]) -> str:
        """Mean over draws, with the range when the draws disagree."""
        array = np.asarray(values, dtype=float)
        return f"{array.mean():.4f} [{array.min():.4f},{array.max():.4f}]"

    names = [name for name, _, _ in candidates(args.blocks, args.buffers)]
    logger.info(
        f"{'candidate':>20} {'train':>6} {'val':>5} {'drop':>5} {'blocks':>7} "
        f"{'sharedNodes':>26} {'gapP10':>7} {'touch':>6} {'maxDev':>26} {'rareWays':>10}"
    )
    for name in names:
        draws = [r for r in rows if r.name == name]
        leak = [x for x in leaks if x.name == name]
        gap = [x for x in gaps if x.name == name]
        mix = [x for x in mixes if x.name == name]
        logger.info(
            f"{name:>20} "
            f"{np.mean([r.n_train for r in draws]):>6.0f} "
            f"{np.mean([r.n_val for r in draws]):>5.0f} "
            f"{np.mean([r.n_dropped for r in draws]):>5.0f} "
            f"{(draws[0].n_blocks or 0):>7} "
            f"{spread([x.shared_node_fraction for x in leak]):>26} "
            f"{np.mean([x.p10_m for x in gap]):>7.0f} "
            f"{np.mean([x.touching_fraction for x in gap]):>6.3f} "
            f"{spread([x.max_deviation for x in mix]):>26} "
            f"{np.mean([x.rarest_ways for x in mix]):>10.0f}"
        )

    header = " ".join(f"{'mdd@' + k:>9}" for k in rho_keys)
    logger.info(
        f"{'candidate':>20} {'regime':>8} {'nodes':>7} {'units':>6} {'rare':>5} "
        f"{'sem':>26} {header}"
    )
    for name in names:
        for regime in args.regimes:
            cells = [p for p in powers if p["name"] == name and p["regime"] == regime]
            mdds = " ".join(
                f"{np.mean([c['min_detectable'][k] for c in cells]):>9.4f}"
                for k in rho_keys
            )
            logger.info(
                f"{name:>20} {regime:>8} "
                f"{np.mean([c['n_nodes'] for c in cells]):>7.0f} "
                f"{np.mean([c['n_units'] for c in cells]):>6.0f} "
                f"{np.mean([c['min_class_units'] for c in cells]):>5.0f} "
                f"{spread([c['macro_f1_sem'] for c in cells]):>26} {mdds}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "params": {
                    "resolution": args.resolution,
                    "extract": str(args.extract),
                    "extract_md5": extract_md5,
                    "network_type": args.network_type,
                    "pad_m": args.pad_m,
                    "val_fraction": args.val_fraction,
                    "blocks": list(args.blocks),
                    "buffers": list(args.buffers),
                    "spacing": args.spacing,
                    "accuracy": args.accuracy,
                    "regimes": list(args.regimes),
                    "rhos": list(args.rhos),
                    "bootstrap": args.bootstrap,
                    "draws": args.draws,
                    "alpha": args.alpha,
                    "power": args.power,
                    "seeds": list(args.seeds),
                    "limit": args.limit,
                    "classes": list(CLASSES),
                },
                "elapsed_seconds": time.perf_counter() - started,
                "n_chips": len(chips),
                "chip_width_m": chip_width,
                "n_osm_ways": len(way_table),
                "candidates": [asdict(r) for r in rows],
                "leakage": [asdict(x) for x in leaks],
                "adjacency": [asdict(x) for x in gaps],
                "composition": [asdict(x) for x in mixes],
                "power": powers,
            },
            indent=2,
        )
    )
    logger.info(f"wrote {args.out} in {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()

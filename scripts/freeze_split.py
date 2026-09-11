"""Draw one split and record it, so everything downstream agrees on the same one.

A split is reproducible from its splitter, its seed and the list of chips it was
drawn over. The chip list lives in an untracked data directory, so a chip added
or removed silently produces a different split under the same seed. Writing the
assignment out, with a hash of what produced it, is what makes the frozen split
something a later run can be checked against rather than merely re-derived.

The default is the configuration ``scripts/split_sweep.py`` measured and left on
the frontier: 2560 m blocks with a 1000 m training margin, at seed 0. It leaks
1.3% of validation nodes against the shipped random split's 45.4%, and pays 178
training chips for that.
"""

import argparse
from pathlib import Path

from loguru import logger

from geo_graphs import spacenet, split


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi-root", type=Path, default=Path("data/AOI_2_Vegas"))
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument(
        "--splitter", default="buffered", choices=sorted(split.SPLIT_REGISTRY)
    )
    parser.add_argument("--block-m", type=float, default=2560.0)
    parser.add_argument("--buffer-m", type=float, default=1000.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--name",
        default=None,
        help="label recorded in the file; defaults to the sweep's naming",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("splits/buffered_2560_1000_seed0.json")
    )
    args = parser.parse_args()

    chips = spacenet.find_chips(args.aoi_root)
    tile_seq = [spacenet.chip_tile(c.image_path, args.resolution) for c in chips]
    ids = [c.image_id for c in chips]
    placement = split.placement_from_tiles(ids, tile_seq)

    kwargs: dict[str, float] = {}
    if args.splitter in ("blocked", "buffered"):
        kwargs["block_m"] = args.block_m
    if args.splitter == "buffered":
        kwargs["buffer_m"] = args.buffer_m

    name = args.name or (
        args.splitter
        if not kwargs
        else f"{args.splitter}-{args.block_m:.0f}"
        + (f"+{args.buffer_m:.0f}" if args.splitter == "buffered" else "")
    )
    frozen = split.freeze(
        name=name,
        splitter=args.splitter,
        kwargs=kwargs,
        placement=placement,
        val_fraction=args.val_fraction,
        seed=args.seed,
        source={
            "aoi_root": str(args.aoi_root),
            "resolution": args.resolution,
            "n_chips": len(ids),
            "chips_sha256": split.digest(ids),
        },
    )
    split.write_frozen(args.out, frozen)
    logger.info(
        f"{frozen.name} seed {args.seed}: {len(frozen.assignment.train)} train, "
        f"{len(frozen.assignment.val)} val, {len(frozen.assignment.dropped)} dropped "
        f"of {len(ids)} chips"
    )
    logger.info(f"digest {frozen.digest}")
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()

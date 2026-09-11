"""Splitters: whether a held-out block stays whole, and whether a buffer buffers."""

import json
from pathlib import Path

import numpy as np
import pytest
from pyproj import CRS

from geo_graphs import spacenet, split, tiles

UTM = CRS.from_epsg(32611)


def lattice(cols: int = 36, rows: int = 28, pitch: float = 320.0) -> split.Placement:
    """A regular grid of chips, which is roughly how SpaceNet tiles an AOI."""
    ids, tile_seq = [], []
    for index in range(cols * rows):
        ids.append(f"img{index}")
        tile_seq.append(
            tiles.Tile(
                crs=UTM,
                x_min=float(index % cols) * pitch,
                y_max=float(index // cols) * pitch,
                height=int(pitch),
                width=int(pitch),
                resolution=1.0,
            )
        )
    return split.placement_from_tiles(ids, tile_seq)


def side_of(assignment: split.Assignment) -> dict[str, str]:
    return (
        {i: "train" for i in assignment.train}
        | {i: "val" for i in assignment.val}
        | {i: "dropped" for i in assignment.dropped}
    )


def test_random_splitter_holds_out_the_requested_fraction():
    placement = lattice(cols=1, rows=100)
    assignment = split.SPLIT_REGISTRY["random"]().split(placement, 0.2, 0)

    assert len(assignment.val) == 20
    assert len(assignment.train) == 80
    assert not set(assignment.train) & set(assignment.val)
    assert set(assignment.train) | set(assignment.val) == set(placement.ids)
    assert assignment.dropped == ()


def test_random_splitter_shuffles_rather_than_taking_a_contiguous_tail():
    """SpaceNet chip numbers run along the ground.

    A contiguous tail would be one neighbourhood held out, not a sample of the
    city, and would flatter or punish the model depending on what is there.
    """
    placement = lattice(cols=1, rows=100)
    assignment = split.SPLIT_REGISTRY["random"]().split(placement, 0.2, 0)
    assert assignment.val != placement.ids[:20]


def test_random_splitter_returns_the_shuffled_order_not_the_input_order():
    """Not cosmetic: `build_dataset` draws crop windows per sample in id order
    from one seeded stream, so a permutation is a different training run."""
    placement = lattice(cols=1, rows=100)
    assignment = split.SPLIT_REGISTRY["random"]().split(placement, 0.2, 0)

    in_input_order = tuple(i for i in placement.ids if i in set(assignment.train))
    assert assignment.train != in_input_order


def test_random_splitter_always_holds_out_at_least_one():
    placement = lattice(cols=1, rows=3)
    assignment = split.SPLIT_REGISTRY["random"]().split(placement, 0.01, 0)

    assert len(assignment.val) == 1
    assert len(assignment.train) == 2


def test_placement_takes_the_centre_not_the_corner():
    placement = split.placement_from_tiles(
        ("a",), [tiles.Tile(UTM, x_min=1000.0, y_max=5000.0, height=400, width=300)]
    )
    assert placement.x == pytest.approx([1150.0])
    assert placement.y == pytest.approx([4800.0])


def test_placement_refuses_mixed_projections():
    """Block and buffer distances are metres in one grid; two grids is nonsense."""
    mixed = [
        tiles.Tile(UTM, 0.0, 0.0, 10, 10),
        tiles.Tile(CRS.from_epsg(32612), 0.0, 0.0, 10, 10),
    ]
    with pytest.raises(ValueError, match="several CRSs"):
        split.placement_from_tiles(("a", "b"), mixed)


def test_block_index_cuts_at_the_block_size():
    placement = split.placement_from_tiles(
        ("a", "b", "c"),
        [
            tiles.Tile(UTM, 0.0, 0.0, 2, 2),
            tiles.Tile(UTM, 500.0, 0.0, 2, 2),
            tiles.Tile(UTM, 1500.0, 0.0, 2, 2),
        ],
    )
    blocks = split.block_index(placement, block_m=1000.0)

    assert blocks[0] == blocks[1]
    assert blocks[2] != blocks[0]


def test_block_index_refuses_a_nonpositive_block():
    with pytest.raises(ValueError, match="must be positive"):
        split.block_index(lattice(), block_m=0.0)


def test_blocked_splitter_never_cuts_a_block():
    """The whole point: a block is on one side or the other, never both."""
    placement = lattice()
    assignment = split.SPLIT_REGISTRY["blocked"](block_m=1280.0).split(placement, 0.2, 0)
    blocks = split.block_index(placement, 1280.0)
    side = side_of(assignment)

    by_block: dict[int, set[str]] = {}
    for sample_id, block in zip(placement.ids, blocks, strict=True):
        by_block.setdefault(int(block), set()).add(side[sample_id])
    assert all(len(sides) == 1 for sides in by_block.values())


def test_blocked_splitter_lands_near_the_requested_fraction():
    placement = lattice()
    assignment = split.SPLIT_REGISTRY["blocked"](block_m=1280.0).split(placement, 0.2, 0)

    achieved = len(assignment.val) / len(placement.ids)
    assert 0.2 <= achieved <= 0.2 + 16 / len(placement.ids)  # one 4x4 block of overshoot


def test_blocked_splitter_keeps_every_sample():
    placement = lattice()
    assignment = split.SPLIT_REGISTRY["blocked"](block_m=1280.0).split(placement, 0.2, 0)

    assert set(assignment.train) | set(assignment.val) == set(placement.ids)
    assert not set(assignment.train) & set(assignment.val)
    assert assignment.dropped == ()


def test_buffered_splitter_opens_a_real_gap():
    placement = lattice()
    assignment = split.SPLIT_REGISTRY["buffered"](block_m=1280.0, buffer_m=700.0).split(
        placement, 0.2, 0
    )
    at = dict(
        zip(placement.ids, np.column_stack([placement.x, placement.y]), strict=True)
    )
    val = np.array([at[i] for i in assignment.val])

    for sample_id in assignment.train:
        gap = np.hypot(*(val - at[sample_id]).T).min()
        assert gap >= 700.0


def test_buffered_splitter_takes_the_margin_out_of_training():
    """Validation keeps its size; evaluation noise is the binding constraint."""
    placement = lattice()
    plain = split.SPLIT_REGISTRY["blocked"](block_m=1280.0).split(placement, 0.2, 0)
    buffered = split.SPLIT_REGISTRY["buffered"](block_m=1280.0, buffer_m=700.0).split(
        placement, 0.2, 0
    )

    assert buffered.val == plain.val
    assert len(buffered.train) < len(plain.train)
    assert len(buffered.dropped) == len(plain.train) - len(buffered.train)


def test_a_zero_buffer_drops_nothing():
    placement = lattice()
    buffered = split.SPLIT_REGISTRY["buffered"](block_m=1280.0, buffer_m=0.0).split(
        placement, 0.2, 0
    )
    assert buffered.dropped == ()


@pytest.mark.parametrize("key", sorted(split.SPLIT_REGISTRY))
def test_splitters_are_reproducible_and_seed_dependent(key):
    placement = lattice()
    first = split.SPLIT_REGISTRY[key]().split(placement, 0.2, seed=0)
    again = split.SPLIT_REGISTRY[key]().split(placement, 0.2, seed=0)
    other = split.SPLIT_REGISTRY[key]().split(placement, 0.2, seed=1)

    assert first == again
    assert first != other


def test_registry_holds_factories_that_satisfy_the_protocol():
    assert sorted(split.SPLIT_REGISTRY) == ["blocked", "buffered", "random"]
    first, second = split.SPLIT_REGISTRY["blocked"](), split.SPLIT_REGISTRY["blocked"]()

    assert first is not second
    assert isinstance(first, split.Splitter)
    assert isinstance(first, split.BlockedSplitter)


def test_hold_out_groups_always_takes_at_least_one():
    group = np.arange(100)
    assert split.hold_out_groups(group, val_fraction=0.0001, seed=0).sum() == 1


def test_a_block_smaller_than_the_spacing_degenerates_to_per_sample():
    """Below the chip pitch every chip is its own block, which is a random split."""
    placement = lattice()
    assignment = split.SPLIT_REGISTRY["blocked"](block_m=1.0).split(placement, 0.2, 0)

    assert len(assignment.val) == round(len(placement.ids) * 0.2)


FROZEN = Path("splits/buffered_2560_1000_seed0.json")
needs_frozen = pytest.mark.skipif(
    not FROZEN.is_file(), reason="the frozen split has not been drawn"
)
needs_vegas = pytest.mark.skipif(
    not Path("data/AOI_2_Vegas").is_dir(), reason="Vegas AOI not extracted"
)


def frozen_of(placement: split.Placement, tmp_path: Path) -> Path:
    """Freeze a blocked split over a synthetic lattice and write it out."""
    path = tmp_path / "frozen.json"
    split.write_frozen(
        path,
        split.freeze(
            name="blocked-1280",
            splitter="blocked",
            kwargs={"block_m": 1280.0},
            placement=placement,
            val_fraction=0.2,
            seed=0,
            source={"aoi_root": "synthetic", "n_chips": len(placement.ids)},
        ),
    )
    return path


def test_a_frozen_split_round_trips(tmp_path):
    placement = lattice()
    path = frozen_of(placement, tmp_path)
    frozen = split.read_frozen(path)

    assert frozen.name == "blocked-1280"
    assert frozen.splitter == "blocked"
    assert frozen.params["block_m"] == 1280.0
    assert frozen.params["seed"] == 0
    assert set(frozen.assignment.train) | set(frozen.assignment.val) == set(placement.ids)


def test_a_frozen_split_is_reproduced_by_its_own_recipe(tmp_path):
    """What freezing is for: the recipe and the record have to still agree."""
    placement = lattice()
    frozen = split.read_frozen(frozen_of(placement, tmp_path))

    assert split.rebuild(frozen, placement) == frozen.assignment


def test_rebuilding_over_a_changed_sample_list_disagrees(tmp_path):
    """One chip more and the same seed draws a different split; that is the point."""
    frozen = split.read_frozen(frozen_of(lattice(), tmp_path))

    assert split.rebuild(frozen, lattice(rows=29)) != frozen.assignment


def test_a_hand_edited_frozen_split_is_refused(tmp_path):
    path = frozen_of(lattice(), tmp_path)
    raw = json.loads(path.read_text())
    raw["val"].append(raw["train"].pop())
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="edited by hand"):
        split.read_frozen(path)


def test_the_digest_covers_which_side_an_id_is_on(tmp_path):
    """A digest over the union would miss an id moving between sides."""
    a = split.Assignment(train=("a", "b"), val=("c",), dropped=())
    b = split.Assignment(train=("a",), val=("b", "c"), dropped=())
    assert split._assignment_digest(a) != split._assignment_digest(b)


def test_digest_depends_on_order():
    assert split.digest(["a", "b"]) != split.digest(["b", "a"])


@needs_frozen
@needs_vegas
def test_the_committed_split_still_matches_the_data_on_disk():
    """The guard the freeze exists for.

    Failing here means the chip list moved: a chip added, removed or reordered
    under `data/AOI_2_Vegas`. The frozen ids stay authoritative; this test says
    the recipe no longer reproduces them.
    """
    frozen = split.read_frozen(FROZEN)
    chips = spacenet.find_chips(Path(str(frozen.source["aoi_root"])))
    ids = [c.image_id for c in chips]

    assert split.digest(ids) == frozen.source["chips_sha256"]
    assert len(ids) == frozen.source["n_chips"]

    placement = split.placement_from_tiles(
        ids,
        [
            spacenet.chip_tile(c.image_path, float(frozen.source["resolution"]))
            for c in chips
        ],
    )
    assert split.rebuild(frozen, placement) == frozen.assignment

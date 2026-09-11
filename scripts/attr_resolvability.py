"""Can a road-class ablation resolve its arms before any model is trained?

The gap-closing line died on this question rather than on the idea: the effect
was real at the link level and smaller than the noise at the level the metric
reported. So the attribute-inference project answers it first, at a cost of
minutes, using only counting and simulation.

What is measured
----------------
Nodes are placed along the OSM ``drive`` network inside every SpaceNet chip and
labelled with the way's ``highway`` class. That is the whole supervision signal
the project would have, so the size and shape of it bounds what any model
trained on it could demonstrate. Three things come out:

1. **Counts.** Nodes, source ways and same-class components per class per split.
   The rarest class is what a macro average is bottlenecked by, and no amount of
   node density creates more ways than OSM drew.
2. **The noise floor.** A model at a stated per-unit accuracy is simulated, its
   macro-F1 is bootstrapped over chips, and the standard error of that statistic
   is reported. This is the *evaluation* noise, and it does not depend on
   training anything.
3. **The smallest detectable difference.** From the standard error, the effect
   size a two-arm comparison would need at 80% power and alpha 0.05.

What is assumed
---------------
The simulated model is right on a unit with probability ``accuracy`` and
otherwise predicts a class drawn from the label prior, excluding the true one.
Prior-proportional confusion is the neutral choice: a hierarchy-aware confusion
(``tertiary`` mistaken for ``secondary`` more often than for ``motorway``) is
more realistic and would invent structure this script has no evidence for. It
would *lower* macro-F1 and raise the noise floor, so the numbers here are the
optimistic end.

The correlation *regime* is the thing the answer turns on, so it is swept rather
than assumed. Errors drawn per node treat a 400 m street sampled every 20 m as
20 independent observations; errors drawn per way treat it as one. Reality is
between them, and the two bracket the answer. Halving the node spacing doubles
the node count and leaves the way count untouched, so the two regimes disagree
about whether spacing buys anything at all.

Pairing is stated, not measured. Two arms scored on the same chips have
correlated errors, which shrinks the standard error of their difference by
``sqrt(2(1 - rho))``. ``rho`` is a property of the two models, which do not
exist yet, so the detectable difference is tabulated across a range of it.

This script trains nothing and scores no model. Its numbers are about the
dataset and the metric, not about anything learnable from imagery.
"""

import argparse
import hashlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger

from geo_graphs import geograph, osm, spacenet

Split = Literal["train", "val"]

#: ``highway`` values folded into each reported class.
#:
#: Link roads join their parent: a motorway ramp is imagery of a motorway, and
#: splitting them would add three classes with a few hundred metres each.
#: ``living_street`` joins ``residential`` for the same reason. Everything the
#: ``drive`` filter admits and this map omits lands in :data:`OTHER` and is
#: reported rather than silently dropped.
CLASS_MAP: dict[str, str] = {
    "motorway": "motorway",
    "motorway_link": "motorway",
    "trunk": "trunk",
    "trunk_link": "trunk",
    "primary": "primary",
    "primary_link": "primary",
    "secondary": "secondary",
    "secondary_link": "secondary",
    "tertiary": "tertiary",
    "tertiary_link": "tertiary",
    "residential": "residential",
    "living_street": "residential",
    "unclassified": "unclassified",
}

#: The class vocabulary, in descending functional order.
#:
#: ``trunk`` is in the map and not here: the Las Vegas AOI has none, and a class
#: with no support would enter a macro average as a constant zero. A run over an
#: AOI that has trunk roads reports them under :data:`OTHER` and the vocabulary
#: needs revisiting.
CLASSES: tuple[str, ...] = (
    "motorway",
    "primary",
    "secondary",
    "tertiary",
    "residential",
    "unclassified",
)

#: Where a drivable class outside :data:`CLASSES` is counted.
OTHER = "other"

#: Node spacings swept, in metres. 20 m is the working default; the sweep is
#: what says whether that choice matters.
DEFAULT_SPACINGS: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0)

#: Per-unit accuracies the simulated model is run at.
DEFAULT_ACCURACIES: tuple[float, ...] = (0.5, 0.7, 0.9)

#: Between-arm correlations the detectable difference is tabulated across.
#: 0 is two unrelated models; 0.9 is two seeds of one architecture.
DEFAULT_RHOS: tuple[float, ...] = (0.0, 0.5, 0.8, 0.9)

#: Correlation regimes for the simulated errors. See the module docstring.
REGIMES: tuple[str, ...] = ("node", "way", "component")

#: Endpoints closer than this share a junction, in pixels. The pieces come out
#: of one dissolve, so their shared endpoints are equal and this is only guarding
#: float formatting.
_JOIN_TOL = 1e-3


@dataclass(frozen=True, slots=True)
class ChipNodes:
    """Every labelled node on one chip, at one spacing.

    Attributes:
        chip_id: The chip these nodes came from.
        split: Which side of the checkpoint's split that chip is on.
        spacing: Node spacing in metres the nodes were placed at.
        klass: ``(N,)`` index into :data:`CLASSES` per node.
        way: ``(N,)`` OSM way id per node, as an index into a run-wide table.
        component: ``(N,)`` same-class connected component per node, as an
            index into a run-wide table. Components are chip-local: pieces are
            clipped at the chip edge, so a street crossing two chips is two
            components.
        n_other: Nodes that would have been placed on a drivable class outside
            :data:`CLASSES`, and were not.
        length_by_class: ``(len(CLASSES),)`` clipped length per class, in
            metres. Length is how a road network's class mix is normally
            reported, and it is what the node counts are derived from.
        length_m: Clipped length on this chip over :data:`CLASSES`, in metres.
    """

    chip_id: str
    split: Split
    spacing: float
    klass: np.ndarray
    way: np.ndarray
    component: np.ndarray
    n_other: int
    length_by_class: np.ndarray
    length_m: float


@dataclass(frozen=True, slots=True)
class ClassCount:
    """One class's share of one split at one spacing.

    Attributes:
        split: Which split these counts come from.
        spacing: Node spacing in metres.
        klass: The class, or :data:`OTHER`.
        n_nodes: Nodes carrying this label.
        n_ways: Distinct OSM ways they lie on.
        n_components: Distinct same-class components they lie on.
        length_m: Clipped length of this class, in metres.
        node_fraction: Share of the split's nodes.
    """

    split: Split
    spacing: float
    klass: str
    n_nodes: int
    n_ways: int
    n_components: int
    length_m: float
    node_fraction: float


@dataclass(frozen=True, slots=True)
class ClusterSizes:
    """How many nodes share a correlated error, under one unit definition.

    The design effect a correlated regime pays is roughly the mean cluster size,
    so this is what separates "20 m spacing gives 70,000 observations" from "it
    gives 4,000 observations sampled 17 times each".

    Attributes:
        split: Which split these clusters come from.
        spacing: Node spacing in metres.
        unit: ``"way"`` or ``"component"``.
        n_clusters: Distinct clusters.
        mean_nodes: Mean nodes per cluster.
        median_nodes: Median nodes per cluster.
        p90_nodes: 90th percentile nodes per cluster.
        max_nodes: Largest cluster, in nodes.
    """

    split: Split
    spacing: float
    unit: str
    n_clusters: int
    mean_nodes: float
    median_nodes: float
    p90_nodes: float
    max_nodes: int


@dataclass(frozen=True, slots=True)
class Resolvability:
    """The noise floor and the detectable difference for one configuration.

    Attributes:
        split: Which split the comparison would be scored on.
        spacing: Node spacing in metres.
        accuracy: Per-unit accuracy the simulated model was run at.
        regime: Which unit carried one correctness draw.
        n_nodes: Nodes scored.
        n_units: Units the errors were drawn on, which is the effective sample.
        macro_f1: Macro-F1 on the whole split, averaged over simulation draws.
        macro_f1_spread: Standard deviation of that macro-F1 across draws. This
            is variation between simulated *models*, not evaluation noise, and
            it is reported only so a reader can see how much one draw would
            have misled them.
        macro_f1_sem: Bootstrap standard error of macro-F1 over chips, averaged
            over simulation draws. This is the evaluation noise the detectable
            difference is computed from.
        sem_spread: Standard deviation of that standard error across draws.
        min_detectable: Smallest macro-F1 difference a two-arm comparison could
            call at 80% power and alpha 0.05, per between-arm correlation. Keyed
            by ``rho`` rendered as a string, because JSON keys are strings.
        min_class_nodes: Nodes in the rarest class.
        min_class_units: Units in the rarest class, which is the binding
            constraint on that class's F1 and so on the macro average.
    """

    split: Split
    spacing: float
    accuracy: float
    regime: str
    n_nodes: int
    n_units: int
    macro_f1: float
    macro_f1_spread: float
    macro_f1_sem: float
    sem_spread: float
    min_detectable: dict[str, float]
    min_class_nodes: int
    min_class_units: int


@dataclass(frozen=True, slots=True)
class SplitBaseline:
    """What a model that has learned nothing already scores.

    A task whose trivial baseline is near the achievable ceiling has nothing to
    ablate, whatever the noise floor says.

    Attributes:
        split: Which split.
        spacing: Node spacing in metres.
        n_chips: Chips contributing nodes.
        n_chips_empty: Chips with no drivable OSM way at all.
        n_nodes: Nodes in the split.
        majority_class: The most common class.
        majority_share: Its share of the nodes, which is the accuracy of always
            predicting it.
        majority_macro_f1: Macro-F1 of always predicting it.
        length_m: Clipped ``drive`` length, in metres.
    """

    split: Split
    spacing: float
    n_chips: int
    n_chips_empty: int
    n_nodes: int
    majority_class: str
    majority_share: float
    majority_macro_f1: float
    length_m: float


def _z(p: float) -> float:
    """The standard normal quantile at ``p``, by bisection on ``erf``."""
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def detectable_difference(sem: float, rho: float, alpha: float, power: float) -> float:
    """The smallest true difference a two-arm comparison would call.

    Args:
        sem: Standard error of one arm's score.
        rho: Correlation between the two arms' scores. Zero is an unpaired
            comparison; higher is the shared evaluation set helping.
        alpha: Two-sided significance level.
        power: Probability of calling a difference of this size.

    Returns:
        The difference in the units ``sem`` is in.
    """
    return (_z(1.0 - alpha / 2.0) + _z(power)) * sem * math.sqrt(2.0 * (1.0 - rho))


def _same_class_components(
    ways: Sequence[osm.TaggedWay], klass: np.ndarray
) -> np.ndarray:
    """Group pieces that touch and share a class into one component.

    Args:
        ways: Noded, clipped pieces on one chip.
        klass: ``(len(ways),)`` class index per piece.

    Returns:
        ``(len(ways),)`` component label, as a piece index that represents it.
    """
    parent = list(range(len(ways)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen: dict[tuple[int, int, int], int] = {}
    for i, way in enumerate(ways):
        for end in (way.pts[0], way.pts[-1]):
            key = (
                int(klass[i]),
                round(float(end[0]) / _JOIN_TOL),
                round(float(end[1]) / _JOIN_TOL),
            )
            other = seen.setdefault(key, i)
            if (a := find(i)) != (b := find(other)):
                parent[b] = a
    return np.array([find(i) for i in range(len(ways))], dtype=np.int64)


def chip_nodes(
    ways: Sequence[osm.TaggedWay],
    chip_id: str,
    split: Split,
    spacing: float,
    way_table: dict[str, int],
    component_table: dict[tuple[str, int], int],
) -> ChipNodes:
    """Place labelled nodes along one chip's ways.

    A piece shorter than the spacing still gets one node rather than none:
    noding cuts a street at every junction along it, so dropping short pieces
    would drop the ones nearest intersections, which is where the classes
    differ most.

    Args:
        ways: Noded, clipped pieces on this chip.
        chip_id: The chip.
        split: Which split it is on.
        spacing: Target metres between nodes.
        way_table: Run-wide OSM way id to index, extended in place.
        component_table: Run-wide ``(chip, local component)`` to index,
            extended in place.

    Returns:
        The chip's nodes.
    """
    classes = [CLASS_MAP.get(w.highway, OTHER) for w in ways]
    index = {name: i for i, name in enumerate(CLASSES)}
    keep = [i for i, name in enumerate(classes) if name in index]
    lengths = np.array([geograph.polyline_length(w.pts) for w in ways], dtype=float)

    kept_ways = [ways[i] for i in keep]
    klass = np.array([index[classes[i]] for i in keep], dtype=np.int64)
    counts = (
        np.maximum(1, np.rint(lengths[keep] / spacing).astype(np.int64))
        if keep
        else np.zeros(0, dtype=np.int64)
    )
    local = _same_class_components(kept_ways, klass)

    way_ids = np.array(
        [
            way_table.setdefault(w.osm_id or f"{chip_id}:{i}", len(way_table))
            for i, w in enumerate(kept_ways)
        ],
        dtype=np.int64,
    )
    component_ids = np.array(
        [
            component_table.setdefault((chip_id, int(c)), len(component_table))
            for c in local
        ],
        dtype=np.int64,
    )
    n_other = int(
        np.maximum(
            1,
            np.rint(
                lengths[[i for i, n in enumerate(classes) if n not in index]] / spacing
            ).astype(np.int64),
        ).sum()
    )
    return ChipNodes(
        chip_id=chip_id,
        split=split,
        spacing=spacing,
        klass=np.repeat(klass, counts),
        way=np.repeat(way_ids, counts),
        component=np.repeat(component_ids, counts),
        n_other=n_other,
        length_by_class=np.bincount(klass, weights=lengths[keep], minlength=len(CLASSES)),
        length_m=float(lengths[keep].sum()),
    )


def macro_f1(confusion: np.ndarray) -> np.ndarray:
    """Macro-F1 over the classes with support, for a stack of confusions.

    Args:
        confusion: ``(..., C, C)`` counts, truth on the first axis of the pair.

    Returns:
        ``(...)`` macro-F1. A class with no truth support is left out of the
        average rather than entering it as a zero, which would report the
        vocabulary rather than the model.
    """
    tp = np.diagonal(confusion, axis1=-2, axis2=-1)
    support = confusion.sum(axis=-1)
    predicted = confusion.sum(axis=-2)
    denom = support + predicted
    f1 = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1), 0.0)
    present = support > 0
    return f1.sum(axis=-1) / np.maximum(present.sum(axis=-1), 1)


def simulate(
    klass: np.ndarray, unit: np.ndarray, accuracy: float, rng: np.random.Generator
) -> np.ndarray:
    """Predictions from a model right at ``accuracy`` on each unit.

    Args:
        klass: ``(N,)`` true class index per node.
        unit: ``(N,)`` which unit each node's error is drawn on, already
            compacted to ``0..U-1``.
        accuracy: Probability a unit is predicted correctly.
        rng: Source of randomness.

    Returns:
        ``(N,)`` predicted class index.
    """
    n_units = int(unit.max()) + 1 if len(unit) else 0
    truth = np.zeros(n_units, dtype=np.int64)
    truth[unit] = klass

    prior = np.bincount(klass, minlength=len(CLASSES)).astype(float)
    prior /= prior.sum()

    predicted = truth.copy()
    wrong = np.flatnonzero(rng.random(n_units) >= accuracy)
    # Rejection rather than renormalising per unit: one vectorized draw against
    # the whole prior, redrawing only the units that landed on their own class.
    while len(wrong):
        draw = rng.choice(len(CLASSES), size=len(wrong), p=prior)
        predicted[wrong] = draw
        wrong = wrong[draw == truth[wrong]]
    return predicted[unit]


def chip_confusions(
    klass: np.ndarray, predicted: np.ndarray, chip: np.ndarray, n_chips: int
) -> np.ndarray:
    """One confusion matrix per chip, so a chip bootstrap is a matrix sum."""
    c = len(CLASSES)
    flat = chip * c * c + klass * c + predicted
    return np.bincount(flat, minlength=n_chips * c * c).reshape(n_chips, c, c)


def bootstrap_weights(n_chips: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """How many times each chip is drawn, in each of ``n_boot`` resamples.

    Resampling chips with replacement is a multinomial draw over chip weights,
    so a bootstrap becomes one matrix product instead of ``n_boot`` index
    gathers. The weights are drawn once per split and reused across every
    spacing, accuracy and regime: common random numbers, so differences between
    those configurations are not obscured by the resampling scheme changing
    underneath them.

    Args:
        n_chips: Chips in the split.
        n_boot: Bootstrap resamples.
        rng: Source of randomness.

    Returns:
        ``(n_boot, n_chips)`` counts, each row summing to ``n_chips``.
    """
    return rng.multinomial(n_chips, np.full(n_chips, 1.0 / n_chips), size=n_boot)


def bootstrap_sem(chip_cm: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Macro-F1 on the whole split, and its standard error over chip resamples.

    Args:
        chip_cm: ``(n_chips, C, C)`` per-chip confusions.
        weights: ``(n_boot, n_chips)`` from :func:`bootstrap_weights`.

    Returns:
        ``(macro_f1, sem)``.
    """
    n_chips = len(chip_cm)
    point = float(macro_f1(chip_cm.sum(axis=0)))
    stacked = (weights @ chip_cm.reshape(n_chips, -1)).reshape(
        len(weights), len(CLASSES), len(CLASSES)
    )
    return point, float(macro_f1(stacked).std(ddof=1))


def _compact(values: np.ndarray) -> np.ndarray:
    """Relabel arbitrary integer ids to a dense ``0..n-1``."""
    return np.unique(values, return_inverse=True)[1]


def summarize_classes(
    nodes: Sequence[ChipNodes], split: Split, spacing: float
) -> list[ClassCount]:
    """Count nodes, ways and components per class over one split."""
    rows = [n for n in nodes if n.split == split and n.spacing == spacing]
    klass = np.concatenate([r.klass for r in rows]) if rows else np.zeros(0, np.int64)
    way = np.concatenate([r.way for r in rows]) if rows else np.zeros(0, np.int64)
    comp = np.concatenate([r.component for r in rows]) if rows else np.zeros(0, np.int64)
    total = max(len(klass), 1)
    lengths = (
        np.sum([r.length_by_class for r in rows], axis=0)
        if rows
        else np.zeros(len(CLASSES))
    )

    out = [
        ClassCount(
            split=split,
            spacing=spacing,
            klass=name,
            n_nodes=int((klass == i).sum()),
            n_ways=len(np.unique(way[klass == i])),
            n_components=len(np.unique(comp[klass == i])),
            length_m=float(lengths[i]),
            node_fraction=float((klass == i).sum() / total),
        )
        for i, name in enumerate(CLASSES)
    ]
    out.append(
        ClassCount(
            split=split,
            spacing=spacing,
            klass=OTHER,
            n_nodes=sum(r.n_other for r in rows),
            n_ways=0,
            n_components=0,
            length_m=0.0,
            node_fraction=float(sum(r.n_other for r in rows) / total),
        )
    )
    return out


def summarize_clusters(
    nodes: Sequence[ChipNodes], split: Split, spacing: float, unit: str
) -> ClusterSizes:
    """Cluster-size distribution under one unit definition."""
    rows = [n for n in nodes if n.split == split and n.spacing == spacing]
    ids = np.concatenate([getattr(r, unit) for r in rows]) if rows else np.zeros(0)
    sizes = np.bincount(_compact(ids)) if len(ids) else np.zeros(1)
    return ClusterSizes(
        split=split,
        spacing=spacing,
        unit=unit,
        n_clusters=len(sizes),
        mean_nodes=float(sizes.mean()),
        median_nodes=float(np.median(sizes)),
        p90_nodes=float(np.percentile(sizes, 90)),
        max_nodes=int(sizes.max()),
    )


def summarize_baseline(
    nodes: Sequence[ChipNodes], split: Split, spacing: float
) -> SplitBaseline:
    """What always predicting the majority class already scores."""
    rows = [n for n in nodes if n.split == split and n.spacing == spacing]
    klass = np.concatenate([r.klass for r in rows]) if rows else np.zeros(0, np.int64)
    counts = np.bincount(klass, minlength=len(CLASSES))
    top = int(counts.argmax())

    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    confusion[:, top] = counts
    return SplitBaseline(
        split=split,
        spacing=spacing,
        n_chips=len(rows),
        n_chips_empty=sum(1 for r in rows if len(r.klass) == 0),
        n_nodes=len(klass),
        majority_class=CLASSES[top],
        majority_share=float(counts[top] / max(len(klass), 1)),
        majority_macro_f1=float(macro_f1(confusion)),
        length_m=float(sum(r.length_m for r in rows)),
    )


def resolvability(
    nodes: Sequence[ChipNodes],
    split: Split,
    spacing: float,
    accuracy: float,
    regime: str,
    rhos: Sequence[float],
    weights: np.ndarray,
    n_draws: int,
    alpha: float,
    power: float,
    rng: np.random.Generator,
) -> Resolvability:
    """Simulate models, bootstrap their macro-F1, and price a comparison.

    The simulated model is itself a random draw, and under a correlated regime
    it is drawn on few enough units that one draw's score is not representative.
    ``n_draws`` models are simulated and their bootstrap standard errors
    averaged, so the reported noise floor is the typical one rather than
    whichever the seed happened to land on.
    """
    rows = [n for n in nodes if n.split == split and n.spacing == spacing]
    klass = np.concatenate([r.klass for r in rows])
    chip = np.repeat(np.arange(len(rows)), [len(r.klass) for r in rows])
    unit = _compact(
        np.arange(len(klass))
        if regime == "node"
        else np.concatenate([getattr(r, regime) for r in rows])
    )

    draws = [
        bootstrap_sem(
            chip_confusions(klass, simulate(klass, unit, accuracy, rng), chip, len(rows)),
            weights,
        )
        for _ in range(n_draws)
    ]
    points = np.array([d[0] for d in draws])
    sems = np.array([d[1] for d in draws])
    sem = float(sems.mean())

    per_class_units = [len(np.unique(unit[klass == i])) for i in range(len(CLASSES))]
    per_class_nodes = np.bincount(klass, minlength=len(CLASSES))
    return Resolvability(
        split=split,
        spacing=spacing,
        accuracy=accuracy,
        regime=regime,
        n_nodes=len(klass),
        n_units=int(unit.max()) + 1 if len(unit) else 0,
        macro_f1=float(points.mean()),
        macro_f1_spread=float(points.std(ddof=1)) if n_draws > 1 else 0.0,
        macro_f1_sem=sem,
        sem_spread=float(sems.std(ddof=1)) if n_draws > 1 else 0.0,
        min_detectable={
            f"{rho:g}": detectable_difference(sem, rho, alpha, power) for rho in rhos
        },
        min_class_nodes=int(per_class_nodes.min()),
        min_class_units=int(min(per_class_units)),
    )


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-json",
        type=Path,
        default=Path("outputs/vegas_best.json"),
        help="run artifact supplying the train/val split the model was fit on",
    )
    parser.add_argument("--aoi-root", type=Path, default=Path("data/AOI_2_Vegas"))
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--extract", type=Path, default=osm.DEFAULT_EXTRACT)
    parser.add_argument(
        "--network-type", default="drive", choices=sorted(osm.NETWORK_FILTERS)
    )
    parser.add_argument("--pad-m", type=float, default=250.0)
    parser.add_argument(
        "--spacings",
        type=float,
        nargs="+",
        default=list(DEFAULT_SPACINGS),
        help="node spacings in metres; the sweep is the point of the script",
    )
    parser.add_argument(
        "--accuracies",
        type=float,
        nargs="+",
        default=list(DEFAULT_ACCURACIES),
        help="per-unit accuracies the simulated model is run at",
    )
    parser.add_argument(
        "--rhos",
        type=float,
        nargs="+",
        default=list(DEFAULT_RHOS),
        help="between-arm correlations the detectable difference is tabulated across",
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=list(REGIMES),
        choices=list(REGIMES),
        help="which unit carries one correctness draw",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=4000, help="chip resamples per configuration"
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=20,
        help="simulated models per configuration, averaged over",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="chips per split; 0 runs every chip, which is what a reported number needs",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/attr_resolvability.json")
    )
    args = parser.parse_args()

    started = time.perf_counter()
    run = json.loads(args.run_json.read_text())
    limit = args.limit or None
    ids_by_split: dict[Split, list[str]] = {
        "train": list(run["config"]["train_ids"])[:limit],
        "val": list(run["config"]["val_ids"])[:limit],
    }
    spacings = tuple(sorted(float(s) for s in args.spacings))
    rng = np.random.default_rng(args.seed)

    chips = {chip.image_id: chip for chip in spacenet.find_chips(args.aoi_root)}
    requested = [(s, i) for s, ids in ids_by_split.items() for i in ids]
    logger.info(
        f"{len(ids_by_split['train'])} train + {len(ids_by_split['val'])} val chips, "
        f"spacings {spacings}, {args.network_type} network"
    )

    # Read-once, clip-many, as in the OSM cross-check: the cost of a bbox read
    # is the file, not the extent, so one read serves every chip.
    extract_md5 = hashlib.md5(args.extract.read_bytes()).hexdigest()
    tile_by_id = {
        i: spacenet.chip_tile(chips[i].image_path, args.resolution) for _, i in requested
    }
    extract = osm.read_extract(
        osm.covering_bounds(list(tile_by_id.values()), args.pad_m), args.extract
    )
    way_source = osm.WAY_SOURCE_REGISTRY["pbf"](extract=extract)

    way_table: dict[str, int] = {}
    component_table: dict[tuple[str, int], int] = {}
    nodes: list[ChipNodes] = []
    for n, (split, chip_id) in enumerate(requested, start=1):
        ways = way_source.ways(tile_by_id[chip_id], args.network_type, args.pad_m)
        nodes.extend(
            chip_nodes(ways, chip_id, split, spacing, way_table, component_table)
            for spacing in spacings
        )
        if n % 100 == 0 or n == len(requested):
            logger.info(f"{n}/{len(requested)} chips, {len(way_table)} distinct OSM ways")

    splits: tuple[Split, ...] = ("train", "val")
    # A road drawn as one OSM way and cut by the split reaches both sides, so
    # part of it is memorised rather than predicted. Counted here because the
    # eventual model's val score depends on it and nothing else reports it.
    by_split = {
        s: {
            int(w)
            for n in nodes
            if n.split == s and n.spacing == spacings[0]
            for w in np.unique(n.way)
        }
        for s in splits
    }
    shared_ways = by_split["train"] & by_split["val"]
    logger.info(
        f"{len(shared_ways)} OSM ways reach both splits, of "
        f"{len(by_split['train'])} train and {len(by_split['val'])} val"
    )

    baselines = [summarize_baseline(nodes, s, sp) for s in splits for sp in spacings]
    class_counts = [
        c for s in splits for sp in spacings for c in summarize_classes(nodes, s, sp)
    ]
    clusters = [
        summarize_clusters(nodes, s, sp, u)
        for s in splits
        for sp in spacings
        for u in ("way", "component")
    ]
    weights = {
        s: bootstrap_weights(len(ids_by_split[s]), args.bootstrap, rng) for s in splits
    }
    rows = [
        resolvability(
            nodes,
            s,
            sp,
            a,
            r,
            tuple(args.rhos),
            weights[s],
            args.draws,
            args.alpha,
            args.power,
            rng,
        )
        for s in splits
        for sp in spacings
        for a in sorted(args.accuracies)
        for r in args.regimes
    ]

    for b in baselines:
        logger.info(
            f"{b.split:>5} s={b.spacing:>4.0f}m: {b.n_nodes:>7} nodes on "
            f"{b.n_chips} chips "
            f"({b.n_chips_empty} empty), {b.length_m / 1000:.1f} km, majority "
            f"{b.majority_class} {b.majority_share:.1%}, majority macro-F1 "
            f"{b.majority_macro_f1:.4f}"
        )

    logger.info(
        f"{'split':>5} {'s':>4} {'class':>13} {'nodes':>8} {'ways':>7} "
        f"{'comps':>7} {'km':>8} {'share':>7}"
    )
    for c in class_counts:
        if c.spacing == 20.0:
            logger.info(
                f"{c.split:>5} {c.spacing:>4.0f} {c.klass:>13} {c.n_nodes:>8} "
                f"{c.n_ways:>7} {c.n_components:>7} {c.length_m / 1000:>8.2f} "
                f"{c.node_fraction:>7.4f}"
            )

    logger.info(
        f"{'split':>5} {'s':>4} {'unit':>10} {'clusters':>9} {'mean':>7} "
        f"{'med':>6} {'p90':>6} {'max':>6}"
    )
    for k in clusters:
        logger.info(
            f"{k.split:>5} {k.spacing:>4.0f} {k.unit:>10} {k.n_clusters:>9} "
            f"{k.mean_nodes:>7.2f} {k.median_nodes:>6.1f} "
            f"{k.p90_nodes:>6.1f} {k.max_nodes:>6}"
        )

    rho_keys = [f"{rho:g}" for rho in args.rhos]
    header = " ".join(f"{'mdd@' + k:>9}" for k in rho_keys)
    logger.info(
        f"{'split':>5} {'s':>4} {'acc':>5} {'regime':>10} {'nodes':>7} {'units':>7} "
        f"{'minU':>6} {'macroF1':>8} {'sem':>8} {header}"
    )
    for r in rows:
        mdds = " ".join(f"{r.min_detectable[k]:>9.4f}" for k in rho_keys)
        logger.info(
            f"{r.split:>5} {r.spacing:>4.0f} {r.accuracy:>5.2f} {r.regime:>10} "
            f"{r.n_nodes:>7} {r.n_units:>7} {r.min_class_units:>6} "
            f"{r.macro_f1:>8.4f} {r.macro_f1_sem:>8.4f} {mdds}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "run_json": str(args.run_json),
                "params": {
                    "resolution": args.resolution,
                    "extract": str(args.extract),
                    "extract_md5": extract_md5,
                    "network_type": args.network_type,
                    "pad_m": args.pad_m,
                    "spacings": list(spacings),
                    "accuracies": sorted(args.accuracies),
                    "rhos": list(args.rhos),
                    "regimes": list(args.regimes),
                    "bootstrap": args.bootstrap,
                    "draws": args.draws,
                    "alpha": args.alpha,
                    "power": args.power,
                    "seed": args.seed,
                    "limit": args.limit,
                    "classes": list(CLASSES),
                    "class_map": CLASS_MAP,
                    "confusion_model": "prior-proportional, excluding the true class",
                },
                "elapsed_seconds": time.perf_counter() - started,
                "n_chips": {k: len(v) for k, v in ids_by_split.items()},
                "n_osm_ways": len(way_table),
                "n_ways_by_split": {s: len(v) for s, v in by_split.items()},
                "n_ways_in_both_splits": len(shared_ways),
                "baselines": [asdict(b) for b in baselines],
                "class_counts": [asdict(c) for c in class_counts],
                "cluster_sizes": [asdict(k) for k in clusters],
                "resolvability": [asdict(r) for r in rows],
            },
            indent=2,
        )
    )
    logger.info(f"wrote {args.out} in {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()

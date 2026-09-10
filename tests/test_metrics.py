import networkx as nx
import numpy as np
import pytest

from geo_graphs import geograph, metrics
from tests.conftest import barbell_graph, curvy_graph, drop_edges, grid_graph


def shifted(G: nx.MultiGraph, dx: float, dy: float = 0.0) -> nx.MultiGraph:
    """The same graph translated, so every road moves by ``(dx, dy)``.

    On an axis-aligned grid a translation along one axis alone leaves the roads
    parallel to that axis lying on top of themselves, so the *perpendicular*
    displacement is ``dx`` for half the roads and zero for the other half. Equal
    ``dx`` and ``dy`` displace every road by the same amount instead.
    """
    offset = np.array([dx, dy])
    return geograph.build(
        [geograph.oriented_pts(G, u, v, k) + offset for u, v, k in G.edges(keys=True)]
    )


def test_identity_scores_one():
    """The invariant that caught the polyline-orientation bug."""
    curvy = curvy_graph()
    assert metrics.apls(curvy, curvy).score == pytest.approx(1.0)


def test_empty_proposal_scores_zero():
    assert metrics.apls(curvy_graph(), nx.MultiGraph()).score == pytest.approx(0.0)


def test_score_decreases_with_damage():
    curvy = curvy_graph()
    scores = [
        metrics.apls(curvy, drop_edges(curvy, f)).score for f in (0.0, 0.1, 0.25, 0.5)
    ]
    assert scores == sorted(scores, reverse=True)


def test_harmonic_mean_punishes_lopsided_proposals():
    """Half a network recovered perfectly still has to score badly.

    A proposal keeping only half the roads is nearly flawless in the
    proposal-to-truth direction, since everything it does claim is real. The
    harmonic mean is what stops that from averaging out to a decent score.
    """
    curvy = curvy_graph()
    result = metrics.apls(curvy, drop_edges(curvy, 0.5))
    assert result.prop_to_gt > 2 * result.gt_to_prop

    arithmetic = 0.5 * (result.gt_to_prop + result.prop_to_gt)
    assert result.score < arithmetic - 0.05
    # the combined score sits nearer the failing direction than the flattering one
    assert result.score - result.gt_to_prop < result.prop_to_gt - result.score


def test_deterministic():
    curvy = curvy_graph()
    H = drop_edges(curvy, 0.2)
    assert metrics.apls(curvy, H).score == metrics.apls(curvy, H).score


def length_error(score: float, n_pairs: int, unlanded: int, no_path: int) -> float:
    """Pairs' worth of score lost to length disagreement alone."""
    return (1.0 - score) * n_pairs - unlanded - no_path


def test_pair_counts_describe_the_pairs_the_score_averages():
    """The counts are instrumentation, so they have to add up to the score.

    A pair scores 0 for one of three reasons. Two of them are counted directly;
    the third is whatever deficit is left over, and it can be neither negative
    nor larger than the number of pairs that actually reached a comparison.
    """
    curvy = curvy_graph()
    for proposal in (curvy, drop_edges(curvy, 0.3), shifted(curvy, 12.0, 12.0)):
        r = metrics.apls(curvy, proposal)
        for score, pairs, unlanded, no_path in (
            (r.gt_to_prop, r.n_pairs_gt, r.n_pairs_unlanded_gt, r.n_pairs_no_path_gt),
            (
                r.prop_to_gt,
                r.n_pairs_prop,
                r.n_pairs_unlanded_prop,
                r.n_pairs_no_path_prop,
            ),
        ):
            residual = length_error(score, pairs, unlanded, no_path)
            assert residual >= -1e-9
            assert residual <= pairs - unlanded - no_path + 1e-9


def test_identity_lands_every_control_point():
    curvy = curvy_graph()
    r = metrics.apls(curvy, curvy)
    assert (r.n_landed_gt, r.n_landed_prop) == (r.n_control_gt, r.n_control_prop)
    assert (r.n_pairs_unlanded_gt, r.n_pairs_no_path_gt) == (0, 0)


def test_severing_the_network_is_counted_as_lost_routes_not_lost_landings():
    """The two ways a pair scores 0 are different failures and stay apart."""
    barbell = barbell_graph()
    bridge = max(barbell.edges(keys=True), key=lambda e: barbell.edges[e]["length"])
    severed = barbell.copy()
    severed.remove_edge(*bridge)

    r = metrics.apls(barbell, severed)
    assert r.n_pairs_no_path_gt > 0
    assert r.n_pairs_unlanded_gt == 0  # every road is still drawn, just disconnected


def test_max_snap_prices_displacement_only_by_refusing_to_land():
    """The module docstring's third property, as an assertion.

    Every road is moved 12 m off its line. At the committed ``max_snap`` the
    control points still land and the routes still measure the same length, so
    the score barely moves. Below the displacement nothing lands at all and the
    score collapses to zero -- and the collapse is entirely unlanded pairs, so
    the metric never charges for the 12 m as displacement.
    """
    grid = grid_graph()
    moved = shifted(grid, 12.0, 12.0)

    far = metrics.apls(grid, moved, max_snap=25.0)
    assert far.score > 0.9
    assert far.n_landed_gt == far.n_control_gt
    assert far.n_pairs_unlanded_gt == 0

    near = metrics.apls(grid, moved, max_snap=5.0)
    assert near.n_landed_gt == 0
    assert near.score == pytest.approx(0.0)
    assert near.n_pairs_unlanded_gt == near.n_pairs_gt
    # Lowering the radius removed landings and nothing else: the pair set is
    # fixed by the source graph and the spacing, so it cannot move.
    assert near.n_pairs_gt == far.n_pairs_gt


def test_lowering_max_snap_cannot_raise_the_score():
    """A smaller radius can only drop a landing, never find a nearer edge."""
    curvy = curvy_graph()
    moved = shifted(curvy, 9.0, 9.0)
    results = [metrics.apls(curvy, moved, max_snap=r) for r in (5.0, 10.0, 15.0, 25.0)]

    assert [r.gt_to_prop for r in results] == sorted(r.gt_to_prop for r in results)
    assert [r.prop_to_gt for r in results] == sorted(r.prop_to_gt for r in results)
    assert [r.n_landed_gt for r in results] == sorted(r.n_landed_gt for r in results)
    assert len({r.n_pairs_gt for r in results}) == 1


def test_iou_bounds():
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((10, 10), dtype=bool)
    a[:5], b[:5] = True, True
    assert metrics.iou(a, b) == pytest.approx(1.0)
    b[:] = False
    b[5:] = True
    assert metrics.iou(a, b) == pytest.approx(0.0)


def test_severing_a_bridge_costs_far_more_than_its_length():
    """The project's central claim, as an assertion.

    Cutting the single edge joining two halves removes a small share of the
    network's length but makes every crossing route infinite, so APLS has to
    fall an order of magnitude further than the length lost. This is the gap a
    pixel metric cannot see: a two-pixel break under a tree shadow looks
    negligible to IoU and severs the route.
    """
    barbell = barbell_graph()
    bridge = max(barbell.edges(keys=True), key=lambda e: barbell.edges[e]["length"])
    lost_fraction = barbell.edges[bridge]["length"] / geograph.total_length(barbell)

    severed = barbell.copy()
    severed.remove_edge(*bridge)
    assert nx.number_connected_components(severed) == 2

    assert lost_fraction < 0.10
    assert 1.0 - metrics.apls(barbell, severed).score > 3 * lost_fraction


def test_buffer_length_identity_scores_one():
    curvy = curvy_graph()
    result = metrics.buffer_length_prf(curvy, curvy, buffer=5.0)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)


def test_buffer_length_recall_tracks_the_length_that_went_missing():
    """Coverage is length-weighted, so half the roads removed costs about half."""
    grid = grid_graph()
    kept = drop_edges(grid, 0.5)
    fraction = geograph.total_length(kept) / geograph.total_length(grid)

    result = metrics.buffer_length_prf(grid, kept, buffer=5.0)
    assert result.precision == pytest.approx(1.0)  # nothing invented
    # The surviving edges cover a little past their own ends, where a removed
    # edge left a junction behind, so recall sits just above the kept fraction.
    assert fraction <= result.recall < fraction + 0.05


def test_buffer_length_barely_notices_a_severed_bridge():
    """The metric this suite exists to add, against the one it does not replace.

    Cutting the single edge joining two halves is the failure APLS is built to
    punish. Buffer coverage charges only the bridge's own length for it, which
    is the whole point of adding a length-weighted metric: the two disagree by
    an order of magnitude on the same damage.
    """
    barbell = barbell_graph()
    bridge = max(barbell.edges(keys=True), key=lambda e: barbell.edges[e]["length"])
    severed = barbell.copy()
    severed.remove_edge(*bridge)

    coverage_lost = 1.0 - metrics.buffer_length_prf(barbell, severed, buffer=5.0).recall
    apls_lost = 1.0 - metrics.apls(barbell, severed).score
    assert coverage_lost < 0.05
    assert apls_lost > 8 * coverage_lost


def test_buffer_length_sees_displacement_only_through_the_buffer():
    """Below the buffer, geometric error is invisible by construction."""
    grid = grid_graph()
    moved = shifted(grid, 8.0)
    assert metrics.buffer_length_prf(grid, moved, buffer=5.0).f1 < 0.75
    assert metrics.buffer_length_prf(grid, moved, buffer=10.0).f1 > 0.95


def test_buffer_length_empty_proposal_scores_zero():
    result = metrics.buffer_length_prf(curvy_graph(), nx.MultiGraph(), buffer=10.0)
    assert result.recall == pytest.approx(0.0)
    assert result.f1 == pytest.approx(0.0)


def test_junction_identity_scores_one():
    curvy = curvy_graph()
    result = metrics.junction_prf(curvy, curvy, radius=5.0)
    assert result.n_matched == result.n_truth == result.n_proposal
    assert result.f1 == pytest.approx(1.0)
    assert result.degree_agreement == pytest.approx(1.0)
    assert result.mean_offset == pytest.approx(0.0)


def test_junction_degree_agreement_falls_when_a_crossing_loses_an_arm():
    """A four-way crossing recovered as a T still matches on position."""
    grid = grid_graph()
    crossing = next(
        e
        for e in grid.edges(keys=True)
        if grid.degree(e[0]) == 4 and grid.degree(e[1]) == 4
    )
    damaged = grid.copy()
    damaged.remove_edge(*crossing)

    result = metrics.junction_prf(grid, damaged, radius=5.0)
    assert result.n_matched == result.n_truth  # both ends are still junctions
    assert result.f1 == pytest.approx(1.0)
    assert 0.0 < (result.degree_agreement or 0.0) < 1.0


def test_junction_matching_is_one_to_one():
    """Two truth junctions cannot both be satisfied by one proposal junction.

    Greedy nearest-neighbour matching would score this 1.0 by using the single
    proposal junction twice, which is the inflation the assignment prevents.
    """
    truth = geograph.build(
        [
            np.array([[0.0, 0.0], [0.0, -50.0]]),
            np.array([[0.0, 0.0], [-50.0, 0.0]]),
            np.array([[0.0, 0.0], [6.0, 0.0]]),
            np.array([[6.0, 0.0], [6.0, 50.0]]),
            np.array([[6.0, 0.0], [56.0, 0.0]]),
        ]
    )
    proposal = geograph.build(
        [
            np.array([[3.0, 0.0], [3.0, -50.0]]),
            np.array([[3.0, 0.0], [-50.0, 0.0]]),
            np.array([[3.0, 0.0], [53.0, 0.0]]),
        ]
    )

    result = metrics.junction_prf(truth, proposal, radius=10.0)
    assert (result.n_truth, result.n_proposal, result.n_matched) == (2, 1, 1)
    assert result.recall == pytest.approx(0.5)
    assert result.precision == pytest.approx(1.0)


def test_junction_scores_one_when_neither_graph_has_a_junction():
    line = geograph.build([np.array([[0.0, 0.0], [100.0, 0.0]])])
    result = metrics.junction_prf(line, line, radius=5.0)
    assert (result.n_truth, result.n_proposal) == (0, 0)
    assert result.f1 == pytest.approx(1.0)
    assert result.degree_agreement is None


def test_junction_beyond_the_radius_does_not_match():
    grid = grid_graph()
    result = metrics.junction_prf(grid, shifted(grid, 8.0), radius=5.0)
    assert result.n_truth > 0
    assert result.n_matched == 0
    assert result.f1 == pytest.approx(0.0)
    assert result.degree_agreement is None

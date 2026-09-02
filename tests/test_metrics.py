import networkx as nx
import numpy as np
import pytest

from geo_graphs import geograph, metrics
from tests.conftest import barbell_graph, curvy_graph, drop_edges


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

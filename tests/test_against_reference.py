"""Validate our APLS against the vendored SpaceNet reference implementation.

Our version exists because it works on our own graph type and can be taken
apart for diagnosis, but the reference is the definition of record. These tests
pin the agreement so a change to our implementation cannot quietly drift away
from the published metric.

On the default settings the two agree to a few parts in 100,000 — close enough
that the tolerance here is a real regression test rather than a formality. The
remaining residual is the snapping step: the reference searches edges belonging
to the 20 nearest *nodes*, while we index every edge directly, so the two very
occasionally choose different edges for a control point.
"""

import pytest

from geo_graphs import geograph, metrics
from tests.conftest import curvy_graph, drop_edges
from tests.reference.oracle import reference_apls

#: Agreement on default settings. Empirically the worst case is ~3e-5.
TOLERANCE = 1e-4

#: Agreement once control point sampling deliberately diverges.
UNIFORM_SAMPLING_TOLERANCE = 0.08

DAMAGE = [0.0, 0.05, 0.15, 0.30, 0.50]


@pytest.mark.parametrize("frac", DAMAGE)
def test_matches_reference_under_damage(frac):
    curvy = curvy_graph()
    proposal = drop_edges(curvy, frac)
    ours = metrics.apls(curvy, proposal).score
    theirs = reference_apls(curvy, proposal)[0]
    assert ours == pytest.approx(theirs, abs=TOLERANCE)


def test_matches_reference_on_identity():
    curvy = curvy_graph()
    assert reference_apls(curvy, curvy)[0] == pytest.approx(1.0)
    assert metrics.apls(curvy, curvy).score == pytest.approx(1.0)


@pytest.mark.parametrize("frac", DAMAGE)
def test_uniform_sampling_is_a_different_estimator(frac):
    """Dense sampling is for diagnosis, not for reporting.

    Splitting every edge, including straight ones, weights the average toward
    long straight roads and drifts from the published number by up to ~0.06.
    Keeping that measured stops it being mistaken for a comparable score.
    """
    curvy = curvy_graph()
    proposal = drop_edges(curvy, frac)
    uniform = metrics.apls(curvy, proposal, sampling="uniform").score
    theirs = reference_apls(curvy, proposal)[0]
    assert uniform == pytest.approx(theirs, abs=UNIFORM_SAMPLING_TOLERANCE)


def test_sampling_choice_changes_control_point_count():
    curvy = curvy_graph()
    dense = geograph.densify(curvy, 50.0, sampling="uniform").number_of_nodes()
    sparse = geograph.densify(curvy, 50.0, sampling="reference").number_of_nodes()
    assert sparse < dense


def test_reference_uses_harmonic_mean():
    """Pins the combination rule, which the reference's own logging misstates.

    It prints ``Total APLS Metric = Mean(a + b)`` but computes
    ``scipy.stats.hmean``. The label is indistinguishable from correct whenever
    the two directions are close, which is exactly the regime of a healthy
    proposal; trusting it costs ~0.4 APLS on a lopsided one.
    """
    curvy = curvy_graph()
    proposal = drop_edges(curvy, 0.5)
    total, fwd, rev = reference_apls(curvy, proposal)

    harmonic = 2 * fwd * rev / (fwd + rev)
    arithmetic = 0.5 * (fwd + rev)

    assert total == pytest.approx(harmonic, abs=1e-6)
    assert abs(total - arithmetic) > 0.1

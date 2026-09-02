import dataclasses
import json

import pytest

from geo_graphs.roundtrip import RoundTripReport, run

FIELDS = {f.name for f in dataclasses.fields(RoundTripReport)}


def _report(**overrides) -> RoundTripReport:
    base = dict.fromkeys(FIELDS, 0.0)
    base.update(lat=36.0, lon=-115.0, size_m=256.0)
    base.update(overrides)
    return RoundTripReport(**base)  # type: ignore[arg-type]


def test_report_is_immutable():
    report = _report()
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.apls = 1.0  # type: ignore[misc]


def test_report_serializes_to_json():
    """The report is the machine-readable artifact acceptance runs off."""
    restored = json.loads(json.dumps(dataclasses.asdict(_report(apls=0.5))))
    assert restored["apls"] == 0.5
    assert set(restored) == FIELDS


@pytest.mark.network
def test_run_returns_a_well_formed_report():
    """A turned-down run proves plumbing, not quality.

    Deliberately asserts only structure and bounds — never a score threshold,
    because these settings are not the ones any reported number comes from.
    """
    report = run(36.1699, -115.1398, size_m=256.0, resolution=2.0)

    assert report.size_m == 256.0
    assert report.truth_nodes > 0
    assert report.truth_length_m > 0.0
    assert 0.0 <= report.mask_iou <= 1.0
    for score in (report.apls, report.apls_gt_to_prop, report.apls_prop_to_gt):
        assert 0.0 <= score <= 1.0


@pytest.mark.network
def test_sampling_choice_reaches_the_metric():
    """The cost knob has to be a real parameter, not an edit you remember."""
    args = (36.1699, -115.1398)
    kwargs = {"size_m": 256.0, "resolution": 2.0}
    reference = run(*args, sampling="reference", **kwargs)
    uniform = run(*args, sampling="uniform", **kwargs)
    assert reference.apls != uniform.apls

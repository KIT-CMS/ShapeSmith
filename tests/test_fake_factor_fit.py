"""Fits of measured ratios against TauFakeFactors' smooth_function (tests/reference/make_fake_factor_reference.py)."""
import json
from pathlib import Path

import numpy as np
import pytest

from shapesmith.measurements.fake_factors import Fit
from shapesmith.measurements.fake_factors.fit import constant_fit, fit, sparsify
from shapesmith.measurements.fake_factors.hist import Hist

REFERENCE = json.loads((Path(__file__).parent / "reference" / "fake_factor_root.json").read_text())
EDGES = np.array([0.0, 1.0, 2.0, 3.5, 5.0, 8.0])
FITS = {  # TauFakeFactors option -> Fit
    "binwise": Fit("binwise"), "smoothed": Fit("smoothed", 2.0), "binwise#[0]+smoothed": Fit("smoothed", 2.0, binwise_left=1),
    "binwise#[-1]+smoothed": Fit("smoothed", 2.0, binwise_right=1), "binwise#[2]+smoothed": Fit("smoothed", 2.0, binwise_left=3),
}


def ratio(values, errors, means, suppressed) -> Hist:
    values, errors = np.array(values), np.array(errors)
    return Hist(EDGES, values, errors**2, np.ones_like(values), np.array(means), values, errors, np.array(suppressed))


def _close(a, b) -> bool:
    return np.allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-15) and len(a) == len(b)


@pytest.mark.parametrize("case", REFERENCE["fits"], ids=lambda c: f"{c['option']}_{c['values'][0]}_{c['mc_shifted'] is not None}")
def test_fit_matches_tau_fake_factors(case):
    mc = [ratio(v, case["errors"], case["means"], case["suppressed"]) for v in case["mc_shifted"]] if case["mc_shifted"] else None
    fitted = fit(ratio(case["values"], case["errors"], case["means"], case["suppressed"]), FITS[case["option"]], mc)
    expected = case["result"]
    assert fitted.reset == expected["reset"]
    assert (fitted.p_value is None) == (expected["p_value"] is None)
    if expected["p_value"] is not None:
        assert fitted.p_value == pytest.approx(expected["p_value"], rel=1e-9)
    assert _close(fitted.stored.edges, expected["stored"]["edges"]) and _close(fitted.stored.values, expected["stored"]["values"])
    assert list(fitted.stored_variations) == list(expected["stored_variations"])
    for name, step in expected["stored_variations"].items():
        assert _close(fitted.stored_variations[name].edges, step["edges"]), name
        assert _close(fitted.stored_variations[name].values, step["values"]), name


def test_the_reference_covers_resets_with_and_without_mc_shifts():
    resets = [(c["mc_shifted"] is not None) for c in REFERENCE["fits"] if c["result"]["reset"]]
    assert True in resets and False in resets


@pytest.mark.parametrize("case", [c for c in REFERENCE["fits"] if c["constants"]][:3])
def test_constant_fit_is_roots_pol0(case):
    for values, constant in zip(case["mc_shifted"], case["constants"]):
        assert constant_fit(ratio(values, case["errors"], case["means"], case["suppressed"])) == pytest.approx(constant, rel=1e-12)


def test_a_constant_fit_without_errors_fails_to_one():
    assert constant_fit(ratio([2.0, 3.0, 1.0, 1.0, 1.0], [0.0] * 5, [0.5, 1.5, 2.5, 4.0, 6.0], [0.0] * 5)) == 1.0


def test_a_reset_ratio_is_stored_as_one_bin_of_one():
    fitted = fit(ratio([1.01, 0.99, 1.0, 1.02, 0.98], [0.2] * 5, [0.5, 1.5, 2.7, 4.2, 6.1], [0.2] * 5), Fit("smoothed", 2.0))
    assert fitted.reset and fitted.stored.edges.tolist() == [0.0, 8.0] and fitted.stored.values.tolist() == [1.0]
    assert fitted.stored_variations["SystMCShiftUp"].values.tolist() == [1.0]  # nothing subtracted: 1


def test_sparsify_keeps_an_edge_per_step_of_the_range_and_the_last_bin():
    edges, values = sparsify(np.arange(6.0), np.array([0.0, 0.0, 0.5, 1.0, 1.0]))
    assert edges.tolist() == [0.0, 2.0, 3.0, 4.0, 5.0] and values.tolist() == [0.0, 0.5, 1.0, 1.0]

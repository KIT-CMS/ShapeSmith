"""The fake-factor histogram against ROOT with TauFakeFactors' patches (tests/reference/make_fake_factor_reference.py)."""
import json
from pathlib import Path

import numpy as np
import pytest

from shapesmith.measurements.fake_factors.hist import Hist

REFERENCE = json.loads((Path(__file__).parent / "reference" / "fake_factor_root.json").read_text())
FIELDS = ("values", "errors", "counts", "means", "base_values", "base_errors", "suppressed_errors")


def assert_matches(hist: Hist, reference: dict) -> None:
    for name in FIELDS:
        assert np.allclose(getattr(hist, name), reference[name], rtol=1e-12, atol=1e-15), name


def filled(case: dict, region: str) -> dict[str, Hist]:
    return {s: Hist.fill(case["edges"], np.array(e["x"], dtype=np.float32), np.array(e["w"])) for s, e in case["events"][region].items()}


def subtracted(hists: dict[str, Hist], shift: float | None) -> Hist:
    result = hists["data"]
    for name in ("mc_a", "mc_b", "mc_c"):
        result = result.add(hists[name] if shift is None else hists[name].shifted(shift), -1.0)
    return result


@pytest.fixture(params=range(len(REFERENCE["hists"])), ids=lambda i: f"case{i}")
def case(request):
    return REFERENCE["hists"][request.param]


def test_fill_matches_root_and_the_centre_of_mass(case):
    for region in ("sr", "ar"):
        for sample, hist in filled(case, region).items():
            assert_matches(hist, case["filled"][region][sample])


def test_subtraction_and_shifts(case):
    for region in ("sr", "ar"):
        hists = filled(case, region)
        for label, shift in (("nominal", None), ("up", 1.0), ("down", -1.0)):
            assert_matches(subtracted(hists, shift), case["yields"][region][label])


def test_ratio_scale_and_the_bin_rules(case):
    sr, ar = subtracted(filled(case, "sr"), None), subtracted(filled(case, "ar"), None)
    ratio = sr.divide(ar)
    assert_matches(ratio, case["ratio"])
    assert_matches(ratio.scaled(1.7), case["scaled"])
    assert_matches(ratio.with_empty_bins_set_to_one(), case["closure"])
    hists = filled(case, "ar")
    assert_matches(subtracted(hists, None).with_negative_bins_zeroed(), case["qcd"])


def test_the_small_case_has_bins_set_to_one():
    closure = REFERENCE["hists"][1]["closure"]
    assert 1.0 in closure["values"] and 1.0 in closure["errors"]

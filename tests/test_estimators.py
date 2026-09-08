import dataclasses

from shapesmith.estimators import run_estimate
from shapesmith.estimators.abcd import clip_negative_bins
from shapesmith.histogram import Histogram
from shapesmith.histograms import INCLUSIVE, HistKey, HistogramSet
from shapesmith.model import Estimator
from tests.mini_analysis import build


def _hist(values, variances=None):
    return Histogram([0.0, 1.0, 2.0], values, variances if variances is not None else values)


def _key(process, region, category=INCLUSIVE, variation="Nominal"):
    return HistKey("mt", category, process, region, variation, "x")


def test_fake_factor_estimate_subtracts_available_processes():
    analysis = build()
    hset = HistogramSet()
    hset.add(_key("data", "anti_iso"), _hist([10.0, 20.0]))
    hset.add(_key("ZTT", "anti_iso"), _hist([1.0, 2.0]))
    # ZL missing on purpose -> skipped with a warning
    added = run_estimate(hset, analysis, ["mt"])
    assert added == [_key("jetFakes", "nominal")]
    out = hset.get(_key("jetFakes", "nominal"))
    assert out.values.tolist() == [9.0, 18.0] and out.variances.tolist() == [11.0, 22.0]


def test_abcd_estimate():
    analysis = dataclasses.replace(
        build(),
        estimator=Estimator("abcd", regions={"B": "abcd_anti_iso", "C": "abcd_same_sign", "D": "abcd_same_sign_anti_iso"}, subtract=("ZTT",), output="QCD"),
    )
    hset = HistogramSet()
    for region, data, mc in (("abcd_anti_iso", (8.0, 8.0), (2.0, 2.0)), ("abcd_same_sign", (6.0, 6.0), (2.0, 2.0)), ("abcd_same_sign_anti_iso", (4.0, 4.0), (2.0, 2.0))):
        hset.add(_key("data", region), _hist(list(data)))
        hset.add(_key("ZTT", region), _hist(list(mc)))
    run_estimate(hset, analysis, ["mt"])
    assert hset.get(_key("QCD", "nominal")).values.tolist() == [12.0, 12.0]  # B=(6,6), C=8, D=4 -> factor 2


def test_abcd_factor_zero_when_control_regions_empty():
    analysis = dataclasses.replace(
        build(),
        estimator=Estimator("abcd", regions={"B": "abcd_anti_iso", "C": "abcd_same_sign", "D": "abcd_same_sign_anti_iso"}, subtract=(), output="QCD"),
    )
    hset = HistogramSet()
    hset.add(_key("data", "abcd_anti_iso"), _hist([5.0, 5.0]))
    hset.add(_key("data", "abcd_same_sign"), _hist([0.0, 0.0]))
    hset.add(_key("data", "abcd_same_sign_anti_iso"), _hist([1.0, 1.0]))
    run_estimate(hset, analysis, ["mt"])
    assert hset.get(_key("QCD", "nominal")).values.tolist() == [0.0, 0.0]


def test_clip_negative_bins_keeps_integral():
    assert clip_negative_bins(_hist([-1.0, 5.0])).values.tolist() == [0.0, 4.0]
    assert clip_negative_bins(_hist([-3.0, 1.0])).values.tolist() == [0.0, 0.0]


def test_estimate_skips_categories_without_data_input():
    hset = HistogramSet()
    hset.add(_key("ZTT", "anti_iso"), _hist([1.0, 2.0]))
    assert run_estimate(hset, build(), ["mt"]) == []

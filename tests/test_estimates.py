import dataclasses

import pytest

from shapesmith.estimates import clip_negative_bins, run_estimates
from shapesmith.histogram import INCLUSIVE, HistKey, Histogram, HistogramSet, part_of
from shapesmith.model import ABCD, ColumnVariation, DataMinus, VariationSum
from tests.mini_analysis import build, build_embedding


def _hist(values, variances=None):
    return Histogram([0.0, 1.0, 2.0], values, variances if variances is not None else values)


def _key(process, region, category=INCLUSIVE, variation="Nominal"):
    return HistKey("mt", category, process, region, variation, "x")


def _with_estimators(analysis, *estimators):
    return dataclasses.replace(analysis, channels={"mt": dataclasses.replace(analysis.channel("mt"), estimators=estimators)})


def test_data_minus_subtracts_available_processes():
    hset = HistogramSet()
    hset[_key("data", "anti_iso")] = _hist([10.0, 20.0])
    hset[_key("ZTT", "anti_iso")] = _hist([1.0, 2.0])
    # ZL missing on purpose -> skipped with a warning
    added = run_estimates(hset, build(), ["mt"])
    assert added == [_key("jetFakes", "nominal")]
    out = hset[_key("jetFakes", "nominal")]
    assert out.values.tolist() == [9.0, 18.0] and out.variances.tolist() == [11.0, 22.0]


def test_data_minus_propagates_column_variations_with_nominal_fallback():
    hset = HistogramSet()
    hset[_key("data", "anti_iso")] = _hist([10.0, 20.0])
    hset[_key("data", "anti_iso", variation="CMS_ffStatUp")] = _hist([12.0, 24.0])
    hset[_key("EMB", "anti_iso")] = _hist([1.0, 2.0])
    hset[_key("EMB", "anti_iso", variation="CMS_tesUp")] = _hist([2.0, 3.0])
    hset[_key("ZL", "anti_iso")] = _hist([0.5, 0.5])
    hset[_key("ZL", "anti_iso", variation="CMS_puUp")] = _hist([5.0, 5.0])  # a weight variation: not propagated
    run_estimates(hset, build_embedding(), ["mt"])
    assert {k.variation for k in hset.select(process="jetFakes")} == {"Nominal", "CMS_ffStatUp", "CMS_tesUp"}
    assert hset[_key("jetFakes", "nominal")].values.tolist() == [8.5, 17.5]
    assert hset[_key("jetFakes", "nominal", variation="CMS_ffStatUp")].values.tolist() == [10.5, 21.5]  # data varied, EMB and ZL nominal
    assert hset[_key("jetFakes", "nominal", variation="CMS_tesUp")].values.tolist() == [7.5, 16.5]  # an EMB-only variation reaches jetFakes


def test_template_shift_uses_the_auxiliary_template():
    hset = HistogramSet()
    hset[_key("EMB", "nominal")] = _hist([10.0, 10.0])
    hset[_key("ZTT", "nominal")] = _hist([2.0, 4.0])
    added = run_estimates(hset, build_embedding(), ["mt"])
    assert set(added) == {_key("EMB", "nominal", variation=f"CMS_htt_emb_ttbar_2018{d}") for d in ("Up", "Down")}
    assert hset[_key("EMB", "nominal", variation="CMS_htt_emb_ttbar_2018Up")].values.tolist() == pytest.approx([10.2, 10.4])
    assert hset[_key("EMB", "nominal", variation="CMS_htt_emb_ttbar_2018Down")].values.tolist() == pytest.approx([9.8, 9.6])


def test_abcd_estimate():
    analysis = _with_estimators(build(), ABCD("QCD", "abcd_anti_iso", "abcd_same_sign", "abcd_same_sign_anti_iso", ("ZTT",)))
    hset = HistogramSet()
    for region, data, mc in (("abcd_anti_iso", (8.0, 8.0), (2.0, 2.0)), ("abcd_same_sign", (6.0, 6.0), (2.0, 2.0)), ("abcd_same_sign_anti_iso", (4.0, 4.0), (2.0, 2.0))):
        hset[_key("data", region)] = _hist(list(data))
        hset[_key("ZTT", region)] = _hist(list(mc))
    run_estimates(hset, analysis, ["mt"])
    assert hset[_key("QCD", "nominal")].values.tolist() == [12.0, 12.0]  # B=(6,6), C=8, D=4 -> factor 2


def test_abcd_factor_zero_when_control_regions_empty():
    analysis = _with_estimators(build(), ABCD("QCD", "abcd_anti_iso", "abcd_same_sign", "abcd_same_sign_anti_iso", ()))
    hset = HistogramSet()
    hset[_key("data", "abcd_anti_iso")] = _hist([5.0, 5.0])
    hset[_key("data", "abcd_same_sign")] = _hist([0.0, 0.0])
    hset[_key("data", "abcd_same_sign_anti_iso")] = _hist([1.0, 1.0])
    run_estimates(hset, analysis, ["mt"])
    assert hset[_key("QCD", "nominal")].values.tolist() == [0.0, 0.0]


def test_clip_negative_bins_keeps_integral():
    assert clip_negative_bins(_hist([-1.0, 5.0])).values.tolist() == [0.0, 4.0]
    assert clip_negative_bins(_hist([-3.0, 1.0])).values.tolist() == [0.0, 0.0]


def test_estimate_skips_categories_without_data_input():
    hset = HistogramSet()
    hset[_key("ZTT", "anti_iso")] = _hist([1.0, 2.0])
    assert run_estimates(hset, build(), ["mt"]) == []


def test_data_minus_can_clip_negative_bins():
    analysis = build()
    estimator = dataclasses.replace(analysis.channel("mt").estimators[0], clip_negative=True)
    hset = HistogramSet()
    hset[_key("data", "anti_iso")] = _hist([2.0, 10.0])
    hset[_key("ZTT", "anti_iso")] = _hist([3.0, 1.0])
    run_estimates(hset, _with_estimators(analysis, estimator), ["mt"])
    assert hset[_key("jetFakes", "nominal")].values.tolist() == pytest.approx([0.0, 8.0])  # (-1, 9) keeps the integral 8


def test_template_shift_also_shifts_every_template_variation():
    hset = HistogramSet()
    hset[_key("EMB", "nominal")] = _hist([10.0, 10.0])
    hset[_key("EMB", "nominal", variation="es+2")] = _hist([11.0, 9.0])
    hset[_key("EMB", "nominal", variation="CMS_tesUp")] = _hist([12.0, 8.0])  # a shape variation: not shifted
    hset[_key("ZTT", "nominal")] = _hist([2.0, 4.0])
    run_estimates(hset, build_embedding(), ["mt"])
    assert hset[_key("EMB", "nominal", variation="CMS_htt_emb_ttbar_2018Up@es+2")].values.tolist() == pytest.approx([11.2, 9.4])
    assert hset[_key("EMB", "nominal", variation="CMS_htt_emb_ttbar_2018Down@es+2")].values.tolist() == pytest.approx([10.8, 8.6])
    assert not hset.select(process="EMB", variation="CMS_htt_emb_ttbar_2018Up@CMS_tesUp")


def _with_tes_parts(analysis):
    """The tau ES of the embedding as one nuisance from two parts, e.g. CROWN shifts per decay mode."""
    channel = analysis.channel("mt")
    parts = tuple(
        ColumnVariation(part_of(f"CMS_tes{d}", dm), f"__tes_{dm}{d}", applies_to=("mc", "embedding")) for dm in ("dm10", "dm11") for d in ("Up", "Down")
    )
    variations = tuple(v for v in channel.variations if not v.name.startswith("CMS_tes")) + parts
    estimators = (DataMinus("jetFakes", "anti_iso", ("EMB", "ZL")), VariationSum("CMS_tes", ("dm10", "dm11")))
    return dataclasses.replace(analysis, channels={"mt": dataclasses.replace(channel, variations=variations, estimators=estimators)})


def test_variation_sum_adds_the_differences_of_its_parts():
    hset = HistogramSet()
    hset[_key("EMB", "nominal")] = _hist([10.0, 10.0], [1.0, 1.0])
    hset[_key("EMB", "nominal", variation=part_of("CMS_tesUp", "dm10"))] = _hist([11.0, 10.0], [2.0, 2.0])
    hset[_key("EMB", "nominal", variation=part_of("CMS_tesUp", "dm11"))] = _hist([10.0, 13.0], [2.0, 2.0])
    hset[_key("EMB", "nominal", variation=part_of("CMS_tesDown", "dm10"))] = _hist([9.0, 10.0])
    hset[_key("ZL", "nominal")] = _hist([5.0, 5.0])  # no part: no summed variation
    run_estimates(hset, _with_tes_parts(build_embedding()), ["mt"])
    up = hset[_key("EMB", "nominal", variation="CMS_tesUp")]
    assert up.values.tolist() == [11.0, 13.0] and up.variances.tolist() == [1.0, 1.0]  # the nominal variances
    assert hset[_key("EMB", "nominal", variation="CMS_tesDown")].values.tolist() == [9.0, 10.0]  # the one part present
    assert not hset.select(process="ZL", variation="CMS_tesUp")


def test_variation_sum_reaches_the_data_minus_output_through_its_parts():
    hset = HistogramSet()
    hset[_key("data", "anti_iso")] = _hist([10.0, 20.0])
    hset[_key("EMB", "anti_iso")] = _hist([1.0, 2.0])
    hset[_key("EMB", "anti_iso", variation=part_of("CMS_tesUp", "dm10"))] = _hist([2.0, 2.0])
    hset[_key("EMB", "anti_iso", variation=part_of("CMS_tesUp", "dm11"))] = _hist([1.0, 4.0])
    run_estimates(hset, _with_tes_parts(build_embedding()), ["mt"])
    # jetFakes = data - EMB, nominal (9, 18), parts (8, 18) and (9, 16)
    assert hset[_key("jetFakes", "nominal", variation="CMS_tesUp")].values.tolist() == [8.0, 16.0]


"""The W+jets fake factors on the synthetic tests/ff_mini.py analysis (build_w, v7 spec 6.8): the W fake factor from
data minus the other MC and the same-sign QCD, the DR->SR correction from W MC, the non-closures with the same-sign
QCD weighted, the three-process fractions, the payload names of the friend; plus the placeholder edges of an era
without tables and the luminosity share of a measurement over several eras."""
import gzip
import json
import logging

import correctionlib
import numpy as np
import pytest

from shapesmith.config import NtupleConfig, RunConfig
from shapesmith.events import Query, load
from shapesmith.measurements import run_measure
from shapesmith.measurements.fake_factors.hist import Hist
from shapesmith.model import Sample
from shapesmith.skim import normalisation
from shapesmith.validate import validate
from tests import ff_mini
from tests.test_fake_factors import evaluator

PT_EDGES, CATEGORY = (30.0, 50.0, 80.0, 150.0), (1.5, 2.5)


@pytest.fixture(scope="module")
def measured_w(tmp_path_factory):
    root = tmp_path_factory.mktemp("ff_w")
    ff_mini.write_skims(root / "skims")
    config = RunConfig(analysis="tests.ff_mini:build_w", era="2018", channels=["mt"], ntuples=NtupleConfig(base=str(root)), skim_dir=root / "skims", output_dir=root / "out", workers=1)
    analysis = ff_mini.build_w(config)
    validate(analysis)
    output = run_measure(config, analysis, ["mt"])
    payloads = {name: json.loads(gzip.decompress((output / f"{name}_mt.json.gz").read_bytes())) for name in ("fake_factors", "FF_corrections", "FF_for_DRtoSR")}
    record = json.loads((output / "measurement_mt.json").read_text())
    return config, analysis, output, payloads, record


def hist(config, analysis, process, region, variable="pt_2", edges=PT_EDGES, category=CATEGORY, weights=None) -> Hist:
    events = load(config, analysis, Query("mt", process, region), [variable, "n_jets"])
    n_jets = events.frame["n_jets"].to_numpy()
    selected = (n_jets >= category[0]) & (n_jets < category[1])
    w = events.weights if weights is None else weights
    return Hist.fill(edges, events.frame[variable].to_numpy()[selected], w[selected])


def qcd_same_sign(config, analysis, region, **options) -> Hist:
    h = hist(config, analysis, "data", region, **options)
    for process in ff_mini.ALL_MC:
        h = h.add(hist(config, analysis, process, region, **options), -1.0)
    return h.with_negative_bins_zeroed()


def test_the_output_directory_is_that_of_the_variant(measured_w):
    _, _, output, _, _ = measured_w
    assert output.parent.name == "fake_factors_w"


def test_the_w_fake_factors_are_data_minus_mc_minus_same_sign_qcd(measured_w):
    config, analysis, _, _, record = measured_w
    yields = {}
    for region in ("Wjets_sr_like", "Wjets_ar_like"):
        h = hist(config, analysis, "data", region)
        for process in ff_mini.W_SUBTRACT:
            h = h.add(hist(config, analysis, process, region), -1.0)
        yields[region] = h.add(qcd_same_sign(config, analysis, f"{region}_ss"), -1.0)
    expected = yields["Wjets_sr_like"].values / yields["Wjets_ar_like"].values
    assert np.allclose(record["Wjets_fake_factors"][0]["y"], expected, rtol=1e-12)
    without_qcd = hist(config, analysis, "data", "Wjets_sr_like").add(hist(config, analysis, "ZT", "Wjets_sr_like"), -1.0)
    assert not np.allclose(record["Wjets_fake_factors"][0]["numerator"], without_qcd.add(hist(config, analysis, "TTJ", "Wjets_sr_like"), -1.0).values)
    assert record["Wjets_fake_factors"][0]["mc_shifted"] is not None  # the QCD is shifted with the MC


def test_the_w_dr_sr_correction_comes_from_w_mc(measured_w):
    config, analysis, _, payloads, record = measured_w
    orthogonal = record["for_DRtoSR/Wjets_fake_factors"][0]
    sr, ar = hist(config, analysis, "WJ", "Wjets_sr_like"), hist(config, analysis, "WJ", "Wjets_ar_like")
    assert np.allclose(orthogonal["y"], sr.values / ar.values, rtol=1e-12) and orthogonal["mc_shifted"] is None
    ff = evaluator(payloads, "Wjets_fake_factors")  # the measured one; the orthogonal one of FF_for_DRtoSR has the same name
    orthogonal_ff = correctionlib.CorrectionSet.from_string(json.dumps(payloads["FF_for_DRtoSR"]))["Wjets_fake_factors"]
    assert ff.evaluate(40.0, 2.0, "nominal") != orthogonal_ff.evaluate(40.0, 2.0, "nominal")
    events = load(config, analysis, Query("mt", "WJ", "Wjets_ar"), ["pt_2", "n_jets", "mt_1"])
    weights = events.weights * orthogonal_ff.evaluate(events.frame["pt_2"].to_numpy(np.float64), events.frame["n_jets"].to_numpy(np.float64), "nominal")
    predicted = hist(config, analysis, "WJ", "Wjets_ar", "mt_1", (0.0, 25.0, 50.0, 70.0), weights=weights)
    correction = record["Wjets_DR_SR"][0]
    assert np.allclose(correction["denominator"], predicted.values, rtol=1e-9) and correction["mc_shifted"] is None
    assert np.allclose(correction["numerator"], hist(config, analysis, "WJ", "Wjets_sr", "mt_1", (0.0, 25.0, 50.0, 70.0)).values, rtol=1e-12)
    dr_sr = evaluator(payloads, "Wjets_DR_SR_correction")
    assert dr_sr.evaluate(30.0, 2.0, "Wjets_DR_SR_CorrSystMCShiftUp") == dr_sr.evaluate(30.0, 2.0, "nominal")


def test_the_w_non_closure_subtracts_the_weighted_same_sign_qcd(measured_w):
    config, analysis, _, payloads, record = measured_w
    edges = (0.0, 30.0, 60.0, 200.0)
    ff, dr_sr = evaluator(payloads, "Wjets_fake_factors"), evaluator(payloads, "Wjets_DR_SR_correction")

    def weighted(process, region):
        events = load(config, analysis, Query("mt", process, region), ["pt_2", "n_jets", "mt_1", "met"])
        frame = events.frame
        w = events.weights * ff.evaluate(frame["pt_2"].to_numpy(np.float64), frame["n_jets"].to_numpy(np.float64), "nominal")
        w = w * dr_sr.evaluate(frame["mt_1"].to_numpy(np.float64), frame["n_jets"].to_numpy(np.float64), "nominal")
        return hist(config, analysis, process, region, "met", edges, weights=w)

    predicted = weighted("data", "Wjets_ar_like")
    for process in ff_mini.W_SUBTRACT:
        predicted = predicted.add(weighted(process, "Wjets_ar_like"), -1.0)
    qcd = weighted("data", "Wjets_ar_like_ss")
    for process in ff_mini.ALL_MC:
        qcd = qcd.add(weighted(process, "Wjets_ar_like_ss"), -1.0)
    predicted = predicted.add(qcd.with_negative_bins_zeroed(), -1.0)
    assert np.allclose(record["Wjets_non_closure_met"][0]["denominator"], predicted.values, rtol=1e-9)


def test_three_process_fractions_and_the_payload_names(measured_w):
    _, _, _, payloads, record = measured_w
    fractions = evaluator(payloads, "process_fractions")
    for key in ("nominal", "process_fractionsfrac_Wjets_up", "process_fractionsfrac_QCD_down"):
        total = sum(fractions.evaluate(process, 40.0, 2.0, key) for process in ("QCD", "Wjets", "ttbar"))
        assert total == pytest.approx(1.0)
    assert list(record["process_fractions"][0]["fractions"]) == ["QCD", "Wjets", "ttbar_J"]
    fake_factor_names = [c["name"] for c in payloads["fake_factors"]["corrections"]]
    assert fake_factor_names == ["QCD_fake_factors", "Wjets_fake_factors", "ttbar_fake_factors", "process_fractions"]
    corrections = {c["name"] for c in payloads["FF_corrections"]["corrections"]}
    assert {"Wjets_DR_SR_correction", "Wjets_non_closure_met_correction", "QCD_DR_SR_correction"} <= corrections
    assert "Wjets_compound_correction" in {c["name"] for c in payloads["FF_corrections"]["compound_corrections"]}
    keys = [item["key"] for item in next(c for c in payloads["fake_factors"]["corrections"] if c["name"] == "Wjets_fake_factors")["data"]["content"]]
    assert "WjetsSystMCShiftUp" in keys and "WjetsStatShiftDown" in keys


def test_suggest_binning_of_the_w_dr_sr_uses_w_mc_and_logs_json(measured_w, caplog, tmp_path):
    config, analysis, _, _, _ = measured_w
    config = config.model_copy(update={"output_dir": tmp_path / "suggest"})
    with caplog.at_level(logging.INFO, logger="shapesmith"):
        run_measure(config, analysis, ["mt"], suggest_binning=True)
    line = next(r.getMessage() for r in caplog.records if "suggested edges (JSON)" in r.getMessage())
    suggested = json.loads(line.split("(JSON): ", 1)[1])
    assert len(suggested["Wjets_DR_SR"]) == 2 and suggested["Wjets_DR_SR"][0][0] == 0.0 and suggested["Wjets_DR_SR"][0][-1] == 70.0
    assert len(suggested["Wjets_fake_factors"]) == 2 and "Wjets_non_closure_met" in suggested


def test_placeholder_edges_allow_only_suggest_binning(measured_w, tmp_path):
    import dataclasses

    config, analysis, _, _, _ = measured_w
    config = config.model_copy(update={"output_dir": tmp_path / "placeholder"})
    placeholder = dataclasses.replace(analysis, measurement=dataclasses.replace(analysis.measurement, placeholder_edges="no 2017 tables"))
    with pytest.raises(ValueError, match=r"placeholders \(no 2017 tables\)"):
        run_measure(config, placeholder, ["mt"])
    run_measure(config, placeholder, ["mt"], suggest_binning=True)


def test_a_sample_carries_its_share_of_the_luminosity():
    half = Sample("TT_2016preVFP", "TT", "mc", xsec=2.0, nevents=100, generator_weight=0.5, era="2016preVFP", lumi_scale=19520 / 36330)
    assert half.norm_weight == pytest.approx(2.0 / 50 * 19520 / 36330)
    assert normalisation(half)["lumi_scale"] == pytest.approx(0.5373, abs=1e-4)
    assert "lumi_scale" not in normalisation(Sample("TT", "TT", "mc", 2.0, 100, 0.5))  # earlier skims stay reusable
    assert Sample("DATA", "data", "data", lumi_scale=0.5).norm_weight == 1.0

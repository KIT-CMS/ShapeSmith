"""The fake-factor measurement chain on the synthetic tests/ff_mini.py analysis: ratios, SystMCShift, fractions,
the chaining of the corrections and the payload conventions of the CROWN fake-factor friend."""
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
from shapesmith.validate import validate
from tests import ff_mini

DIRECTIONS = ("Up", "Down", "_up", "_down")


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    root = tmp_path_factory.mktemp("ff")
    ff_mini.write_skims(root / "skims")
    config = RunConfig(analysis="tests.ff_mini:build", era="2018", channels=["mt"], ntuples=NtupleConfig(base=str(root)), skim_dir=root / "skims", output_dir=root / "out", workers=1)
    analysis = ff_mini.build(config)
    validate(analysis)
    output = run_measure(config, analysis, ["mt"])
    payloads = {name: json.loads(gzip.decompress((output / f"{name}_mt.json.gz").read_bytes())) for name in ("fake_factors", "FF_corrections")}
    record = json.loads((output / "measurement_mt.json").read_text())
    return config, analysis, output, payloads, record


def hist(config, analysis, process, region, variable, edges, category_edges, weights=None) -> Hist:
    events = load(config, analysis, Query("mt", process, region), [variable, "n_jets"])
    n_jets = events.frame["n_jets"].to_numpy()
    selected = (n_jets >= category_edges[0]) & (n_jets < category_edges[1])
    w = events.weights if weights is None else weights
    return Hist.fill(edges, events.frame[variable].to_numpy()[selected], w[selected])


def evaluator(payloads, name):
    for payload in payloads.values():
        cset = correctionlib.CorrectionSet.from_string(json.dumps(payload))
        if name in {c["name"] for c in payload["corrections"]}:
            return cset[name]
        if name in {c["name"] for c in payload.get("compound_corrections") or []}:
            return cset.compound[name]
    raise KeyError(name)


def test_the_qcd_fake_factors_are_data_minus_mc_over_data_minus_mc(measured):
    config, analysis, _, _, record = measured
    edges, category = (30.0, 50.0, 80.0, 150.0), (1.5, 2.5)
    yields = {region: hist(config, analysis, "data", region, "pt_2", edges, category) for region in ("QCD_sr_like", "QCD_ar_like")}
    for region in yields:
        for process in ff_mini.SUBTRACT:
            yields[region] = yields[region].add(hist(config, analysis, process, region, "pt_2", edges, category), -1.0)
    expected = yields["QCD_sr_like"].values / yields["QCD_ar_like"].values
    assert np.allclose(record["QCD_fake_factors"][0]["y"], expected, rtol=1e-12)


def test_systmcshift_subtracts_the_mc_shifted_by_its_statistical_error_not_a_scaled_mc(measured):
    config, analysis, _, payloads, _ = measured
    edges, category = (30.0, 50.0, 80.0, 150.0), (1.5, 2.5)
    shifted = {}
    for region in ("QCD_sr_like", "QCD_ar_like"):
        h = hist(config, analysis, "data", region, "pt_2", edges, category)
        for process in ff_mini.SUBTRACT:
            h = h.add(hist(config, analysis, process, region, "pt_2", edges, category).shifted(1.0), -1.0)
        shifted[region] = h
    ratio = shifted["QCD_sr_like"].values / shifted["QCD_ar_like"].values
    ff = evaluator(payloads, "QCD_fake_factors")
    nominal = ff.evaluate(40.0, 2.0, "nominal")
    up = ff.evaluate(40.0, 2.0, "QCDSystMCShiftUp")
    assert up != nominal and np.isclose(up, ratio[0], rtol=0.15)  # the smoothed curve follows the shifted ratio
    assert (up - nominal) * (ratio[0] - nominal) > 0


def test_the_ttbar_fake_factors_carry_the_data_scale_and_no_mc_shift(measured):
    _, _, _, payloads, record = measured
    ff = evaluator(payloads, "ttbar_fake_factors")
    for pt in (35.0, 90.0):
        assert ff.evaluate(pt, 3.0, "ttbarSystMCShiftUp") == ff.evaluate(pt, 3.0, "nominal") == ff.evaluate(pt, 3.0, "ttbarSystMCShiftDown")
    scale = record["ttbar_data_scale"][0]["factor"]
    raw = np.array(record["ttbar_fake_factors"][0]["numerator"]) / np.array(record["ttbar_fake_factors"][0]["denominator"])
    assert np.allclose(record["ttbar_fake_factors"][0]["y"], raw * scale, rtol=1e-12) and scale != 1.0


def test_the_fractions_sum_to_one_in_every_variation(measured):
    _, _, _, payloads, _ = measured
    fractions = evaluator(payloads, "process_fractions")
    for key in ("nominal", "process_fractionsfrac_QCD_up", "process_fractionsfrac_ttbar_J_down"):
        for x, n_jets in ((40.0, 2.0), (100.0, 4.0)):
            assert fractions.evaluate("QCD", x, n_jets, key) + fractions.evaluate("ttbar", x, n_jets, key) == pytest.approx(1.0)


def test_each_category_of_the_split_reads_its_own_payload_bin(measured):
    _, _, _, payloads, record = measured
    ff = evaluator(payloads, "QCD_fake_factors")
    values = [ff.evaluate(35.0, n_jets, "nominal") for n_jets in (2.0, 3.0, 5.0)]
    assert values[0] != values[1] and values[1] == values[2]


def test_the_compound_correction_is_the_product_of_the_non_closures(measured):
    _, _, _, payloads, _ = measured
    compound = evaluator(payloads, "QCD_compound_correction")
    parts = [evaluator(payloads, "QCD_non_closure_tau_decaymode_2_correction"), evaluator(payloads, "QCD_non_closure_met_correction")]
    for dm, met, n_jets in ((0.0, 20.0, 2.0), (10.0, 90.0, 3.0)):
        product = parts[0].evaluate(dm, n_jets, "nominal") * parts[1].evaluate(met, n_jets, "nominal")
        assert compound.evaluate(dm, met, n_jets, "nominal") == pytest.approx(product)


def test_the_non_closures_are_measured_on_top_of_the_dr_sr_correction(measured):
    config, analysis, _, payloads, record = measured
    edges, category = (-0.5, 0.5, 9.5, 10.5), (1.5, 2.5)
    ff, dr_sr = evaluator(payloads, "QCD_fake_factors"), evaluator(payloads, "QCD_DR_SR_correction")
    predicted = None
    for process in ("data", *ff_mini.SUBTRACT):
        events = load(config, analysis, Query("mt", process, "QCD_ar_like"), ["pt_2", "pt_tautau", "n_jets", "tau_decaymode_2"])
        frame = events.frame
        weights = events.weights * ff.evaluate(frame["pt_2"].to_numpy(np.float64), frame["n_jets"].to_numpy(np.float64), "nominal")
        weights = weights * dr_sr.evaluate(frame["pt_tautau"].to_numpy(np.float64), frame["n_jets"].to_numpy(np.float64), "nominal")
        h = hist(config, analysis, process, "QCD_ar_like", "tau_decaymode_2", edges, category, weights)
        predicted = h if predicted is None else predicted.add(h, -1.0)
    assert np.allclose(record["QCD_non_closure_tau_decaymode_2"][0]["denominator"], predicted.values, rtol=1e-9)


def _keys(node) -> list[str]:
    return [item["key"] for item in node["content"]]


def test_the_payloads_follow_the_friend_conventions(measured):
    _, _, output, payloads, _ = measured
    names = {c["name"] for p in payloads.values() for c in p["corrections"]}
    assert {"QCD_fake_factors", "ttbar_fake_factors", "process_fractions", "QCD_DR_SR_correction", "QCD_non_closure_met_correction", "ttbar_non_closure_met_correction"} <= names
    for payload in payloads.values():
        assert "$schema" not in payload and json.loads(payload["description"])["channel"] == "mt"
        for correction in payload["corrections"]:
            inputs = correction["inputs"]
            assert inputs[-1] == {"name": "syst", "type": "string"} and correction["data"]["nodetype"] == "category" and correction["data"]["input"] == "syst"
            assert all(v["type"] == ("string" if v["name"] == "process" else "real") for v in inputs[:-1])
            assert correction["data"].get("default") is not None
            keys = [k for k in _keys(correction["data"]) if not k.endswith("nominal")]
            assert all(k.endswith(DIRECTIONS) for k in keys)
            for key in keys:
                partner = next(key[: -len(d)] + o for d, o in (("Up", "Down"), ("Down", "Up"), ("_up", "_down"), ("_down", "_up")) if key.endswith(d))
                assert partner in keys, key
            if correction["name"] == "process_fractions":
                assert _keys(correction["data"]["default"]) == ["QCD", "ttbar"]
    by_name = {c["name"]: c for c in payloads["FF_corrections"]["corrections"]}
    for compound in payloads["FF_corrections"]["compound_corrections"]:
        process = compound["name"].removesuffix("_compound_correction")
        for member in compound["stack"]:
            assert f"{process}_non_closure_CorrStatShiftUp" in _keys(by_name[member]["data"])
    assert (output / "FF_for_DRtoSR_mt.json.gz").exists() and list((output / "plots" / "mt").glob("*.png"))


def test_suggest_binning_writes_nothing(measured, caplog, tmp_path):
    config, analysis, _, _, _ = measured
    config = config.model_copy(update={"output_dir": tmp_path / "suggest"})
    with caplog.at_level(logging.INFO, logger="shapesmith"):
        run_measure(config, analysis, ["mt"], suggest_binning=True)
    assert "mt QCD_fake_factors category 0: suggested [30.0," in caplog.text and not (tmp_path / "suggest").exists()


def test_merge_is_refused(measured, tmp_path):
    config, analysis, _, _, _ = measured
    config = config.model_copy(update={"output_dir": tmp_path / "merge"})
    with pytest.raises(ValueError, match="--merge is not supported"):
        run_measure(config, analysis, ["mt"], merge=True)

"""The tau-leg trigger measurement (v7 spec E33, v7_tt_emb_trigger 12.2/12.5/12.6): card text and rateParams, the fit
with a fake CMSSW, the counting, the payload schema and the merge checks, on tests/trigger_mini.py."""
import json
import logging
import re

import correctionlib
import numpy as np
import pytest
import uproot

from shapesmith import cmssw, payloads
from shapesmith.config import CombineConfig, NtupleConfig, RunConfig
from shapesmith.histogram import Histogram
from shapesmith.measurements import MeasureContext, run_measure
from shapesmith.measurements.tau_trigger import Bin, check, merge, merged_values
from shapesmith.measurements.tau_trigger.cards import TriggerBin, card_text, fake_kappas, rate_parameters
from shapesmith.measurements.tau_trigger.counting import Count, data_efficiency, efficiency
from shapesmith.measurements.tau_trigger.fit import COARSE_RANGE, fit_bin, scan_command, workspace_command
from shapesmith.measurements.tau_trigger.payload import PT_MAX, check as check_payload, correction_set, edges_of, payload_name
from shapesmith.validate import validate
from tests import trigger_mini

EDGES = (30.0, 60.0, 90.0)


def _h(*values):
    return Histogram(EDGES, list(values), list(values))


def _bin():
    backgrounds = {"pass": {"ZJ": _h(2.0, 2.0), "QCD": _h(1.0, 1.0), "TTL": _h(0.0, 0.0)}, "fail": {"ZJ": _h(8.0, 8.0), "QCD": _h(4.0, 4.0), "TTL": _h(1.0, 1.0)}}
    return TriggerBin("DM0_PT40_45", {"pass": _h(20.0, 20.0), "fail": _h(15.0, 15.0)}, _h(25.0, 25.0), backgrounds, {"emb_esUp": _h(26.0, 24.0), "emb_esDown": _h(24.0, 26.0)})


def test_the_card_has_pass_and_fail_with_the_efficiency_as_rate_parameter():
    text = card_text(_bin(), "EMB", {"norm_zj": (("ZJ",), 1.2), "norm_tt": (("TTL",), 1.1)}, ("ZJ", "QCD"), 0.1)
    lines = text.splitlines()
    assert "shapes * * shapes.root $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC" in lines
    assert "bin DM0_PT40_45_pass DM0_PT40_45_fail" in lines and "observation 40.0 30.0" in lines
    # TTL has no pass yield and stays out of the pass bin; the signal is in both, every process index positive
    assert "bin DM0_PT40_45_pass DM0_PT40_45_pass DM0_PT40_45_pass DM0_PT40_45_fail DM0_PT40_45_fail DM0_PT40_45_fail DM0_PT40_45_fail" in lines
    assert "process EMB ZJ QCD EMB ZJ QCD TTL" in lines and "process 1 2 3 1 2 3 4" in lines
    assert "norm_zj lnN - 1.2 - - 1.2 - -" in lines and "norm_tt lnN - - - - - - 1.1" in lines
    assert "CMS_tauTrg_fake_efficiency lnN - 1.1000 1.1000 - 0.9750 0.9750 -" in lines  # fail: 1 - 0.1 * P / F
    assert "emb_es shape 1 - - 1 - - -" in lines
    assert lines[-4:] == rate_parameters("DM0_PT40_45", "EMB") + ["* autoMCStats 0"]
    assert rate_parameters("DM0_PT40_45", "EMB") == [
        "eff rateParam DM0_PT40_45_pass EMB 0.8 [0,1]", "eff_fail rateParam DM0_PT40_45_fail EMB (1-@0) eff", "r_EMB rateParam * EMB 1 [0,5]"]
    assert fake_kappas(_bin(), "ZJ", 0.1) == pytest.approx((1.1, 0.975))
    empty = _bin()
    empty.signal = _h(0.0, 0.0)
    with pytest.raises(ValueError, match="signal template is empty"):
        card_text(empty, "EMB", {}, (), 0.1)


def test_counting_efficiencies():
    assert efficiency(Count(80.0, 80.0), Count(20.0, 20.0)) == pytest.approx({"value": 0.8, "error": 0.04, "pass": 80.0, "fail": 20.0})
    counts = {("data", "os", "pass"): Count(100.0, 100.0), ("data", "os", "fail"): Count(60.0, 60.0), ("data", "ss", "pass"): Count(10.0, 10.0),
              ("data", "ss", "fail"): Count(20.0, 20.0), ("ZJ", "os", "pass"): Count(10.0, 1.0), ("ZJ", "os", "fail"): Count(20.0, 2.0),
              ("ZJ", "ss", "pass"): Count(15.0, 1.0), ("ZJ", "ss", "fail"): Count(5.0, 1.0)}
    eff = data_efficiency(lambda process, region, side: counts[process, region, side], "data", ("ZJ",), ("ZJ",), "os", "ss")
    assert eff["pass"] == pytest.approx(100 - 10 - 0) and eff["fail"] == pytest.approx(60 - 20 - 15)  # QCD in pass clipped at 0
    assert eff["value"] == pytest.approx(90 / 115)


def _results():
    """Two classes; Medium with DM10 and DM11 from one bin."""
    def bins(eff):
        return [(40.0, 60.0, {"nom": eff, "up": eff + 0.02, "down": eff - 0.02}), (60.0, None, {"nom": eff + 0.05, "up": eff + 0.06, "down": eff + 0.03})]
    return {wp: {"eff_data": {dm: bins(base) for dm in (-1, 0, 1, 10, 11)}, "eff_mc": {dm: bins(base + 0.05) for dm in (-1, 0, 1, 10, 11)}} for wp, base in (("Medium", 0.8), ("VVVLoose_not_Medium", 0.6))}


def test_the_payload_schema_follows_kit_and_pog_with_flow_clamp():
    cset = correction_set(_results(), {"era": "2018"})
    payload = json.loads(payloads.dumps(cset))
    (correction,) = payload["corrections"]
    assert correction["name"] == "tau_trigger" and [(v["name"], v["type"]) for v in correction["inputs"]] == [
        ("pt", "real"), ("dm", "int"), ("trigtype", "string"), ("wp", "string"), ("corrtype", "string"), ("syst", "string")]
    evaluator = correctionlib.CorrectionSet.from_string(payloads.dumps(cset))["tau_trigger"]
    assert evaluator.evaluate(50.0, 1, "ditau", "Medium", "eff_data", "nom") == pytest.approx(0.8)
    assert evaluator.evaluate(30.0, -1, "ditau", "Medium", "eff_data", "up") == pytest.approx(0.82)  # below the first edge: clamp
    assert evaluator.evaluate(5000.0, 11, "ditau", "VVVLoose_not_Medium", "eff_mc", "down") == pytest.approx(0.68)  # beyond PT_MAX: clamp
    assert payload_name("2016preVFP") == "ditau_trigger_DeepTau2018v2p5_embedding2016preVFPUL.json.gz"
    assert check_payload(cset, _results()) == []
    assert edges_of(_results()["Medium"]["eff_data"][0], "x") == [40.0, 60.0, PT_MAX]
    with pytest.raises(ValueError, match="not contiguous at 60 / 70"):
        edges_of([(40.0, 60.0, {}), (70.0, None, {})], "Medium DM0")
    missing = _results()
    del missing["Medium"]["eff_data"][11]
    with pytest.raises(ValueError, match=r"Medium eff_data: no bins for the decay modes \[11\]"):
        correction_set(missing, {})


def test_the_payload_check_finds_values_outside_and_unordered():
    bad = _results()
    bad["Medium"]["eff_data"][0] = [(40.0, 60.0, {"nom": 1.02, "up": 1.03, "down": 1.0}), (60.0, None, {"nom": 0.8, "up": 0.7, "down": 0.75})]
    problems = check_payload(correction_set(bad, {}), bad)
    assert any("outside (0, 1]" in p for p in problems) and any("down <= nom <= up violated" in p for p in problems)


class FakeCombine:
    """cmssw.run for the trigger fit: the delta NLL of a Gaussian in eff around `mean` with `sigma` (the fine scans may
    have their own width)."""

    def __init__(self, mean=0.8, sigma=0.02, fine_sigma=None):
        self.mean, self.sigma, self.fine_sigma, self.commands = mean, sigma, fine_sigma, []

    def __call__(self, commands, config, cwd, timeout=None):
        for command in commands:
            self.commands.append(command)
            name = re.search(r" -n \.(\S+)", command)
            if name is None:
                continue
            name = name.group(1)
            if "--algo grid" in command:
                low, high = (float(x) for x in re.search(r"--setParameterRanges eff=([-\d.e]+),([-\d.e]+)", command).groups())
                x = np.linspace(low, high, 41)
                sigma = self.sigma if name == "coarse" or self.fine_sigma is None else self.fine_sigma
                self.write(cwd, name, np.r_[self.mean, x], np.r_[0.0, 0.5 * ((x - self.mean) / sigma) ** 2])
            elif "--algo singles" in command:
                self.write(cwd, name, np.array([self.mean, self.mean - self.sigma, self.mean + self.sigma]), np.zeros(3))

    @staticmethod
    def write(cwd, name, eff, dnll):
        with uproot.recreate(cwd / f"higgsCombine.{name}.MultiDimFit.mH125.root") as f:
            f["limit"] = {"eff": eff.astype(np.float32), "deltaNLL": dnll.astype(np.float32)}


def test_the_fit_is_the_profile_interval_with_rescans_and_fails_at_the_boundary(tmp_path, monkeypatch):
    runner = FakeCombine()
    monkeypatch.setattr(cmssw, "run", runner)
    result = fit_bin(None, tmp_path)
    assert result["problems"] == [] and result["widenings"] == 0
    assert (result["fit"]["best"], result["fit"]["low"], result["fit"]["high"]) == pytest.approx((0.8, 0.78, 0.82), abs=1e-3)
    assert runner.commands[0] == workspace_command() + " > workspace.log 2>&1" and "--X-allow-no-signal" in runner.commands[0]
    assert runner.commands[1].startswith(scan_command(COARSE_RANGE, 0.8, "coarse"))
    assert "--redefineSignalPOIs eff --floatOtherPOIs 1" in runner.commands[2] and result["singles"]["best"] == pytest.approx(0.8)
    assert any("--algo none" in c and "--freezeParameters eff" in c for c in runner.commands)
    wide = FakeCombine(fine_sigma=0.06)  # the fine profile wider than the window of the coarse one
    monkeypatch.setattr(cmssw, "run", wide)
    result = fit_bin(None, tmp_path / "wide")
    assert result["widenings"] >= 1 and result["fit"]["low"] == pytest.approx(0.74, abs=2e-3) and result["problems"] == []
    monkeypatch.setattr(cmssw, "run", FakeCombine(mean=0.995, sigma=0.01))
    result = fit_bin(None, tmp_path / "edge")
    assert result["fit"] is None and any("eff: no crossing of delta NLL = 0.5 on the high side" in p for p in result["problems"])


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    root = tmp_path_factory.mktemp("trigger")
    trigger_mini.write_skims(root / "skims")
    config = RunConfig(analysis="tests.trigger_mini:build", era="2018", channels=["mt"], ntuples=NtupleConfig(base=str(root)), skim_dir=root / "skims",
                       output_dir=root / "out", workers=1, combine=CombineConfig(cmssw_dir="/unused"))
    analysis = trigger_mini.build(config)
    validate(analysis)
    return config, analysis


def test_measure_writes_cards_results_and_plots(mini, monkeypatch):
    config, analysis = mini
    monkeypatch.setattr(cmssw, "run", FakeCombine())
    output = run_measure(config, analysis, ["mt"])
    directory = output / "Medium" / "nominal"
    record = json.loads((directory / "results.json").read_text())
    assert set(record["bins"]) == {b.name for b in trigger_mini.BINS} and record["wp_class"] == "Medium"
    entry = record["bins"]["DM0_PT40_60"]
    assert entry["fit"]["best"] == pytest.approx(0.8, abs=1e-3) and entry["problems"] == []
    assert entry["mc"]["value"] == pytest.approx(trigger_mini.EFFICIENCY[0], abs=0.03)  # the genuine MC taus of DM0
    assert entry["emb_monitor"]["value"] == pytest.approx(trigger_mini.EFFICIENCY[0], abs=0.03)
    assert 0.0 < entry["counting"]["value"] < 1.0 and entry["counting"]["error"] > 0
    card = (directory / "fits" / "DM0_PT40_60" / "card.txt").read_text()
    assert "eff rateParam DM0_PT40_60_pass EMB 0.8 [0,1]" in card and "emb_es_shift shape" in card and "QCD" in card
    with uproot.open(directory / "fits" / "DM0_PT40_60" / "shapes.root") as f:
        signal_pass, signal_fail = f["DM0_PT40_60_pass/EMB"].values(), f["DM0_PT40_60_fail/EMB"].values()
        assert np.array_equal(signal_pass, signal_fail)  # one template, pass + fail
        assert f["DM0_PT40_60_pass/EMB_emb_es_shiftUp"].values().sum() != signal_pass.sum()
    binning = json.loads((output / "Medium" / "nominal" / "binning.json").read_text())["mt"]
    assert binning["DM0_PT40_60_pass"]["m_vis"]["edges"] == binning["DM0_PT40_60_fail"]["m_vis"]["edges"]  # EqualData with the bin cut
    assert (directory / "report.md").exists() and (directory / "plots" / "eff_DM0.png").exists() and (directory / "plots" / "eff_DMall.png").exists()


def test_refit_reuses_the_shapes_and_the_counting(mini, monkeypatch):
    config, analysis = mini
    monkeypatch.setattr(cmssw, "run", FakeCombine())
    run_measure(config, analysis, ["mt"])
    monkeypatch.setattr(cmssw, "run", FakeCombine(mean=0.7))
    directory = run_measure(config, analysis, ["mt"], refit=True) / "Medium" / "nominal"
    record = json.loads((directory / "results.json").read_text())
    assert record["provenance"]["refit"] is True and record["bins"]["DMall_PT60"]["fit"]["best"] == pytest.approx(0.7, abs=1e-3)
    assert record["bins"]["DMall_PT60"]["counting"] is not None


def test_suggest_binning_merges_from_the_top(mini, caplog):
    config, analysis = mini
    import dataclasses

    loose = dataclasses.replace(analysis, measurement=dataclasses.replace(analysis.measurement, max_relative_error=0.5))
    strict = dataclasses.replace(analysis, measurement=dataclasses.replace(analysis.measurement, max_relative_error=1e-6))
    with caplog.at_level(logging.INFO, logger="shapesmith"):
        run_measure(config.model_copy(update={"output_dir": config.output_dir / "suggest"}), loose, ["mt"], suggest_binning=True)
        run_measure(config.model_copy(update={"output_dir": config.output_dir / "suggest"}), strict, ["mt"], suggest_binning=True)
    lines = [r.getMessage() for r in caplog.records if "suggested edges (JSON)" in r.getMessage()]
    assert json.loads(lines[0].split("(JSON): ")[1]) == {"DM0": [40.0, 60.0, None], "DMall": [40.0, 60.0, None]}  # both bins qualify
    assert json.loads(lines[1].split("(JSON): ")[1]) == {"DM0": [40.0, None], "DMall": [40.0, None]}  # none qualifies: one bin
    assert not (config.output_dir / "suggest").exists()


MERGE_BINS = (*trigger_mini.BINS, Bin("DM1_PT40", (1,), (40.0, None)), Bin("DM1011_PT40", (10, 11), (40.0, None)))


def _write_record(directory, wp, variant, eff, problems=(), counting=None):
    bins = {}
    for b in MERGE_BINS:
        bins[b.name] = {"decay_modes": list(b.decay_modes), "pt": list(b.pt), "fit": {"best": eff, "low": eff - 0.02, "high": eff + 0.02}, "problems": list(problems),
                        "warnings": [], "counting": counting, "mc": {"value": min(eff + 0.05, 0.99), "error": 0.01}, "emb_monitor": None}
    path = directory / wp / variant / "results.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"wp_class": wp, "variant": variant, "bins": bins, "provenance": {"run": f"{wp}/{variant}"}}))


def test_merge_adds_the_variants_and_checks_the_payload(mini, tmp_path):
    config, analysis = mini
    measurement = __import__("dataclasses").replace(analysis.measurement, systematic_variants={"tag_pt": 0.5, "signal_mc": 1.0})
    _write_record(tmp_path, "Medium", "nominal", 0.80, counting={"value": 0.70, "error": 0.01})
    _write_record(tmp_path, "Medium", "tag_pt", 0.84)
    context = MeasureContext(config, analysis, ["mt"], tmp_path, merge=True)
    path = merge(measurement, context)
    assert path.name == "ditau_trigger_DeepTau2018v2p5_embedding2018UL.json.gz"
    evaluator = correctionlib.CorrectionSet.from_string(payloads.dumps(payloads.read(path)))["tau_trigger"]
    # up: the fit 0.02, half the tag_pt change 0.02 and the fit-counting difference 0.10 (beyond 1 sigma) in quadrature
    expected = (0.02 ** 2 + 0.02 ** 2 + 0.10 ** 2) ** 0.5
    assert evaluator.evaluate(50.0, 0, "ditau", "Medium", "eff_data", "up") == pytest.approx(0.8 + expected)
    assert evaluator.evaluate(50.0, 10, "ditau", "Medium", "eff_data", "down") == evaluator.evaluate(50.0, 11, "ditau", "Medium", "eff_data", "down") == pytest.approx(0.8 - expected)  # one DM10+11 bin
    assert evaluator.evaluate(80.0, -1, "ditau", "Medium", "eff_mc", "nom") == pytest.approx(0.85)
    report = (tmp_path / "report.md").read_text()
    assert "Medium: variant signal_mc missing" in report and (tmp_path / "plots" / "Medium" / "eff_DM0.png").exists()
    _, _, warnings = merged_values(measurement, {"nominal": json.loads((tmp_path / "Medium" / "nominal" / "results.json").read_text())})
    assert len(warnings) == 2


def test_merge_refuses_problems_and_a_missing_nominal(mini, tmp_path):
    config, analysis = mini
    _write_record(tmp_path / "a", "Medium", "nominal", 0.8, problems=["eff: no crossing of delta NLL = 0.5 on the high side"])
    with pytest.raises(ValueError, match="Medium/nominal/DM0_PT40_60: eff: no crossing"):
        merge(analysis.measurement, MeasureContext(config, analysis, ["mt"], tmp_path / "a", merge=True))
    _write_record(tmp_path / "b", "Medium", "tag_pt", 0.8)
    with pytest.raises(ValueError, match="Medium: the nominal variant is missing"):
        merge(analysis.measurement, MeasureContext(config, analysis, ["mt"], tmp_path / "b", merge=True))
    _write_record(tmp_path / "c", "Medium", "nominal", 0.995)  # up beyond 1 is cut at 1; the payload stays valid
    path = merge(analysis.measurement, MeasureContext(config, analysis, ["mt"], tmp_path / "c", merge=True))
    assert correctionlib.CorrectionSet.from_string(payloads.dumps(payloads.read(path)))["tau_trigger"].evaluate(50.0, 0, "ditau", "Medium", "eff_data", "up") == 1.0


def test_check_names_what_is_missing(mini):
    _, analysis = mini
    import dataclasses

    broken = dataclasses.replace(analysis.measurement, bins=(*analysis.measurement.bins, Bin("DM1_PT40_60", (1,), (40.0, 60.0))), signal_template="es_unknown")
    with pytest.raises(ValueError, match="category DM1_PT40_60_pass is missing.*the signal template es_unknown"):
        check(analysis, broken)

"""The statistics of the NN phase (spec 7.2, SH9-SH12): combine options of the run configuration, the masks of
blinding.json in every fit to data, the statistical checks (GoF, impacts, a-posteriori limit, injection, HybridNew)
with CMSSW faked by synthetic combine outputs, and the variants of the datacards."""
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import uproot
from pydantic import ValidationError
from typer.testing import CliRunner

from shapesmith import checks, cmssw, combine_options, masking
from shapesmith.background_fit import fit_commands
from shapesmith.checks import Target
from shapesmith.cli import app
from shapesmith.config import CombineConfig
from shapesmith.limits import collect, limit_commands, limit_overview, write_summary
from shapesmith.model import CrossSection
from shapesmith.plotting import fit_plots
from shapesmith.plotting.fit_plots import card_binning, fit_stack, pull_plot, shown_data
from tests.mini_analysis import build
from tests.test_cli import _write_config

runner = CliRunner()
XS = CrossSection("ggF HH", 30.40, ("QCDscale_HH", "PDF_alphas_HH"))
QUANTILES = [0.025, 0.16, 0.5, 0.84, 0.975]


def _limit_tree(path, quantiles, limits, extra=None):
    with uproot.recreate(path) as f:
        f["limit"] = {"quantileExpected": np.array(quantiles, dtype=np.float32), "limit": np.array(limits, dtype=np.float64), **(extra or {})}


def _card(directory: Path, bins=("htt_mt_1_2018", "htt_mt_2hi_2018"), nuisances=("lumi", "QCDscale_HH")) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["imax *", "jmax *", "kmax *", "shapes * * common/htt_input_2018.root $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC",
             "bin " + " ".join(bins), "observation " + " ".join("10" for _ in bins),
             "bin " + " ".join(b for b in bins for _ in (0, 1)), "process " + " HH ZTT" * len(bins), "process " + " 0 1" * len(bins), "rate " + " -1" * 2 * len(bins)]
    lines += [f"{name} lnN " + " 1.02" * 2 * len(bins) for name in nuisances] + ["* autoMCStats 0"]
    (directory / "combined.txt").write_text("\n".join(lines) + "\n")
    return directory


def _blinding(directory: Path, masked=("htt_mt_2hi_2018",)) -> None:
    content = {"edge_rule": "S/B(mu=1) > 0.005", "masked": list(masked), "categories": {name: {"lo": name.replace("hi", "lo"), "edge": 0.8, "s_over_b_first_hi_bin": 0.006} for name in masked}, "asimov_data": True}
    (directory / "blinding.json").write_text(json.dumps(content))


# -- combine options (SH10) ------------------------------------------------------------------------------------------


def test_the_minimizer_options_replace_built_in_ones_and_keep_repeated_flags():
    merged = combine_options.merge("combine -M FitDiagnostics -t -1 --cminDefaultMinimizerStrategy 0 --X-rtd A --X-rtd B -n .x", "--cminDefaultMinimizerStrategy 1")
    assert merged == "combine -M FitDiagnostics -t -1 --X-rtd A --X-rtd B -n .x --cminDefaultMinimizerStrategy 1"
    assert combine_options.merge("combine --setParameterRanges r=-40,40", "--setParameterRanges r=-5,20") == "combine --setParameterRanges r=-5,20"
    assert combine_options.split_options("combine -t -1 --setParameterRanges r=-40,40") == [["combine"], ["-t", "-1"], ["--setParameterRanges", "r=-40,40"]]


def test_precise_limits_are_the_default_of_asymptotic_limits_only():
    """R1: AsymptoticLimits converged (--rAbsAcc/--rRelAcc 1e-5, minimizer tolerance 1e-4) by default; combine.precise
    false restores combine's defaults; never on Significance, the fit or HybridNew."""
    default = limit_commands()
    [limit] = [c for c in default if "-M AsymptoticLimits" in c]
    assert limit.endswith("--setParameterRanges r=-40,40 " + combine_options.PRECISE_LIMIT)
    assert all("--rAbsAcc" not in c for c in default if "-M AsymptoticLimits" not in c)
    plain = limit_commands(combine=CombineConfig(cmssw_dir="/x", precise=False))
    assert all("--rAbsAcc" not in c and "--cminDefaultMinimizerTolerance" not in c for c in plain)
    hybrid = checks.hybridnew_commands(None, [], 100, 0.05, 1.0)[-1]
    assert "--rAbsAcc 1 --rRelAcc 0.05" in hybrid and "0.00001" not in hybrid


def test_r_range_and_minimizer_from_the_run_configuration():
    combine = CombineConfig(cmssw_dir="/x", r_range=(-3000, 3000), minimizer="--cminDefaultMinimizerStrategy 1 --X-rtd MINIMIZER_analytic")
    script = "\n".join(limit_commands(combine=combine) + fit_commands(combine=combine))
    assert "--setParameterRanges r=-3000,3000" in script and "r=-40,40" not in script
    assert "--cminDefaultMinimizerStrategy 0" not in script and script.count("--cminDefaultMinimizerStrategy 1") == 4
    with pytest.raises(ValidationError, match="only minimizer options"):
        CombineConfig(cmssw_dir="/x", minimizer="--setParameters mask_a=0")  # would undo the masks
    with pytest.raises(ValidationError, match="only minimizer options"):
        CombineConfig(cmssw_dir="/x", minimizer="-t -1")
    with pytest.raises(ValidationError, match="low < high"):
        CombineConfig(cmssw_dir="/x", r_range=(5, -5))


def test_default_fit_options_leave_the_resolved_configuration_unchanged():
    """The versions/ hash of an existing run YAML stays: the new combine fields dump only when set."""
    assert CombineConfig(cmssw_dir="/x").model_dump() == {"cmssw_dir": "/x", "scram_arch": "el9_amd64_gcc12"}
    assert CombineConfig(cmssw_dir="/x", timeout=60, precise=False).model_dump() == {"cmssw_dir": "/x", "scram_arch": "el9_amd64_gcc12", "precise": False, "timeout": 60.0}


def test_a_hanging_combine_call_is_killed_at_the_configured_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(cmssw, "script", lambda combine, cwd, commands: "; ".join([f"cd {cwd}", *commands]))
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        cmssw.run(["sleep 30"], CombineConfig(cmssw_dir="/unused", timeout=1), tmp_path)  # no explicit timeout: combine.timeout
    assert time.monotonic() - start < 10
    assert CombineConfig(cmssw_dir="/x").timeout == 21600.0  # never unlimited by default


def test_parallel_jobs_run_at_most_combine_parallel_at_a_time_and_raise_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(cmssw, "script", lambda combine, cwd, commands: "; ".join([f"cd {cwd}", *commands]))
    combine = CombineConfig(cmssw_dir="/unused", parallel=2, timeout=20)
    jobs = [[f"touch running_{i}; sleep 0.5; ls running_* | wc -l > count_{i}; rm running_{i}"] for i in range(4)]
    cmssw.run_parallel(jobs, combine, tmp_path)
    assert max(int((tmp_path / f"count_{i}").read_text()) for i in range(4)) <= 2
    with pytest.raises(subprocess.CalledProcessError):
        cmssw.run_parallel([["true"], ["exit 3"], ["true"]], combine, tmp_path)


# -- blinding.json (the contract with `datacards`) -------------------------------------------------------------------


def test_a_missing_blinding_json_masks_nothing_and_a_mask_of_a_missing_bin_is_an_error(tmp_path, caplog):
    card_dir = _card(tmp_path / "mt")
    assert masking.read(card_dir).masked == () and masking.data_masks(card_dir) == []
    _blinding(card_dir, ("htt_mt_2hi_2018",))
    assert masking.data_masks(card_dir) == ["mask_htt_mt_2hi_2018=1"]
    _blinding(card_dir, ("htt_mt_3hi_2018",))
    with pytest.raises(ValueError, match="does not have: htt_mt_3hi_2018"):
        masking.data_masks(card_dir)
    _blinding(card_dir, ())
    with caplog.at_level(logging.WARNING, logger="shapesmith.masking"):
        assert masking.data_masks(card_dir) == []
    assert "htt_mt_2hi_2018 are named like blinded channels but blinding.json does not mask them" in caplog.text


def test_unblinding_needs_the_flag_and_is_logged_loudly(tmp_path, caplog):
    card_dir = _card(tmp_path / "mt")
    _blinding(card_dir)
    with caplog.at_level(logging.WARNING, logger="shapesmith.masking"):
        assert masking.data_masks(card_dir, unblind=True) == []
    assert "UNBLINDED fit to data" in caplog.text and "htt_mt_2hi_2018" in caplog.text and "Asimov" in caplog.text


def test_the_combination_merges_the_masked_bins_of_every_era(tmp_path):
    a, b, c = _card(tmp_path / "2017", ("htt_mt_1_2017", "htt_mt_2hi_2017")), _card(tmp_path / "2018", ("htt_mt_1_2018", "htt_mt_2hi_2018")), _card(tmp_path / "2016", ("htt_mt_1_2016",))
    _blinding(a, ("htt_mt_2hi_2017",))
    _blinding(b, ("htt_mt_2hi_2018",))  # c: no blinding.json, nothing masked
    merged = masking.write_merged([a, b, c], tmp_path / "run2")
    assert merged.masked == ("htt_mt_2hi_2017", "htt_mt_2hi_2018")
    content = json.loads((tmp_path / "run2" / "blinding.json").read_text())
    assert content["masked"] == ["htt_mt_2hi_2017", "htt_mt_2hi_2018"] and content["edge_rule"] == "S/B(mu=1) > 0.005" and content["asimov_data"] is True
    assert set(content["categories"]) == {"htt_mt_2hi_2017", "htt_mt_2hi_2018"} and len(content["sources"]) == 2
    rule = json.loads((b / "blinding.json").read_text())
    (b / "blinding.json").write_text(json.dumps({**rule, "edge_rule": "S/B(mu=1) > 0.01"}))
    assert masking.merge([masking.read(a), masking.read(b)])["edge_rule"] == ["S/B(mu=1) > 0.005", "S/B(mu=1) > 0.01"]
    assert masking.merge([masking.read(c)]) == {"edge_rule": None, "masked": [], "categories": {}, "asimov_data": False}


def test_the_background_only_fit_masks_the_blinded_channels():
    plain = fit_commands()
    assert plain[0] == "text2workspace.py combined.txt -o workspace.root -m 125" and "mask_" not in "\n".join(plain)
    masked = fit_commands(masks=["mask_htt_mt_2hi_2018=1"])
    assert masked[0].endswith("--channel-masks")
    assert "--setParameters r=0,mask_htt_mt_2hi_2018=1 --freezeParameters r" in masked[1]
    assert all("mask_" not in c for c in limit_commands())  # prefit Asimov limits: no masks


# -- limits on the cross section (SH12) ------------------------------------------------------------------------------


def test_the_cross_section_limit_freezes_the_theory_nuisances_of_the_card():
    commands = limit_commands(frozen=["QCDscale_HH", "PDF_alphas_HH"])
    xs = commands[-1]
    assert "-M AsymptoticLimits" in xs and "-n .LimitXS" in xs and "-t -1" in xs
    assert "--setParameters QCDscale_HH=0,PDF_alphas_HH=0 --freezeParameters QCDscale_HH,PDF_alphas_HH" in xs
    assert len(limit_commands(frozen=[])) == 4  # no theory nuisance in the card (only statistics): no second limit


def test_collect_converts_the_limits_to_the_cross_section_in_fb(tmp_path):
    card_dir = tmp_path / "all"
    card_dir.mkdir()
    _limit_tree(card_dir / "higgsCombine.Limit.AsymptoticLimits.mH125.root", QUANTILES, [5.0, 7.0, 10.0, 14.0, 20.0])
    _limit_tree(card_dir / "higgsCombine.LimitXS.AsymptoticLimits.mH125.root", QUANTILES, [4.0, 6.0, 9.0, 13.0, 18.0])
    result = collect(card_dir, cross_section=XS, frozen=["QCDscale_HH"])
    assert result["xs_limit"]["quantity"] == "sigma(ggF HH)" and result["xs_limit"]["unit"] == "fb" and result["xs_limit"]["frozen"] == ["QCDscale_HH"]
    assert result["xs_limit"]["limit"]["exp_median"] == pytest.approx(9.0 * 30.40)
    stat_only = collect(card_dir, cross_section=XS, frozen=[])
    assert stat_only["xs_limit"]["limit"]["exp_median"] == pytest.approx(10.0 * 30.40)
    assert "xs_limit" not in collect(card_dir)
    _, md_path, _ = write_summary({"all": result}, tmp_path)
    assert "σ(ggF HH) / fb: median [68% band]" in md_path.read_text() and "| 273.6 [182.4, 395.2] |" in md_path.read_text()
    assert (tmp_path / "limits_xs.pdf").exists()


def test_limit_overview_per_channel_era_and_combination(tmp_path):
    for era, median in (("2017", 12.0), ("2018", 10.0)):
        (tmp_path / era).mkdir()
        limit = {"exp_m2": median / 2, "exp_m1": median * 0.7, "exp_median": median, "exp_p1": median * 1.4, "exp_p2": median * 2}
        (tmp_path / era / "limits.json").write_text(json.dumps({"mt": {"limit": limit}, "all": {"limit": limit, "xs_limit": {"quantity": "sigma(ggF HH)", "limit": {k: v * 30.4 for k, v in limit.items()}}}}))
    (tmp_path / "run2").mkdir()
    (tmp_path / "run2" / "limits.json").write_text(json.dumps({"combination": {"limit": {"exp_m2": 4.0, "exp_m1": 5.0, "exp_median": 7.0, "exp_p1": 9.0, "exp_p2": 12.0}}}))
    result = limit_overview([("2017", tmp_path / "2017" / "limits.json"), ("2018", tmp_path / "2018" / "limits.json"), ("combined", tmp_path / "run2" / "limits.json")], tmp_path / "overview")
    assert list(result["limit"]) == ["mt 2017", "all 2017", "mt 2018", "all 2018", "combined"]
    assert list(result["xs_limit"]) == ["all 2017", "all 2018"] and result["xs_quantity"] == ["sigma(ggF HH)"]
    text = (tmp_path / "overview" / "limit_overview.md").read_text()
    assert "| combined | 7.000 | [5.000, 9.000] | [4.000, 12.000] | - |" in text and "| all 2018 | 10.000 |" in text
    assert (tmp_path / "overview" / "limit_overview.pdf").exists() and (tmp_path / "overview" / "limit_overview_xs.pdf").exists()


# -- the checks (SH11): command lines ----------------------------------------------------------------------------


MASKS = ["mask_htt_mt_2hi_2018=1"]


def test_gof_masks_the_observed_fit_and_the_toys():
    first, jobs = checks.gof_commands(None, MASKS, toys=60)
    assert first[0] == "text2workspace.py ../combined.txt -o workspace.root -m 125 --channel-masks"
    assert "--algo saturated --setParameters r=0,mask_htt_mt_2hi_2018=1 --freezeParameters r -n .obs" in first[1]
    assert [len(job) for job in jobs] == [1, 1, 1]
    assert all("mask_htt_mt_2hi_2018=1" in job[0] and "--toysFrequentist" in job[0] for job in jobs)
    assert ["-t 25 -s 1000", "-t 25 -s 1001", "-t 10 -s 1002"] == [job[0].split("-n .toys ")[1].split(" --toys")[0] for job in jobs]


def test_impacts_on_data_are_masked_and_blind_on_asimov_without_masks():
    data = checks.impacts_commands(None, MASKS)
    assert all("mask_htt_mt_2hi_2018=1" in c for c in data[1:4]) and data[-1].startswith("plotImpacts.py -i impacts.json -o impacts --blind")
    assert "--doFits --parallel 8" in data[2]
    asimov = checks.impacts_commands(CombineConfig(cmssw_dir="/x", parallel=3), MASKS, asimov=True)
    assert all("mask_" not in c for c in asimov) and all("-t -1 --expectSignal 1" in c for c in asimov[1:4]) and "--parallel 3" in asimov[2]
    assert "--blind" not in asimov[-1] and "--channel-masks" not in asimov[0]


def test_the_aposteriori_limit_fits_masked_and_limits_on_all_channels():
    """Spec 7.2: background-only fit to the data with the hi channels masked, snapshot saved, then AsymptoticLimits -t -1
    --toysFrequentist --bypassFrequentistFit on the snapshot; the masks set to 0 there, so that the hi bins count."""
    commands = checks.aposteriori_commands(None, MASKS, frozen=["QCDscale_HH"])
    assert commands[0].endswith("--channel-masks")
    assert "-M MultiDimFit -d workspace.root -m 125 --setParameters r=0,mask_htt_mt_2hi_2018=1 --freezeParameters r --saveWorkspace -n .BackgroundOnlySnapshot" in commands[1]
    limit, xs = commands[2], commands[3]
    for command in (limit, xs):
        assert f"-M AsymptoticLimits -d {checks.SNAPSHOT} --snapshotName MultiDimFit --toysFrequentist --bypassFrequentistFit" in command
        assert "-t -1 --expectSignal 0" in command and combine_options.PRECISE_LIMIT in command and "mask_htt_mt_2hi_2018=1" not in command
    assert "--setParameters mask_htt_mt_2hi_2018=0 -m" in limit
    assert "--setParameters mask_htt_mt_2hi_2018=0,QCDscale_HH=0" in xs and "--freezeParameters QCDscale_HH" in xs


def test_injection_toys_of_the_prefit_model_need_no_masks_the_aposteriori_ones_unmask_the_snapshot():
    first, jobs = checks.injection_commands(None, MASKS, [0.0, 1.0, 8.73], toys=30)
    assert first == ["text2workspace.py ../combined.txt -o workspace.root -m 125"] and all("mask_" not in job[0] for job in jobs)
    assert len(jobs) == 6 and "--expectSignal 8.73 -s 2201 --skipBOnlyFit -n .mu8p73_s2201" in jobs[-1][0]
    first, jobs = checks.injection_commands(None, MASKS, [1.0], toys=10, a_posteriori=True)
    assert "mask_htt_mt_2hi_2018=1" in first[1] and "--saveWorkspace" in first[1]
    assert "--snapshotName MultiDimFit --toysFrequentist --bypassFrequentistFit --setParameters mask_htt_mt_2hi_2018=0" in jobs[0][0]


def test_hybridnew_on_the_asimov_dataset_from_zero():
    commands = checks.hybridnew_commands(CombineConfig(cmssw_dir="/x", r_range=(-20, 50)), MASKS, 500, 0.02, 0.05)
    assert commands[0] == "text2workspace.py ../combined.txt -o workspace.root -m 125"
    assert "-M HybridNew -d workspace.root -m 125 --LHCmode LHC-limits -t -1 --expectSignal 0 -T 500" in commands[1] and commands[1].count("--setParameterRanges") == 1
    assert "--setParameterRanges r=0,50" in commands[1] and "mask_" not in commands[1]
    assert "mask_htt_mt_2hi_2018=0" in checks.hybridnew_commands(None, MASKS, 500, 0.02, 0.05, a_posteriori=True)[-1]


# -- the checks: summaries of synthetic combine outputs --------------------------------------------------------------


def test_gof_p_value_from_the_toys(tmp_path):
    _limit_tree(tmp_path / "higgsCombine.obs.GoodnessOfFit.mH125.root", [-1], [5.0])
    _limit_tree(tmp_path / "higgsCombine.toys.GoodnessOfFit.mH125.1000.root", [-1] * 4, [1.0, 2.0, 6.0, 7.0])
    _limit_tree(tmp_path / "higgsCombine.toys.GoodnessOfFit.mH125.1001.root", [-1] * 4, [3.0, 4.0, 8.0, float("nan")])
    result = checks.gof_result(tmp_path)
    assert result["statistic"] == 5.0 and result["toys"] == 7 and result["failed_toys"] == 1 and result["p_value"] == pytest.approx(3 / 7)


def _impacts_json(path):
    path.write_text(json.dumps({"POIs": [{"name": "r", "fit": [0.5, 1.0, 1.6]}], "params": [
        {"name": "lumi", "fit": [-0.6, 0.2, 1.0], "prefit": [-1.0, 0.0, 1.0], "r": [0.9, 1.0, 1.2], "impact_r": 0.2, "type": "Gaussian"},
        {"name": "CMS_pu", "fit": [-1.4, -1.0, -0.6], "prefit": [-1.0, 0.0, 1.0], "r": [0.6, 1.0, 1.5], "impact_r": 0.5, "type": "Gaussian"},
    ]}))


def test_impacts_ranking(tmp_path):
    _impacts_json(tmp_path / "impacts.json")
    result = checks.impacts_result(tmp_path)
    assert [entry["name"] for entry in result["ranking"]] == ["CMS_pu", "lumi"]
    pu = result["ranking"][0]
    assert pu["pull"] == pytest.approx(-1.0) and pu["constraint"] == pytest.approx(0.4) and pu["r_down"] == pytest.approx(-0.4) and pu["r_up"] == pytest.approx(0.5)


def _toys_file(path, r, low, high, status):
    with uproot.recreate(path) as f:
        f["tree_fit_sb"] = {"r": np.array(r), "rLoErr": np.array(low), "rHiErr": np.array(high), "fit_status": np.array(status, dtype=np.int32)}


def test_injection_pulls_use_the_error_towards_the_injected_value(tmp_path):
    _toys_file(tmp_path / "fitDiagnostics.mu1_s2100.root", [0.0, 3.0, 2.0], [0.5, 2.0, 1.0], [1.0, 4.0, 1.0], [0, 0, -1])
    toys = checks.injection_toys(tmp_path, 1.0)
    assert toys["pull"].tolist() == [-1.0, 1.0] and toys["failed"][0] == 1  # (0 - 1) / 1 (upper), (3 - 1) / 2 (lower)
    summary = checks.injection_summary(1.0, toys)
    assert summary["r_mean"] == 1.5 and summary["pull_mean"] == 0.0 and summary["biased"] is False
    entries = {f"mu{m}": {"mu": float(m), "r_mean": 2.0 * m, "r_mean_error": 0.1} for m in (0, 1, 2)}
    assert checks.linearity(entries) == pytest.approx({"slope": 2.0, "slope_error": 0.0707, "offset": 0.0, "offset_error": 0.0913}, abs=1e-3)


# -- the commands, with CMSSW faked ----------------------------------------------------------------------------------


def _stats_config(tmp_path):
    config = tmp_path / "run.yaml"
    config.write_text(f"""
analysis: tests.mini_analysis:build
era: "2018"
channels: [mt]
ntuples: {{base: {tmp_path / 'ntuples'}}}
skim_dir: {tmp_path / 'skims'}
output_dir: {tmp_path / 'out'}
combine: {{cmssw_dir: /unused, parallel: 2}}
""")
    return config


def test_gof_cli_masks_and_summarises(tmp_path, monkeypatch):
    config = _stats_config(tmp_path)
    card_dir = _card(tmp_path / "out" / "datacards" / "s0" / "mt")
    _blinding(card_dir)
    calls = []

    def fake_run(commands, combine, cwd, timeout=None):
        calls.append(list(commands))
        if any("-n .obs" in c for c in commands):
            _limit_tree(cwd / "higgsCombine.obs.GoodnessOfFit.mH125.root", [-1], [4.0])
        for command in commands:
            if "-n .toys" in command:
                seed = command.split(" -s ")[1].split()[0]
                _limit_tree(cwd / f"higgsCombine.toys.GoodnessOfFit.mH125.{seed}.root", [-1] * 2, [1.0, 9.0])

    monkeypatch.setattr(cmssw, "run", fake_run)
    result = runner.invoke(app, ["gof", "-c", str(config), "--variant", "s0", "--toys", "50"])
    assert result.exit_code == 0, result.output
    assert len(calls) == 3 and all("mask_htt_mt_2hi_2018=1" in c for call in calls for c in call if c.startswith("combine"))
    assert "| mt | 4 | 4 | 0.500 |" in result.output and "htt_mt_2hi_2018" in result.output
    assert (card_dir / "gof" / "gof.pdf").exists() and json.loads((card_dir.parent / "gof.json").read_text())["mt"]["p_value"] == 0.5
    unblinded = runner.invoke(app, ["gof", "-c", str(config), "--variant", "s0", "--toys", "25", "--unblind"])
    assert unblinded.exit_code == 0 and all("mask_" not in c for c in calls[-1]) and "UNBLINDED" in unblinded.output


def test_fit_masks_only_the_fit_to_data_and_unblind_needs_the_background_only_fit(tmp_path, monkeypatch):
    config = _stats_config(tmp_path)
    card_dir = _card(tmp_path / "out" / "datacards" / "mt")
    _blinding(card_dir)
    calls = []

    def fake_run(commands, combine, cwd, timeout=None):
        calls.append(list(commands))
        values = {"status": 0, "covariance_quality": 3, "prefit": {"lumi": [0.0, 1.0]}, "postfit": {"lumi": [0.1, 0.9]}, "norm_prefit": {}, "norm_fit_b": {}, "data": {},
                  "shapes": {"prefit": {"htt_mt_2hi_2018": {"ZTT": [[5.0], [1.0]], "total_background": [[5.0], [1.0]], "data": [[6.0], [2.0], [2.5]]}}, "fit_b": {}}}
        (cwd / "fitDiagnostics.BackgroundOnly.json").write_text(json.dumps(values))

    monkeypatch.setattr(cmssw, "run", fake_run)
    result = runner.invoke(app, ["fit", "-c", str(config), "--background-only"])
    assert result.exit_code == 0, result.output
    assert "--setParameters r=0,mask_htt_mt_2hi_2018=1" in calls[0][1] and "Masked (blinded) channels, not in the fit: htt_mt_2hi_2018" in result.output
    assert (card_dir / "plots" / "pulls.pdf").exists() and (card_dir / "plots" / "prefit" / "htt_mt_2hi_2018.png").exists()
    assert runner.invoke(app, ["fit", "-c", str(config), "--unblind"]).exit_code != 0  # the limits use the Asimov dataset


def test_aposteriori_and_injection_cli(tmp_path, monkeypatch):
    config = _stats_config(tmp_path)
    card_dir = _card(tmp_path / "out" / "datacards" / "mt")
    _blinding(card_dir)
    (card_dir.parent / "limits.json").write_text(json.dumps({"mt": {"limit": dict(zip(["exp_m2", "exp_m1", "exp_median", "exp_p1", "exp_p2"], [4.0, 6.0, 8.0, 11.0, 15.0]))}}))
    calls = []

    def fake_run(commands, combine, cwd, timeout=None):
        calls.append(list(commands))
        for command in commands:
            if "-n .APosteriori" in command:
                name = command.split("-n ")[1].split()[0]
                _limit_tree(cwd / f"higgsCombine{name}.AsymptoticLimits.mH125.root", QUANTILES, [4.5, 6.5, 8.5, 11.5, 15.5])
            if "-M FitDiagnostics" in command:
                name = command.split("-n .")[1].split()[0]
                mu = float(command.split("--expectSignal ")[1].split()[0])
                _toys_file(cwd / f"fitDiagnostics.{name}.root", [mu - 0.5, mu + 0.5], [1.0, 1.0], [1.0, 1.0], [0, 0])

    monkeypatch.setattr(cmssw, "run", fake_run)
    result = runner.invoke(app, ["limit-aposteriori", "-c", str(config)])
    assert result.exit_code == 0, result.output
    assert "| mt | 8.5000 | [6.5000, 11.5000] | [4.5000, 15.5000] | 8.0000 | 1.0625 |" in result.output
    result = runner.invoke(app, ["injection", "-c", str(config), "--toys", "2"])
    assert result.exit_code == 0, result.output
    injected = json.loads((card_dir.parent / "injection.json").read_text())["mt"]["injected"]
    assert [entry["mu"] for entry in injected.values()] == [0.0, 1.0, 8.5]  # the a-posteriori median
    assert all("mask_" not in c for call in calls[-1:] for c in call)
    assert (card_dir / "injection" / "injection.pdf").exists()


# -- variants of the datacards (SH9) ---------------------------------------------------------------------------------


def test_datacards_and_fits_of_a_variant_sit_beside_the_default_layout(tmp_path, monkeypatch):
    config = _write_config(tmp_path)
    for command in ("skim", "hist", "estimate"):
        assert runner.invoke(app, [command, "-c", str(config)]).exit_code == 0
    assert runner.invoke(app, ["datacards", "-c", str(config)]).exit_code == 0
    assert runner.invoke(app, ["datacards", "-c", str(config), "--variant", "stat_only", "--no-systematics"]).exit_code == 0
    cards = tmp_path / "out" / "datacards"
    assert (cards / "mt" / "combined.txt").exists() and (cards / "stat_only" / "mt" / "combined.txt").exists()  # default layout unchanged
    assert " lnN " in (cards / "mt" / "combined.txt").read_text() and " lnN " not in (cards / "stat_only" / "mt" / "combined.txt").read_text()
    for bad in ("mt", "all", "Stat-Only"):
        assert runner.invoke(app, ["datacards", "-c", str(config), "--variant", bad]).exit_code != 0
    runs = []
    monkeypatch.setattr(cmssw, "run", lambda commands, combine, cwd, timeout=None: runs.append(cwd))
    config.write_text(config.read_text() + "combine: {cmssw_dir: /unused}\n")
    assert runner.invoke(app, ["fit", "-c", str(config), "--variant", "stat_only"]).exit_code == 0
    assert runs == [cards / "stat_only" / "mt"] and (cards / "stat_only" / "limits.md").exists() and not (cards / "limits.md").exists()


# -- plots -----------------------------------------------------------------------------------------------------------


def test_a_masked_bin_shows_no_data(tmp_path):
    entries = {"ZTT": [[5.0, 3.0], [1.0, 1.0]], "jetFakes": [[2.0, 1.0], [0.5, 0.5]], "HH": [[0.1, 0.3], [0.0, 0.0]], "total_background": [[7.0, 4.0], [1.2, 1.1]], "data": [[8.0, 3.0], [2.8, 1.7], [3.3, 2.3]]}
    assert shown_data(entries, masked=True) is None and shown_data(entries, masked=False)[0].tolist() == [8.0, 3.0]
    for masked in (False, True):
        paths = fit_stack(entries, build(), "htt_mt_2hi_2018", tmp_path / str(masked) / "bin", masked, np.array([0.5, 0.8, 1.0]))
        assert [p.suffix for p in paths] == [".pdf", ".png"] and all(p.stat().st_size > 1000 for p in paths)
    nuisances = {"lumi": {"pull": 0.5, "constraint": 0.8}, "free": {"pull": None, "constraint": None}, "dropped": {"pull": -1.2, "constraint": None}}
    assert pull_plot(nuisances, tmp_path / "pulls.pdf", "mt").exists()
    assert pull_plot({"free": {"pull": None, "constraint": None}}, tmp_path / "none.pdf", "mt") is None


def test_the_fit_plot_of_a_card_bin_shows_its_own_bins(tmp_path, monkeypatch):
    """FitDiagnostics saves every bin in the binning of the largest one: a 2-bin hi channel of a card whose lo channel
    has 4 bins comes with 2 empty bins appended (the data with the Poisson error of 0), which the edges cut off."""
    padded = {"ZTT": [[5.0, 3.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]], "total_background": [[5.0, 3.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
              "data": [[5.0, 3.0, 0.0, 0.0], [2.2, 1.6, 0.0, 0.0], [2.8, 2.3, 1.84, 1.84]]}
    edges = np.array([0.6, 0.8, 1.0])
    trimmed, kept = card_binning(padded, edges)
    assert trimmed["data"] == [[5.0, 3.0], [2.2, 1.6], [2.8, 2.3]] and trimmed["ZTT"] == [[5.0, 3.0], [1.0, 1.0]] and kept.tolist() == edges.tolist()
    assert card_binning(padded, None)[1] is None and card_binning(padded, np.linspace(0, 1, 5))[0] is padded
    full = {**padded, "ZTT": [[5.0, 3.0, 0.5, 0.0], [1.0, 1.0, 0.1, 0.0]]}  # a non-empty bin beyond the edges: not padding
    assert card_binning(full, edges) == (full, None)
    figures = []
    monkeypatch.setattr(fit_plots.plt, "close", figures.append)
    fit_stack(padded, build(), "htt_mt_1hi_2018", tmp_path / "bin", False, edges)
    single = {process: [part[:1] + [0.0] * 3 for part in entry] for process, entry in padded.items()}  # one bin of four
    fit_stack(single, build(), "htt_mt_1hi_2018", tmp_path / "one", True, np.array([0.318, 1.0]))
    assert [(f.axes[1].get_xlim(), f.axes[1].get_xlabel()) for f in figures] == [((0.6, 1.0), "NN output"), ((0.318, 1.0), "NN output")]


def test_check_summaries_keep_the_cards_of_earlier_runs(tmp_path):
    checks.write_summary("hybridnew", {"mt": {"asimov": "prefit", "limit": 9.0, "limit_error": 0.2, "toys": 500, "asymptotic_median": 8.8, "ratio": 9.0 / 8.8}}, tmp_path)
    checks.write_summary("hybridnew", {"all": {"asimov": "prefit", "limit": 6.0, "limit_error": 0.1, "toys": 500}}, tmp_path)
    text = (tmp_path / "hybridnew.md").read_text()
    assert "| mt | prefit | 9 ± 0.2 | 8.8 | 1.023 | 500 |" in text and "| all | prefit | 6 ± 0.1 | - | - | 500 |" in text
    assert set(json.loads((tmp_path / "hybridnew.json").read_text())) == {"mt", "all"}
    assert Target("all", tmp_path).card == tmp_path / "combined.txt"

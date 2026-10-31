"""The GoF of single variables (input_gof.py, NN spec 7.1, NV1): the binning of equal prediction, the cards of a control
variable in a region, the toy commands, the p-values (toys and the fitted chi-square tail), the failure rule and the
command with CMSSW faked by synthetic combine outputs."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import uproot
from typer.testing import CliRunner

from shapesmith import checks, input_gof
from shapesmith.cli import app
from shapesmith.config import CombineConfig
from shapesmith.datacards import collect_bin
from shapesmith.expressions import columns_in, evaluate
from shapesmith.histogram import INCLUSIVE, HistKey, Histogram, HistogramSet
from shapesmith.input_gof import CardTarget, GofBinning
from shapesmith.model import Category
from tests.mini_analysis import M_VIS, build
from tests.test_cli import _write_config

runner = CliRunner()
EDGES = tuple(float(x) for x in np.linspace(0.0, 200.0, 41))  # 40 fine bins of m_vis


# -- binning ---------------------------------------------------------------------------------------------------------


def test_equal_prediction_edges_close_a_bin_at_the_target_and_merge_a_sparse_last_bin():
    edges = list(range(11))
    flat = [10.0] * 10
    assert input_gof.equal_prediction_edges(edges, flat, GofBinning(n_bins=5, min_prediction=5.0)) == [0, 2, 4, 6, 8, 10]
    # the minimum wins over total / n_bins
    assert input_gof.equal_prediction_edges(edges, flat, GofBinning(n_bins=10, min_prediction=30.0)) == [0, 3, 6, 10]
    # a falling tail, target 108 / 4 = 27: 40 | 30 | 20 + 10 | the last bin (8 events) joins its neighbour
    falling = [40.0, 30.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.0, 0.0, 0.0]
    assert input_gof.equal_prediction_edges(edges, falling, GofBinning(n_bins=4, min_prediction=10.0)) == [0, 1, 2, 10]
    assert input_gof.equal_prediction_edges(edges, [0.0] * 10) == [0, 10]


def test_integer_variables_keep_one_bin_per_value_and_merge_sparse_tails():
    edges = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    assert input_gof.is_integer_binning(edges) and not input_gof.is_integer_binning(EDGES)
    counts = [3.0, 200.0, 150.0, 60.0, 25.0, 4.0]  # n_jets 0 and 5 sparse
    # n_bins would merge the values; integer variables ignore it
    assert input_gof.equal_prediction_edges(edges, counts, GofBinning(n_bins=2, min_prediction=20.0), integer=True) == [-0.5, 1.5, 2.5, 3.5, 5.5]
    assert input_gof.equal_prediction_edges(edges, counts, GofBinning(n_bins=2, min_prediction=20.0)) == [-0.5, 2.5, 5.5]  # as a continuous one: 221 per bin


def _hset(region: str = "nominal", empty_down: bool = False) -> HistogramSet:
    rng = np.random.default_rng(1)
    shape = np.exp(-np.linspace(0.0, 200.0, 40) / 60.0)
    hset = HistogramSet()

    def put(process, values, variation="Nominal"):
        hset[HistKey("mt", INCLUSIVE, process, region, variation, "m_vis")] = Histogram(EDGES, values, values * 0.01)

    put("ZTT", 300 * shape)
    put("ZTT", 300 * shape * 1.03, "CMS_puUp")
    put("ZTT", 300 * shape * 0.97, "CMS_puDown")
    put("ZL", 50 * shape)
    put("ZL", 50 * shape * 1.01, "CMS_pseudo_jes_2018Up")
    put("ZL", 0.0 * shape if empty_down else 50 * shape * 0.99, "CMS_pseudo_jes_2018Down")
    put("jetFakes", 40 * shape[::-1] / shape[::-1].max() * shape.max())
    put("HH", 0.1 * shape)
    put("data", rng.poisson(300 * shape + 50 * shape + 40 * shape[::-1] / shape[::-1].max() * shape.max()).astype(float))
    return hset


def test_the_card_merges_the_prediction_in_the_region_and_leaves_out_excluded_and_empty_pairs(tmp_path):
    """The edges come from the histograms (the control variable only names them): bins of about equal total
    background, the processes of the region, the shapes of the pairs, and its plot."""
    analysis, target, binning = build(), CardTarget("2018", "mt", "btag0", "m_vis"), GofBinning(n_bins=8, min_prediction=10.0)
    hset = _hset("btag0")
    record = input_gof.write_card(hset, analysis, target, tmp_path, binning)
    card = tmp_path / "btag0" / "2018" / "mt" / "m_vis" / "combined.txt"
    text = card.read_text()
    assert record["card"] == str(card) and record["bins"] <= 8 and json.loads((card.parent / "record.json").read_text())["edges"] == record["edges"]
    process_line = next(line for line in text.splitlines() if line.startswith("process"))
    assert process_line.split()[1:] == ["HH", "ZTT", "ZL", "jetFakes"]
    assert "CMS_pu shape - 1 - -" in text and "CMS_pseudo_jes_2018 shape - - 1 -" in text
    assert (card.parent / "plot.png").exists()
    total = sum(hset[HistKey("mt", INCLUSIVE, p, "btag0", "Nominal", "m_vis")].rebin(record["edges"]).values for p in ("ZTT", "ZL", "jetFakes"))
    assert total.min() >= 10.0 and total.max() < 2.5 * total.sum() / record["bins"]
    excluded = input_gof.write_card(hset, analysis, target, tmp_path / "x", binning, exclude=("CMS_pseudo",), plot=False)
    assert "CMS_pseudo" not in (tmp_path / "x" / target.name / "combined.txt").read_text() and excluded["edges"] == record["edges"]
    empty = input_gof.write_card(_hset("btag0", empty_down=True), analysis, target, tmp_path / "e", binning, plot=False)
    assert empty["dropped_pairs"] == ["ZL: CMS_pseudo_jes_2018"] and "CMS_pseudo" not in (tmp_path / "e" / target.name / "combined.txt").read_text()
    assert input_gof.write_card(hset, analysis, CardTarget("2018", "mt", "nominal", "m_vis"), tmp_path / "n", binning) is None  # nothing there
    # a variable that is constant in the region (one bin) gets no card
    constant = HistogramSet()
    for key, h in hset.items():
        values = np.zeros_like(h.values)
        values[3] = h.values.sum()
        constant[key] = Histogram(EDGES, values, values * 0.01)
    assert input_gof.write_card(constant, analysis, target, tmp_path / "c", binning, plot=False) is None and not (tmp_path / "c").exists()


def test_collect_bin_reads_the_region_and_the_given_edges():
    category = Category(INCLUSIVE, "True", M_VIS)
    b = collect_bin(_hset("btag0"), build(), "mt", 0, category, None, region="btag0", edges=[0.0, 100.0, 200.0])
    assert list(b.data.edges) == [0.0, 100.0, 200.0] and set(b.processes) == {"HH", "ZTT", "ZL", "jetFakes"}
    assert list(b.variations["ZTT"]) == ["CMS_puDown", "CMS_puUp"]
    assert collect_bin(_hset("btag0"), build(), "mt", 0, category, None) is None  # the nominal region has no histograms


def test_the_uncertainty_band_holds_statistics_shapes_and_lnn():
    analysis = build()
    b, _ = input_gof.card_bin(_hset(), analysis, CardTarget("2018", "mt", "nominal", "m_vis"), GofBinning(n_bins=4, min_prediction=10.0))
    band = input_gof.prediction_uncertainty(analysis, b)
    ztt, zl, fakes = b.processes["ZTT"].values, b.processes["ZL"].values, b.processes["jetFakes"].values
    statistics = sum(h.variances for p, h in b.processes.items() if p != "HH")
    half = lambda kappa: (kappa - 1 / kappa) / 2  # noqa: E731
    # CMS_pu 3 % of ZTT, the pseudo-shift 1 % of ZL; lumi on every process (*), zjXsec on ZTT + ZL, ffNorm on jetFakes
    expected = np.sqrt(statistics + (0.03 * ztt) ** 2 + (0.01 * zl) ** 2 + (half(1.025) * (ztt + zl + fakes)) ** 2 + (half(1.02) * (ztt + zl)) ** 2 + (half(1.1) * fakes) ** 2)
    assert np.allclose(band, expected)


# -- GoF jobs --------------------------------------------------------------------------------------------------------


def test_each_card_runs_its_gof_in_one_job_with_its_own_seed(tmp_path):
    card_dir = tmp_path / "nominal" / "2018" / "mt" / "m_vis"
    card_dir.mkdir(parents=True)
    (card_dir / "combined.txt").write_text("imax *\nbin htt_mt_1_2018\n")
    (card_dir / "gof").mkdir()
    stale = card_dir / "gof" / "higgsCombine.toys.GoodnessOfFit.mH125.99.root"
    stale.write_text("old")
    seed = input_gof.card_seed("nominal/2018/mt/m_vis")
    commands = input_gof.gof_job(card_dir, None, 300, seed)
    assert commands[0] == f"cd {(card_dir / 'gof').resolve()}" and commands[1].startswith("text2workspace.py ../combined.txt")
    assert sum("-t 300" in c for c in commands) == 1 and f"-s {seed} --toysFrequentist" in commands[-1] and not stale.exists()
    assert seed != input_gof.card_seed("nominal/2018/mt/pt_2") and seed == input_gof.card_seed("nominal/2018/mt/m_vis")
    # the minimizer options of the run reach the observed fit and the toys
    fallback = CombineConfig(cmssw_dir="cmssw", minimizer="--cminFallbackAlgo Minuit2,Migrad,0:0.1")
    fits = [c for c in input_gof.gof_job(card_dir, fallback, 300, seed) if "-M GoodnessOfFit" in c]
    assert len(fits) == 2 and all("--cminFallbackAlgo Minuit2,Migrad,0:0.1" in c for c in fits)
    first, jobs = checks.gof_commands(None, [], 50, 25)
    assert [job[0].split("-s ")[1].split()[0] for job in jobs] == [str(checks.GOF_SEED), str(checks.GOF_SEED + 1)]


# -- p-values --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("x,k,expected", [
    (1.0, 1.0, 0.31731050786291115), (10.0, 10.0, 0.44049328506521257), (30.0, 10.0, 0.000856641210775301),
    (100.0, 20.0, 1.2596084591660847e-12), (3.5, 17.3, 0.99983893781636), (250.0, 30.0, 1.5157414203687308e-36),
])
def test_chi2_tail_against_scipy(x, k, expected):
    assert input_gof.chi2_sf(x, k) == pytest.approx(expected, rel=1e-10)


def test_the_p_value_comes_from_the_toys_while_they_resolve_it_and_else_from_the_fitted_tail():
    rng = np.random.default_rng(5)
    toys = 1.3 * rng.chisquare(12.0, 4000)
    k, s = input_gof.scaled_chi2(toys)
    assert k == pytest.approx(12.0, rel=0.1) and s == pytest.approx(1.3, rel=0.1)
    central = input_gof.p_values(toys, 1.3 * 12.0)
    assert central["method"] == "toys" and central["p"] == central["p_toys"] and central["p_fit"] == pytest.approx(central["p_toys"], abs=0.03)
    tail = input_gof.p_values(toys, 1.3 * 60.0)
    assert tail["method"] == "chi2 tail" and tail["toys_above"] < input_gof.MIN_TOYS_ABOVE
    assert tail["p"] == pytest.approx(input_gof.chi2_sf(60.0, 12.0), rel=0.5) and tail["p"] < 1e-6
    assert input_gof.p_values(np.array([1.0, 2.0]), 5.0)["method"] == "toys"  # no fit with too few toys
    # a few toys of failed fits far in the tail move the moments, not the quantiles
    spoiled = np.concatenate([toys, [900.0, 1200.0, 400.0, 3000.0]])
    k, s = input_gof.scaled_chi2(spoiled)
    assert k == pytest.approx(12.0, rel=0.15) and s == pytest.approx(1.3, rel=0.15)
    mean, variance = spoiled.mean(), spoiled.var()
    assert 2 * mean**2 / variance < 3.0  # the moments: k below 3


def test_chi2_quantiles_invert_the_tail():
    for p, k in ((0.5, 1.0), (0.1, 17.5), (1e-6, 20.0)):
        assert input_gof.chi2_sf(input_gof.chi2_isf(p, k), k) == pytest.approx(p, rel=1e-6)


def test_ks_uniform_against_scipy():
    values = np.random.default_rng(3).uniform(size=400) ** 1.3
    distance, p = input_gof.ks_uniform(values)
    assert distance == pytest.approx(0.08888642950028403) and p == pytest.approx(0.0033644303070512437, rel=0.05)
    assert input_gof.ks_uniform(np.linspace(0.005, 0.995, 100))[1] == pytest.approx(1.0)


def test_failure_rule_bonferroni_and_two_eras():
    entries = []
    for variable, ps in (("pt_1", (0.5, 0.3, 0.7, 0.2)), ("met", (0.04, 0.03, 0.5, 0.9)), ("m_vis", (1e-6, 0.4, 0.6, 0.8)), ("eta_1", (0.04, 0.6, 0.6, 0.9))):
        entries += [{"variable": variable, "channel": "mt", "region": "nominal", "era": era, "p": p} for era, p in zip(("2016preVFP", "2016postVFP", "2017", "2018"), ps)]
    entries.append({"variable": "eta_1", "channel": "et", "region": "nominal", "era": "2017", "p": 0.03})  # another channel: no second era
    rule = input_gof.failure_rule(entries)
    assert rule["tests"] == 17 and rule["threshold"] == pytest.approx(0.05 / 17)
    verdict = {name: record["failed"] for name, record in rule["variables"].items()}
    assert verdict == {"pt_1": False, "met": True, "m_vis": True, "eta_1": False}
    assert rule["variables"]["met"]["reasons"] == ["p < 0.05 in 2 eras of mt nominal (2016postVFP, 2016preVFP)"]
    assert "< 0.0029" in rule["variables"]["m_vis"]["reasons"][0]
    # a variant of some tests of a larger family keeps the threshold of the family
    larger = input_gof.failure_rule(entries, n_tests=50000)
    assert larger["threshold"] == pytest.approx(1e-6) and larger["tests"] == 17 and larger["variables"]["m_vis"]["failed"] is False
    assert larger["below_alpha"] == rule["below_alpha"] == 5 and larger["expected_below_alpha"] == pytest.approx(0.05 * 17)  # of these tests


def _gof_outputs(directory, observed, toys):
    directory.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(directory / "higgsCombine.obs.GoodnessOfFit.mH125.root") as f:
        f["limit"] = {"limit": np.array([observed], dtype=np.float64)}
    with uproot.recreate(directory / "higgsCombine.toys.GoodnessOfFit.mH125.1234.root") as f:
        f["limit"] = {"limit": np.asarray(toys, dtype=np.float64)}


def test_the_command_writes_cards_collects_the_gof_and_keeps_earlier_results(tmp_path, monkeypatch):
    """input-gof with CMSSW faked: cards from the control histograms of a region, the results of fake GoF outputs, the
    summary merged with the results of an earlier run into the same output."""
    config = _write_config(tmp_path)
    _hset("btag0").save(tmp_path / "out" / "control_shapes.root")
    calls = []

    def fake_run_parallel(jobs, combine, cwd):
        calls.append(jobs)
        for job in jobs:
            _gof_outputs(Path(job[0].split(" ", 1)[1]), 40.0, np.random.default_rng(2).chisquare(20.0, 200))

    monkeypatch.setattr("shapesmith.cmssw.run_parallel", fake_run_parallel)
    out = tmp_path / "gof"
    out.mkdir()
    (out / "input_gof.json").write_text(json.dumps({"results": [{"era": "2017", "channel": "mt", "region": "btag0", "variable": "m_vis", "p": 0.01, "method": "toys"}]}))
    result = runner.invoke(app, ["input-gof", "-c", str(config), "-o", str(out), "--region", "btag0", "--toys", "200", "--bins", "6", "--min-prediction", "10", "--no-plots"])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1 and len(calls[0]) == 1 and (out / "btag0" / "2018" / "mt" / "m_vis" / "combined.txt").exists()
    summary = json.loads((out / "input_gof.json").read_text())
    assert [r["era"] for r in summary["results"]] == ["2017", "2018"] and summary["rule"]["tests"] == 2
    entry = summary["results"][1]
    expected = float(np.mean(np.random.default_rng(2).chisquare(20.0, 200) >= 40.0))
    assert entry["statistic"] == 40.0 and entry["toys"] == 200 and entry["p"] == pytest.approx(expected if entry["toys_above"] >= 10 else entry["p_fit"])
    # with the earlier p = 0.01 of 2017, a second era below 0.05 fails the variable
    assert summary["rule"]["variables"]["m_vis"]["failed"] == (entry["p"] < 0.05)
    assert "| m_vis |" in (out / "input_gof.md").read_text() and (out / "pvalues.pdf").exists()
    rerun = runner.invoke(app, ["input-gof", "-c", str(config), "-o", str(out), "--region", "btag0", "--skip-cards", "--skip-combine"])
    assert rerun.exit_code == 0, rerun.output
    assert json.loads((out / "input_gof.json").read_text())["results"][1]["statistic"] == 40.0 and len(calls) == 1
    # tail toys also with --skip-combine: one job of the 200 missing toys
    tail = runner.invoke(app, ["input-gof", "-c", str(config), "-o", str(out), "--region", "btag0", "--skip-cards", "--skip-combine", "--tail-toys", "400", "--tail-below", "1"])
    assert tail.exit_code == 0, tail.output
    assert len(calls) == 2 and len(calls[1]) == 1 and " -t 200 " in calls[1][0][1]


def test_tail_toys_go_to_the_cards_below_the_threshold_and_their_p_values_use_all_toys(tmp_path, monkeypatch):
    """A card below tail_below gets jobs of TAIL_JOB_TOYS toys up to tail_toys, with seeds apart from its first toys and
    continuing earlier tail toys; a card above gets none; the p-value of the card then comes from all its toys."""
    rng = np.random.default_rng(5)
    records = []
    for variable, observed in (("m_vis", 75.0), ("pt_2", 20.0)):
        card_dir = tmp_path / "nominal" / "2018" / "mt" / variable
        _gof_outputs(card_dir / "gof", observed, rng.chisquare(20.0, 300))
        records.append({"era": "2018", "channel": "mt", "region": "nominal", "variable": variable, "card": str(card_dir / "combined.txt")})
    results = [{**r, **input_gof.card_result(Path(r["card"]).parent)} for r in records]
    assert results[0]["p"] < 1e-5 and results[0]["method"] == "chi2 tail" and results[1]["p"] > 0.1
    calls = []

    def fake_run_parallel(jobs, combine, cwd):
        calls.append(jobs)
        for job in jobs:
            directory = Path(job[0].split(" ", 1)[1])
            seed, toys = int(job[1].split(" -s ")[1].split()[0]), int(job[1].split(" -t ")[1].split()[0])
            with uproot.recreate(directory / f"higgsCombine.toys.GoodnessOfFit.mH125.{seed}.root") as f:
                f["limit"] = {"limit": rng.chisquare(20.0, toys)}

    monkeypatch.setattr("shapesmith.cmssw.run_parallel", fake_run_parallel)
    updated = input_gof.run_tail_toys(results, None, tmp_path, 1300, below=1e-3)
    seed = input_gof.card_seed("nominal/2018/mt/m_vis")
    (jobs,) = calls
    assert [int(job[1].split(" -s ")[1].split()[0]) for job in jobs] == [seed + input_gof.TAIL_SEED_OFFSET, seed + input_gof.TAIL_SEED_OFFSET + 1]
    assert all(job[0] == f"cd {(tmp_path / 'nominal' / '2018' / 'mt' / 'm_vis' / 'gof').resolve()}" and " -t 500 " in job[1] for job in jobs)
    assert updated[0]["toys"] == 1300 and updated[1] == results[1]  # 300 + 2 x 500, the card above untouched
    all_toys = checks.gof_toys(tmp_path / "nominal" / "2018" / "mt" / "m_vis" / "gof")
    assert updated[0]["p_fit"] == pytest.approx(input_gof.p_values(all_toys, 75.0)["p_fit"]) and updated[0]["p_fit"] != results[0]["p_fit"]
    # a second pass to more toys continues the seeds of the card
    input_gof.run_tail_toys(updated, None, tmp_path, 1800, below=1e-3)
    assert int(calls[1][0][1].split(" -s ")[1].split()[0]) == seed + input_gof.TAIL_SEED_OFFSET + 2 and len(calls[1]) == 1


def test_cards_with_a_lower_statistic_under_the_alternative_options_run_again_with_them(tmp_path, monkeypatch):
    """Every card gets a second observed fit with the alternative options; a card whose statistic is lower there (its
    background-only fit failed with the options of the run) runs its GoF again with them, recorded in minimizer.json,
    which its tail toys use; the other cards keep their GoF."""
    records = []
    for variable in ("m_vis", "pt_2"):
        card_dir = tmp_path / "nominal" / "2018" / "mt" / variable
        card_dir.mkdir(parents=True)
        records.append({"era": "2018", "channel": "mt", "region": "nominal", "variable": variable, "card": str(card_dir / "combined.txt")})
    calls = []

    def fake_run_parallel(jobs, combine, cwd):
        calls.append(jobs)
        for job in jobs:
            directory = Path(job[0].split(" ", 1)[1])
            alternative = any("MINIMIZER_no_analytic" in command for command in job[1:])
            statistic = 60.0 if alternative or directory.parent.name == "pt_2" else 2484.0  # m_vis: a failed fit without the alternative
            if " -n .alt " in job[1]:
                with uproot.recreate(directory / "higgsCombine.alt.GoodnessOfFit.mH125.root") as f:
                    f["limit"] = {"limit": np.array([statistic + (0.05 if directory.parent.name == "pt_2" else 0.0)])}
            elif " -t 500 " in job[1]:  # a tail toy job
                seed = int(job[1].split(" -s ")[1].split()[0])
                with uproot.recreate(directory / f"higgsCombine.toys.GoodnessOfFit.mH125.{seed}.root") as f:
                    f["limit"] = {"limit": np.random.default_rng(seed).chisquare(20.0, 500)}
            else:
                _gof_outputs(directory, statistic, np.random.default_rng(3).chisquare(20.0, 300))

    monkeypatch.setattr("shapesmith.cmssw.run_parallel", fake_run_parallel)
    combine = CombineConfig(cmssw_dir="cmssw", minimizer="--cminFallbackAlgo Minuit2,Migrad,0:0.1")
    results = input_gof.run_gof(records, combine, 300, tmp_path, tail_toys=800, tail_below=1e-3, alt_minimizer="--X-rtd MINIMIZER_no_analytic")
    first, alternatives, again, tails = calls
    assert len(first) == 2 and all("MINIMIZER_no_analytic" in job[1] and "> alt.log" in job[1] for job in alternatives)
    assert len(again) == 1 and again[0][0].endswith("m_vis/gof") and "MINIMIZER_no_analytic" in again[0][2]
    m_vis, pt_2 = results
    assert m_vis["statistic"] == m_vis["alt_statistic"] == 60.0 and m_vis["minimizer"] == "--cminFallbackAlgo Minuit2,Migrad,0:0.1 --X-rtd MINIMIZER_no_analytic"
    assert pt_2["statistic"] == 60.0 and pt_2["alt_statistic"] == pytest.approx(60.05) and pt_2["minimizer"] is None  # higher there: kept
    # both are far in the tail: one tail job each, m_vis with its alternative options
    options = {job[0].split("/")[-2]: job[1] for job in tails}
    assert "MINIMIZER_no_analytic" in options["m_vis"] and "MINIMIZER_no_analytic" not in options["pt_2"] and m_vis["toys"] == pt_2["toys"] == 800
    # a new GoF of the card removes its alternative options
    input_gof.gof_job(tmp_path / "nominal" / "2018" / "mt" / "m_vis", combine, 300, 1)
    assert not (tmp_path / "nominal/2018/mt/m_vis/gof" / input_gof.MINIMIZER).exists()


# -- expressions -----------------------------------------------------------------------------------------------------


def test_long_expressions_on_float32_columns_fall_back_to_the_python_engine():
    """pandas passes every literal of a float32 expression to numexpr as an array: a long one exceeds numexpr's inputs
    and registers; the python engine evaluates it."""
    frame = pd.DataFrame({name: np.linspace(1.0, 2.0, 5, dtype=np.float32) for name in ("a", "b")})
    expr = " + ".join(f"{1.0 + i / 7!r} * a * sin(b)" for i in range(150))
    expected = sum((1.0 + i / 7) for i in range(150)) * frame["a"].to_numpy(np.float64) * np.sin(frame["b"].to_numpy(np.float64))
    assert np.allclose(evaluate(frame, expr), expected, rtol=1e-5)
    assert columns_in("floor(x) + ceil(y)") == {"x", "y"} and np.array_equal(evaluate(pd.DataFrame({"x": [1.5, -0.5], "y": [0.2, 0.0]}), "floor(x) + ceil(y)"), [2.0, -1.0])
    assert math.isclose(evaluate(pd.DataFrame({"x": [2.0]}), "x ** 0.5")[0], math.sqrt(2.0))


# -- flatness and shape compatibility --------------------------------------------------------------------------------


def _score(values, variances=None):
    edges = np.linspace(0.0, 1.0, 101)
    values = np.asarray(values, dtype=float)
    return Histogram(edges, values, values * 0.0 if variances is None else variances)


def test_flatness_ignores_the_normalisation_of_each_range_and_finds_a_slope():
    x = (np.arange(100) + 0.5) / 100
    prediction = _score(1000.0 * np.ones(100), 1.0 * np.ones(100))
    ranges = [(0.0, 0.2), (0.2, 0.6), (0.6, 1.0)]
    # data = 1.3 x the prediction below 0.2, 0.8 x above: every range flat
    scaled = _score(np.where(x < 0.2, 1300.0, 800.0))
    flat = input_gof.flatness(scaled, prediction, ranges, GofBinning(5, 10.0))
    assert [r["bins"] for r in flat] == [5, 5, 5] and all(r["chi2"] == pytest.approx(0.0, abs=1e-9) for r in flat)
    assert [r["scale"] for r in flat] == pytest.approx([1.3, 0.8, 0.8]) and all(r["slope"] == pytest.approx(0.0, abs=1e-9) for r in flat)
    # a 10 % rise across the last range: chi2 and slope see it, the other ranges stay flat
    tilted = _score(np.where(x < 0.6, 1000.0, 1000.0 * (0.95 + 0.25 * (x - 0.6))))
    result = input_gof.flatness(tilted, prediction, ranges, GofBinning(5, 10.0))
    assert result[2]["p"] < 1e-4 and result[2]["z"] > 5 and result[2]["slope"] == pytest.approx(0.1, rel=0.05)
    assert result[0]["p"] == pytest.approx(1.0) and result[1]["chi2"] == pytest.approx(0.0, abs=1e-9)
    assert input_gof.flatness(_score(np.zeros(100)), prediction, ranges[:1])[0]["p"] is None
    with pytest.raises(ValueError, match="not bounded by fine edges"):
        input_gof.flatness(tilted, prediction, [(0.0, 0.205)])


def test_flatness_p_values_are_uniform_for_poisson_data():
    rng = np.random.default_rng(7)
    prediction = _score(200.0 * np.ones(100), 0.0 * np.ones(100))
    p = [input_gof.flatness(_score(rng.poisson(200.0, 100)), prediction, [(0.0, 1.0)], GofBinning(10, 20.0))[0]["p"] for _ in range(400)]
    assert input_gof.ks_uniform(p)[1] > 0.01 and 0.03 < np.mean(np.array(p) < 0.05) < 0.08


def test_shape_compatibility_with_the_statistics_and_the_shifts():
    base = np.array([100.0, 80.0, 60.0, 40.0, 20.0])
    first = Histogram(range(6), 2 * base, 2 * base)  # the same shape, other normalisation
    second = Histogram(range(6), base, base)
    assert input_gof.shape_compatibility(first, second)["chi2"] == pytest.approx(0.0, abs=1e-12)
    tilted = Histogram(range(6), base * [1.2, 1.1, 1.0, 0.9, 0.8] * 100, base * 100)
    big = Histogram(range(6), base * 100, base * 100)
    alone = input_gof.shape_compatibility(tilted, big)
    assert alone["ndf"] == 4 and alone["p"] < 1e-10
    # a shift of the second histogram of the size of the tilt covers it
    up, down = Histogram(range(6), base * [1.2, 1.1, 1.0, 0.9, 0.8] * 100, base), Histogram(range(6), base * [0.8, 0.9, 1.0, 1.1, 1.2] * 100, base)
    covered = input_gof.shape_compatibility(tilted, big, {"ff": (up, down)})
    assert covered["p"] > 0.5 and covered["shifts"] == 1
    # the covariance of a unit-sum shape: J diag(var) J^T, its rows sum to 0
    shape, covariance = input_gof._normalised(second)
    assert shape.sum() == pytest.approx(1.0) and np.allclose(covariance.sum(axis=1), 0.0)


def test_merge_low_statistics_by_effective_events():
    edges = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    weighted = Histogram(edges, [10.0, 10.0, 10.0, 10.0, 1.0], [10.0, 10.0, 10.0, 10.0, 1.0])  # n_eff = sum per bin
    heavy = Histogram(edges, [10.0, 10.0, 10.0, 10.0, 10.0], [30.0, 1.0, 1.0, 1.0, 1.0])  # first bin n_eff = 3.3, with the second 12.9
    assert input_gof.merge_low_statistics(edges, [weighted], 10.0) == [0.0, 1.0, 2.0, 3.0, 5.0]
    assert input_gof.merge_low_statistics(edges, [weighted, heavy], 10.0) == [0.0, 2.0, 3.0, 5.0]


def test_flatness_with_shifts_that_cover_a_slope():
    x = (np.arange(100) + 0.5) / 100
    prediction = _score(1000.0 * np.ones(100), np.ones(100))
    tilted = _score(1000.0 * (0.95 + 0.1 * x))
    up, down = _score(1000.0 * (0.9 + 0.2 * x)), _score(1000.0 * (1.1 - 0.2 * x))
    [plain] = input_gof.flatness(tilted, prediction, [(0.0, 1.0)], GofBinning(5, 10.0))
    [covered] = input_gof.flatness(tilted, prediction, [(0.0, 1.0)], GofBinning(5, 10.0), {"btag": (up, down)})
    assert plain["p"] < 1e-6 and covered["p_shifts"] > 0.5 and covered["chi2"] == pytest.approx(plain["chi2"])
    [flat] = input_gof.flatness(tilted, prediction, [(0.0, 1.0)], GofBinning(5, 10.0), {"none": (prediction, prediction)})
    assert flat["chi2_shifts"] == pytest.approx(plain["chi2"])

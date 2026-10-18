"""The profile-scan intervals of the tau-ID/ES measurement (v7 spec 5.4, U11), with a fake CMSSW that writes the combine
outputs of a Gaussian likelihood: interval from the 1D profiles, rescans on a widened range, the abort at the range
edge, the singles cross-check and its flags, refit from synced shapes, the merge checks and the ES transition of the
payload."""
import json
import re

import correctionlib
import numpy as np
import pytest
import uproot

from shapesmith import cmssw, payloads
from shapesmith.config import CombineConfig, NtupleConfig, RunConfig
from shapesmith.histogram import HistKey, HistogramSet
from shapesmith.measurements import MeasureContext
from shapesmith.measurements.tau_id_es import TauIdEsMeasurement, combine, fit_category, measure, merge, merge_problems, report
from shapesmith.measurements.tau_id_es.combine import Interval, Profile
from shapesmith.measurements.tau_id_es.grid import CATEGORIES, grid
from shapesmith.measurements.tau_id_es.payload import ES_TRANSITION, correction_set
from shapesmith.measurements.tau_id_es.synced import write_synced
from tests.test_tau_id_es import _analysis, _h

GRID = grid(-200, 200, 2)
RANGES = re.compile(r"--setParameterRanges r_EMB_(\w+?)=([-\d.e]+),([-\d.e]+):ES_\w+?=([-\d.e]+),([-\d.e]+)")


def _write(path, **columns):
    with uproot.recreate(path) as f:
        f["limit"] = {name: np.asarray(values, dtype=np.float32) for name, values in columns.items()}


class FakeCombine:
    """cmssw.run for the commands of fit_category: the delta NLL of independent Gaussians in r and ES (the 1D scans
    may have their own widths), the singles fit at `singles` (r and ES as (best, low, high))."""

    def __init__(self, r=(0.95, 0.05), es=(-1.0, 1.5), profile_sigma=None, singles=None, fail=()):
        self.r, self.es, self.profile_sigma, self.singles, self.fail = r, es, profile_sigma or {}, singles, fail
        self.commands = []

    def dnll(self, poi, x, sigma=None):
        mean, width = self.r if poi == "r" else self.es
        return 0.5 * ((x - mean) / (sigma or width)) ** 2

    def __call__(self, commands, config, cwd, timeout=None):
        for command in commands:
            self.commands.append(command)
            if any(word in command for word in self.fail):
                raise cmssw.subprocess.CalledProcessError(1, command)
            self.run(command, cwd)

    def run(self, command, cwd):
        name = re.search(r" -n \.(\S+)", command)
        if command.startswith("MorphingTauID2017"):
            category = re.search(r"--categories=(\S+)", command).group(1)
            cards = combine.card_dir(cwd, category)
            cards.mkdir(parents=True, exist_ok=True)
            (cards / "htt_mt_7_Run2018.txt").write_text(f"process W ZL EMB_{category}\n")
            (cards / "htt_mm_100_Run2018.txt").write_text("process W MUEMB\n")
            return
        if name is None:
            return
        name = name.group(1)
        category, r_low, r_high, es_low, es_high = RANGES.search(command).groups() if "--setParameterRanges" in command else (None,) * 5
        r_poi, es_poi = f"r_EMB_{category}", f"ES_{category}"
        if name.startswith(("scan_2D_", "envelope_")):
            points = (17, 17) if name.startswith("scan_2D_") else (combine.PROFILE_POINTS, combine.ENVELOPE_ES_POINTS)
            r, es = (a.ravel() for a in np.meshgrid(np.linspace(float(r_low), float(r_high), points[0]), np.linspace(float(es_low), float(es_high), points[1])))
            _write(combine.output_path(cwd, name), **{r_poi: np.r_[self.r[0], r], es_poi: np.r_[self.es[0], es], "deltaNLL": np.r_[0.0, self.dnll("r", r) + self.dnll("es", es)]})
        elif name.startswith("profile_"):
            poi = re.search(r"--redefineSignalPOIs (\S+)", command).group(1)
            key, other = ("r", es_poi) if poi == r_poi else ("es", r_poi)
            low, high = (float(r_low), float(r_high)) if key == "r" else (float(es_low), float(es_high))
            x = np.linspace(low, high, combine.PROFILE_POINTS)
            tracked = np.full(len(x) + 1, (float(es_low) + float(es_high)) / 2 if key == "r" else (float(r_low) + float(r_high)) / 2)
            _write(combine.output_path(cwd, name), **{poi: np.r_[x[0], x], "deltaNLL": np.r_[0.0, self.dnll(key, x, self.profile_sigma.get(key))], f"trackedParam_{other}": tracked})
        elif name.startswith("singles_"):
            r, es = self.singles or ((self.r[0], self.r[0] - self.r[1], self.r[0] + self.r[1]), (self.es[0], self.es[0] - self.es[1], self.es[0] + self.es[1]))
            _write(combine.output_path(cwd, name), **{r_poi: [r[0], r[1], r[2], r[0], r[0]], es_poi: [es[0], es[0], es[0], es[1], es[2]]})


@pytest.fixture
def fake(monkeypatch):
    def install(**options):
        runner = FakeCombine(**options)
        monkeypatch.setattr(cmssw, "run", runner)
        return runner
    return install


def test_profile_best_value_crossings_and_widening():
    x = np.linspace(0.6, 1.4, 41)
    profile = Profile.of(x, 0.5 * ((x - 0.937) / 0.1) ** 2)
    assert profile.best == pytest.approx(0.937, abs=1e-9)  # the parabola through the lowest point and its neighbours
    assert profile.crossings(0.5) == pytest.approx((0.837, 1.037), abs=1e-3)
    assert combine.widened(profile, (0.11, 2.89)) is None
    one_sided = Profile.of(x, 0.5 * ((x - 1.35) / 0.1) ** 2)  # no crossing above
    assert combine.widened(one_sided, (0.11, 2.89)) == pytest.approx((0.6, 2.2))  # the high side moves out by the range width
    assert combine.widened(one_sided, (0.11, 1.4)) is None  # already at the bound: nothing to widen
    interval, problems = combine.profile_interval("r", one_sided, (0.11, 1.4))
    assert interval is None and problems == ["r: no crossing of delta NLL = 0.5 on the high side within the scan range [0.6, 1.4] (bounds [0.11, 1.4])"]
    edge = Profile.of(x, 0.5 * ((x - 0.6) / 0.05) ** 2)
    assert any("best value 0.6 at an end" in p for p in combine.profile_interval("r", edge, (0.11, 2.89))[1])


def test_the_envelope_of_a_2d_grid_is_the_r_profile_with_the_es_at_its_minimum():
    r, es = (a.ravel() for a in np.meshgrid(np.linspace(0.8, 1.2, 5), np.linspace(-4.0, 4.0, 5)))
    dnll = 0.5 * ((r - 1.0) / 0.1) ** 2 + 0.5 * ((es - (r - 1.0) * 20) / 1.0) ** 2  # the ES of the minimum moves with r
    profile = combine.Scan(r, es, dnll).profile(0)
    assert profile.values.tolist() == pytest.approx([0.8, 0.9, 1.0, 1.1, 1.2]) and profile.profiled.tolist() == [-4.0, -2.0, 0.0, 2.0, 4.0]
    assert profile.dnll.tolist() == pytest.approx([2.0, 0.5, 0.0, 0.5, 2.0])
    assert combine.profile_interval("r", profile, (0.11, 2.89), "ES", (-4.0, 4.0))[1] == []  # at its window ends only outside [0.9, 1.1]
    assert combine.profile_interval("r", profile, (0.11, 2.89), "ES", (-2.0, 2.0))[1] == ["r: the profiled ES reaches an end of its range [-2, 2] inside the interval"]


def test_a_profiled_poi_at_its_range_inside_the_interval_is_a_problem():
    x = np.linspace(0.6, 1.4, 41)
    profiled = np.where(np.abs(x - 1.0) < 0.05, -10.0, 0.0)
    profile = Profile.of(x, 0.5 * ((x - 1.0) / 0.1) ** 2, profiled)
    _, problems = combine.profile_interval("r", profile, (0.11, 2.89), "ES", (-10.0, 6.0))
    assert problems == ["r: the profiled ES reaches an end of its range [-10, 6] inside the interval"]
    assert combine.profile_interval("r", profile, (0.11, 2.89), "ES", (-12.0, 6.0))[1] == []


def test_the_result_is_the_profile_interval_with_singles_as_cross_check(tmp_path, fake):
    runner = fake()
    fit = fit_category(None, "DM0", tmp_path / "synced", tmp_path, "2018", GRID)
    assert fit["problems"] == [] and fit["warnings"] == []
    assert (fit["sf"]["best"], fit["sf"]["low"], fit["sf"]["high"]) == pytest.approx((0.95, 0.90, 1.00), abs=2e-3)
    assert (fit["es"]["best"], fit["es"]["low"], fit["es"]["high"]) == pytest.approx((-1.0, -2.5, 0.5), abs=2e-2)
    assert fit["profiles"]["r"]["widenings"] == fit["profiles"]["ES"]["widenings"] == 0 and fit["profiles"]["r"]["points"] == 41
    assert fit["singles"]["sf"]["best"] == pytest.approx(0.95) and fit["singles"]["deviations"] == []
    names = [re.search(r" -n \.(\S+)", c).group(1) for c in runner.commands if " -n ." in c]
    assert names == ["scan_2D_DM0", "profile_ES_DM0", "envelope_r_EMB_DM0", "bestfit_DM0", "singles_DM0"]
    envelope = next(c for c in runner.commands if ".envelope_r_EMB_DM0" in c)
    assert "--redefineSignalPOIs r_EMB_DM0,ES_DM0" in envelope and "--gridPoints 41,25" in envelope  # both POIs fixed per point
    bestfit = next(c for c in runner.commands if ".bestfit_DM0" in c)
    assert f"r_EMB_DM0={fit['sf']['best']!r},ES_DM0={fit['es']['best']!r}" in bestfit  # the postfit at the profile minima
    postfit = next(c for c in runner.commands if c.startswith("PostFitShapesFromWorkspace"))
    assert f"--freeze r_EMB_DM0={fit['sf']['best']!r},ES_DM0={fit['es']['best']!r}" in postfit  # the frozen POIs are not in the fit result


def test_an_open_side_is_rescanned_on_a_widened_range(tmp_path, fake):
    runner = fake(profile_sigma={"es": 6.0})  # the 1D profile is wider than the window from the 2D scan
    fit = fit_category(None, "DM0", tmp_path / "synced", tmp_path, "2018", GRID)
    assert fit["profiles"]["ES"]["widenings"] == 1 and fit["problems"] == []
    assert fit["es"]["low"] == pytest.approx(-7.0, abs=0.1) and fit["es"]["high"] == pytest.approx(5.0, abs=0.1)
    assert fit["ranges"]["ES"][0] < fit["scan_2d"]["ranges"]["ES"][0]
    assert sum(" -n .profile_ES_DM0 " in c for c in runner.commands) == 1 and sum(" -n .profile_ES_DM0_1 " in c for c in runner.commands) == 1


def test_the_window_is_taken_above_the_lowest_grid_point():
    x = np.linspace(0.11, 2.89, 17)  # the r axis of the 2D scan: a spacing of 0.17 against a width of 0.02
    coarse = Profile.of(x, 0.5 * ((x - 0.8965) / 0.02) ** 2)
    assert coarse.dnll.min() > combine.LEVEL  # every grid point above the 2 sigma level of the fit
    low, high = coarse.window(0.2, (0.11, 2.89))
    assert low < 0.8965 < high and high - low < 1.0  # around the minimum, not the whole scan range (v7 2016preVFP Medium/Tight DM0)
    fine = Profile.of(np.linspace(0.6, 1.2, 41), 0.5 * ((np.linspace(0.6, 1.2, 41) - 0.9) / 0.1) ** 2)
    assert fine.window(0.0, (0.11, 2.89)) == pytest.approx((0.9 - 0.1 * np.sqrt(6.18), 0.9 + 0.1 * np.sqrt(6.18)), abs=2e-3)  # a point at the minimum: as before


def test_a_profile_that_does_not_resolve_its_interval_is_scanned_on_a_narrower_range():
    x = np.linspace(0.11, 2.89, 41)  # spacing 0.07 against an interval of +-0.02
    coarse = Profile.of(x, 0.5 * ((x - 0.8965) / 0.02) ** 2 - 0.5 * ((0.8964949 - 0.8965) / 0.02) ** 2)
    narrower = combine.refined(coarse, (0.11, 2.89))
    assert narrower is not None and narrower[0] < 0.8765 and narrower[1] > 0.9165 and narrower[1] - narrower[0] < 0.5
    x = np.linspace(0.6, 1.2, 41)
    assert combine.refined(Profile.of(x, 0.5 * ((x - 0.9) / 0.1) ** 2), (0.11, 2.89)) is None  # 13 points inside: resolved
    assert combine.refined(Profile.of(x, 0.5 * ((x - 1.15) / 0.1) ** 2), (0.11, 2.89)) is None  # an open side: widened's case
    between = Profile.of(np.array([0.7, 0.8, 0.9, 1.0]), np.array([30.0, 0.7, 0.9, 30.0]))  # no grid point below 0.5
    assert between.crossings(0.5) == (None, None) and combine.widened(between, (0.11, 2.89)) is not None
    low, high = combine.refined(between, (0.11, 2.89))
    assert low < 0.8 and high > 0.9 and high - low < 0.4


def test_a_best_value_outside_its_interval_is_a_problem():
    profile = Profile.of(np.array([0.736, 0.805, 0.875, 0.944, 1.014]), np.array([14.93, 4.59, 0.326, 1.582, 7.826]))  # v7 2016preVFP Medium/Tight DM0 r
    assert profile.best > profile.crossings(0.5)[1]
    interval, problems = combine.profile_interval("r", profile, (0.11, 2.89))
    assert interval is None and problems == [f"r: best value {profile.best:g} outside its interval [{profile.crossings(0.5)[0]:g}, {profile.crossings(0.5)[1]:g}] (the scan does not resolve the interval)"]


def test_a_narrow_likelihood_on_the_coarse_2d_grid_gives_resolved_intervals(tmp_path, fake):
    runner = fake(r=(0.8965, 0.02), es=(-1.71, 0.7))  # every point of the 17 x 17 grid far above the fit
    fit = fit_category(None, "DM0", tmp_path / "synced", tmp_path, "2018", GRID)
    assert fit["problems"] == []
    assert (fit["sf"]["best"], fit["sf"]["low"], fit["sf"]["high"]) == pytest.approx((0.8965, 0.8765, 0.9165), abs=2e-3)
    assert (fit["es"]["best"], fit["es"]["low"], fit["es"]["high"]) == pytest.approx((-1.71, -2.41, -1.01), abs=2e-2)
    for name, (low, high) in fit["ranges"].items():
        bounds = combine.bounds(GRID)[0 if name == "r" else 1]
        assert high - low < 0.5 * (bounds[1] - bounds[0])  # not the whole scan range
    for name, key in (("r", "sf"), ("ES", "es")):
        values = np.linspace(*fit["ranges"][name], combine.PROFILE_POINTS)
        assert np.count_nonzero((values > fit[key]["low"]) & (values < fit[key]["high"])) >= combine.MIN_INTERVAL_POINTS
    assert any(" -n .envelope_r_EMB_DM0" in c for c in runner.commands)


def test_an_interval_that_stays_at_the_range_edge_is_a_problem_naming_its_side(tmp_path, fake):
    fake(es=(19.6, 1.0))
    fit = fit_category(None, "DM1011", tmp_path / "synced", tmp_path, "2018", GRID)
    assert fit["es"] is None and any("ES: no crossing of delta NLL = 0.5 on the high side" in p for p in fit["problems"])
    record = {"vsjet_wp": "Tight", "vsele_wp": "VVLoose", "categories": {"DM1011": fit}}
    assert any(p.startswith("Tight_VVLoose/DM1011: ES: no crossing") and "high side" in p for p in merge_problems(record))


def test_a_singles_fit_off_the_profile_is_flagged_not_blocking(tmp_path, fake):
    fake(singles=((0.95, 0.90, 1.00), (5.2, 3.0, 5.6)))  # the v5 failure: singles in another minimum
    fit = fit_category(None, "DM1011", tmp_path / "synced", tmp_path, "2018", GRID)
    assert fit["problems"] == [] and any(w.startswith("ES best: singles 5.2 vs profile") for w in fit["warnings"])
    text = report([{"vsjet_wp": "Tight", "vsele_wp": "VVLoose", "categories": {"DM1011": fit}}])
    assert "| Tight_VVLoose | DM1011 | 0.95 [" in text and "| 5.2 [3, 5.6] |" in text and "check: ES best: singles 5.2" in text
    failing = fake(fail=(".singles_",))
    fit = fit_category(None, "DM1011", tmp_path / "synced2", tmp_path / "second", "2018", GRID)
    assert fit["problems"] == [] and fit["warnings"][0].startswith("singles cross-check failed") and failing.commands


def test_a_singles_fit_without_a_result_is_a_warning(tmp_path, fake, monkeypatch):
    fake()

    def empty(path, category):  # the limit tree of a singles fit that wrote no entry
        raise IndexError("index 0 is out of bounds for axis 0 with size 0")

    monkeypatch.setattr(combine, "read_singles", empty)
    fit = fit_category(None, "DM0", tmp_path / "synced", tmp_path, "2018", GRID)
    assert fit["problems"] == [] and fit["sf"] is not None
    assert fit["warnings"] == ["singles cross-check failed: no result in its output (IndexError: index 0 is out of bounds for axis 0 with size 0)"]


def _context(tmp_path, analysis, **flags):
    config = RunConfig(analysis="tests.mini_analysis:build", era="2018", channels=["mt", "mm"], ntuples=NtupleConfig(base="/unused"), skim_dir=tmp_path / "skims",
                       output_dir=tmp_path / "out", workers=1, combine=CombineConfig(cmssw_dir="/unused"))
    return MeasureContext(config, analysis, ["mt", "mm"], tmp_path / "out" / "tau_id_es" / "2018", **flags)


def test_refit_runs_the_fits_from_the_stored_synced_shapes(tmp_path, fake):
    analysis = _analysis()
    measurement = TauIdEsMeasurement("Tight", "VVLoose", (-2, 2))
    context = _context(tmp_path, analysis, refit=True)
    with pytest.raises(FileNotFoundError, match="synced shapes of an earlier run are missing"):
        measure(measurement, context)
    hset = HistogramSet()
    for channel, category, process in (("mt", "DM0", "data"), ("mt", "DM0", "EMB"), ("mm", "control_region", "data"), ("mm", "control_region", "MUEMB")):
        hset[HistKey(channel, category, process, "nominal", "Nominal", "m_vis")] = _h(1.0)
    write_synced(hset, analysis, context.output / "Tight_VVLoose" / "synced")
    runner = fake(es=(0.0, 0.02))  # within the ES range of the grid (-2, 2): +-0.1 %
    directory = measure(measurement, context)
    record = json.loads((directory / "results.json").read_text())
    assert record["provenance"]["refit"] is True and record["categories"]["DM0"]["problems"] == []
    assert not (directory / "shapes.root").exists() and (directory / "report.md").exists()
    assert any(c.startswith("MorphingTauID2017") and f"--base_path={(directory / 'synced').resolve()}" in c for c in runner.commands)
    assert (directory / "plots" / "profiles" / "DM0_ES.png").exists() and (directory / "plots" / "scans" / "DM0.png").exists()


def test_merge_refuses_an_interval_at_its_fit_range(tmp_path, mini_run):
    config, analysis = mini_run
    fits = {name: {"sf": {"best": 0.9, "low": 0.85, "high": 0.95}, "es": {"best": -1.0, "low": -3.0, "high": 1.0}, "ranges": {"r": [0.5, 1.5], "ES": [-5.0, 5.0]}, "problems": []} for name in CATEGORIES}
    fits["DM0"]["ranges"]["ES"] = [-3.0, 5.0]
    (tmp_path / "Tight_VVLoose").mkdir()
    (tmp_path / "Tight_VVLoose" / "results.json").write_text(json.dumps({"vsjet_wp": "Tight", "vsele_wp": "VVLoose", "categories": fits, "provenance": {}}))
    with pytest.raises(ValueError, match=r"Tight_VVLoose/DM0: ES: interval \[-3, 1\] at the fit range \[-3, 5\] \(low side\)"):
        merge(MeasureContext(config, analysis, ["mt"], tmp_path, merge=True))
    assert (tmp_path / "report.md").exists()


def _results(es_low=-5.0, es_high=0.5):
    result = {}
    for name in CATEGORIES:
        es = es_low if "PT20_40" in name else es_high if "PT40_200" in name else -1.0
        result[name] = (Interval(0.9, 0.85, 0.95), Interval(es, es - 1.0, es + 1.0))
    return {("Medium", "VVLoose"): result}


def test_the_pt_binned_es_changes_linearly_across_the_40_gev_edge():
    cset = correctionlib.CorrectionSet.from_string(payloads.dumps(correction_set(_results(), {})))
    es = cset["tau_energy_scale"]

    def value(pt, syst="nom"):
        return es.evaluate(pt, 0.3, 1, 5, "DeepTau2018v2p5", "Medium", "VVLoose", syst)

    low, high = (ES_TRANSITION[0] * 40, ES_TRANSITION[1] * 40)
    assert (low, high) == pytest.approx((36.0, 42.0))
    assert value(30.0) == value(35.9) == pytest.approx(0.95) and value(42.0) == value(100.0) == pytest.approx(1.005)
    assert value(39.0) == pytest.approx(0.95 + (1.005 - 0.95) * 3 / 6)
    assert value(36.0, "up") == pytest.approx(0.96) and value(41.999, "down") == pytest.approx(0.995, abs=1e-4)
    corrected = [pt * value(pt) for pt in np.linspace(35.0, 43.0, 161)]
    assert np.all(np.diff(corrected) > 0)  # no gap, no fold: the corrected pT is continuous and increasing
    steps = correctionlib.CorrectionSet.from_string(payloads.dumps(correction_set(_results(), {}, es_transition=False)))["tau_energy_scale"]
    assert steps.evaluate(39.0, 0.3, 1, 5, "DeepTau2018v2p5", "Medium", "VVLoose", "nom") == pytest.approx(0.95)
    assert cset["tau_energy_scale_dm_binned"].evaluate(39.0, 0.3, 1, 5, "DeepTau2018v2p5", "Medium", "VVLoose", "nom") == pytest.approx(0.99)

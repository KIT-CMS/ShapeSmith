"""ScoreBinning categories through the chain (tests/nn_mini.py): the fine bins and unweighted counts of hist, the
counts of the estimates, the data-blind or thesis edges of datacards, the lo/hi split with Asimov data and
blinding.json, the clipping of negative jet fakes, the blinded plots and gallery yields."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import uproot

from shapesmith.config import FriendConfig, NtupleConfig, RunConfig
from shapesmith.datacards import BLINDING, NN_RECORD, bin_categories, bin_name, write_datacard
from shapesmith.estimates import run_estimates
from shapesmith.fill import run_hist
from shapesmith.histogram import NOMINAL_VARIATION, HistKey, Histogram, HistogramSet, raw_variable
from shapesmith.model import AnalysisError, Category, DataMinus, ScoreBinning, Variable
from shapesmith.nn_binning import BinningRules, auto_rebin, resolve_edges
from shapesmith.plotting import stack as stack_module
from shapesmith.plotting.stack import plot_stack, run_plot, stack
from shapesmith.plotting.style import edge_label
from shapesmith.score_bins import resolve_category
from shapesmith.skim import run_skim
from shapesmith.store import read_skims
from shapesmith.validate import validate
from shapesmith.web import gallery
from tests import mini_analysis, nn_mini

CARD_BINS = ["htt_mt_1lo_2018", "htt_mt_1hi_2018", "htt_mt_2_2018", "htt_mt_3_2018"]
BACKGROUNDS = ("ZTT", "TT", "jetFakes")


def _chain(root: Path, method: str = "data_blind"):
    nn_mini.make_nn_dataset(root / "data")
    config = RunConfig(
        analysis="tests.nn_mini:build", era="2018", channels=["mt"], switches={"method": method},
        ntuples=NtupleConfig(base=str(root / "data" / "CROWNRun"), friends=[FriendConfig(base=str(root / "data" / "CROWNFriends" / "nn"))]),
        skim_dir=root / "skims", output_dir=root / "out", workers=1,
    )
    analysis = nn_mini.build(config)
    validate(analysis)
    run_skim(config, analysis)
    filled = run_hist(config, analysis, ["mt"], False, None, True, None, root / "out" / "shapes.root")
    hset = HistogramSet(filled)
    run_estimates(hset, analysis, ["mt"])
    hset.save(root / "out" / "shapes.root")
    card = write_datacard(hset, analysis, ["mt"], root / "out" / "datacards" / "mt")
    return config, analysis, filled, hset, card


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """skim -> hist -> estimate -> datacards of the synthetic NN analysis, once for the module."""
    return _chain(tmp_path_factory.mktemp("nn"))


def _record(card: Path) -> dict:
    return json.loads((card.parent / NN_RECORD).read_text())["mt"]


def _shapes(card: Path) -> dict[str, Histogram]:
    with uproot.open(card.parent / "common" / "htt_input_2018.root") as f:
        return {name: Histogram.from_root(f[name]) for name, cls in f.classnames(recursive=True, cycle=False).items() if cls.startswith("TH1")}


def test_hist_fills_the_fine_bins_and_the_unweighted_counts(chain):
    config, analysis, filled, _, _ = chain
    key = HistKey("mt", "HH", "data", "anti_iso", NOMINAL_VARIATION, "NN_score")
    assert len(filled[key].values) == 200
    raw = filled[HistKey("mt", "HH", "data", "anti_iso", NOMINAL_VARIATION, raw_variable("NN_score"))]
    frame = read_skims(config.skim_dir, "mt", ["DATA_A"], None)
    selected = frame[(frame.q_1 * frame.q_2 < 0) & (frame.id_tight < 0.5) & (frame.predicted_class == 0) & (frame.n_bjets < 3)]
    assert (selected.fake_factor == 0).sum() > 0  # events of weight 0 are not counted
    np.testing.assert_array_equal(raw.values, np.histogram(selected.predicted_max_value[selected.fake_factor != 0], bins=nn_mini.FINE)[0])
    assert not np.array_equal(raw.values, filled[key].values)  # weighted by the fake factor
    expected = {(p, r) for p in ("data", "ZTT", "TT") for r in ("nominal", "anti_iso")} | {("HH", "nominal")}
    for category in nn_mini.CLASSES:
        assert {(k.process, k.region) for k in filled if k.category == category and k.variable == raw_variable("NN_score")} == expected
    assert not any(k.variable == raw_variable("NN_score") and k.variation != NOMINAL_VARIATION for k in filled)
    assert not any(k.variable == raw_variable("score") for k in filled)  # fixed edges: no counts


def test_estimate_gives_the_jet_fakes_the_counts_of_the_anti_isolated_data(chain):
    _, _, filled, hset, _ = chain
    for category in nn_mini.CLASSES:
        jet_fakes = hset[HistKey("mt", category, "jetFakes", "nominal", NOMINAL_VARIATION, raw_variable("NN_score"))]
        np.testing.assert_array_equal(jet_fakes.values, filled[HistKey("mt", category, "data", "anti_iso", NOMINAL_VARIATION, raw_variable("NN_score"))].values)
    assert not any(k.process == "jetFakes" and k.variable == raw_variable("NN_score") and k.variation != NOMINAL_VARIATION for k in hset)


def test_the_card_splits_the_hh_category_at_the_blinding_edge(chain):
    _, _, _, _, card = chain
    lines = card.read_text().splitlines()
    assert next(line for line in lines if line.startswith("bin ")).split()[1:] == CARD_BINS
    record = _record(card)["HH"]
    split, edges = record["blinding"]["first_hi_bin"], record["edges"]
    assert 0 < split < len(edges) - 1 and record["card_bins"] == CARD_BINS[:2]
    ratios = record["blinding"]["s_over_b_per_bin"]
    assert ratios[split] > 0.005 and all(r <= 0.005 for r in ratios[:split])
    assert json.loads((card.parent / BLINDING).read_text()) == {
        "edge_rule": "S/B(mu=1) > 0.005",
        "masked": ["htt_mt_1hi_2018"],
        "categories": {"htt_mt_1hi_2018": {"lo": "htt_mt_1lo_2018", "edge": edges[split], "s_over_b_first_hi_bin": ratios[split]}},
        "asimov_data": True,
    }
    assert "blinding" not in _record(card)["ZTT"] and set(_record(card)) == set(nn_mini.CLASSES)


def test_the_hi_channel_has_asimov_data_and_the_lo_channel_the_data(chain):
    _, _, _, hset, card = chain
    shapes = _shapes(card)
    record = _record(card)["HH"]
    split, edges = record["blinding"]["first_hi_bin"], record["edges"]
    hi = sum(shapes[f"htt_mt_1hi_2018/{p}"].values for p in BACKGROUNDS)
    np.testing.assert_allclose(shapes["htt_mt_1hi_2018/data_obs"].values, hi, rtol=1e-12)
    data = hset[HistKey("mt", "HH", "data", "nominal", NOMINAL_VARIATION, "NN_score")].rebin(edges)
    np.testing.assert_array_equal(shapes["htt_mt_1lo_2018/data_obs"].values, data.values[:split])
    assert not np.allclose(shapes["htt_mt_1hi_2018/data_obs"].values, data.values[split:])  # the data stay hidden
    np.testing.assert_allclose(np.concatenate([shapes[f"htt_mt_1{p}_2018/data_obs"].edges[:-1] for p in ("lo", "hi")] + [[1.0]]), edges)
    observation = next(line for line in card.read_text().splitlines() if line.startswith("observation")).split()[1:]
    assert float(observation[1]) == shapes["htt_mt_1hi_2018/data_obs"].values.sum() and observation[0] == f"{data.values[:split].sum():.1f}"
    for p in ("HH", "ZTT", "TT", "jetFakes"):  # the parts of every template and variation are its rebinned whole
        whole = hset[HistKey("mt", "HH", p, "nominal", NOMINAL_VARIATION, "NN_score")].rebin(edges).values
        np.testing.assert_allclose(np.concatenate([shapes[f"htt_mt_1lo_2018/{p}"].values, shapes[f"htt_mt_1hi_2018/{p}"].values]), whole)
    assert "htt_mt_1hi_2018/HH_CMS_puUp" in shapes


def test_the_edges_are_those_of_the_pure_function_and_never_read_data(chain):
    _, analysis, _, hset, card = chain
    category = analysis.channel("mt").categories[0]

    def nominal(process, variable="NN_score"):
        return hset[HistKey("mt", "HH", process, "nominal", NOMINAL_VARIATION, variable)]

    backgrounds = {p: (nominal(p).values, nominal(p).variances, nominal(p, raw_variable("NN_score")).values) for p in BACKGROUNDS}
    edges = resolve_edges(nn_mini.FINE, nominal("HH").values, backgrounds, category.variable.edges.rules)
    assert list(edges) == _record(card)["HH"]["edges"] and len(edges) > 2
    altered = HistogramSet(hset)
    for key in [k for k in hset if k.process == "data"]:
        altered[key] = Histogram(hset[key].edges, hset[key].values * 7.0 + 3.0, hset[key].variances)
    view = resolve_category(altered, analysis, "mt", category)
    assert view.edges == list(edges) and view.split == _record(card)["HH"]["blinding"]["first_hi_bin"]


def test_the_thesis_binning_rebins_the_start_edges_with_autorebin(tmp_path):
    _, analysis, _, hset, card = _chain(tmp_path, "thesis")
    for index, name in enumerate(nn_mini.CLASSES):
        binning = analysis.channel("mt").categories[index].variable.edges
        total = Histogram.empty(binning.thesis)
        for p in BACKGROUNDS:
            total.add(hset[HistKey("mt", name, p, "nominal", NOMINAL_VARIATION, "NN_score")].rebin(binning.thesis))
        record = _record(card)[name]
        assert record["method"] == "thesis" and record["edges"] == list(auto_rebin(binning.thesis, total.values, total.variances))
        assert set(record["edges"]) <= set(binning.thesis)
    assert len(_record(card)["HH"]["edges"]) < len(nn_mini.THESIS_HH_EDGES)  # AutoRebin merged


def test_the_plot_shows_the_prediction_in_the_blinded_bins(chain, tmp_path):
    _, analysis, _, hset, card = chain
    record = _record(card)["HH"]
    split = record["blinding"]["first_hi_bin"]
    shown = stack(hset, analysis, "mt", "HH", "NN_score")
    data = hset[HistKey("mt", "HH", "data", "nominal", NOMINAL_VARIATION, "NN_score")].rebin(record["edges"])
    assert shown.blinded_from == split and list(shown.groups[0][1].edges) == record["edges"]
    np.testing.assert_array_equal(shown.data.values[:split], data.values[:split])
    np.testing.assert_allclose(shown.data.values[split:], shown.background[split:])
    shapes = _shapes(card)
    np.testing.assert_allclose(shown.background, sum(np.concatenate([shapes[f"htt_mt_1lo_2018/{p}"].values, shapes[f"htt_mt_1hi_2018/{p}"].values]) for p in BACKGROUNDS))
    np.testing.assert_allclose(shown.signal.values, hset[HistKey("mt", "HH", "HH", "nominal", NOMINAL_VARIATION, "NN_score")].rebin(record["edges"]).values)
    anti = stack(hset, analysis, "mt", "HH", "NN_score", "anti_iso")
    assert anti.blinded_from is None and list(anti.data.edges) == record["edges"]
    np.testing.assert_array_equal(anti.data.values, hset[HistKey("mt", "HH", "data", "anti_iso", NOMINAL_VARIATION, "NN_score")].rebin(record["edges"]).values)
    assert stack(hset, analysis, "mt", "ZTT", "NN_score").blinded_from is None
    files = run_plot(hset, analysis, ["mt"], False, None, None, tmp_path)
    assert sorted(p.name for p in files if p.suffix == ".png") == ["HH_NN_score.png", "TT_NN_score.png", "ZTT_NN_score.png"]


def test_the_plot_draws_the_card_bins_with_equal_widths(chain, tmp_path, monkeypatch):
    """The plot of a score category draws the bins of its card with equal widths (positions 0..n), the edges as tick
    labels, events per bin and the blinded bins shaded from their index; the drawn yields are the stack's.
    --normalize-by-bin-width keeps the score axis (dN/dx)."""
    _, analysis, _, hset, card = chain
    record = _record(card)["HH"]
    edges, split = record["edges"], record["blinding"]["first_hi_bin"]
    figures, drawn, original = [], [], stack_module.hep.histplot
    monkeypatch.setattr(stack_module.plt, "close", figures.append)
    monkeypatch.setattr(stack_module.hep, "histplot", lambda values, *args, **kwargs: drawn.append((values, kwargs["bins"])) or original(values, *args, **kwargs))
    plot_stack(hset, analysis, "mt", "HH", "NN_score", tmp_path)
    ax, rax = figures[-1].axes
    n = len(edges) - 1
    assert rax.get_xlim() == (0.0, n) and [t.get_text() for t in rax.get_xticklabels()] == [edge_label(e) for e in edges]
    assert rax.get_xlabel().endswith("(card bins)") and ax.get_ylabel() == "Events / bin" and 2 < n < len(edges) * 2
    stacked, bins = drawn[0]
    assert list(bins) == list(range(n + 1))
    shown = stack(hset, analysis, "mt", "HH", "NN_score")
    np.testing.assert_allclose(np.sum(stacked, axis=0), shown.background)
    np.testing.assert_allclose(drawn[-1][0], shown.data.values)  # the data with the prediction in the hi bins
    shaded = [patch for patch in ax.patches if patch.get_label() == "Blinded: prediction as data"]
    assert len(shaded) == 1 and shaded[0].get_x() == pytest.approx(split) and shaded[0].get_width() == pytest.approx(n - split)
    plot_stack(hset, analysis, "mt", "HH", "NN_score", tmp_path, normalize_by_bin_width=True)
    ax, rax = figures[-1].axes
    assert rax.get_xlim() == pytest.approx((edges[0], edges[-1])) and ax.get_ylabel() == "dN/dx" and list(drawn[-1][1]) == pytest.approx(edges)
    np.testing.assert_allclose(drawn[-1][0], shown.data.values / np.diff(edges))
    plot_stack(hset, analysis, "mt", "ZTT", "NN_score", tmp_path)  # a category without blinding: the same axis
    view_edges = _record(card)["ZTT"]["edges"]
    assert figures[-1].axes[1].get_xlim() == (0.0, len(view_edges) - 1)


def test_the_gallery_yields_hide_the_blinded_data(chain):
    config, analysis, _, hset, card = chain
    plots = {"mt": {("nominal", "HH", "NN_score"): {}, ("anti_iso", "HH", "NN_score"): {}}}
    yields = gallery._yields(config, analysis, plots)["mt"]
    record = _record(card)["HH"]
    split = record["blinding"]["first_hi_bin"]
    data = hset[HistKey("mt", "HH", "data", "nominal", NOMINAL_VARIATION, "NN_score")].rebin(record["edges"])
    prediction = yields["nominal"]["HH"]["prediction"]
    hi = sum(_shapes(card)[f"htt_mt_1hi_2018/{p}"].values.sum() for p in BACKGROUNDS)
    assert yields["nominal"]["HH"]["data"] == pytest.approx(data.values[:split].sum() + hi)
    assert prediction == pytest.approx(sum(_shapes(card)[f"htt_mt_1{part}_2018/{p}"].values.sum() for part in ("lo", "hi") for p in BACKGROUNDS))
    assert yields["anti_iso"]["HH"]["data"] == pytest.approx(hset[HistKey("mt", "HH", "data", "anti_iso", NOMINAL_VARIATION, "NN_score")].sum())


def _hand_made(binning: ScoreBinning, jet_fakes, signal=None, data=None, clip_negative=False):
    """The mini analysis with one ScoreBinning category on 10 fine bins and a hand-made histogram set."""
    analysis = mini_analysis.build()
    mt = analysis.channel("mt")
    estimators = (DataMinus("jetFakes", "anti_iso", ("ZTT", "ZL"), clip_negative=clip_negative),)
    channel = dataclasses.replace(mt, categories=(Category("sig", "cls == 0", Variable("score", "score", binning)),), estimators=estimators)
    analysis = dataclasses.replace(analysis, channels={"mt": channel})
    edges = list(binning.fine)
    hset = HistogramSet()

    def put(process, values, variation=NOMINAL_VARIATION, raw=None):
        values = np.asarray(values, dtype=float)
        hset[HistKey("mt", "sig", process, "nominal", variation, "score")] = Histogram(edges, values, np.abs(values) * 0.01)
        if variation == NOMINAL_VARIATION:
            hset[HistKey("mt", "sig", process, "nominal", variation, raw_variable("score"))] = Histogram(edges, np.full(len(values), 50.0) if raw is None else raw, np.zeros(len(values)))

    put("data", data if data is not None else [30.0] * 10)
    put("ZTT", [20.0, 18, 16, 14, 12, 10, 8, 6, 4, 2])
    put("ZL", [5.0] * 10)
    put("jetFakes", jet_fakes)
    put("jetFakes", np.asarray(jet_fakes) * 1.1, "CMS_ffUp")
    put("jetFakes", np.asarray(jet_fakes) * 0.9, "CMS_ffDown")
    put("HH", signal if signal is not None else [0.001, 0.002, 0.004, 0.008, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8])
    validate(analysis)
    return analysis, hset


FINE10 = tuple(float(round(x, 2)) for x in np.linspace(0.0, 1.0, 11))
THESIS10 = ScoreBinning(FINE10, thesis=FINE10, blinding=0.005, clip=("jetFakes",))  # AutoRebin keeps all ten bins


def test_negative_jet_fakes_left_after_the_rebinning_are_clipped_and_recorded(tmp_path, caplog):
    jet_fakes = [3.0, 3, 3, 3, -1.0, 3, 3, 3, 3, 3]
    analysis, hset = _hand_made(THESIS10, jet_fakes)
    with caplog.at_level("INFO", logger="shapesmith.datacards"):
        card = write_datacard(hset, analysis, ["mt"], tmp_path / "mt")
    clipped = json.loads((card.parent / "template_changes.json").read_text())["clipped"]
    assert [(c["process"], c["variation"], c["bins"]) for c in clipped] == [("jetFakes", "Nominal", [4]), ("jetFakes", "CMS_ffDown", [4]), ("jetFakes", "CMS_ffUp", [4])]
    assert clipped[0]["values"] == [-1.0] and clipped[0]["integral"] == pytest.approx(26.0)
    assert "3 templates with negative bins after the rebinning clipped at 0" in caplog.text
    shapes = _shapes(card)
    whole = np.concatenate([shapes[f"htt_mt_1{p}_2018/jetFakes"].values for p in ("lo", "hi")])
    assert whole.min() >= 0.0 and whole.sum() == pytest.approx(26.0) and whole[4] == 0.0
    up = np.concatenate([shapes[f"htt_mt_1{p}_2018/jetFakes_CMS_ffUp"].values for p in ("lo", "hi")])
    assert up.min() >= 0.0 and up.sum() == pytest.approx(26.0 * 1.1)
    assert np.concatenate([shapes[f"htt_mt_1{p}_2018/ZTT"].values for p in ("lo", "hi")]).tolist() == [20.0, 18, 16, 14, 12, 10, 8, 6, 4, 2]


def test_the_data_blind_rules_merge_away_a_negative_bin_before_clipping(tmp_path):
    rules = BinningRules(n_start=10, required={"jetFakes": ("jetFakes",)})
    analysis, hset = _hand_made(ScoreBinning(FINE10, rules=rules, clip=("jetFakes",)), [3.0, 3, 3, 3, 3, 3, 3, 3, 1.0, -0.5], signal=[0.25] * 10)
    card = write_datacard(hset, analysis, ["mt"], tmp_path / "mt")
    assert _record(card)["sig"]["edges"] == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    assert json.loads((card.parent / "template_changes.json").read_text())["clipped"] == []
    assert not (card.parent / BLINDING).exists()  # no blinding configured


def test_estimator_outputs_that_clip_are_clipped_after_the_rebinning(tmp_path):
    """DataMinus with clip_negative leaves the fine bins of a ScoreBinning category to datacards."""
    analysis, hset = _hand_made(THESIS10, [0.0] * 10, clip_negative=True)
    for key in [k for k in hset if k.process == "jetFakes"]:
        del hset[key]
    for process, values in (("data", [6.0, 6, 6, 6, 6, 6, 6, 6, 6, 6]), ("ZTT", [2.0, 2, 2, 2, 8, 2, 2, 2, 2, 2]), ("ZL", [1.0] * 10)):
        hset[HistKey("mt", "sig", process, "anti_iso", NOMINAL_VARIATION, "score")] = Histogram(list(FINE10), values, values)
        hset[HistKey("mt", "sig", process, "anti_iso", NOMINAL_VARIATION, raw_variable("score"))] = Histogram(list(FINE10), values, values)
    run_estimates(hset, analysis, ["mt"])
    fine = hset[HistKey("mt", "sig", "jetFakes", "nominal", NOMINAL_VARIATION, "score")]
    assert fine.values[4] == -3.0  # not clipped in the estimate
    np.testing.assert_array_equal(hset[HistKey("mt", "sig", "jetFakes", "nominal", NOMINAL_VARIATION, raw_variable("score"))].values, [6.0] * 10)
    card = write_datacard(hset, analysis, ["mt"], tmp_path / "mt")
    assert [c["bins"] for c in json.loads((card.parent / "template_changes.json").read_text())["clipped"]] == [[4]]


@pytest.mark.parametrize("threshold,bins,masked", [
    (1.0, ["htt_mt_1_2018"], []),  # no bin above: one channel with the data, nothing masked
    (1e-9, ["htt_mt_1hi_2018"], ["htt_mt_1hi_2018"]),  # every bin above: no lo channel
])
def test_a_category_without_or_with_only_blinded_bins(tmp_path, threshold, bins, masked):
    binning = dataclasses.replace(THESIS10, blinding=threshold)
    analysis, hset = _hand_made(binning, [3.0] * 10)
    card = write_datacard(hset, analysis, ["mt"], tmp_path / "mt")
    assert next(line for line in card.read_text().splitlines() if line.startswith("bin ")).split()[1:] == bins
    blinding = json.loads((card.parent / BLINDING).read_text())
    assert blinding["masked"] == masked and blinding["asimov_data"] is True
    if masked:
        assert blinding["categories"]["htt_mt_1hi_2018"]["lo"] is None and blinding["categories"]["htt_mt_1hi_2018"]["edge"] == 0.0
        np.testing.assert_allclose(_shapes(card)["htt_mt_1hi_2018/data_obs"].values, [28.0, 26, 24, 22, 20, 18, 16, 14, 12, 10])


def test_the_record_is_strict_json(tmp_path, caplog):
    """A bin without background has no relative uncertainty: null, not Infinity; a short signal category is logged."""
    analysis, hset = _hand_made(ScoreBinning(FINE10, rules=BinningRules(n_start=10), blinding=0.005), [0.0] * 10)
    for process in ("ZTT", "ZL"):
        hset[HistKey("mt", "sig", process, "nominal", NOMINAL_VARIATION, "score")] = Histogram(list(FINE10), np.zeros(10), np.zeros(10))
    with caplog.at_level("WARNING", logger="shapesmith.score_bins"):
        card = write_datacard(hset, analysis, ["mt"], tmp_path / "mt")
    record = json.loads((card.parent / NN_RECORD).read_text(), parse_constant=lambda constant: pytest.fail(f"{constant} in the record"))
    assert record["mt"]["sig"]["edges"] == [0.0, 1.0] and record["mt"]["sig"]["bins"][0]["rel_uncertainty"] is None
    assert "only 1 bins in a category with blinding" in caplog.text and "fail a rule" in caplog.text


def test_cards_without_score_categories_stay_as_they_were(tmp_path):
    """No nn_binning.json, no blinding.json (a stale one is removed) and no "clipped" in template_changes.json."""
    from tests.test_datacards import _hset

    for name in (NN_RECORD, BLINDING):
        (tmp_path / "mt").mkdir(exist_ok=True)
        (tmp_path / "mt" / name).write_text("{}")
    card = write_datacard(_hset(), mini_analysis.build(), ["mt"], tmp_path / "mt")
    assert not (card.parent / NN_RECORD).exists() and not (card.parent / BLINDING).exists()
    assert set(json.loads((card.parent / "template_changes.json").read_text())) == {"asymmetry_ratio", "no_ops", "symmetrised"}


def test_bin_names_of_the_parts():
    analysis, _ = _hand_made(THESIS10, [3.0] * 10)
    assert bin_name(analysis, "mt", 0, "hi") == "htt_mt_1hi_2018" and bin_name(analysis, "mt", 0) == "htt_mt_1_2018"
    assert bin_categories(analysis) == {"htt_mt_1_2018": ("mt", "sig"), "htt_mt_1lo_2018": ("mt", "sig lo"), "htt_mt_1hi_2018": ("mt", "sig hi")}
    assert bin_categories(mini_analysis.build()) == {"htt_mt_1_2018": ("mt", "sig"), "htt_mt_2_2018": ("mt", "bkg")}


def test_missing_unweighted_counts_are_reported(tmp_path):
    analysis, hset = _hand_made(ScoreBinning(FINE10, rules=BinningRules(n_start=10, required={"Z": ("ZTT",)})), [3.0] * 10)
    del hset[HistKey("mt", "sig", "ZTT", "nominal", NOMINAL_VARIATION, raw_variable("score"))]
    with pytest.raises(ValueError, match="no unweighted counts"):
        write_datacard(hset, analysis, ["mt"], tmp_path / "mt")


@pytest.mark.parametrize("binning,match", [
    (ScoreBinning(FINE10), "exactly one of rules and thesis"),
    (ScoreBinning(FINE10, rules=BinningRules(), thesis=FINE10), "exactly one of rules and thesis"),
    (ScoreBinning(FINE10, thesis=(0.0, 0.33, 1.0)), "thesis edges are not among the fine edges"),
    (ScoreBinning(FINE10, rules=BinningRules(required={"Top": ("TTL",)})), "TTL is no background"),
    (ScoreBinning(FINE10, rules=BinningRules(low=1.0)), "rules.low"),
    (ScoreBinning(FINE10, thesis=FINE10, clip=("QCD",)), "clips QCD"),
    (ScoreBinning(FINE10, thesis=FINE10, blinding=0.0), "blinding S/B must be > 0"),
    (ScoreBinning((0.0,), thesis=(0.0,)), "fine edges"),
])
def test_invalid_score_binnings_are_reported(binning, match):
    analysis = mini_analysis.build()
    mt = analysis.channel("mt")
    channel = dataclasses.replace(mt, categories=(Category("sig", "cls == 0", Variable("score", "score", binning)),))
    with pytest.raises(AnalysisError, match=match):
        validate(dataclasses.replace(analysis, channels={"mt": channel}))
    control = dataclasses.replace(mt, variables={"score": Variable("score", "score", ScoreBinning(FINE10, thesis=FINE10))})
    with pytest.raises(AnalysisError, match="for categories only"):
        validate(dataclasses.replace(analysis, channels={"mt": control}))

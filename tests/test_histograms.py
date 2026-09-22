import dataclasses

import numpy as np
import pytest

from shapesmith.histograms import INCLUSIVE, HistKey, HistogramSet, bookings, resolve_regions, run_hist, targets
from shapesmith.io.skims import read_skims
from shapesmith.model import Region, Selection


def test_histkey_roundtrip():
    key = HistKey("mt", "sig", "ZTT", "nominal", "CMS_puUp", "score")
    assert key.directory == "mt_sig" and key.object_name == "ZTT#nominal#CMS_puUp#score"
    assert HistKey.parse(key.path) == key
    assert HistKey.parse("mt_inclusive/data#anti_iso#Nominal#m_vis") == HistKey("mt", INCLUSIVE, "data", "anti_iso", "Nominal", "m_vis")


def test_bookings_follow_process_kinds(mini_run):
    _, analysis = mini_run
    by_process = {b.process.name: b for b in bookings(analysis, "mt", systematics=True)}
    assert by_process["data"].regions == ("nominal", "anti_iso") and by_process["data"].variations == ()
    assert by_process["HH"].regions == ("nominal",) and [v.name for v in by_process["HH"].variations] == ["CMS_puUp", "CMS_puDown"]
    assert by_process["ZTT"].regions == ("nominal", "anti_iso") and len(by_process["ZTT"].variations) == 2
    assert all(b.variations == () for b in bookings(analysis, "mt", systematics=False))


def test_explicit_regions_include_signal_and_all_expands_per_channel(mini_run):
    _, analysis = mini_run
    channel = analysis.channel("mt")
    other = dataclasses.replace(channel, name="et", regions=channel.regions[:1])
    analysis = dataclasses.replace(analysis, channels={"mt": channel, "et": other})

    assert resolve_regions(analysis, "mt", ["all"]) == ("nominal", "same_sign", "anti_iso")
    assert resolve_regions(analysis, "et", ["all"]) == ("nominal", "same_sign")
    assert {b.process.name: b.regions for b in bookings(analysis, "mt", True, ["same_sign"])}["HH"] == ("same_sign",)
    with pytest.raises(KeyError, match="channel mt: unknown region 'missing'"):
        bookings(analysis, "mt", False, ["missing"])


def test_targets(mini_run):
    _, analysis = mini_run
    assert [(t.category, t.cut, t.variable.name) for t in targets(analysis, control=False, variables=None)] == [("sig", "cls == 0", "score"), ("bkg", "cls == 1", "score")]
    assert [(t.category, t.cut, t.variable.name) for t in targets(analysis, control=True, variables=None)] == [(INCLUSIVE, None, "m_vis")]
    with pytest.raises(KeyError):
        targets(analysis, control=True, variables=["nope"])


def test_run_hist_control_matches_numpy(mini_run):
    config, analysis = mini_run
    hset = run_hist(config, analysis, ["mt"], control=True, variables=None, systematics=True, processes=None, output=config.output_dir / "control_shapes.root", force=True)
    frame = read_skims(config.skim_dir, "mt", ["ZTT_1"], None)
    sel = (frame["veto"] < 0.5) & ((frame["q_1"] * frame["q_2"]) < 0) & (frame["id_tight"] > 0.5) & (frame["gen_match"] == 5)
    weight = frame["norm_weight"] * analysis.lumi_pb * frame["trg_wgt"] * frame["puweight"]
    expected, _ = np.histogram(frame.loc[sel, "m_vis"], bins=[0, 50, 100, 200], weights=weight[sel])
    h = hset.get(HistKey("mt", INCLUSIVE, "ZTT", "nominal", "Nominal", "m_vis"))
    assert np.allclose(h.values, expected) and h.edges.tolist() == [0.0, 50.0, 100.0, 200.0]
    data = read_skims(config.skim_dir, "mt", ["DATA_A"], None)
    dsel = (data["veto"] < 0.5) & ((data["q_1"] * data["q_2"]) < 0) & (data["id_tight"] > 0.5)
    assert hset.get(HistKey("mt", INCLUSIVE, "data", "nominal", "Nominal", "m_vis")).sum() == pytest.approx(dsel.sum())
    anti = (data["veto"] < 0.5) & ((data["q_1"] * data["q_2"]) < 0) & (data["id_tight"] < 0.5) & (data["id_loose"] > 0.5)
    assert hset.get(HistKey("mt", INCLUSIVE, "data", "anti_iso", "Nominal", "m_vis")).sum() == pytest.approx(data.loc[anti, "fake_factor"].sum())
    assert not hset.keys(process="data", variation="CMS_puUp")
    up = hset.get(HistKey("mt", INCLUSIVE, "ZTT", "nominal", "CMS_puUp", "m_vis"))
    assert np.allclose(up.values, expected * 1.05)
    assert (config.output_dir / "control_shapes.root").exists() and (config.output_dir / "control_shapes.json").exists()
    reloaded = HistogramSet.load(config.output_dir / "control_shapes.root")
    assert len(reloaded) == len(hset)
    again = reloaded.get(HistKey("mt", INCLUSIVE, "ZTT", "nominal", "Nominal", "m_vis"))
    assert np.allclose(again.values, expected) and np.allclose(again.variances, h.variances) and again.edges.tolist() == [0.0, 50.0, 100.0, 200.0]


def test_run_hist_explicit_region_uses_event_selection_for_every_process(mini_run):
    config, analysis = mini_run
    hset = run_hist(
        config,
        analysis,
        ["mt"],
        control=True,
        variables=None,
        systematics=False,
        processes=None,
        output=config.output_dir / "same_sign.root",
        force=True,
        regions=["same_sign"],
    )

    data = read_skims(config.skim_dir, "mt", ["DATA_A"], None)
    same_sign = (data["veto"] < 0.5) & ((data["q_1"] * data["q_2"]) > 0) & (data["id_tight"] > 0.5)
    assert hset.get(HistKey("mt", INCLUSIVE, "data", "same_sign", "Nominal", "m_vis")).sum() == pytest.approx(same_sign.sum())
    assert hset.has(HistKey("mt", INCLUSIVE, "HH", "same_sign", "Nominal", "m_vis"))
    assert {key.region for key in hset.keys()} == {"same_sign"}


def test_region_replacement_of_baseline_weight_is_mc_only(mini_run):
    config, analysis = mini_run
    channel = analysis.channel("mt")
    baseline = Selection(channel.baseline.cuts, {**channel.baseline.weights, "mc_only": "puweight"})
    regions = tuple(
        Region(region.name, region.replace_cuts, {**region.add_weights, "mc_only": "puweight_up"})
        if region.name == "same_sign"
        else region
        for region in channel.regions
    )
    analysis = dataclasses.replace(analysis, channels={"mt": dataclasses.replace(channel, baseline=baseline, regions=regions)})

    hset = run_hist(
        config,
        analysis,
        ["mt"],
        True,
        None,
        False,
        ["data", "ztt"],
        config.output_dir / "weight_replacement.root",
        force=True,
        regions=["same_sign"],
    )

    data = read_skims(config.skim_dir, "mt", ["DATA_A"], None)
    selected_data = (data["veto"] < 0.5) & ((data["q_1"] * data["q_2"]) > 0) & (data["id_tight"] > 0.5)
    assert hset.get(HistKey("mt", INCLUSIVE, "data", "same_sign", "Nominal", "m_vis")).sum() == pytest.approx(selected_data.sum())

    mc = read_skims(config.skim_dir, "mt", ["ZTT_1"], None)
    selected_mc = (mc["veto"] < 0.5) & ((mc["q_1"] * mc["q_2"]) > 0) & (mc["id_tight"] > 0.5) & (mc["gen_match"] == 5)
    expected = mc.loc[selected_mc, "norm_weight"] * analysis.lumi_pb * mc.loc[selected_mc, "trg_wgt"] * mc.loc[selected_mc, "puweight"] * mc.loc[selected_mc, "puweight_up"]
    assert hset.get(HistKey("mt", INCLUSIVE, "ZTT", "same_sign", "Nominal", "m_vis")).sum() == pytest.approx(expected.sum())


def test_run_hist_force_keeps_other_processes(mini_run):
    config, analysis = mini_run
    output = config.output_dir / "partial.root"
    first = run_hist(config, analysis, ["mt"], control=True, variables=None, systematics=False, processes=None, output=output, force=True)
    again = run_hist(config, analysis, ["mt"], control=True, variables=None, systematics=False, processes=["hh"], output=output, force=True)
    assert len(again) == len(first) and set(again.keys()) == set(first.keys())


def test_run_hist_partial_force_keeps_other_variables_and_regions(mini_run):
    config, analysis = mini_run
    analysis = dataclasses.replace(analysis, control_variables={**analysis.control_variables, "score": analysis.categories[0].variable})
    output = config.output_dir / "partial_regions.root"
    first = run_hist(config, analysis, ["mt"], True, None, False, None, output, force=True, regions=["all"])
    kept_key = HistKey("mt", INCLUSIVE, "HH", "nominal", "Nominal", "score")
    kept_values = first.get(kept_key).values.copy()

    again = run_hist(config, analysis, ["mt"], True, ["m_vis"], False, ["hh"], output, force=True, regions=["same_sign"])

    assert set(again.keys()) == set(first.keys())
    assert np.array_equal(again.get(kept_key).values, kept_values)


def test_run_hist_categories(mini_run):
    config, analysis = mini_run
    hset = run_hist(config, analysis, ["mt"], control=False, variables=None, systematics=False, processes=["hh", "data"], output=config.output_dir / "shapes.root", force=True)
    assert {k.category for k in hset.keys()} == {"sig", "bkg"}
    assert {k.process for k in hset.keys()} == {"HH", "data"}
    assert hset.keys(process="HH", region="anti_iso") == []
    assert hset.get(HistKey("mt", "sig", "HH", "nominal", "Nominal", "score")).edges.tolist() == [0.0, 0.5, 1.0]

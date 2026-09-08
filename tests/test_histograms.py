import numpy as np
import pytest

from shapesmith.histograms import INCLUSIVE, HistKey, HistogramSet, bookings, run_hist, targets
from shapesmith.io.skims import read_skims


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


def test_run_hist_force_keeps_other_processes(mini_run):
    config, analysis = mini_run
    output = config.output_dir / "partial.root"
    first = run_hist(config, analysis, ["mt"], control=True, variables=None, systematics=False, processes=None, output=output, force=True)
    again = run_hist(config, analysis, ["mt"], control=True, variables=None, systematics=False, processes=["hh"], output=output, force=True)
    assert len(again) == len(first) and set(again.keys()) == set(first.keys())


def test_run_hist_categories(mini_run):
    config, analysis = mini_run
    hset = run_hist(config, analysis, ["mt"], control=False, variables=None, systematics=False, processes=["hh", "data"], output=config.output_dir / "shapes.root", force=True)
    assert {k.category for k in hset.keys()} == {"sig", "bkg"}
    assert {k.process for k in hset.keys()} == {"HH", "data"}
    assert hset.keys(process="HH", region="anti_iso") == []
    assert hset.get(HistKey("mt", "sig", "HH", "nominal", "Nominal", "score")).edges.tolist() == [0.0, 0.5, 1.0]

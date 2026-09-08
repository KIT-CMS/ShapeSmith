import uproot

from shapesmith.histogram import Histogram
from shapesmith.histograms import HistKey, HistogramSet
from shapesmith.sync import run_sync, synced_processes
from tests.mini_analysis import build


def _hist(value):
    return Histogram([0.0, 0.5, 1.0], [value, value], [value, value])


def test_synced_processes():
    assert synced_processes(build()) == ("ZTT", "ZL", "jetFakes", "HH")


def test_run_sync_layout(tmp_path):
    analysis = build()
    hset = HistogramSet()
    for category in ("sig", "bkg"):
        hset.add(HistKey("mt", category, "data", "nominal", "Nominal", "score"), _hist(3.0))
        hset.add(HistKey("mt", category, "data", "anti_iso", "Nominal", "score"), _hist(1.0))
        hset.add(HistKey("mt", category, "ZTT", "nominal", "Nominal", "score"), _hist(2.0))
        hset.add(HistKey("mt", category, "ZTT", "nominal", "CMS_puUp", "score"), _hist(2.1))
        hset.add(HistKey("mt", category, "ZTT", "anti_iso", "Nominal", "score"), _hist(0.2))
        hset.add(HistKey("mt", category, "HH", "nominal", "Nominal", "score"), _hist(0.1))
        hset.add(HistKey("mt", category, "jetFakes", "nominal", "Nominal", "score"), _hist(0.8))
    hset.add(HistKey("mt", "inclusive", "ZTT", "nominal", "Nominal", "m_vis"), _hist(5.0))  # control shape, never synced
    files = run_sync(hset, analysis, ["mt"], tmp_path)
    assert files == [tmp_path / "synced" / "htt_mt.inputs-Run2018.root"]
    with uproot.open(files[0]) as f:
        names = sorted(k for k in f.keys(recursive=True, cycle=False) if "/" in k)
        assert names == sorted(f"mt_{c}/{p}" for c in ("sig", "bkg") for p in ("data_obs", "ZTT", "ZTT_CMS_puUp", "HH", "jetFakes"))
        assert f["mt_sig/data_obs"].values().tolist() == [3.0, 3.0] and f["mt_sig/data_obs"].axis().edges().tolist() == [0.0, 0.5, 1.0]

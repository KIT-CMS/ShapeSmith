import uproot

from shapesmith.datacards import bin_name, rebin_edges, run_datacards, shape_processes, write_datacard
from shapesmith.histogram import Histogram
from shapesmith.histograms import HistKey, HistogramSet
from tests.mini_analysis import build


def _hist(values, edges=(0.0, 0.25, 0.5, 0.75, 1.0)):
    return Histogram(list(edges), values, [v * 0.1 for v in values])


def _hset():
    hset = HistogramSet()
    for category in ("sig", "bkg"):
        hset.add(HistKey("mt", category, "data", "nominal", "Nominal", "score"), _hist([20, 10, 5, 2]))
        hset.add(HistKey("mt", category, "ZTT", "nominal", "Nominal", "score"), _hist([10, 5, 2, 0.4]))
        hset.add(HistKey("mt", category, "ZTT", "nominal", "CMS_puUp", "score"), _hist([11, 5.5, 2.2, 0.44]))
        hset.add(HistKey("mt", category, "ZTT", "nominal", "CMS_puDown", "score"), _hist([9, 4.5, 1.8, 0.36]))
        hset.add(HistKey("mt", category, "ZL", "nominal", "Nominal", "score"), _hist([5, 3, 1, 0.3]))
        hset.add(HistKey("mt", category, "jetFakes", "nominal", "Nominal", "score"), _hist([4, 2, 1, 0.2]))
        hset.add(HistKey("mt", category, "HH", "nominal", "Nominal", "score"), _hist([0.01, 0.02, 0.05, 0.1]))
        hset.add(HistKey("mt", category, "HH", "nominal", "CMS_puUp", "score"), _hist([0.011, 0.022, 0.055, 0.11]))
        hset.add(HistKey("mt", category, "HH", "nominal", "CMS_puDown", "score"), _hist([0.009, 0.018, 0.045, 0.09]))
    return hset


def test_bin_name_and_shape_processes():
    analysis = build()
    assert bin_name(analysis, "mt", 0) == "htt_mt_1_2018"
    assert shape_processes(analysis) == ("ZTT", "ZL", "HH")


def test_rebin_edges_merges_from_the_right():
    total = _hist([10, 5, 2, 0.9])  # last bin below threshold 1.0 -> merged with its left neighbour
    assert rebin_edges(total, 1.0) == [0.0, 0.25, 0.5, 1.0]
    assert rebin_edges(total, 100.0) == [0.0, 1.0]
    assert rebin_edges(total, None) == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_write_datacard(tmp_path):
    analysis = build()
    path = write_datacard(_hset(), analysis, ["mt"], tmp_path / "mt", systematics=True, min_background=1.0)
    text = path.read_text()
    assert "imax * number of bins" in text
    assert "shapes * htt_mt_1_2018 common/htt_input_2018.root htt_mt_1_2018/$PROCESS htt_mt_1_2018/$PROCESS_$SYSTEMATIC" in text
    observation = [line for line in text.splitlines() if line.startswith("observation")][0]
    assert observation.split()[1:] == ["37.0", "37.0"]
    process_line = [line for line in text.splitlines() if line.startswith("process")][0]
    assert process_line.split()[1:5] == ["HH", "ZTT", "ZL", "jetFakes"]
    index_line = [line for line in text.splitlines() if line.startswith("process")][1]
    assert index_line.split()[1:5] == ["0", "1", "2", "3"]
    assert "lumi lnN 1.025 1.025 1.025 1.025" in text
    assert "zjXsec lnN - 1.02 1.02 -" in text
    assert "ffNorm_mt lnN - - - 1.1" in text
    assert "CMS_pu shape 1 1 - -" in text  # HH and ZTT carry the variation, ZL and jetFakes do not
    assert text.strip().endswith("* autoMCStats 0")
    with uproot.open(tmp_path / "mt" / "common" / "htt_input_2018.root") as f:
        assert f["htt_mt_1_2018/data_obs"].values().tolist() == [20.0, 10.0, 7.0]  # last two bins merged (bkg 0.9 < 1)
        assert f["htt_mt_1_2018/data_obs"].axis().edges().tolist() == [0.0, 0.25, 0.5, 1.0]
        assert f["htt_mt_1_2018/ZTT_CMS_puUp"].values().tolist() == [11.0, 5.5, 2.64]
        assert "htt_mt_2_2018/HH" in f


def test_write_datacard_without_systematics(tmp_path):
    text = write_datacard(_hset(), build(), ["mt"], tmp_path / "mt", systematics=False, min_background=None).read_text()
    assert "lnN" not in text and "shape" not in text.replace("shapes *", "")


def test_run_datacards_final_states(tmp_path):
    paths = run_datacards(_hset(), build(), {"mt": ["mt"], "all": ["mt"]}, tmp_path, systematics=True, min_background=1.0)
    assert set(paths) == {"mt", "all"} and all(p.exists() for p in paths.values())

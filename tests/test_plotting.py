import pytest

from shapesmith.histogram import Histogram
from shapesmith.histograms import INCLUSIVE, HistKey, HistogramSet
from shapesmith.plotting.stack import _group_histogram, default_signal_scale, run_plot
from shapesmith.plotting.style import axis_label, grouped_backgrounds
from tests.mini_analysis import build


def _hist(values, edges=(0.0, 50.0, 100.0, 200.0)):
    return Histogram(list(edges), values, [v * 0.5 for v in values])


def test_grouped_backgrounds_and_axis_label():
    analysis = build()
    assert grouped_backgrounds(analysis) == [("jetFakes", ["jetFakes"]), ("Z", ["ZTT", "ZL"])]
    assert axis_label(analysis, "mt", "m_vis") == r"$m_{vis}$ / GeV"
    assert axis_label(analysis, "mt", "unknown_var") == "unknown_var"


def test_default_signal_scale():
    hset = HistogramSet()
    hset.add(HistKey("mt", INCLUSIVE, "ZTT", "nominal", "Nominal", "m_vis"), _hist([100, 50, 20]))
    hset.add(HistKey("mt", INCLUSIVE, "HH", "nominal", "Nominal", "m_vis"), _hist([0.1, 0.2, 0.05]))
    assert default_signal_scale(hset, build(), "mt", INCLUSIVE, "m_vis") in (100.0, 200.0)


def test_group_histogram_reads_only_the_selected_region():
    hset = HistogramSet()
    hset.add(HistKey("mt", INCLUSIVE, "ZTT", "nominal", "Nominal", "m_vis"), _hist([100, 100, 100]))
    hset.add(HistKey("mt", INCLUSIVE, "ZTT", "same_sign", "Nominal", "m_vis"), _hist([3, 2, 1]))
    hset.add(HistKey("mt", INCLUSIVE, "jetFakes", "nominal", "Nominal", "m_vis"), _hist([50, 50, 50]))

    selected = _group_histogram(hset, "mt", INCLUSIVE, ["ZTT", "jetFakes"], "m_vis", region="same_sign")

    assert selected.values.tolist() == [3, 2, 1]


def test_run_plot_writes_files(tmp_path):
    analysis = build()
    hset = HistogramSet()
    for process, values in (("data", [40, 25, 10]), ("ZTT", [20, 12, 5]), ("ZL", [10, 8, 2]), ("jetFakes", [8, 4, 2]), ("HH", [0.05, 0.1, 0.02])):
        hset.add(HistKey("mt", INCLUSIVE, process, "nominal", "Nominal", "m_vis"), _hist(values))
    files = run_plot(hset, analysis, ["mt"], control=True, category=None, variables=["m_vis"], output_dir=tmp_path, blind=False, log=False, signal_scale=None, normalize_by_bin_width=False)
    assert sorted(p.name for p in files) == ["inclusive_m_vis.pdf", "inclusive_m_vis.png"]
    assert all(p.stat().st_size > 1000 for p in files)
    blind = run_plot(hset, analysis, ["mt"], control=True, category=None, variables=["m_vis"], output_dir=tmp_path / "blind", blind=True, log=True, signal_scale=10.0, normalize_by_bin_width=True)
    assert len(blind) == 2


def test_run_plot_routes_non_nominal_region_and_needs_its_histograms(tmp_path):
    analysis = build()
    hset = HistogramSet()
    for process, values in (("data", [8, 5, 2]), ("ZTT", [4, 3, 1]), ("HH", [0.5, 0.2, 0.1])):
        hset.add(HistKey("mt", INCLUSIVE, process, "same_sign", "Nominal", "m_vis"), _hist(values))
    hset.add(HistKey("mt", INCLUSIVE, "jetFakes", "nominal", "Nominal", "m_vis"), _hist([100, 100, 100]))

    files = run_plot(
        hset,
        analysis,
        ["mt"],
        control=True,
        category=None,
        variables=["m_vis"],
        output_dir=tmp_path,
        region="same_sign",
    )

    assert {path.relative_to(tmp_path).as_posix() for path in files} == {
        "mt/same_sign/inclusive_m_vis.pdf",
        "mt/same_sign/inclusive_m_vis.png",
    }
    assert all(path.stat().st_size > 1000 for path in files)


def test_non_nominal_plot_rejects_missing_data_unless_blinded(tmp_path):
    analysis = build()
    hset = HistogramSet()
    hset.add(HistKey("mt", INCLUSIVE, "ZTT", "same_sign", "Nominal", "m_vis"), _hist([4, 3, 1]))

    with pytest.raises(ValueError, match=r"data.*same_sign.*hist --regions same_sign"):
        run_plot(hset, analysis, ["mt"], True, None, ["m_vis"], tmp_path, region="same_sign")
    assert run_plot(hset, analysis, ["mt"], True, None, ["m_vis"], tmp_path, region="same_sign", blind=True)


def test_non_nominal_plot_rejects_region_without_histograms(tmp_path):
    with pytest.raises(ValueError, match=r"no histograms.*same_sign.*hist --regions same_sign"):
        run_plot(HistogramSet(), build(), ["mt"], True, None, ["m_vis"], tmp_path, region="same_sign")

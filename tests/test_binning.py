import dataclasses
import json

import numpy as np
import pytest

from shapesmith.binning import equal_data_edges
from shapesmith.fill import Target, bookings, fill_booking, run_hist
from shapesmith.histogram import INCLUSIVE, HistKey
from shapesmith.model import Category, EqualData, Variable
from shapesmith.store import read_skims

RULE = EqualData(4, 20.0, 180.0)


def smhtt_ul_edges(values, percentiles, low=30.0, high=160.0):
    """smhtt_ul gof/build_binning.py (tauID_SFs_dev): the m_vis cut of get_data_selection, then get_1d_binning."""
    arr = values[(low < values) & (values < high)]
    edges = np.percentile(arr, percentiles)
    edges = sorted(set(float(x) for x in edges))
    edges = [e - 1e-4 for e in edges]
    edges[-1] += 2e-4
    return edges


@pytest.mark.parametrize("percentiles", [
    [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],  # the three lists of build_binning.py
    [0.0, 12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5, 100.0],
    [0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
])
def test_edges_are_those_of_smhtt_ul(percentiles):
    values = np.random.default_rng(7).gamma(9.0, 9.0, 20000)
    assert list(equal_data_edges(values, EqualData(len(percentiles) - 1, 30.0, 160.0))) == smhtt_ul_edges(values, percentiles)


def test_edges_keep_the_range_open_and_drop_duplicates():
    assert equal_data_edges(np.array([30.0, 40.0, 50.0, 160.0, np.nan]), EqualData(1, 30.0, 160.0)) == pytest.approx((40.0 - 1e-4, 50.0 + 1e-4), abs=1e-12)
    assert equal_data_edges(np.repeat([1.0, 2.0, 3.0], 100), EqualData(10, 0.0, 5.0)) == pytest.approx((1.0 - 1e-4, 2.0 - 1e-4, 3.0 + 1e-4), abs=1e-12)
    with pytest.raises(ValueError, match=r"no data in \(30.0, 160.0\)"):
        equal_data_edges(np.array([10.0, 200.0]), EqualData(10, 30.0, 160.0))
    with pytest.raises(ValueError, match="fewer than two distinct edges from 3 values"):
        equal_data_edges(np.array([50.0, 50.0, 50.0]), EqualData(10, 30.0, 160.0))


def _with_categories(analysis, categories, variables=None):
    channel = analysis.channel("mt")
    channel = dataclasses.replace(channel, categories=categories, variables=channel.variables if variables is None else variables)
    return dataclasses.replace(analysis, channels={"mt": channel})


def _equal_data(analysis, rule=RULE):
    m_vis = Variable("m_vis", "m_vis", rule)
    return _with_categories(analysis, (Category("sig", "cls == 0", m_vis), Category("bkg", "cls == 1", m_vis)))


def test_run_hist_fills_every_region_and_variation_into_the_edges_of_the_data(mini_run):
    config, analysis = mini_run
    hset = run_hist(config, _equal_data(analysis), ["mt"], control=False, variables=None, systematics=True, processes=None, output=config.output_dir / "shapes.root")

    data = read_skims(config.skim_dir, "mt", ["DATA_A"], None)
    nominal = (data["veto"] < 0.5) & ((data["q_1"] * data["q_2"]) < 0) & (data["id_tight"] > 0.5)
    record = json.loads((config.output_dir / "binning.json").read_text())
    for cls, category in enumerate(("sig", "bkg")):
        values = data.loc[nominal & (data["cls"] == cls), "m_vis"].to_numpy()
        expected = equal_data_edges(values, RULE)
        in_range = int(((values > RULE.low) & (values < RULE.high)).sum())
        keys = hset.select(category=category)
        assert {(k.region, k.variation) for k in keys} >= {("nominal", "Nominal"), ("anti_iso", "Nominal"), ("nominal", "CMS_puUp")}
        assert all(hset[k].edges.tolist() == list(expected) for k in keys)
        assert hset[HistKey("mt", category, "data", "nominal", "Nominal", "m_vis")].sum() == pytest.approx(in_range)  # every data value is inside
        assert record["mt"][category]["m_vis"] == {"rule": {"n_bins": 4, "low": 20.0, "high": 180.0}, "n_data": in_range, "edges": list(expected)}


def test_the_record_keeps_the_entries_of_other_runs(mini_run):
    config, analysis = mini_run
    analysis = _equal_data(analysis)
    run_hist(config, analysis, ["mt"], False, None, False, ["data"], config.output_dir / "shapes.root")
    control = dataclasses.replace(analysis.channel("mt"), variables={"m_vis": Variable("m_vis", "m_vis", EqualData(3, 0.0, 200.0))})
    run_hist(config, dataclasses.replace(analysis, channels={"mt": control}), ["mt"], True, None, False, ["data"], config.output_dir / "control_shapes.root")

    record = json.loads((config.output_dir / "binning.json").read_text())
    assert sorted(record["mt"]) == sorted(["sig", "bkg", INCLUSIVE])
    assert record["mt"][INCLUSIVE]["m_vis"]["rule"] == {"n_bins": 3, "low": 0.0, "high": 200.0}


def test_fixed_edges_write_no_record(mini_run):
    config, analysis = mini_run
    run_hist(config, analysis, ["mt"], False, None, False, ["data"], config.output_dir / "plain" / "shapes.root")
    assert not (config.output_dir / "plain" / "binning.json").exists()


def test_a_category_without_data_names_itself(mini_run):
    config, analysis = mini_run
    empty = _with_categories(analysis, (Category("none", "cls == 7", Variable("m_vis", "m_vis", RULE)),))
    with pytest.raises(ValueError, match=r"mt/none/m_vis: equal-data binning: no data in \(20.0, 180.0\)"):
        run_hist(config, empty, ["mt"], False, None, False, None, config.output_dir / "shapes.root")


def test_fill_booking_refuses_unresolved_edges(mini_run):
    config, analysis = mini_run
    analysis = _equal_data(analysis)
    channel = analysis.channel("mt")
    booking = bookings(channel, False)[0]
    with pytest.raises(ValueError, match="mt: the equal-data binning of m_vis, m_vis is not resolved"):
        fill_booking(config, analysis, "mt", booking, [Target(c.name, c.cut, c.variable) for c in channel.categories])

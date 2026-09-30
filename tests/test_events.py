import dataclasses

import numpy as np
import pytest

from shapesmith.events import Query, load, select
from shapesmith.model import ColumnVariation, Process, Region, Selection, WeightVariation
from shapesmith.store import read_skims
from tests.mini_analysis import build

CHANNEL = build().channel("mt")
ZTT = CHANNEL.process("ZTT")
SKIM = ["veto < 0.5", "id_loose > 0.5"]


def test_nominal_selection_has_channel_process_and_skim_cuts_and_the_process_weights():
    cuts, weights = select(CHANNEL, ZTT, CHANNEL.region("nominal"))
    assert cuts == ["veto < 0.5", "(q_1 * q_2) < 0", "id_tight > 0.5", "gen_match == 5", *SKIM]
    assert weights == ["trg_wgt", "puweight"]


def test_region_replaces_cuts_and_weights_in_place_and_appends_its_weights():
    region = Region("r", replace_cuts={"os": "(q_1 * q_2) > 0"}, add_weights={"ff": "fake_factor"}, replace_weights={"trg": "1.0"})
    cuts, weights = select(CHANNEL, ZTT, region)
    assert cuts[1] == "(q_1 * q_2) > 0" and weights == ["1.0", "puweight", "fake_factor"]
    data_cuts, data_weights = select(CHANNEL, CHANNEL.process("data"), region)
    assert data_weights == ["fake_factor"]  # data carries no trg weight: nothing to replace


def test_weight_variation_applies_only_to_processes_with_its_weights():
    up = WeightVariation("CMS_puUp", {"pu": "puweight_up"})
    assert select(CHANNEL, ZTT, CHANNEL.region("nominal"), up)[1] == ["trg_wgt", "puweight_up"]
    assert select(CHANNEL, CHANNEL.process("data"), CHANNEL.region("nominal"), up) is None


def test_column_variation_rewrites_every_cut_and_weight():
    channel = dataclasses.replace(CHANNEL, skim={"iso_loose": "id_loose > 0.5"})
    process = Process("X", "DY", "background", "Z", Selection(cuts={"gen": "gen_match == 5"}, weights={"w": "trg_wgt"}))
    variation = ColumnVariation("tesUp", "__tesUp")
    available = {"id_loose__tesUp", "gen_match__tesUp", "trg_wgt__tesUp", "q_1__tesUp"}
    cuts, weights = select(channel, process, channel.region("nominal"), variation, available)
    assert cuts == ["veto < 0.5", "(q_1__tesUp * q_2) < 0", "id_tight > 0.5", "gen_match__tesUp == 5", "id_loose__tesUp > 0.5"]
    assert weights == ["trg_wgt__tesUp"]


def test_load_returns_selected_events_with_weights(mini_run):
    config, analysis = mini_run
    events = load(config, analysis, Query("mt", "ZTT"), ["m_vis", "event"])
    frame = read_skims(config.skim_dir, "mt", ["ZTT_1"], None)
    selected = frame[(frame["veto"] < 0.5) & ((frame["q_1"] * frame["q_2"]) < 0) & (frame["id_tight"] > 0.5) & (frame["gen_match"] == 5)]
    assert events.frame["event"].tolist() == selected["event"].tolist() and "m_vis" in events.frame
    expected = selected["norm_weight"].to_numpy() * analysis.lumi_pb * selected["trg_wgt"].to_numpy().astype(np.float64) * selected["puweight"].to_numpy().astype(np.float64)
    assert np.allclose(events.weights, expected)
    with pytest.raises(ValueError, match="does not apply"):
        load(config, analysis, Query("mt", "data", variation=WeightVariation("CMS_puUp", {"pu": "x"})))

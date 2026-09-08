import dataclasses

import pytest

from shapesmith import model
from tests.mini_analysis import build


def test_mini_analysis_validates():
    analysis = build()
    analysis.validate()
    assert analysis.process("ZTT").kind == "true_tau"
    assert [p.name for p in analysis.processes_of_kind("data")] == ["data"]
    assert analysis.backgrounds() == ("ZTT", "ZL", "jetFakes")
    assert [s.nick for s in analysis.samples_for("DY", "mt")] == ["ZTT_1"]
    assert [s.nick for s in analysis.samples_for("data", "mt")] == ["DATA_A"]
    assert analysis.samples_for("data", "et") == ()


def test_sample_norm_weight():
    mc = model.Sample("X", "G", "mc", xsec=10.0, nevents=100, generator_weight=0.5)
    assert mc.norm_weight == pytest.approx(10.0 / (100 * 0.5))
    assert model.Sample("D", "data", "data").norm_weight == 1.0


def test_validate_reports_all_problems():
    analysis = build()
    broken = dataclasses.replace(
        analysis,
        processes=analysis.processes + (model.Process("x", "NOGROUP", "ZTT", "other", "Z"),),
        estimator=dataclasses.replace(analysis.estimator, regions={"anti_iso": "does_not_exist"}),
    )
    with pytest.raises(model.AnalysisError) as excinfo:
        broken.validate()
    message = str(excinfo.value)
    assert "duplicate process name" in message and "ZTT" in message
    assert "NOGROUP" in message
    assert "does_not_exist" in message


def test_validate_checks_region_and_variation_targets():
    analysis = build()
    channel = analysis.channels["mt"]
    bad_region = dataclasses.replace(channel, regions=channel.regions + (model.Region("bad", {"no_such_cut": "1"}),))
    with pytest.raises(model.AnalysisError, match="no_such_cut"):
        dataclasses.replace(analysis, channels={"mt": bad_region}).validate()
    with pytest.raises(model.AnalysisError, match="no_such_weight"):
        dataclasses.replace(analysis, weight_variations=(model.WeightVariation("v", {"no_such_weight": "1"}),)).validate()


def test_dataclasses_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        build().signal = "other"


def test_process_selection_per_channel():
    shared = model.Process("x", "G", "X", "other", "rare", model.Selection(cuts={"a": "a > 1"}))
    assert shared.selection_for("mt") is shared.selection_for("tt")
    per_channel = model.Process("y", "G", "Y", "other", "rare", {"mt": model.Selection(cuts={"a": "a > 1"}), "tt": model.Selection(cuts={"a": "a > 2"})})
    assert per_channel.selection_for("tt").cuts["a"] == "a > 2"
    with pytest.raises(KeyError):
        per_channel.selection_for("et")

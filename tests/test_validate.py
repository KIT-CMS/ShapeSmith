import dataclasses

import pytest

from shapesmith import model
from shapesmith.validate import validate
from tests.mini_analysis import build, build_embedding


def _with_channel(analysis, **changes):
    channel = dataclasses.replace(analysis.channel("mt"), **changes)
    return dataclasses.replace(analysis, channels={"mt": channel})


def test_mini_analysis_validates():
    for analysis in (build(), build_embedding()):
        validate(analysis)
    channel = build().channel("mt")
    assert channel.backgrounds() == ("ZTT", "ZL", "jetFakes")
    assert channel.data() == "data" and channel.kind_of(channel.process("ZTT")) == "mc"
    assert [s.nick for s in channel.samples_of("DY")] == ["ZTT_1"]


def test_auxiliary_processes_are_no_backgrounds():
    assert build_embedding().channel("mt").backgrounds() == ("ZL", "EMB", "jetFakes")


def test_sample_norm_weight():
    mc = model.Sample("X", "G", "mc", xsec=10.0, nevents=100, generator_weight=0.5)
    assert mc.norm_weight == pytest.approx(10.0 / (100 * 0.5))
    assert model.Sample("D", "data", "data").norm_weight == 1.0


def test_validate_reports_all_problems():
    analysis = build()
    channel = analysis.channel("mt")
    broken = _with_channel(
        analysis,
        processes=channel.processes + (model.Process("ZTT", "NOGROUP", "background", "Z"),),
        estimators=(model.DataMinus("jetFakes", "does_not_exist", ("ZTT", "nope")),),
    )
    with pytest.raises(model.AnalysisError) as excinfo:
        validate(broken)
    message = str(excinfo.value)
    assert "channel mt: duplicate process ZTT" in message
    assert "no samples for group NOGROUP" in message
    assert "region does_not_exist does not exist" in message and "subtracts unknown process nope" in message


def test_validate_checks_region_targets():
    channel = build().channel("mt")
    for region, problem in (
        (model.Region("bad", {"no_such_cut": "1"}), "replaces unknown cut no_such_cut"),
        (model.Region("bad", replace_weights={"no_such_weight": "1"}), "replaces unknown weight no_such_weight"),
        (model.Region("bad", add_weights={"pu": "1"}), "adds weight pu, which a process carries"),
    ):
        with pytest.raises(model.AnalysisError, match=problem):
            validate(_with_channel(build(), regions=channel.regions + (region,)))


def test_validate_checks_variations():
    channel = build().channel("mt")
    for variations, problem in (
        ((model.WeightVariation("vUp", {"no_such_weight": "1"}), model.WeightVariation("vDown", {"pu": "1"})), "replaces unknown weight no_such_weight"),
        ((model.ColumnVariation("tesUp", "__tesUp"),), "its partner tesDown is missing"),
        ((model.ColumnVariation("tesDown", "__tesDown", regions=("nowhere",)), model.ColumnVariation("tesUp", "__tesUp")), "unknown region nowhere"),
        ((model.ColumnVariation("xUp", "__xUp", {"m_vis": "m_vis"}), model.ColumnVariation("xDown", "")), "needs exactly one of suffix and derived"),
        ((model.ColumnVariation("CMS_puUp", "__puUp"),), "duplicate variation CMS_puUp"),
        ((model.ColumnVariation("aUp", "__aUp", applies_to=("simulation",)), model.ColumnVariation("aDown", "__aDown")), "unknown sample kind simulation"),
    ):
        with pytest.raises(model.AnalysisError, match=problem):
            validate(_with_channel(build(), variations=channel.variations + variations))


def test_a_variation_without_direction_is_a_template():
    channel = build().channel("mt")
    grid = model.ColumnVariation("emb1p002", derived={"m_vis": "m_vis * 1.002"}, applies_to=("mc",), regions=("nominal",))
    validate(_with_channel(build(), variations=channel.variations + (grid,)))


def test_validate_checks_estimators_and_auxiliary_processes():
    analysis = build_embedding()
    channel = analysis.channel("mt")
    with pytest.raises(model.AnalysisError, match="subtracts the auxiliary process ZTT"):
        validate(_with_channel(analysis, estimators=(model.DataMinus("jetFakes", "anti_iso", ("ZTT",)),)))
    with pytest.raises(model.AnalysisError, match="template shift t: unknown process TTT"):
        validate(_with_channel(analysis, estimators=(model.TemplateShift("t", "EMB", "TTT", 0.1),)))
    with pytest.raises(model.AnalysisError, match="ML export: ZTT is an auxiliary process"):
        validate(dataclasses.replace(analysis, ml=dataclasses.replace(analysis.ml, processes=("ZTT",))))
    mixed = channel.samples + (model.Sample("DATA_B", "EMB", "data"),)
    with pytest.raises(model.AnalysisError, match="process EMB: samples of different kinds"):
        validate(_with_channel(analysis, samples=mixed))


def test_dataclasses_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        build().signal = "other"

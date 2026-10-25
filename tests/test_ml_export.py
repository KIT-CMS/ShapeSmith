import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from shapesmith.config import FriendConfig, NtupleConfig, RunConfig
from shapesmith.estimates import run_estimates
from shapesmith.events import Query, load
from shapesmith.fill import run_hist
from shapesmith.histogram import INCLUSIVE, HistKey
from shapesmith.ml_export import (
    BOOKKEEPING, CLASSES, EVENTS, Rows, class_scheme, export_blocks, export_process, merge_eras, repeated_keys, run_ml_export, summary,
)
from shapesmith.model import AnalysisError, DataMinus
from shapesmith.skim import exported_by, needed_columns, run_skim
from shapesmith.testing import make_mini_dataset
from shapesmith.validate import validate
from tests.mini_analysis import build

MAIN = ("ZTT", "ZL", "HH", "jetFakes")


def _read(config, channel="mt") -> pd.DataFrame:
    return pd.read_feather(config.ml_dir / channel / EVENTS)


@pytest.fixture
def mini(mini_run):
    """The mini run, exporting inline (a pool of spawned workers only in test_workers_keep_the_row_order)."""
    config, analysis = mini_run
    return config.model_copy(update={"workers": 1}), analysis


def _with_ml(analysis, **changes):
    return dataclasses.replace(analysis, ml=dataclasses.replace(analysis.ml, **changes))


def _run(tmp_path, analysis, workers=1) -> RunConfig:
    """A mini run of `analysis`, skimmed (so that the skims hold the columns of its ML export)."""
    make_mini_dataset(tmp_path / "data", n_files=2, n_events=400)
    config = RunConfig(
        analysis="tests.mini_analysis:build", era="2018", channels=["mt"],
        ntuples=NtupleConfig(base=str(tmp_path / "data" / "CROWNRun"), friends=[FriendConfig(base=str(tmp_path / "data" / "CROWNFriends" / "nn"))]),
        skim_dir=tmp_path / "skims", output_dir=tmp_path / "out", ml_dir=tmp_path / "ml", workers=workers,
    )
    run_skim(config, analysis)
    return config


def test_export_process_layout(mini):
    config, analysis = mini
    frame = export_process(config, analysis, "mt", "ZTT")
    assert list(frame.columns) == ["event", *BOOKKEEPING, "m_vis", "score"]
    assert frame["event"].dtype == np.uint64 and frame["class_index"].dtype == np.int8 and frame["weight"].dtype == np.float64
    assert frame["m_vis"].dtype == frame["score"].dtype == np.float32
    assert set(frame["process"]) == {"ZTT"} and set(frame["era"]) == {"2018"} and set(frame["channel"]) == {"mt"} and set(frame["sample_nick"]) == {"ZTT_1"}
    assert (frame["class_index"] == 1).all() and not frame["is_subtraction"].any()  # Z, the second class
    assert frame["thesis_training"].tolist() == np.resize([True, True, False, False], len(frame)).tolist()
    fakes = export_process(config, analysis, "mt", "jetFakes")  # data in the anti-isolated region, weighted with fake factors
    assert set(fakes["process"]) == {"jetFakes"} and set(fakes["sample_nick"]) == {"DATA_A"} and (fakes["class_index"] == 2).all()
    assert ((fakes["weight"] > 0) & (fakes["weight"] < 1.0)).all()


def test_run_ml_export_writes_one_file_per_channel_and_the_classes(mini):
    config, analysis = mini
    files = run_ml_export(config, analysis, ["mt"])
    assert files == [config.ml_dir / "mt" / EVENTS, config.ml_dir / CLASSES]
    frame = _read(config)
    assert frame["row"].tolist() == list(range(len(frame)))
    assert all(frame[c].dtype == "category" for c in ("era", "channel", "sample_nick", "process"))
    main = frame[~frame["is_subtraction"]]
    assert list(dict.fromkeys(main["process"])) == list(MAIN)  # in the order of the configuration
    assert main.groupby("process", observed=True)["class_index"].unique().map(list).to_dict() == {"ZTT": [1], "ZL": [1], "HH": [0], "jetFakes": [2]}
    assert len(main) == sum(len(export_process(config, analysis, "mt", name)) for name in MAIN)
    scheme = json.loads((config.ml_dir / CLASSES).read_text())
    assert scheme == {
        "profile": "mini", "classes": ["HH", "Z", "jetFakes"], "variables": ["m_vis", "score"], "mc_only": [], "identifiers": ["event"],
        "regions": {"jetFakes": "anti_iso"}, "eras": ["2018"],
        "channels": {"mt": {"processes": {"HH": ["HH"], "Z": ["ZTT", "ZL"], "jetFakes": ["jetFakes"]}, "subtraction": {"jetFakes": ["ZTT", "ZL"]}}},
    }


def test_the_class_order_is_the_configured_one(mini):
    """Not the alphabetical one of the label columns before (B 0.4): class_index is the position in `classes`."""
    config, analysis = mini
    analysis = _with_ml(analysis, classes=("jetFakes", "Z", "HH"))
    run_ml_export(config, analysis, ["mt"])
    frame = _read(config)
    assert set(frame.loc[frame["process"] == "HH", "class_index"]) == {2} and set(frame.loc[frame["process"] == "jetFakes", "class_index"]) == {0}
    assert json.loads((config.ml_dir / CLASSES).read_text())["classes"] == ["jetFakes", "Z", "HH"]


def test_the_subtraction_rows_complete_the_jet_fake_estimate(mini, tmp_path):
    """The jetFakes rows (data in anti_iso) plus the subtraction rows (ZTT and ZL in anti_iso, negated) add up to the
    DataMinus estimate of the histograms."""
    config, analysis = mini
    run_ml_export(config, analysis, ["mt"])
    frame = _read(config)
    subtraction = frame[frame["is_subtraction"]]
    assert list(dict.fromkeys(subtraction["process"])) == ["ZTT", "ZL"] and (subtraction["class_index"] == 2).all()
    for name in ("ZTT", "ZL"):  # the negated weights of the process in anti_iso
        rows = subtraction[subtraction["process"] == name]
        assert len(rows) and rows["weight"].tolist() == (-load(config, analysis, Query("mt", name, "anti_iso")).weights).tolist()
    hset = run_hist(config, analysis, ["mt"], True, ["m_vis"], False, None, tmp_path / "control.root")
    run_estimates(hset, analysis, ["mt"])
    estimate = hset[HistKey("mt", INCLUSIVE, "jetFakes", "nominal", "Nominal", "m_vis")].values.sum()  # m_vis is inside its edges
    assert frame.loc[frame["class_index"] == 2, "weight"].sum() == pytest.approx(estimate, rel=1e-6)
    assert subtraction["weight"].sum() < 0
    without = _with_ml(analysis, subtraction=False)
    assert export_blocks(without, "mt") == export_blocks(analysis, "mt")[: len(MAIN)] and not class_scheme(without, ["mt"])["channels"]["mt"]["subtraction"]
    assert export_blocks(analysis, "mt")[len(MAIN):] == [Rows("mt", p, p, "anti_iso", 2, 1.0, True) for p in ("ZTT", "ZL")]


def test_mc_only_variables_are_nan_in_data_and_embedding_and_never_skimmed_there(tmp_path):
    """gen_match as a simulation-only variable: the data skims do not need it (the data n-tuples have none)."""
    analysis = _with_ml(build(), variables=("m_vis", "gen_match"), mc_only=("gen_match",))
    validate(analysis)
    channel = analysis.channel("mt")
    data, ztt = (next(s for s in channel.samples if s.nick == nick) for nick in ("DATA_A", "ZTT_1"))
    assert "gen_match" in needed_columns(channel, ztt, analysis.ml) and "gen_match" not in needed_columns(channel, data, analysis.ml)
    config = _run(tmp_path, analysis)
    run_ml_export(config, analysis, ["mt"])
    frame = _read(config)
    data_rows = frame["sample_nick"] == "DATA_A"
    assert data_rows.any() and frame.loc[data_rows, "gen_match"].isna().all()
    assert set(frame.loc[~data_rows, "gen_match"]) <= {5.0, 6.0}


def test_the_skims_hold_the_export_columns_except_in_a_control_channel():
    analysis = _with_ml(build(), variables=("m_vis", "score", "q_1"), identifiers=("event", "id_tight"))
    channel = analysis.channel("mt")
    sample = channel.samples[0]
    assert {"q_1", "id_tight"} <= needed_columns(channel, sample, exported_by(analysis, channel))
    assert exported_by(dataclasses.replace(analysis, control_channels=("mt",)), channel) is None


def test_workers_keep_the_row_order(tmp_path):
    analysis = build()
    config = _run(tmp_path, analysis, workers=2)
    run_ml_export(config, analysis, ["mt"])
    parallel = _read(config)
    run_ml_export(config.model_copy(update={"workers": 1}), analysis, ["mt"])
    pd.testing.assert_frame_equal(parallel, _read(config))


def test_another_class_scheme_in_the_same_ml_dir_is_refused(mini):
    config, analysis = mini
    run_ml_export(config, analysis, ["mt"])
    with pytest.raises(ValueError, match="another class scheme .*classes differ"):
        run_ml_export(config, _with_ml(analysis, classes=("Z", "HH", "jetFakes")), ["mt"])


def test_merge_eras(mini, tmp_path):
    config, analysis = mini
    directories = []
    for era in ("2017", "2018"):
        era_config = config.model_copy(update={"ml_dir": tmp_path / "ml" / era})
        run_ml_export(era_config, dataclasses.replace(analysis, era=era), ["mt"])
        directories.append(era_config.ml_dir)
    written = merge_eras(directories, tmp_path / "run2")
    assert written == [tmp_path / "run2" / "mt" / EVENTS, tmp_path / "run2" / CLASSES]
    merged = pd.read_feather(written[0])
    parts = [pd.read_feather(d / "mt" / EVENTS) for d in directories]
    assert len(merged) == sum(len(p) for p in parts) and merged["era"].tolist() == ["2017"] * len(parts[0]) + ["2018"] * len(parts[1])
    assert merged["era"].dtype == "category" and not merged.duplicated(["era", "row"]).any()
    assert json.loads(written[1].read_text())["eras"] == ["2017", "2018"]
    (directories[0] / "versions").mkdir()
    (directories[0] / "versions" / "mini.json").write_text("{}")
    merge_eras(directories, tmp_path / "run2")
    assert (tmp_path / "run2" / "versions" / "2017" / "mini.json").exists() and not (tmp_path / "run2" / "versions" / "2018").exists()
    with pytest.raises(ValueError, match="an era is in more than one export"):
        merge_eras([directories[0], directories[0]], tmp_path / "again")
    with pytest.raises(ValueError, match="channels missing"):
        merge_eras(directories, tmp_path / "again", ["et"])


def test_summary_counts_the_classes_and_repeated_keys():
    frame = pd.DataFrame({"era": "2018", "channel": "mt", "sample_nick": "ZTT_1", "event": [1, 1, 2], "class_index": [1, 1, 2],
                          "is_subtraction": [False, False, True], "weight": [1.0, 2.0, -0.5]})
    assert summary(frame, ("HH", "Z", "jetFakes")) == ["HH: 0 rows, weight 0", "Z: 2 rows, weight 3", "jetFakes: 0 rows, weight 0", "jetFakes (subtraction): 1 rows, weight -0.5"]
    assert repeated_keys(frame, ("event",)) == 1 and repeated_keys(frame.assign(sample_nick=["A", "B", "A"]), ("event",)) == 0


@pytest.mark.parametrize("changes,match", [
    ({"class_of": {"ZTT": "Z", "ZL": "Z", "HH": "HH", "jetFakes": "QCD"}}, "jetFakes has the class QCD, which is not among the classes"),
    ({"classes": ("HH", "Z", "jetFakes", "W")}, "no exported process has the class W"),
    ({"variables": ("m_vis", "weight")}, "the variable weight is a reserved column"),
    ({"variables": ("m_vis", "event")}, "the variable event is a reserved column"),
    ({"mc_only": ("gen_match",)}, "the mc_only variable gen_match is not among the variables"),
    ({"variables": ("m_vis", "m_vis")}, "duplicate variable m_vis"),
])
def test_validate_rejects_an_inconsistent_export(changes, match):
    with pytest.raises(AnalysisError, match=match):
        validate(_with_ml(build(), **changes))


def test_a_process_the_channel_lacks_is_not_exported(mini):
    """EMB of another channel: before, its name fell back to the data of the nominal region (an unblinding)."""
    config, analysis = mini
    ml = dataclasses.replace(analysis.ml, class_of={**analysis.ml.class_of, "EMB": "Z"})
    analysis = dataclasses.replace(analysis, ml=ml)
    assert ml.exported(analysis.channel("mt")) == MAIN
    with pytest.raises(ValueError, match="channel mt: EMB is no signal or background process or estimator output"):
        export_process(config, analysis, "mt", "EMB")
    run_ml_export(config, analysis, ["mt"])
    frame = _read(config)
    assert "EMB" not in set(frame["process"])
    assert (~frame["is_subtraction"]).sum() == sum(len(export_process(config, analysis, "mt", name)) for name in MAIN)


def test_a_control_channel_is_not_exported(mini):
    config, analysis = mini
    analysis = dataclasses.replace(analysis, control_channels=("mt",))
    assert run_ml_export(config, analysis, ["mt"]) == []
    assert not (config.ml_dir / "mt").exists() and not (config.ml_dir / CLASSES).exists()

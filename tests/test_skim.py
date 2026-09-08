import json

import numpy as np
import pandas as pd
import pytest

from shapesmith.config import FriendConfig, NtupleConfig, RunConfig
from shapesmith.io.skims import SkimMissingError, read_manifest, read_skims, skim_path
from shapesmith.model import Selection
from shapesmith.skim import SKIM_COLUMNS, required_columns, run_skim
from shapesmith.testing import make_mini_dataset
from tests.mini_analysis import build


@pytest.fixture
def config(tmp_path):
    make_mini_dataset(tmp_path / "data", n_files=2, n_events=200)
    return RunConfig(
        analysis="tests.mini_analysis:build",
        era="2018",
        channels=["mt"],
        ntuples=NtupleConfig(base=str(tmp_path / "data" / "CROWNRun"), friends=[FriendConfig(base=str(tmp_path / "data" / "CROWNFriends" / "nn"))]),
        skim_dir=tmp_path / "skims",
        output_dir=tmp_path / "out",
        workers=2,
    )


def test_required_columns_cover_every_expression():
    columns = required_columns(build(), "mt")
    assert {"veto", "id_loose", "q_1", "q_2", "id_tight", "trg_wgt", "fake_factor", "gen_match", "puweight", "puweight_up", "puweight_down", "score", "cls", "m_vis", "event"} <= columns
    assert "genWeight" in columns


def test_empty_skims_keep_a_readable_schema(config):
    import dataclasses

    analysis = build(config)
    channel = analysis.channel("mt")
    strict = dataclasses.replace(channel, skim=Selection(cuts={"nothing": "m_vis > 1e9"}))
    analysis = dataclasses.replace(analysis, channels={"mt": strict})
    run_skim(config, analysis, ["mt"])
    frame = read_skims(config.skim_dir, "mt", ["ZTT_1", "DATA_A"], ["m_vis", "sample_nick", "is_mc"])
    assert len(frame) == 0 and str(frame["sample_nick"].dtype) in ("string", "object") and frame["is_mc"].dtype == bool


def test_run_skim_writes_parquet_and_manifest(config):
    analysis = build()
    results = run_skim(config, analysis)
    assert len(results) == 6 and all(r.n_out <= r.n_in == 200 for r in results)
    frame = read_skims(config.skim_dir, "mt", ["ZTT_1"], None)
    assert set(SKIM_COLUMNS) <= set(frame.columns)
    assert frame["sample_nick"].unique().tolist() == ["ZTT_1"] and frame["is_mc"].all() and not frame["is_data"].any()
    expected_norm = 10.0 / (100 * 1.0)
    assert np.allclose(np.abs(frame["norm_weight"]), expected_norm)
    assert set(np.sign(frame["norm_weight"]).unique()) <= {-1.0, 1.0}
    assert (frame["veto"] < 0.5).all() and (frame["id_loose"] > 0.5).all()  # skim selection applied
    assert ((frame["q_1"] * frame["q_2"]) > 0).any()  # both charges kept
    data = read_skims(config.skim_dir, "mt", ["DATA_A"], ["norm_weight", "is_data"])
    assert (data["norm_weight"] == 1.0).all() and data["is_data"].all()
    manifest = read_manifest(config.skim_dir / "mt" / "ZTT_1" / "manifest.json")
    assert manifest["nick"] == "ZTT_1" and len(manifest["files"]) == 2 and manifest["n_out"] == len(frame)
    assert manifest["metadata"]["sample_type"] == "mc"
    assert manifest["normalisation"]["kind"] == "mc" and set(manifest["normalisation"]) == {"kind", "xsec", "nevents", "generator_weight"}


def test_run_skim_skips_existing_files_unless_forced(config):
    first = run_skim(config, build())
    second = run_skim(config, build())
    assert all(r.n_in == 0 for r in second)  # skipped: nothing read
    third = run_skim(config, build(), force=True)
    assert sum(r.n_in for r in third) == sum(r.n_in for r in first)


def test_read_skims_missing_nick(config):
    with pytest.raises(SkimMissingError, match="NOPE"):
        read_skims(config.skim_dir, "mt", ["NOPE"], None)


def test_skim_path():
    assert skim_path(config_dir := __import__("pathlib").Path("/s"), "mt", "N", "N_0.root") == config_dir / "mt" / "N" / "N_0.parquet"

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shapesmith.config import FriendConfig, NtupleConfig, RunConfig
from shapesmith.io.skims import SkimMissingError, read_manifest, read_skims, skim_path, write_manifest
from shapesmith.model import Region, Selection
from shapesmith.skim import SKIM_COLUMNS, columns_for_sample, required_columns, run_skim
from shapesmith.testing import make_mini_dataset
from tests.mini_analysis import build


def _config(tmp_path, n_events=200, workers=2):
    make_mini_dataset(tmp_path / "data", n_files=2, n_events=n_events)
    return RunConfig(
        analysis="tests.mini_analysis:build",
        era="2018",
        channels=["mt"],
        ntuples=NtupleConfig(base=str(tmp_path / "data" / "CROWNRun"), friends=[FriendConfig(base=str(tmp_path / "data" / "CROWNFriends" / "nn"))]),
        skim_dir=tmp_path / "skims",
        output_dir=tmp_path / "out",
        workers=workers,
    )


@pytest.fixture
def config(tmp_path):
    return _config(tmp_path)


def _tighter_skim(analysis):
    import dataclasses

    channel = analysis.channel("mt")
    changed = dataclasses.replace(channel, skim=Selection(cuts={**channel.skim.cuts, "tighter": "m_vis > 40"}))
    return dataclasses.replace(analysis, channels={"mt": changed})


def test_required_columns_cover_every_expression():
    columns = required_columns(build(), "mt")
    assert {"veto", "id_loose", "q_1", "q_2", "id_tight", "trg_wgt", "fake_factor", "gen_match", "puweight", "puweight_up", "puweight_down", "score", "cls", "m_vis", "event"} <= columns
    assert "genWeight" in columns


def test_region_replacement_of_baseline_weight_stays_optional_for_data(config):
    import dataclasses

    analysis = build()
    channel = analysis.channel("mt")
    baseline = Selection(channel.baseline.cuts, {**channel.baseline.weights, "mc_only": "puweight"})
    regions = tuple(
        Region(region.name, region.replace_cuts, {**region.add_weights, "mc_only": "puweight_up"})
        if region.name == "same_sign"
        else region
        for region in channel.regions
    )
    analysis = dataclasses.replace(analysis, channels={"mt": dataclasses.replace(channel, baseline=baseline, regions=regions)})
    data = analysis.samples[0]

    assert "puweight_up" not in columns_for_sample(analysis, "mt", data)
    run_skim(config, analysis)
    assert "puweight_up" not in read_skims(config.skim_dir, "mt", [data.nick], None)


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
    assert manifest["contract"]["selection"] == {"iso_loose": "id_loose > 0.5", "veto": "veto < 0.5"}
    assert "m_vis" in manifest["contract"]["required_columns"]
    assert manifest["contract"]["normalisation"] == manifest["normalisation"]


def test_run_skim_skips_existing_files_unless_forced(config):
    first = run_skim(config, build())
    second = run_skim(config, build())
    assert not any(r.skipped for r in first) and all(r.skipped for r in second)
    assert sum(r.n_in for r in second) == sum(r.n_in for r in first)  # reused files report their recorded counts
    third = run_skim(config, build(), force=True)
    assert not any(r.skipped for r in third) and sum(r.n_in for r in third) == sum(r.n_in for r in first)


def test_run_skim_resumes_after_a_read_error_without_recomputing_finished_files(config, tmp_path):
    broken = tmp_path / "data" / "CROWNRun" / "2018" / "ZTT_1" / "mt" / "ZTT_1_1.root"
    original = broken.read_bytes()
    broken.write_bytes(b"not a ROOT file")
    with pytest.raises(RuntimeError, match="skim failures"):
        run_skim(config, build())
    finished = config.skim_dir / "mt" / "ZTT_1" / "ZTT_1_0.parquet"
    path = config.skim_dir / "mt" / "ZTT_1" / "manifest.json"
    assert set(read_manifest(path)["completed"]) == {"ZTT_1_0.root"}
    written = finished.stat().st_mtime_ns

    broken.write_bytes(original)
    results = run_skim(config, build())
    assert [r.basename for r in results if not r.skipped] == ["ZTT_1_1.root"]
    assert finished.stat().st_mtime_ns == written
    manifest = read_manifest(path)
    assert set(manifest["completed"]) == {"ZTT_1_0.root", "ZTT_1_1.root"} and manifest["n_in"] == 400


def test_run_skim_records_empty_inputs_under_the_new_contract_after_force(tmp_path):
    config = _config(tmp_path, n_events=0)
    analysis = build()
    results = run_skim(config, analysis)
    assert results and all(r.n_in == 0 and not r.skipped for r in results)
    path = config.skim_dir / "mt" / "ZTT_1" / "manifest.json"
    changed = _tighter_skim(analysis)
    run_skim(config, changed, force=True)
    assert read_manifest(path)["contract"]["selection"] == changed.channel("mt").skim.cuts
    assert all(r.skipped for r in run_skim(config, changed))


def test_run_skim_manifest_counts_cover_reused_files(config):
    run_skim(config, build())
    path = config.skim_dir / "mt" / "ZTT_1" / "manifest.json"
    (config.skim_dir / "mt" / "ZTT_1" / "ZTT_1_1.parquet").unlink()
    results = run_skim(config, build())
    assert [r.basename for r in results if not r.skipped] == ["ZTT_1_1.root"]
    manifest = read_manifest(path)
    assert manifest["n_in"] == 400 and manifest["n_out"] == sum(r.n_out for r in results if r.nick == "ZTT_1")


def test_run_skim_keeps_the_previous_parquet_when_a_write_fails(tmp_path, monkeypatch):
    import shapesmith.io.skims as skims

    config = _config(tmp_path, workers=1)  # inline jobs, so the patched writer is used
    run_skim(config, build())
    target = config.skim_dir / "mt" / "ZTT_1" / "ZTT_1_1.parquet"
    rows = len(pd.read_parquet(target))
    real = skims.pq.write_table

    def failing(table, where, **kwargs):
        if Path(where).name.startswith("ZTT_1_1"):
            Path(where).write_bytes(b"partial")
            raise OSError("disk full")
        return real(table, where, **kwargs)

    monkeypatch.setattr(skims.pq, "write_table", failing)
    with pytest.raises(RuntimeError, match="disk full"):
        run_skim(config, build(), force=True)
    assert len(pd.read_parquet(target)) == rows
    assert set(read_manifest(config.skim_dir / "mt" / "ZTT_1" / "manifest.json")["completed"]) == {"ZTT_1_0.root"}
    assert not list((config.skim_dir / "mt" / "ZTT_1").glob("*.tmp"))
    monkeypatch.undo()
    assert [r.basename for r in run_skim(config, build()) if not r.skipped] == ["ZTT_1_1.root"]


def test_write_manifest_replaces_atomically(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    write_manifest(path, {"version": 1})
    real = Path.write_text

    def partial(self, text, *args, **kwargs):
        real(self, text[:5], *args, **kwargs)
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", partial)
    with pytest.raises(OSError):
        write_manifest(path, {"version": 2})
    assert read_manifest(path) == {"version": 1}


@pytest.mark.parametrize("change", ["selection", "columns", "normalisation"])
def test_run_skim_rejects_incompatible_reuse_with_force_guidance(config, change):
    import dataclasses

    analysis = build()
    run_skim(config, analysis)
    if change == "selection":
        analysis = _tighter_skim(analysis)
    elif change == "columns":
        channel = analysis.channel("mt")
        changed = dataclasses.replace(channel, keep_columns=(*channel.keep_columns, "new_required_column"))
        analysis = dataclasses.replace(analysis, channels={"mt": changed})
    else:
        samples = tuple(dataclasses.replace(sample, xsec=sample.xsec * 2) if sample.nick == "ZTT_1" else sample for sample in analysis.samples)
        analysis = dataclasses.replace(analysis, samples=samples)

    with pytest.raises(ValueError, match=r"incompatible.*shapesmith skim --force"):
        run_skim(config, analysis)


def test_run_skim_rejects_legacy_manifest_without_selection_contract(config):
    analysis = build()
    run_skim(config, analysis)
    path = config.skim_dir / "mt" / "ZTT_1" / "manifest.json"
    manifest = read_manifest(path)
    manifest.pop("contract")
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=r"selection was not recorded.*shapesmith skim --force"):
        run_skim(config, analysis)


def test_read_skims_missing_nick(config):
    with pytest.raises(SkimMissingError, match="NOPE"):
        read_skims(config.skim_dir, "mt", ["NOPE"], None)


def test_skim_path():
    assert skim_path(config_dir := __import__("pathlib").Path("/s"), "mt", "N", "N_0.root") == config_dir / "mt" / "N" / "N_0.parquet"

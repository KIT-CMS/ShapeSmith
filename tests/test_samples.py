import json

import pytest

from shapesmith.samples import InventoryEntry, kind_of, normalisation, read_inventory, write_inventory

DB = {
    "TT_old": {"nick": "TT_old", "dbs": "/TT/campaign-v2/NANOAODSIM", "sample_type": "ttbar", "xsec": 88.5, "nevents": 1000, "generator_weight": 0.99, "era": "2018"},
    "DATA_A": {"nick": "DATA_A", "dbs": "/SingleMuon/Run2018A/NANOAOD", "sample_type": "data", "xsec": 1.0, "nevents": 1, "generator_weight": 1.0, "era": "2018"},
}
RENAMED = {
    "TT_new": {**DB["TT_old"], "nick": "TT_new"},
    "DATA_A_v2": {**DB["DATA_A"], "nick": "DATA_A_v2"},
    "DATA_A": DB["DATA_A"],  # the old name kept as a duplicate with identical values
}


def _write(tmp_path, name, db):
    path = tmp_path / name
    path.write_text(json.dumps(db))
    return path


def test_kind_of():
    assert kind_of("data") == "data" and kind_of("embedding") == "embedding" and kind_of("ttbar") == "mc"


def test_inventory_roundtrip(tmp_path):
    database = _write(tmp_path, "old.json", DB)
    entries = write_inventory(tmp_path / "inv.txt", database, ["TT_old", "DATA_A"])
    assert entries == [InventoryEntry("TT_old", "/TT/campaign-v2/NANOAODSIM"), InventoryEntry("DATA_A", "/SingleMuon/Run2018A/NANOAOD")]
    (tmp_path / "inv.txt").write_text("# comment\n" + (tmp_path / "inv.txt").read_text() + "\n")
    assert read_inventory(tmp_path / "inv.txt") == entries
    with pytest.raises(KeyError, match="NOPE"):
        write_inventory(tmp_path / "x.txt", database, ["NOPE"])
    (tmp_path / "bad.txt").write_text("only_one_column\n")
    with pytest.raises(ValueError, match="only_one_column"):
        read_inventory(tmp_path / "bad.txt")


def test_normalisation_by_nick_and_by_dbs(tmp_path):
    inventory = [InventoryEntry("TT_old", "/TT/campaign-v2/NANOAODSIM"), InventoryEntry("DATA_A", "/SingleMuon/Run2018A/NANOAOD")]
    same = normalisation(_write(tmp_path, "old.json", DB), inventory)
    assert same["TT_old"] == {"kind": "mc", "database_nick": "TT_old", "sample_type": "ttbar", "xsec": 88.5, "nevents": 1000, "generator_weight": 0.99}
    renamed = normalisation(_write(tmp_path, "new.json", RENAMED), inventory)
    assert renamed["TT_old"]["database_nick"] == "TT_new" and renamed["TT_old"]["xsec"] == 88.5
    assert renamed["DATA_A"]["database_nick"] == "DATA_A" and renamed["DATA_A"]["kind"] == "data"


def test_normalisation_rejects_ambiguous_and_missing(tmp_path):
    conflicting = {"TT_a": {**DB["TT_old"], "nick": "TT_a"}, "TT_b": {**DB["TT_old"], "nick": "TT_b", "nevents": 2000}}
    with pytest.raises(KeyError, match="TT_old"):
        normalisation(_write(tmp_path, "conflict.json", conflicting), [InventoryEntry("TT_old", "/TT/campaign-v2/NANOAODSIM")])
    with pytest.raises(KeyError, match="GONE"):
        normalisation(_write(tmp_path, "old.json", DB), [InventoryEntry("GONE", "/nowhere")])

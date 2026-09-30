import json

import pytest

from shapesmith.model import AnalysisError
from shapesmith.samples import kind_of, normalisation, read_sample_list

DB = {
    "TT_1": {"nick": "TT_1", "dbs": "/TT/campaign-v2/NANOAODSIM", "sample_type": "ttbar", "xsec": 88.5, "nevents": 1000, "generator_weight": 0.99, "era": "2018"},
    "DATA_A": {"nick": "DATA_A", "dbs": "/SingleMuon/Run2018A/NANOAOD", "sample_type": "data", "xsec": 1.0, "nevents": 1, "generator_weight": 1.0, "era": "2018"},
    "EMB_A": {"nick": "EMB_A", "dbs": "/EmbeddingRun2018A/MuTau/USER", "sample_type": "embedding", "xsec": 1.0, "nevents": 5000, "generator_weight": 1.0, "era": "2018"},
}


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "datasets.json"
    path.write_text(json.dumps(DB))
    return path


def _list(tmp_path, text):
    path = tmp_path / "sample_list.txt"
    path.write_text(text)
    return path


def test_kind_of():
    assert (kind_of("data"), kind_of("embedding"), kind_of("ttbar"), kind_of("dy")) == ("data", "embedding", "mc", "mc")


def test_sample_list_is_read_in_order_and_blank_lines_are_skipped(tmp_path):
    assert read_sample_list(_list(tmp_path, "TT_1\n\nDATA_A\n  \nEMB_A")) == ("TT_1", "DATA_A", "EMB_A")


@pytest.mark.parametrize("line", ["TT_1 /TT/campaign-v2/NANOAODSIM", "# a comment"])
def test_a_line_with_more_than_one_token_raises(tmp_path, line):
    with pytest.raises(ValueError, match=r"sample_list.txt:2: expected one nick per line.*DBS column"):
        read_sample_list(_list(tmp_path, f"DATA_A\n{line}\n"))


def test_a_duplicate_nick_raises(tmp_path):
    with pytest.raises(ValueError, match="sample_list.txt:3: duplicate nick TT_1"):
        read_sample_list(_list(tmp_path, "TT_1\nDATA_A\nTT_1\n"))


def test_normalisation_by_nick(database):
    values = normalisation(database, ("TT_1", "DATA_A", "EMB_A"))
    assert values["TT_1"] == {"kind": "mc", "xsec": 88.5, "nevents": 1000, "generator_weight": 0.99}
    assert values["DATA_A"]["kind"] == "data" and values["EMB_A"] == {"kind": "embedding", "xsec": 1.0, "nevents": 5000, "generator_weight": 1.0}


def test_normalisation_lists_every_missing_nick(database):
    with pytest.raises(AnalysisError, match=r"2 nicks are not in .*renamed.*\nGONE_1\nGONE_2"):
        normalisation(database, ("GONE_1", "TT_1", "GONE_2"))

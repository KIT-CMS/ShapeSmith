import numpy as np
import pytest

from shapesmith.config import FriendConfig, NtupleConfig
from shapesmith.io.discovery import discover, join_url, list_root_files
from shapesmith.io.ntuples import MissingColumnsError, available_columns, num_entries, read_columns, read_metadata
from shapesmith.testing import make_mini_dataset, make_ntuple


@pytest.fixture
def dataset(tmp_path):
    return make_mini_dataset(tmp_path, n_files=2, n_events=50), tmp_path


def _config(root, applies_to=("mc", "data", "embedding")):
    return NtupleConfig(base=str(root / "CROWNRun"), friends=[FriendConfig(base=str(root / "CROWNFriends" / "nn"), applies_to=applies_to)])


def test_join_url():
    assert join_url("", "/data/x.root") == "/data/x.root"
    assert join_url("root://server.example", "/store/x.root") == "root://server.example//store/x.root"


def test_list_and_discover(dataset):
    info, root = dataset
    assert list_root_files("", str(root / "CROWNRun" / "2018" / "ZTT_1" / "mt")) == ["ZTT_1_0.root", "ZTT_1_1.root"]
    files = discover(_config(root), "2018", "ZTT_1", "mt", "mc")
    assert [f.basename for f in files] == ["ZTT_1_0.root", "ZTT_1_1.root"]
    assert files[0].friends == (str(root / "CROWNFriends" / "nn" / "2018" / "ZTT_1" / "mt" / "ZTT_1_0.root"),)
    assert files[0].nick == "ZTT_1" and files[0].channel == "mt"


def test_friend_applies_to_filters_kinds(dataset):
    info, root = dataset
    files = discover(_config(root, applies_to=("mc",)), "2018", "DATA_A", "mt", "data")
    assert files[0].friends == ()


def test_missing_friend_is_an_error(dataset):
    info, root = dataset
    (root / "CROWNFriends" / "nn" / "2018" / "ZTT_1" / "mt" / "ZTT_1_1.root").unlink()
    with pytest.raises(FileNotFoundError, match="ZTT_1_1.root"):
        discover(_config(root), "2018", "ZTT_1", "mt", "mc")


def test_read_columns_joins_friends(dataset):
    info, root = dataset
    ntuple = discover(_config(root), "2018", "ZTT_1", "mt", "mc")[0]
    columns = available_columns(ntuple)
    assert columns["m_vis"].endswith("CROWNRun/2018/ZTT_1/mt/ZTT_1_0.root") and columns["score"].endswith("CROWNFriends/nn/2018/ZTT_1/mt/ZTT_1_0.root")
    frame = read_columns(ntuple, {"m_vis", "score", "event"})
    assert list(sorted(frame.columns)) == ["event", "m_vis", "score"] and len(frame) == 50
    assert frame["m_vis"].dtype == np.float32
    assert num_entries(ntuple.path) == 50
    assert read_metadata(ntuple.path)["sample_type"] == "mc"


def test_missing_columns_are_listed(dataset):
    info, root = dataset
    ntuple = discover(_config(root), "2018", "ZTT_1", "mt", "mc")[0]
    with pytest.raises(MissingColumnsError) as excinfo:
        read_columns(ntuple, {"m_vis", "nope", "also_nope"})
    assert excinfo.value.missing == ["also_nope", "nope"]


def test_friend_length_mismatch_is_an_error(tmp_path):
    main = make_ntuple(tmp_path / "main.root", {"a": np.zeros(3, dtype=np.float32)})
    friend = make_ntuple(tmp_path / "friend.root", {"b": np.zeros(2, dtype=np.float32)})
    from shapesmith.io.discovery import NtupleFile
    with pytest.raises(ValueError, match="entries"):
        read_columns(NtupleFile(str(main), (str(friend),), "X", "mt", "main.root"), {"a", "b"})


def test_missing_columns_error_is_picklable():
    import pickle

    error = MissingColumnsError(["b", "a"], "/x.root")
    back = pickle.loads(pickle.dumps(error))
    assert back.missing == ["a", "b"] and back.path == "/x.root" and str(back) == str(error)

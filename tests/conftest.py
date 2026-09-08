import pytest

from shapesmith.config import FriendConfig, NtupleConfig, RunConfig
from shapesmith.skim import run_skim
from shapesmith.testing import make_mini_dataset
from tests.mini_analysis import build


@pytest.fixture
def mini_run(tmp_path):
    """RunConfig + Analysis for the mini dataset, with skims already written."""
    make_mini_dataset(tmp_path / "data", n_files=2, n_events=400)
    config = RunConfig(
        analysis="tests.mini_analysis:build",
        era="2018",
        channels=["mt"],
        switches={"jet_fakes": "ff", "embedding": False},
        ntuples=NtupleConfig(base=str(tmp_path / "data" / "CROWNRun"), friends=[FriendConfig(base=str(tmp_path / "data" / "CROWNFriends" / "nn"))]),
        skim_dir=tmp_path / "skims",
        output_dir=tmp_path / "out",
        ml_dir=tmp_path / "ml",
        workers=2,
    )
    analysis = build(config)
    run_skim(config, analysis)
    return config, analysis

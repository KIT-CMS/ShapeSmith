from pathlib import Path

import numpy as np
import pytest
import uproot

from shapesmith import cmssw
from shapesmith.config import CombineConfig
from shapesmith.limits import collect, limit_commands, write_summary


def _limit_tree(path, quantiles, limits, r=None):
    with uproot.recreate(path) as f:
        data = {"quantileExpected": np.array(quantiles, dtype=np.float32), "limit": np.array(limits, dtype=np.float64)}
        if r is not None:
            data["r"] = np.array(r, dtype=np.float32)
        f["limit"] = data


def test_cmssw_script_sets_up_the_environment_before_the_commands():
    script = cmssw.script(CombineConfig(cmssw_dir="/cmssw/CMSSW_14_1_9"), Path("/cards/mt"), ["echo one", "echo two"])
    lines = script.splitlines()
    assert lines[:2] == ["set -e", "export SCRAM_ARCH=el9_amd64_gcc12"]
    assert "cd /cmssw/CMSSW_14_1_9/src" in lines and "eval $(scramv1 runtime -sh)" in lines
    assert lines[-3:] == ["cd /cards/mt", "echo one", "echo two"]


def test_cmssw_commands_run_in_a_clean_environment(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/lcg/python")
    monkeypatch.setenv("HOME", "/home/someone")
    environment = cmssw.clean_environment()
    assert "PYTHONPATH" not in environment and environment["HOME"] == "/home/someone" and environment["PATH"] == "/usr/bin:/bin"


def test_cmssw_run_needs_a_combine_configuration(tmp_path):
    with pytest.raises(ValueError, match="needs `combine`"):
        cmssw.run(["true"], None, tmp_path)


def test_limit_commands():
    script = "\n".join(limit_commands())
    assert "text2workspace.py combined.txt -o workspace.root -m 125" in script
    assert "-M AsymptoticLimits" in script and "--expectSignal 0" in script and "-n .Limit" in script
    assert "-M Significance" in script and "-n .Significance" in script
    assert "-M MultiDimFit --algo singles" in script and "-n .Fit" in script
    assert "--setParameterRanges r=-40,40" in script and "-t -1" in script


def test_collect_and_summary(tmp_path):
    card_dir = tmp_path / "mt"
    card_dir.mkdir()
    _limit_tree(card_dir / "higgsCombine.Limit.AsymptoticLimits.mH125.root", [0.025, 0.16, 0.5, 0.84, 0.975, -1], [5.0, 7.0, 10.0, 14.0, 20.0, 11.0])
    _limit_tree(card_dir / "higgsCombine.Significance.Significance.mH125.root", [-1], [0.42])
    _limit_tree(card_dir / "higgsCombine.Fit.MultiDimFit.mH125.root", [-1, 0.32, 0.32], [0, 0, 0], r=[1.0, -4.0, 6.0])
    result = collect(card_dir)
    assert result["limit"] == {"exp_m2": 5.0, "exp_m1": 7.0, "exp_median": 10.0, "exp_p1": 14.0, "exp_p2": 20.0, "observed": 11.0}
    assert result["significance"] == 0.42
    assert result["r"] == {"best": 1.0, "low": -4.0, "high": 6.0}
    json_path, md_path, pdf_path = write_summary({"mt": result, "all": {}}, tmp_path)
    assert json_path.exists() and pdf_path.exists()
    table = md_path.read_text()
    assert "| mt | 10.00 | [7.00, 14.00] | [5.00, 20.00] | 0.42 |" in table
    assert "| all |" in table

import ast
import json

from typer.testing import CliRunner

from shapesmith import cmssw
from shapesmith.background_fit import VALUES_SCRIPT, fit_commands, summarise
from shapesmith.cli import app
from tests.mini_analysis import build

# the values script's output for the mini analysis (bins htt_mt_1_2018 = sig, htt_mt_2_2018 = bkg)
VALUES = {
    "status": 0,
    "covariance_quality": 3,
    "prefit": {"lumi": [0.0, 1.0], "CMS_pu": [0.0, 2.0]},
    "postfit": {"lumi": [-0.5, 0.8], "CMS_pu": [1.0, 1.0], "rate_ZTT": [1.2, 0.1]},
    "norm_prefit": {"htt_mt_1_2018/ZTT": [10.0, 1.0], "htt_mt_1_2018/total_background": [12.0, 1.5], "htt_mt_2_2018/total_background": [6.0, 1.0]},
    "norm_fit_b": {"htt_mt_1_2018/ZTT": [9.0, 0.5], "htt_mt_1_2018/total_background": [11.0, 0.8], "htt_mt_2_2018/total_background": [6.5, 0.5]},
    "data": {"htt_mt_1_2018": 11.0, "htt_mt_2_2018": 7.0},
}


def test_fit_commands():
    script = "\n".join(fit_commands())
    assert "text2workspace.py combined.txt -o workspace.root -m 125" in script
    assert "combine -M FitDiagnostics -d workspace.root -m 125 --setParameters r=0 --freezeParameters r --skipSBFit" in script
    assert "--robustHesse 1" in script
    assert "--saveNormalizations --saveShapes --saveWithUncertainties -n .BackgroundOnly" in script
    assert f"python3 {VALUES_SCRIPT} workspace.root fitDiagnostics.BackgroundOnly.root fitDiagnostics.BackgroundOnly.json" in script


def test_the_values_script_runs_in_the_python_of_cmssw():
    """Python 3.9 and no imports beyond the standard library and PyROOT."""
    tree = ast.parse(VALUES_SCRIPT.read_text(), feature_version=(3, 9))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    assert imported == {"json", "sys", "ROOT"}


def test_summarise():
    result = summarise(VALUES, build())
    assert result["fit_status"] == 0 and result["covariance_quality"] == 3
    assert result["nuisances"]["lumi"] == {"prefit": [0.0, 1.0], "postfit": [-0.5, 0.8], "pull": -0.5, "constraint": 0.8}
    assert result["nuisances"]["CMS_pu"]["pull"] == 0.5 and result["nuisances"]["CMS_pu"]["constraint"] == 0.5  # width 2
    assert result["nuisances"]["rate_ZTT"] == {"prefit": None, "postfit": [1.2, 0.1], "pull": None, "constraint": None}  # free
    sig = result["yields"]["mt"]["sig"]
    assert sig == {"data": 11.0, "prefit": {"ZTT": [10.0, 1.0], "total_background": [12.0, 1.5]}, "postfit": {"ZTT": [9.0, 0.5], "total_background": [11.0, 0.8]}}
    assert result["yields"]["mt"]["bkg"]["postfit"] == {"total_background": [6.5, 0.5]}


def test_fit_background_only_cli(tmp_path, monkeypatch):
    config = tmp_path / "run.yaml"
    config.write_text(f"""
analysis: tests.mini_analysis:build_with_control
era: "2018"
channels: [mt, em]
ntuples: {{base: {tmp_path / 'ntuples'}}}
skim_dir: {tmp_path / 'skims'}
output_dir: {tmp_path / 'out'}
""")
    values = {**VALUES, "data": {**VALUES["data"], "htt_em_1_2018": 5.0}}  # plus the control channel em
    values["norm_prefit"] = {**VALUES["norm_prefit"], "htt_em_1_2018/total_background": [4.0, 1.0]}
    values["norm_fit_b"] = {**VALUES["norm_fit_b"], "htt_em_1_2018/total_background": [4.5, 0.5]}
    runs = []

    def fake_run(commands, combine, cwd):  # combine in CMSSW: writes the values next to the card
        runs.append((commands, cwd))
        (cwd / "fitDiagnostics.BackgroundOnly.json").write_text(json.dumps(values))

    monkeypatch.setattr(cmssw, "run", fake_run)
    (tmp_path / "out" / "datacards" / "mt").mkdir(parents=True)
    result = CliRunner().invoke(app, ["fit", "-c", str(config), "--background-only"])
    assert result.exit_code == 0, result.output
    assert [cwd.name for _, cwd in runs] == ["mt"] and runs[0][0] == fit_commands()  # the one final state, mt with em
    assert "| em | sig | 5 | 4.0 | 4.5 | 1.250 | 1.111 |" in result.output and "| lumi | -0.50 | 0.80 |" in result.output
    summary = json.loads((tmp_path / "out" / "datacards" / "background_fit.json").read_text())
    assert summary["mt"]["yields"]["em"]["sig"]["data"] == 5.0 and summary["mt"]["nuisances"]["CMS_pu"]["pull"] == 0.5
    rerun = CliRunner().invoke(app, ["fit", "-c", str(config), "--background-only", "--skip-combine"])
    assert rerun.exit_code == 0 and len(runs) == 1  # the stored values are summarised again without combine

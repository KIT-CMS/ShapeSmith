import ast
import json

from typer.testing import CliRunner

from shapesmith import cmssw
from shapesmith.background_fit import VALUES_SCRIPT, fit_commands, summarise, summary_lines
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
    assert imported == {"json", "math", "sys", "ROOT"}


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
    assert "| em | sig | 5 | 4.0 ± 1.0 | 4.5 ± 0.5 | 1.250 | 1.111 |" in result.output and "| lumi | -0.50 | 0.80 |" in result.output
    summary = json.loads((tmp_path / "out" / "datacards" / "background_fit.json").read_text())
    assert summary["mt"]["yields"]["em"]["sig"]["data"] == 5.0 and summary["mt"]["nuisances"]["CMS_pu"]["pull"] == 0.5
    rerun = CliRunner().invoke(app, ["fit", "-c", str(config), "--background-only", "--skip-combine"])
    assert rerun.exit_code == 0 and len(runs) == 1  # the stored values are summarised again without combine


class _Arg:
    def __init__(self, name, value, error):
        self.name, self.value, self.error = name, value, error

    def GetName(self):
        return self.name

    def getVal(self):
        return self.value

    def getError(self):
        return self.error


class _Template:
    def __init__(self, errors):
        self.errors = errors

    def __bool__(self):
        return True

    def InheritsFrom(self, name):
        return name == "TH1"

    def GetNbinsX(self):
        return len(self.errors)

    def GetBinError(self, i):
        return self.errors[i - 1]


class _Shapes:
    def __init__(self, templates):
        self.templates = templates

    def Get(self, key):
        return self.templates.get(key)


def _values_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("fit_diagnostics_values", VALUES_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_values_script_writes_an_undefined_error_as_null():
    """robustHesse drops a parameter of a negative eigenvalue from the covariance (error 0); the sampled postfit
    templates that depend on it get NaN errors and combine stores 0 as their normalisation error."""
    values = _values_module()
    parameters = values.parameters([_Arg("dropped", 0.1, 0.0), _Arg("nan", 0.2, float("nan")), _Arg("fine", -0.5, 0.8)])
    assert parameters == {"dropped": [0.1, None], "nan": [0.2, None], "fine": [-0.5, 0.8]}
    shapes = _Shapes({"htt_mt_1_2018/jetFakes": _Template([float("nan")]), "htt_mt_1_2018/EMB": _Template([228.0])})
    norms = values.normalisations([_Arg("htt_mt_1_2018/jetFakes", 6572.4, 0.0), _Arg("htt_mt_1_2018/EMB", 4132.6, 228.0), _Arg("htt_mt_1_2018/total_signal", 0.0, 0.0)], shapes)
    assert norms == {"htt_mt_1_2018/jetFakes": [6572.4, None], "htt_mt_1_2018/EMB": [4132.6, 228.0], "htt_mt_1_2018/total_signal": [0.0, 0.0]}


def test_an_undefined_postfit_error_is_reported_as_undefined_never_as_0(caplog):
    values = json.loads(json.dumps(VALUES))
    values["postfit"]["CMS_pu"] = [1.0, None]
    values["norm_fit_b"]["htt_mt_1_2018/total_background"] = [11.0, None]
    values["norm_fit_b"]["htt_mt_1_2018/ZTT"] = [9.0, float("nan")]  # NaN of an older values script
    with caplog.at_level("WARNING", logger="shapesmith.background_fit"):
        result = summarise(values, build())
    assert result["nuisances"]["CMS_pu"] == {"prefit": [0.0, 2.0], "postfit": [1.0, None], "pull": 0.5, "constraint": None}
    assert result["undefined"] == {"postfit": ["CMS_pu"], "norm_fit_b": ["htt_mt_1_2018/ZTT", "htt_mt_1_2018/total_background"]}
    assert "undefined postfit errors (reported as undefined, not 0): 1 parameters (CMS_pu), 2 normalisations" in caplog.text
    text = "\n".join(summary_lines("mt", result))
    assert "| mt | sig | 11 | 12.0 ± 1.5 | 11.0 ± undefined | 0.917 | 1.000 |" in text
    assert "| CMS_pu | +0.50 | undefined |" in text and "- parameter CMS_pu" in text and "- normalisation htt_mt_1_2018/total_background" in text
    assert json.loads(json.dumps(result))["yields"]["mt"]["sig"]["postfit"]["total_background"] == [11.0, None]

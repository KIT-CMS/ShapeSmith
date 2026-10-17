"""`shapesmith combine` (v7 spec 6.9): combineCards over the cards of several eras, the nuisance report, then the limits
or the background-only fit on the combined card, with CMSSW faked."""
import dataclasses
import json

import numpy as np
import pytest
import uproot
from typer.testing import CliRunner

from shapesmith import cmssw
from shapesmith.background_fit import fit_commands
from shapesmith.cli import app
from shapesmith.combination import card_nuisances, combine_cards_command, foreign_token, validate_command
from shapesmith.limits import limit_commands
from tests.mini_analysis import build

ERAS = ("2017", "2018")


def build_era(config=None):
    """The mini analysis in the era of the run: its bins carry the era (htt_mt_1_<era>)."""
    return dataclasses.replace(build(config), era=config.era if config is not None else "2018")


def _card(era: str, extra: str = "") -> str:
    return (
        "imax * number of bins\njmax *\nkmax *\n" + "-" * 10 + "\n"
        f"shapes * htt_mt_1_{era} common/htt_input_{era}.root htt_mt_1_{era}/$PROCESS htt_mt_1_{era}/$PROCESS_$SYSTEMATIC\n"
        f"bin htt_mt_1_{era} htt_mt_2_{era}\nobservation 10.0 5.0\n"
        f"bin htt_mt_1_{era} htt_mt_1_{era} htt_mt_2_{era}\nprocess HH ZTT ZTT\nprocess 0 1 1\nrate -1 -1 -1\n"
        f"lumi_13TeV_15161718_l lnN 1.0084 1.0084 1.0084\nCMS_pu shape - 1 1\nCMS_ff_QCDStatShift_mt_{era} shape - 1 -\n{extra}* autoMCStats 0\n"
    )


@pytest.fixture
def runs(tmp_path):
    configs = []
    for era in ERAS:
        config = tmp_path / f"run_{era}.yaml"
        config.write_text(f"""
analysis: tests.test_combination:build_era
era: "{era}"
channels: [mt]
ntuples: {{base: {tmp_path / 'ntuples'}}}
skim_dir: {tmp_path / 'skims'}
output_dir: {tmp_path / era}
combine: {{cmssw_dir: /unused}}
""")
        card = tmp_path / era / "datacards" / "all" / "combined.txt"
        card.parent.mkdir(parents=True)
        card.write_text(_card(era, "CMS_ff_QCDStatShift_mt_2017 lnN - 1.01 -\n" if era == "2018" else ""))
        configs.append(config)
    return tmp_path, configs


def test_the_commands_and_the_nuisance_checks(tmp_path):
    command = combine_cards_command([tmp_path / "a" / "combined.txt", tmp_path / "b" / "combined.txt"])
    assert command == f"combineCards.py .={tmp_path / 'a' / 'combined.txt'} .={tmp_path / 'b' / 'combined.txt'} > combined.txt"  # bin names stay
    assert validate_command() == "ValidateDatacards.py combined.txt --jsonFile validation.json --printLevel 1"
    (tmp_path / "card.txt").write_text(_card("2018"))
    assert card_nuisances(tmp_path / "card.txt") == ["lumi_13TeV_15161718_l", "CMS_pu", "CMS_ff_QCDStatShift_mt_2018"]
    eras = ["2016preVFP", "2016postVFP", "2017", "2018"]
    assert foreign_token("CMS_x_2017", "2018", eras) == "2017"
    assert foreign_token("CMS_ff_x_2016", "2016preVFP", eras + ["2016"]) is None  # the year shared by both halves
    assert foreign_token("CMS_x_2016postVFP", "2016preVFP", eras) == "2016postVFP"
    assert foreign_token("CMS_x_2018", "2018", eras) is None and foreign_token("lumi_13TeV_15161718_l", "2018", eras) is None


def test_combine_runs_the_limits_on_the_combined_card(runs, monkeypatch):
    directory, configs = runs
    calls = []

    def fake_run(commands, combine, cwd, timeout=None):
        calls.append((commands, cwd))
        for name, values in (("Limit.AsymptoticLimits", ([0.025, 0.16, 0.5, 0.84, 0.975], [2.0, 3.0, 4.0, 6.0, 8.0])),):
            with uproot.recreate(cwd / f"higgsCombine.{name}.mH125.root") as f:
                f["limit"] = {"quantileExpected": np.array(values[0], dtype=np.float32), "limit": np.array(values[1], dtype=np.float64)}

    monkeypatch.setattr(cmssw, "run", fake_run)
    output = directory / "run2"
    result = CliRunner().invoke(app, ["combine", *[a for c in configs for a in ("-c", str(c))], "-o", str(output), "--validate"])
    assert result.exit_code == 0, result.output
    (commands, cwd), = calls
    assert cwd == output and commands[0] == combine_cards_command([directory / era / "datacards" / "all" / "combined.txt" for era in ERAS])
    assert commands[1].startswith(validate_command()) and commands[2:] == limit_commands()
    assert "| combination | 4.00 | [3.00, 6.00] | [2.00, 8.00] |" in result.output
    report = (output / "nuisances.md").read_text()
    assert "CMS_ff_QCDStatShift_mt_2017 (2018 card): ends in the era 2017 of another run" in report
    assert "| lumi_13TeV_15161718_l | x | x |" in report and "| CMS_ff_QCDStatShift_mt_2018 |  | x |" in report


def test_combine_background_only_maps_the_bins_of_every_era(runs, monkeypatch):
    directory, configs = runs
    values = {
        "status": 0, "covariance_quality": 3, "prefit": {"CMS_pu": [0.0, 1.0]}, "postfit": {"CMS_pu": [0.5, 0.9]},
        "norm_prefit": {f"htt_mt_{i}_{era}/total_background": [4.0, 1.0] for era in ERAS for i in (1, 2)},
        "norm_fit_b": {f"htt_mt_{i}_{era}/total_background": [5.0, 1.0] for era in ERAS for i in (1, 2)},
        "data": {f"htt_mt_{i}_{era}": 6.0 for era in ERAS for i in (1, 2)},
    }

    def fake_run(commands, combine, cwd, timeout=None):
        assert commands[1:] == fit_commands()
        (cwd / "fitDiagnostics.BackgroundOnly.json").write_text(json.dumps(values))

    monkeypatch.setattr(cmssw, "run", fake_run)
    output = directory / "run2_bonly"
    result = CliRunner().invoke(app, ["combine", *[a for c in configs for a in ("-c", str(c))], "-o", str(output), "--background-only"])
    assert result.exit_code == 0, result.output
    summary = json.loads((output / "background_fit.json").read_text())["combination"]
    assert set(summary["yields"]) == {"mt 2017", "mt 2018"} and summary["yields"]["mt 2017"]["sig"]["data"] == 6.0
    assert "| mt 2018 | bkg | 6 | 4.0 | 5.0 | 1.500 | 1.200 |" in result.output


def test_one_run_yaml_for_every_era(runs, monkeypatch):
    """--era loads a run YAML once per era (its {era} filled): the same combination as one YAML per era."""
    directory, configs = runs
    generic = directory / "run.yaml"
    generic.write_text(configs[0].read_text().replace(f"output_dir: {directory / ERAS[0]}", f"output_dir: {directory}/{{era}}"))
    calls = []

    def fake_run(commands, combine, cwd, timeout=None):
        calls.append(commands)
        with uproot.recreate(cwd / "higgsCombine.Limit.AsymptoticLimits.mH125.root") as f:
            f["limit"] = {"quantileExpected": np.array([0.025, 0.16, 0.5, 0.84, 0.975], dtype=np.float32), "limit": np.array([2.0, 3.0, 4.0, 6.0, 8.0])}

    monkeypatch.setattr(cmssw, "run", fake_run)
    output = directory / "run2_generic"
    result = CliRunner().invoke(app, ["combine", "-c", str(generic), *[a for era in ERAS for a in ("--era", era)], "-o", str(output)])
    assert result.exit_code == 0, result.output
    assert calls[0][0] == combine_cards_command([directory / era / "datacards" / "all" / "combined.txt" for era in ERAS])
    assert "CMS_ff_QCDStatShift_mt_2017 (2018 card): ends in the era 2017 of another run" in (output / "nuisances.md").read_text()


def test_two_runs_of_one_era_or_a_missing_card_are_refused(runs, monkeypatch):
    directory, configs = runs
    monkeypatch.setattr(cmssw, "run", lambda *args, **kwargs: None)
    result = CliRunner().invoke(app, ["combine", "-c", str(configs[0]), "-c", str(configs[0]), "-o", str(directory / "x")])
    assert result.exit_code != 0 and "appear in more than one run" in str(result.exception)
    result = CliRunner().invoke(app, ["combine", "-c", str(configs[0]), "-c", str(configs[1]), "-o", str(directory / "y"), "--final-state", "et"])
    assert result.exit_code != 0 and "datacards missing" in str(result.exception)

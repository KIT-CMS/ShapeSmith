import textwrap

import uproot
from typer.testing import CliRunner

from shapesmith import __version__
from shapesmith.cli import app
from shapesmith.testing import make_mini_dataset

runner = CliRunner()


def _write_config(tmp_path):
    make_mini_dataset(tmp_path / "data", n_files=2, n_events=400)
    path = tmp_path / "run.yaml"
    path.write_text(textwrap.dedent(f"""
        analysis: tests.mini_analysis:build
        era: "2018"
        channels: [mt]
        switches: {{jet_fakes: ff, embedding: false}}
        ntuples:
          base: {tmp_path / 'data' / 'CROWNRun'}
          friends:
            - {{base: {tmp_path / 'data' / 'CROWNFriends' / 'nn'}}}
        skim_dir: {tmp_path / 'skims'}
        output_dir: {tmp_path / 'out'}
        ml_dir: {tmp_path / 'ml'}
        workers: 2
    """))
    return path


def _run(*args):
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return result


def test_full_chain(tmp_path):
    config = _write_config(tmp_path)
    _run("validate", "-c", str(config))
    _run("skim", "-c", str(config))
    assert (tmp_path / "skims" / "mt" / "ZTT_1" / "manifest.json").exists()
    assert (tmp_path / "skims" / "versions" / "mini.json").exists()
    _run("hist", "-c", str(config), "--control")
    _run("estimate", "-c", str(config), "--control")
    _run("plot", "-c", str(config), "--control")
    assert (tmp_path / "out" / "plots" / "mt" / "inclusive_m_vis.png").exists()
    _run("hist", "-c", str(config))
    _run("estimate", "-c", str(config))
    _run("sync", "-c", str(config))
    assert (tmp_path / "out" / "synced" / "htt_mt.inputs-Run2018.root").exists()
    _run("datacards", "-c", str(config))
    card = (tmp_path / "out" / "datacards" / "mt" / "combined.txt").read_text()
    assert "jetFakes" in card and "* autoMCStats 0" in card
    _run("plot", "-c", str(config), "--category", "sig", "--blind")
    assert (tmp_path / "out" / "plots" / "mt" / "sig_score.pdf").exists()
    _run("ml-export", "-c", str(config))
    assert (tmp_path / "ml" / "mt" / "fold0_training.feather").exists()
    result = _run("inspect", str(tmp_path / "out" / "shapes.root"))
    assert "jetFakes" in result.output
    assert _run("inspect", "--unchanged", str(tmp_path / "out" / "shapes.root")).output == ""
    with uproot.open(tmp_path / "out" / "shapes.root") as f:
        assert "mt_sig/jetFakes#nominal#Nominal#score" in f


def test_hist_regions_and_plot_region_cli(tmp_path):
    config = _write_config(tmp_path)
    _run("skim", "-c", str(config))
    _run("hist", "-c", str(config), "--control", "--regions", "same_sign")
    _run("plot", "-c", str(config), "--control", "--region", "same_sign")
    assert (tmp_path / "out" / "plots" / "mt" / "same_sign" / "inclusive_m_vis.png").exists()


def test_validate_reports_broken_analysis(tmp_path):
    config = _write_config(tmp_path)
    config.write_text(config.read_text().replace("tests.mini_analysis:build", "tests.mini_analysis:no_such"))
    result = runner.invoke(app, ["validate", "-c", str(config)])
    assert result.exit_code != 0


def test_commands_write_a_log_file(tmp_path):
    config = _write_config(tmp_path)
    _run("validate", "-c", str(config))
    assert not (tmp_path / "out" / "logs").exists()  # validate only reports
    _run("skim", "-c", str(config), "-s", "log_level=debug")
    [log] = (tmp_path / "out" / "logs").glob("skim_*.log")
    text = log.read_text()
    assert f"shapesmith.cli: ShapeSmith {__version__} skim: {config}, overrides log_level=debug" in text
    assert "log level DEBUG" in text and "shapesmith.skim: mt: samples 0 reused, 0 resumed" in text
    assert "shapesmith.parallel: skim: " in text and "jobs done (100 %)" in text
    assert any(" DEBUG " in line and "SpawnProcess" in line and "shapesmith.skim: mt/" in line for line in text.splitlines())  # from the workers
    assert "shapesmith.cli: done in" in text


def test_a_failing_command_logs_its_traceback(tmp_path):
    config = _write_config(tmp_path)
    result = runner.invoke(app, ["hist", "-c", str(config)])  # no skims yet
    assert result.exit_code != 0
    [log] = (tmp_path / "out" / "logs").glob("hist_*.log")
    text = log.read_text()
    assert "ERROR   MainProcess shapesmith.cli: failed after" in text and "SkimMissingError" in text and "Traceback" in text

import textwrap

import uproot
from typer.testing import CliRunner

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
    assert (tmp_path / "skims" / "versions.json").exists()
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
    with uproot.open(tmp_path / "out" / "shapes.root") as f:
        assert "mt_sig/jetFakes#nominal#Nominal#score" in f


def test_example_config(tmp_path):
    _run("example-config", str(tmp_path / "example.yaml"))
    assert "analysis:" in (tmp_path / "example.yaml").read_text()


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

import dataclasses
import json
import textwrap
from dataclasses import dataclass

import pytest
from typer.testing import CliRunner

from shapesmith.cli import app
from shapesmith.events import Query
from shapesmith.measurements import MeasureContext, run_measure
from shapesmith.testing import make_mini_dataset
from tests import mini_analysis


@dataclass(frozen=True)
class CountEvents:
    """A minimal measurement: the weighted ZTT yield per channel, with the payload provenance."""

    name: str = "count_events"

    def run(self, context: MeasureContext) -> None:
        result = {channel: float(context.events(Query(channel, "ZTT")).weights.sum()) for channel in context.channels}
        if context.suggest_binning:
            print(f"suggested: {result}")
            return
        (context.output / "result.json").write_text(json.dumps({"yields": result, "provenance": context.provenance(channel="mt")}))


def build(config=None):
    return dataclasses.replace(mini_analysis.build(config), measurement=CountEvents())


def test_run_measure_needs_a_measurement(mini_run):
    config, analysis = mini_run
    with pytest.raises(ValueError, match="defines no measurement"):
        run_measure(config, analysis, ["mt"])


def test_run_measure_writes_into_its_own_directory(mini_run):
    config, _ = mini_run
    output = run_measure(config, build(config), ["mt"])
    assert output == config.output_dir / "count_events" / "2018"
    result = json.loads((output / "result.json").read_text())
    assert result["yields"]["mt"] > 0
    provenance = result["provenance"]
    assert provenance["analysis"]["name"] == "mini" and provenance["analysis"]["module"] == "tests.mini_analysis:build"
    assert provenance["era"] == "2018" and provenance["channel"] == "mt" and provenance["ntuples"] == config.ntuples.base and provenance["config_sha256"]
    assert (output / "versions" / "mini.json").exists()


def test_measure_command(tmp_path):
    make_mini_dataset(tmp_path / "data", n_files=1, n_events=100)
    config = tmp_path / "run.yaml"
    config.write_text(textwrap.dedent(f"""
        analysis: tests.test_measure:build
        era: "2018"
        channels: [mt]
        ntuples:
          base: {tmp_path / 'data' / 'CROWNRun'}
          friends: [{{base: {tmp_path / 'data' / 'CROWNFriends' / 'nn'}}}]
        skim_dir: {tmp_path / 'skims'}
        output_dir: {tmp_path / 'out'}
        workers: 1
    """))
    runner = CliRunner()
    assert runner.invoke(app, ["skim", "-c", str(config)]).exit_code == 0
    result = runner.invoke(app, ["measure", "-c", str(config), "--suggest-binning"])
    assert result.exit_code == 0 and "suggested" in result.output and not (tmp_path / "out" / "count_events").exists()
    assert runner.invoke(app, ["measure", "-c", str(config)]).exit_code == 0
    assert (tmp_path / "out" / "count_events" / "2018" / "result.json").exists()

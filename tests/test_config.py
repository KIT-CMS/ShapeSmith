import json
import textwrap
from pathlib import Path

import pytest

from shapesmith.config import NtupleConfig, RunConfig, load_analysis, load_config
from shapesmith.provenance import write_versions


def _yaml(tmp_path, text):
    path = tmp_path / "run.yaml"
    path.write_text(textwrap.dedent(text))
    return path


def test_load_config_and_analysis(tmp_path):
    path = _yaml(tmp_path, """
        analysis: tests.mini_analysis:build
        era: "2018"
        channels: [mt]
        switches: {jet_fakes: mc, embedding: false}
        ntuples:
          base: /data/CROWNRun
          friends:
            - {base: /data/CROWNFriends/nn}
            - {base: /data/CROWNFriends/ff, channels: [et, mt]}
        skim_dir: /tmp/skims
        output_dir: /tmp/out
        workers: 2
    """)
    config = load_config(path)
    assert isinstance(config, RunConfig)
    assert config.ntuples.server == "" and config.ntuples.friends[0].applies_to == ("mc", "data", "embedding")
    assert config.ntuples.friends[0].channels is None and config.ntuples.friends[1].channels == ("et", "mt")
    assert config.switches == {"jet_fakes": "mc", "embedding": False}
    analysis = load_analysis(config)
    assert analysis.name == "mini"


def test_bad_import_path(tmp_path):
    path = _yaml(tmp_path, """
        analysis: tests.mini_analysis:no_such_function
        era: "2018"
        channels: [mt]
        ntuples: {base: /data/CROWNRun}
        skim_dir: /tmp/skims
        output_dir: /tmp/out
    """)
    with pytest.raises(AttributeError, match="no_such_function"):
        load_analysis(load_config(path))


def test_missing_required_field(tmp_path):
    path = _yaml(tmp_path, """
        analysis: tests.mini_analysis:build
        channels: [mt]
        ntuples: {base: /data}
        skim_dir: /tmp/skims
        output_dir: /tmp/out
    """)
    with pytest.raises(ValueError, match="era"):
        load_config(path)


def test_relative_paths_resolve_against_the_config_file(tmp_path):
    (tmp_path / "db").mkdir()
    (tmp_path / "db" / "datasets.json").write_text("{}")
    path = _yaml(tmp_path, """
        analysis: tests.mini_analysis:build
        era: "2018"
        channels: [mt]
        ntuples: {base: /data}
        skim_dir: skims
        output_dir: /abs/out
        sample_database: db/datasets.json
    """)
    config = load_config(path)
    assert config.skim_dir == tmp_path / "skims" and config.output_dir == Path("/abs/out")
    assert config.sample_database == tmp_path / "db" / "datasets.json" and config.ml_dir is None


def test_overrides(tmp_path):
    path = _yaml(tmp_path, """
        analysis: tests.mini_analysis:build
        era: "2018"
        channels: [et, mt]
        switches: {jet_fakes: mc}
        ntuples: {base: /data}
        skim_dir: /tmp/skims
        output_dir: /tmp/out
    """)
    config = load_config(path, ["switches.jet_fakes=ff", "workers=4", "channels=[mt]", "ntuples.server=root://x"])
    assert config.switches == {"jet_fakes": "ff"} and config.workers == 4 and config.channels == ["mt"] and config.ntuples.server == "root://x"
    with pytest.raises(ValueError, match="key=value"):
        load_config(path, ["workers"])


def test_one_yaml_for_every_era(tmp_path):
    """`{era}` in any string (paths, switches, friends, web texts) becomes the era of the run, also one given with -s;
    an era written as a number is text."""
    path = _yaml(tmp_path, """
        analysis: tests.mini_analysis:build
        era: "2018"
        channels: [mt]
        switches: {sample_lists: ["sm_{era}_binned", "sm_{era}_embedding"], payload: "out/{era}/x_{era}UL.json", n: 3}
        ntuples:
          base: /data/CROWNRun
          friends: [{base: /data/CROWNFriend/ff, channels: [mt]}]
        skim_dir: /ceph/skims/{era}          # a plain scalar may hold {era}; inside [...] or {...} it is quoted
        output_dir: out/{era}
        web: {dir: "/web/tag/{era}", variant: emb, title: "tag {era} · control plots", about: "keeps {braces} and {era}"}
    """)
    config = load_config(path)
    assert config.switches == {"sample_lists": ["sm_2018_binned", "sm_2018_embedding"], "payload": "out/2018/x_2018UL.json", "n": 3}
    assert config.skim_dir == Path("/ceph/skims/2018") and config.output_dir == tmp_path / "out" / "2018"
    assert config.web.dir == Path("/web/tag/2018") and config.web.title == "tag 2018 · control plots" and config.web.about == "keeps {braces} and 2018"
    other = load_config(path, ["era=2016preVFP"])
    assert other.era == "2016preVFP" and other.skim_dir == Path("/ceph/skims/2016preVFP") and other.switches["sample_lists"][0] == "sm_2016preVFP_binned"
    numeric = load_config(path, ["era=2017"])
    assert numeric.era == "2017" and numeric.output_dir == tmp_path / "out" / "2017"
    assert load_config(path, ["era=2017", "skim_dir=/s/{era}/again"]).skim_dir == Path("/s/2017/again")  # overrides are filled too


def test_write_versions(tmp_path):
    config = RunConfig(analysis="tests.mini_analysis:build", era="2018", channels=["mt"], ntuples=NtupleConfig(base="/data"), skim_dir=tmp_path / "s", output_dir=tmp_path / "o")
    path = write_versions(config, "mini", tmp_path / "o")
    assert path == tmp_path / "o" / "versions" / "mini.json"
    record = json.loads(path.read_text())
    assert record["shapesmith"]["version"] and record["analysis"]["module"] == "tests.mini_analysis:build"
    assert record["sample_database"] is None and record["config"]["channels"] == ["mt"] and record["ntuples"] == "/data"
    assert len(record["config_sha256"]) == 64

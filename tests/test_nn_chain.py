"""The blinding contract through the real chain (the merge of the NN branches): the blinding.json that `datacards`
writes (score categories) is the file every fit to the data reads (masking.py), in the layout of a variant, per era and
merged by `combine`; the summaries of the background-only fits name the lo and hi bins by their category and mark the
masked ones. Synthetic NN analysis of two eras (tests/nn_mini.py), CMSSW faked by the values a fit would write."""
import json
import textwrap

import pytest
import uproot
from typer.testing import CliRunner

from shapesmith import cmssw, masking
from shapesmith.background_fit import OUTPUT
from shapesmith.cards import card_bins
from shapesmith.cli import app
from shapesmith.histogram import Histogram
from tests import nn_mini

runner = CliRunner()
ERAS = ("2017", "2018")
VARIANT = "s0"


def _run(*args: str):
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return result


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """The run YAML of both eras ({era} in its paths) and the root of its outputs, after skim, hist, estimate and
    datacards --variant s0 of each era."""
    root = tmp_path_factory.mktemp("nn_chain")
    for era in ERAS:
        nn_mini.make_nn_dataset(root / "data", era=era)
    config = root / "run.yaml"
    config.write_text(textwrap.dedent(f"""
        analysis: tests.nn_mini:build
        era: "2018"
        channels: [mt]
        ntuples:
          base: {root / 'data' / 'CROWNRun'}
          friends:
            - {{base: {root / 'data' / 'CROWNFriends' / 'nn'}}}
        skim_dir: {root / 'skims' / '{era}'}
        output_dir: {root / 'out' / '{era}'}
        workers: 1
        combine: {{cmssw_dir: /unused}}
    """))
    for era in ERAS:
        for command in ("skim", "hist", "estimate"):
            _run(command, "-c", str(config), "-s", f"era={era}")
        _run("datacards", "-c", str(config), "-s", f"era={era}", "--variant", VARIANT)
    return config, root


def _card_dir(root, era):
    return root / "out" / era / "datacards" / VARIANT / "mt"


def _values(bins: list[str]) -> dict:
    """The values script's output of a background-only fit of a card with these bins."""
    norms = {f"{b}/{process}": [5.0, 1.0] for b in bins for process in ("ZTT", "total_background")}
    return {"status": 0, "covariance_quality": 3, "prefit": {"lumi": [0.0, 1.0]}, "postfit": {"lumi": [0.2, 0.9]},
            "norm_prefit": norms, "norm_fit_b": norms, "data": {b: 6.0 for b in bins}}


def _fake_cmssw(monkeypatch, bins_of):
    """cmssw.run recording its commands and writing the values of a fit of the card bins `bins_of(cwd)`."""
    calls = []

    def fake_run(commands, combine, cwd, timeout=None):
        calls.append(list(commands))
        (cwd / f"{OUTPUT}.json").write_text(json.dumps(_values(bins_of(cwd))))

    monkeypatch.setattr(cmssw, "run", fake_run)
    return calls


def test_the_writer_and_the_reader_of_blinding_json_agree(run):
    """The masked bins are exactly the hi bins of each card, with Asimov data_obs; the reader takes them from the card
    directory of the variant."""
    _, root = run
    for era in ERAS:
        card_dir = _card_dir(root, era)
        content = json.loads((card_dir / "blinding.json").read_text())
        bins = card_bins(card_dir / "combined.txt")
        hi, lo = f"htt_mt_1hi_{era}", f"htt_mt_1lo_{era}"
        assert content["masked"] == [hi] and bins == [lo, hi, f"htt_mt_2_{era}", f"htt_mt_3_{era}"]
        assert content["categories"][hi]["lo"] == lo and content["asimov_data"] is True
        assert masking.read(card_dir).masked == (hi,) and masking.data_masks(card_dir) == [f"mask_{hi}=1"]
        with uproot.open(card_dir / "common" / f"htt_input_{era}.root") as shapes:
            data = Histogram.from_root(shapes[f"{hi}/data_obs"]).values
            backgrounds = sum(Histogram.from_root(shapes[f"{hi}/{p}"]).values for p in ("ZTT", "TT", "jetFakes") if f"{hi}/{p}" in shapes)
        assert data.tolist() == pytest.approx(backgrounds.tolist())  # Asimov: the background prediction, no data
    assert not (root / "out" / "2018" / "datacards" / "mt").exists()  # the variant only


def test_the_background_only_fit_of_a_variant_masks_the_hi_bin_and_labels_the_parts(run, monkeypatch):
    config, root = run
    calls = _fake_cmssw(monkeypatch, lambda cwd: card_bins(cwd / "combined.txt"))
    result = _run("fit", "-c", str(config), "--variant", VARIANT, "--background-only")
    assert calls[0][0].endswith("--channel-masks") and "--setParameters r=0,mask_htt_mt_1hi_2018=1 --freezeParameters r" in calls[0][1]
    summary = json.loads((_card_dir(root, "2018").parent / "background_fit.json").read_text())["mt"]
    assert summary["masked"] == ["htt_mt_1hi_2018"]
    yields = summary["yields"]["mt"]
    assert list(yields) == ["HH lo", "HH hi", "ZTT", "TT"] and yields["HH hi"]["masked"] is True and "masked" not in yields["HH lo"]
    assert "| mt | HH hi | masked |" in result.output and "| mt | HH lo | 6 |" in result.output


def test_the_combination_merges_the_blinding_of_every_era_and_labels_the_parts(run, monkeypatch, tmp_path):
    config, root = run
    eras = [option for era in ERAS for option in ("--era", era)]
    bins = [b for era in ERAS for b in card_bins(_card_dir(root, era) / "combined.txt")]
    calls = _fake_cmssw(monkeypatch, lambda cwd: bins)
    _run("combine", "-c", str(config), *eras, "--final-state", "mt", "--variant", VARIANT, "-o", str(tmp_path / "run2"), "--background-only")
    merged = json.loads((tmp_path / "run2" / "blinding.json").read_text())
    assert merged["masked"] == ["htt_mt_1hi_2017", "htt_mt_1hi_2018"] and len(merged["sources"]) == 2
    fit = next(command for command in calls[0] if "-M FitDiagnostics" in command)
    assert "--setParameters r=0,mask_htt_mt_1hi_2017=1,mask_htt_mt_1hi_2018=1 --freezeParameters r" in fit
    yields = json.loads((tmp_path / "run2" / "background_fit.json").read_text())["combination"]["yields"]
    assert list(yields) == ["mt 2017", "mt 2018"] and list(yields["mt 2018"]) == ["HH lo", "HH hi", "ZTT", "TT"]
    assert yields["mt 2017"]["HH hi"]["masked"] is True and "-" not in {category for entries in yields.values() for category in entries}

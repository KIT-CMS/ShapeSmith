import dataclasses
import json

import numpy as np
import pytest
import uproot

from shapesmith.datacards import bin_name, collect_bin, observation, rebin_edges, run_datacards, symmetric_shift, symmetrisation_reason, symmetrise, write_datacard
from shapesmith.histogram import HistKey, Histogram, HistogramSet, part_of
from shapesmith.model import AnalysisError, LnN
from shapesmith.validate import validate
from tests.mini_analysis import build, build_embedding


def _hist(values, edges=(0.0, 0.25, 0.5, 0.75, 1.0)):
    return Histogram(list(edges), values, [v * 0.1 for v in values])


def _hset():
    hset = HistogramSet()
    for category in ("sig", "bkg"):
        hset[HistKey("mt", category, "data", "nominal", "Nominal", "score")] = _hist([20, 10, 5, 2])
        hset[HistKey("mt", category, "ZTT", "nominal", "Nominal", "score")] = _hist([10, 5, 2, 0.4])
        hset[HistKey("mt", category, "ZTT", "nominal", "CMS_puUp", "score")] = _hist([11, 5.5, 2.2, 0.44])
        hset[HistKey("mt", category, "ZTT", "nominal", "CMS_puDown", "score")] = _hist([9, 4.5, 1.8, 0.36])
        hset[HistKey("mt", category, "ZL", "nominal", "Nominal", "score")] = _hist([5, 3, 1, 0.3])
        hset[HistKey("mt", category, "jetFakes", "nominal", "Nominal", "score")] = _hist([4, 2, 1, 0.2])
        hset[HistKey("mt", category, "HH", "nominal", "Nominal", "score")] = _hist([0.01, 0.02, 0.05, 0.1])
        hset[HistKey("mt", category, "HH", "nominal", "CMS_puUp", "score")] = _hist([0.011, 0.022, 0.055, 0.11])
        hset[HistKey("mt", category, "HH", "nominal", "CMS_puDown", "score")] = _hist([0.009, 0.018, 0.045, 0.09])
    return hset


def test_bin_name():
    assert bin_name(build(), "mt", 0) == "htt_mt_1_2018"


def test_rebin_edges_merges_from_the_right():
    total = _hist([10, 5, 2, 0.9])  # last bin below threshold 1.0 -> merged with its left neighbour
    assert rebin_edges(total, 1.0) == [0.0, 0.25, 0.5, 1.0]
    assert rebin_edges(total, 100.0) == [0.0, 1.0]
    assert rebin_edges(total, None) == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_write_datacard(tmp_path):
    analysis = build()
    path = write_datacard(_hset(), analysis, ["mt"], tmp_path / "mt", systematics=True, min_background=1.0)
    text = path.read_text()
    assert "imax * number of bins" in text
    assert "shapes * htt_mt_1_2018 common/htt_input_2018.root htt_mt_1_2018/$PROCESS htt_mt_1_2018/$PROCESS_$SYSTEMATIC" in text
    observation = [line for line in text.splitlines() if line.startswith("observation")][0]
    assert observation.split()[1:] == ["37.0", "37.0"]
    process_line = [line for line in text.splitlines() if line.startswith("process")][0]
    assert process_line.split()[1:5] == ["HH", "ZTT", "ZL", "jetFakes"]
    index_line = [line for line in text.splitlines() if line.startswith("process")][1]
    assert index_line.split()[1:5] == ["0", "1", "2", "3"]
    assert "lumi lnN 1.025 1.025 1.025 1.025" in text
    assert "zjXsec lnN - 1.02 1.02 -" in text
    assert "ffNorm_mt lnN - - - 1.1" in text
    assert "CMS_pu shape 1 1 - -" in text  # HH and ZTT carry the variation, ZL and jetFakes do not
    assert text.strip().endswith("* autoMCStats 0")
    with uproot.open(tmp_path / "mt" / "common" / "htt_input_2018.root") as f:
        assert f["htt_mt_1_2018/data_obs"].values().tolist() == [20.0, 10.0, 7.0]  # last two bins merged (bkg 0.9 < 1)
        assert f["htt_mt_1_2018/data_obs"].axis().edges().tolist() == [0.0, 0.25, 0.5, 1.0]
        assert f["htt_mt_1_2018/ZTT_CMS_puUp"].values().tolist() == [11.0, 5.5, 2.64]
        assert "htt_mt_2_2018/HH" in f


@pytest.mark.parametrize("total", [4.243, 0.0123456789, 1.0 / 3.0])
def test_the_observation_reads_back_as_the_sum_of_data_obs(total):
    """combine rejects a card whose observation differs from data_obs by more than 0.5 % (htt_mt_1hi_2016preVFP: 4.2 for
    the Asimov sum 4.243 with one decimal)."""
    data = Histogram([0.0, 0.5, 1.0], [total / 4, 3 * total / 4], [total / 4, 3 * total / 4])
    assert float(observation(data)) == data.sum()


def test_an_event_count_keeps_its_one_decimal():
    assert observation(Histogram([0.0, 1.0], [118.0], [118.0])) == "118.0"  # the cards of real data as before (golden card)


def test_write_datacard_without_systematics(tmp_path):
    text = write_datacard(_hset(), build(), ["mt"], tmp_path / "mt", systematics=False, min_background=None).read_text()
    assert "lnN" not in text and "shape" not in text.replace("shapes *", "")


def test_run_datacards_final_states(tmp_path):
    paths = run_datacards(_hset(), build(), {"mt": ["mt"], "all": ["mt"]}, tmp_path, systematics=True, min_background=1.0)
    assert set(paths) == {"mt", "all"} and all(p.exists() for p in paths.values())


def test_auxiliary_processes_and_templates_stay_out_of_the_card(tmp_path):
    hset = _hset()
    for category in ("sig", "bkg"):
        hset[HistKey("mt", category, "EMB", "nominal", "Nominal", "score")] = _hist([3, 2, 1, 0.5])
        hset[HistKey("mt", category, "EMB", "nominal", "emb1p002", "score")] = _hist([3, 2, 1, 0.6])
    text = write_datacard(hset, build_embedding(), ["mt"], tmp_path / "mt").read_text()
    processes = [line for line in text.splitlines() if line.startswith("process")][0].split()[1:]
    assert "ZTT" not in processes and "EMB" in processes  # ZTT is the auxiliary template of the embedding
    assert "emb1p002" not in text


def test_parts_of_a_summed_variation_stay_out_of_the_card_and_the_shapes(tmp_path):
    hset = _hset()
    for category in ("sig", "bkg"):
        hset[HistKey("mt", category, "EMB", "nominal", "Nominal", "score")] = _hist([3, 2, 1, 0.5])
        for d in ("Up", "Down"):
            hset[HistKey("mt", category, "EMB", "nominal", part_of(f"CMS_tes{d}", "dm10"), "score")] = _hist([3, 2, 1, 0.6])
            hset[HistKey("mt", category, "EMB", "nominal", f"CMS_tes{d}", "score")] = _hist([3, 2, 1, 0.6])
    card = write_datacard(hset, build_embedding(), ["mt"], tmp_path / "mt")
    text = card.read_text()
    assert "CMS_tes shape" in " ".join(text.split()) and "%" not in text
    with uproot.open(card.parent / "common" / "htt_input_2018.root") as f:
        names = f.keys(recursive=True, cycle=False)
        assert "htt_mt_1_2018/EMB_CMS_tesUp" in names and not [name for name in names if "%" in name]



def test_lnn_entries_with_one_name_share_a_line(tmp_path):
    """E.g. a lepton-ID nuisance with one value on MC and another on the embedded sample."""
    analysis = dataclasses.replace(build(), lnn=(LnN("eff_m", ("ZTT",), 1.02), LnN("lumi", ("HH",), 1.01), LnN("eff_m", ("ZL",), 1.01)))
    text = write_datacard(_hset(), analysis, ["mt"], tmp_path / "mt").read_text()
    lines = [line for line in text.splitlines() if " lnN " in line]
    assert lines == ["eff_m lnN - 1.02 1.01 - - 1.02 1.01 -", "lumi lnN 1.01 - - - 1.01 - - -"]
    overlap = dataclasses.replace(analysis, lnn=(LnN("eff_m", ("ZTT",), 1.02), LnN("eff_m", ("*",), 1.01)))
    with pytest.raises(ValueError, match="lnN eff_m: two entries apply to ZTT in htt_mt_1_2018"):
        write_datacard(_hset(), overlap, ["mt"], tmp_path / "overlap")


def test_lnn_variations_become_an_lnn_of_the_yield_ratios(tmp_path):
    """A variation of Analysis.lnn_variations: down/up yield ratios per bin and process, neither a shape line nor a
    shapes-file histogram; the processes without it get '-'."""
    analysis = dataclasses.replace(build(), lnn_variations=("CMS_pu",))
    card = write_datacard(_hset(), analysis, ["mt"], tmp_path / "mt")
    text = card.read_text()
    assert "CMS_pu lnN 0.9000/1.1000 0.9000/1.1000 - - 0.9000/1.1000 0.9000/1.1000 - -" in text
    assert "CMS_pu shape" not in text
    with uproot.open(card.parent / "common" / "htt_input_2018.root") as f:
        assert not [name for name in f.keys(recursive=True, cycle=False) if "CMS_pu" in name]


def test_lnn_variations_must_exist():
    with pytest.raises(AnalysisError, match="lnN variation CMS_prefiring: no variations CMS_prefiringUp and CMS_prefiringDown"):
        validate(dataclasses.replace(build(), lnn_variations=("CMS_prefiring",)))
    validate(dataclasses.replace(build(), lnn_variations=("CMS_pu",)))


def _one_bin_hset(down: float, up: float, name: str = "CMS_ff_shift", nominal: float = 10.0):
    """Both categories with one bin, jetFakes Up/Down at down/up times the nominal."""
    hset = HistogramSet()
    edges = (0.0, 1.0)
    for category in ("sig", "bkg"):
        hset[HistKey("mt", category, "data", "nominal", "Nominal", "score")] = Histogram(edges, [20.0], [20.0])
        hset[HistKey("mt", category, "ZTT", "nominal", "Nominal", "score")] = Histogram(edges, [10.0], [1.0])
        hset[HistKey("mt", category, "jetFakes", "nominal", "Nominal", "score")] = Histogram(edges, [nominal], [1.0])
        hset[HistKey("mt", category, "jetFakes", "nominal", f"{name}Up", "score")] = Histogram(edges, [nominal * up], [2.0])
        hset[HistKey("mt", category, "jetFakes", "nominal", f"{name}Down", "score")] = Histogram(edges, [nominal * down], [3.0])
    return hset


def _shape(path, name):
    with uproot.open(path.parent / "common" / "htt_input_2018.root") as f:
        return f[f"htt_mt_1_2018/{name}"].values().tolist()


def test_a_pair_with_an_empty_side_after_the_symmetrisation_is_left_out(tmp_path):
    """Up at 2.5 times the nominal and Down at the nominal: the symmetric Down (-0.5 times the nominal) is clipped at 0,
    whose norm combine rejects; the pair is left out and listed."""
    analysis = dataclasses.replace(build(), symmetrised_variations=("CMS_ff_shift",))
    card = write_datacard(_one_bin_hset(1.0, 2.5), analysis, ["mt"], tmp_path / "mt")
    assert "CMS_ff_shift" not in card.read_text()
    changes = json.loads((card.parent / "template_changes.json").read_text())
    assert changes["empty_pairs"] == ["htt_mt_1_2018/jetFakes: CMS_ff_shift", "htt_mt_2_2018/jetFakes: CMS_ff_shift"] and len(changes["symmetrised"]) == 2
    kept = write_datacard(_one_bin_hset(1.0, 1.5), analysis, ["mt"], tmp_path / "kept")  # Down at 0.5 times the nominal
    assert "CMS_ff_shift" in kept.read_text() and "empty_pairs" not in json.loads((kept.parent / "template_changes.json").read_text())


def test_a_shifted_template_with_a_negative_sum_is_left_out_of_its_bin_and_process(tmp_path, caplog):
    """Simulated events with negative weights (smoke_run2 htt_et_4_2018: ZL CMS_btag_fixedWP_bc_uncorrelated_2018Up
    sums to -0.0128 at a nominal of 0.097, text2workspace stops with "Bogus norm"), not symmetrised: the pair is left out
    of that bin and process only, logged and listed."""
    hset = _one_bin_hset(0.9, 1.1, name="CMS_btag")
    hset[HistKey("mt", "sig", "jetFakes", "nominal", "CMS_btagUp", "score")] = Histogram((0.0, 1.0), [-0.0128], [1.0])
    with caplog.at_level("WARNING", logger="shapesmith.datacards"):
        card = write_datacard(hset, build(), ["mt"], tmp_path / "mt")
    assert "CMS_btag shape - - - 1" in card.read_text()  # columns ZTT, jetFakes of htt_mt_1_2018, then of htt_mt_2_2018
    changes = json.loads((card.parent / "template_changes.json").read_text())
    assert changes["empty_pairs"] == ["htt_mt_1_2018/jetFakes: CMS_btag"] and changes["symmetrised"] == []
    assert "1 shape pairs with a side whose sum is not positive left out" in caplog.text and "htt_mt_1_2018/jetFakes: CMS_btag" in caplog.text
    with uproot.open(card.parent / "common" / "htt_input_2018.root") as f:
        names = f.keys(recursive=True, cycle=False)
        assert "htt_mt_1_2018/jetFakes_CMS_btagUp" not in names and "htt_mt_1_2018/jetFakes_CMS_btagDown" not in names
        assert f["htt_mt_2_2018/jetFakes_CMS_btagUp"].values().tolist() == pytest.approx([11.0])


@pytest.mark.parametrize(("rate", "level"), [(0.0, "INFO"), (-0.25, "WARNING")])
def test_a_process_with_a_rate_up_to_zero_is_left_out_of_its_bin_and_listed(tmp_path, caplog, rate, level):
    """A process whose nominal sums to <= 0 in a bin (simulated events with negative weights, R3 htt_mt_1hi_2018: VVJ
    -0.076) has no column there, only there; it is listed under "left_out_processes" and logged, at WARNING when its
    rate is negative."""
    hset = _one_bin_hset(0.9, 1.1)
    hset[HistKey("mt", "sig", "ZL", "nominal", "Nominal", "score")] = Histogram((0.0, 1.0), [rate], [0.1])
    hset[HistKey("mt", "bkg", "ZL", "nominal", "Nominal", "score")] = Histogram((0.0, 1.0), [2.0], [0.1])
    with caplog.at_level("INFO", logger="shapesmith.score_bins"):
        card = write_datacard(hset, build(), ["mt"], tmp_path / "mt")
    lines = card.read_text().splitlines()
    columns = list(zip([line for line in lines if line.startswith("bin ")][1].split()[1:], next(line for line in lines if line.startswith("process ")).split()[1:]))
    assert columns == [("htt_mt_1_2018", "ZTT"), ("htt_mt_1_2018", "jetFakes"), ("htt_mt_2_2018", "ZTT"), ("htt_mt_2_2018", "ZL"), ("htt_mt_2_2018", "jetFakes")]
    assert json.loads((card.parent / "template_changes.json").read_text())["left_out_processes"] == [{"bin": "htt_mt_1_2018", "process": "ZL", "rate": rate}]
    records = [r for r in caplog.records if "in a card bin left out there" in r.getMessage()]
    assert [r.levelname for r in records] == [level] and records[0].getMessage().endswith(f": htt_mt_1_2018/ZL {rate:.3g}")


@pytest.mark.parametrize(("down", "up", "reason", "after"), [
    (1.2787, 0.9216, "asymmetry 3.6", (1.17855, 0.82145)),  # mt QCD non-closure MCShift 2018: half the distance of Down to Up
    (0.9337, 1.1832, "asymmetry 2.8", (0.87525, 1.12475)),  # StatShift 2018
    (1.0515, 1.0063, "one-sided", (1.0515, 0.9485)),  # QCD DR->SR MCShift et 2018: the larger shift, Up takes the sign of Up - Down
    (0.9719, 0.9898, "one-sided", (0.9719, 1.0281)),
    (1.0, 1.05, "one-sided", (0.95, 1.05)),  # Down exactly nominal
])
def test_one_sided_and_strongly_asymmetric_pairs_are_symmetrised(tmp_path, caplog, down, up, reason, after):
    analysis = dataclasses.replace(build(), symmetrised_variations=("CMS_ff_shift",))
    with caplog.at_level("INFO", logger="shapesmith.datacards"):
        card = write_datacard(_one_bin_hset(down, up), analysis, ["mt"], tmp_path / "mt")
    assert _shape(card, "jetFakes_CMS_ff_shiftDown") == pytest.approx([10.0 * after[0]])
    assert _shape(card, "jetFakes_CMS_ff_shiftUp") == pytest.approx([10.0 * after[1]])
    changes = json.loads((card.parent / "template_changes.json").read_text())
    assert changes["asymmetry_ratio"] == 2.0 and len(changes["symmetrised"]) == 2  # both categories
    record = changes["symmetrised"][0]
    assert record["nuisance"] == "CMS_ff_shift" and record["bin"] == "htt_mt_1_2018" and record["process"] == "jetFakes" and record["reason"] == reason
    assert record["before"] == pytest.approx([down, up]) and record["after"] == pytest.approx(list(after))
    assert "2 templates symmetrised" in caplog.text and f"CMS_ff_shift htt_mt_1_2018/jetFakes ({reason})" in caplog.text


@pytest.mark.parametrize(("down", "up"), [(0.95, 1.06), (1.04, 0.92), (0.98, 1.0399)])
def test_mildly_asymmetric_pairs_and_other_nuisances_stay(tmp_path, down, up):
    """Below the ratio 2 the pair stays, and a one-sided pair outside symmetrised_variations stays too."""
    analysis = dataclasses.replace(build(), symmetrised_variations=("CMS_ff_shift",))
    card = write_datacard(_one_bin_hset(down, up), analysis, ["mt"], tmp_path / "mt")
    assert _shape(card, "jetFakes_CMS_ff_shiftDown") == pytest.approx([10.0 * down]) and _shape(card, "jetFakes_CMS_ff_shiftUp") == pytest.approx([10.0 * up])
    other = write_datacard(_one_bin_hset(1.05, 1.01, name="CMS_other"), analysis, ["mt"], tmp_path / "other")
    assert _shape(other, "jetFakes_CMS_otherDown") == pytest.approx([10.5]) and _shape(other, "jetFakes_CMS_otherUp") == pytest.approx([10.1])
    assert json.loads((other.parent / "template_changes.json").read_text())["symmetrised"] == []


def test_symmetrisation_is_bin_wise_for_shapes():
    """Opposite bins: half the distance of Down to Up; one-sided bins: the larger shift with the sign of Up - Down;
    clipped at 0."""
    nominal = np.array([10.0, 10.0, 10.0, 10.0, 1.0])
    up = np.array([12.0, 11.0, 10.5, 10.0, 0.0])
    down = np.array([9.0, 10.5, 11.0, 10.0, 3.5])
    shift = symmetric_shift(nominal, up, down)
    assert shift.tolist() == pytest.approx([1.5, 1.0, -1.0, 0.0, -1.75])
    assert symmetrisation_reason(-0.1, 0.25, 2.0) == "asymmetry 2.5" and symmetrisation_reason(-0.1, 0.2, 2.0) is None
    assert symmetrisation_reason(0.0, 0.0, 2.0) is None and symmetrisation_reason(0.01, 0.0, 2.0) == "one-sided"
    hset = HistogramSet()
    edges = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    for category in ("sig", "bkg"):
        hset[HistKey("mt", category, "data", "nominal", "Nominal", "score")] = Histogram(edges, nominal, nominal)
        hset[HistKey("mt", category, "jetFakes", "nominal", "Nominal", "score")] = Histogram(edges, nominal, nominal)
        hset[HistKey("mt", category, "jetFakes", "nominal", "CMS_ffUp", "score")] = Histogram(edges, up, up)
        hset[HistKey("mt", category, "jetFakes", "nominal", "CMS_ffDown", "score")] = Histogram(edges, down, down)
    analysis = dataclasses.replace(build(), symmetrised_variations=("CMS_ff",))
    bins = [collect_bin(hset, analysis, "mt", 0, analysis.channel("mt").categories[0], None)]
    (record,) = symmetrise(analysis, bins)  # yields: Down +7.3 %, Up +6.1 %: one-sided
    assert record.reason == "one-sided" and record.clipped == 1
    assert bins[0].variations["jetFakes"]["CMS_ffUp"].values.tolist() == pytest.approx([11.5, 11.0, 9.0, 10.0, 0.0])  # 1 - 1.75 clipped at 0
    assert bins[0].variations["jetFakes"]["CMS_ffDown"].values.tolist() == pytest.approx([8.5, 9.0, 11.0, 10.0, 2.75])
    assert bins[0].variations["jetFakes"]["CMS_ffUp"].variances.tolist() == up.tolist()  # the variances stay


def test_no_op_nuisances_are_left_out(tmp_path, caplog):
    """A pair equal to the nominal in every bin of every process leaves the card and the shapes file; one that moves
    a single process stays."""
    hset = _hset()
    for category in ("sig", "bkg"):
        for process in ("ZTT", "HH"):
            nominal = hset[HistKey("mt", category, process, "nominal", "Nominal", "score")]
            for d in ("Up", "Down"):
                hset[HistKey("mt", category, process, "nominal", f"CMS_noop{d}", "score")] = nominal.copy()
                hset[HistKey("mt", category, process, "nominal", f"CMS_half{d}", "score")] = nominal.copy() if process == "ZTT" else nominal.copy().scale(1.1)
    with caplog.at_level("INFO", logger="shapesmith.datacards"):
        card = write_datacard(hset, build(), ["mt"], tmp_path / "mt")
    text = card.read_text()
    assert "CMS_noop" not in text and "CMS_half shape 1 1 - - 1 1 - -" in text and "CMS_pu shape" in text
    with uproot.open(card.parent / "common" / "htt_input_2018.root") as f:
        assert not [name for name in f.keys(recursive=True, cycle=False) if "CMS_noop" in name]
    assert "1 no-op nuisances left out (Up = Down = nominal in every bin of every process): CMS_noop" in caplog.text
    assert json.loads((card.parent / "template_changes.json").read_text())["no_ops"] == ["CMS_noop"]

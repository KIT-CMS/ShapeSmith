"""Column variations end to end on the mini dataset with CROWN shifts: skim checks and reuse, fill rule, estimates."""
import dataclasses
import json
import shutil

import numpy as np
import pytest
import uproot

from shapesmith.config import FriendConfig, NtupleConfig, RunConfig
from shapesmith.estimates import run_estimates
from shapesmith.fill import bookings, run_hist
from shapesmith.histogram import INCLUSIVE, HistKey
from shapesmith.model import ColumnVariation
from shapesmith.skim import run_skim
from shapesmith.store import read_manifest, read_skims
from shapesmith.testing import make_mini_dataset, make_ntuple
from tests.mini_analysis import build_embedding
from tests.test_ntuples import TREE_WITHOUT_BRANCHES

TES = ("CMS_tesUp", "CMS_tesDown")
FF = ("CMS_ffStatUp", "CMS_ffStatDown")


@pytest.fixture
def config(tmp_path):
    make_mini_dataset(tmp_path / "data", n_files=2, n_events=300, variations=True)
    return RunConfig(
        analysis="tests.mini_analysis:build_embedding",
        era="2018",
        channels=["mt"],
        ntuples=NtupleConfig(base=str(tmp_path / "data" / "CROWNRun"), friends=[FriendConfig(base=str(tmp_path / "data" / "CROWNFriends" / "nn"))]),
        skim_dir=tmp_path / "skims",
        output_dir=tmp_path / "out",
        workers=1,
    )


def _analysis(skim_cut=None, keep=TES + FF):
    analysis = build_embedding()
    channel = analysis.channel("mt")
    variations = tuple(v for v in channel.variations if not isinstance(v, ColumnVariation) or v.name in keep)
    skim = {**channel.skim, "mass": skim_cut} if skim_cut else channel.skim
    return dataclasses.replace(analysis, channels={"mt": dataclasses.replace(channel, variations=variations, skim=skim)})


def _manifest(config, nick):
    return read_manifest(config.skim_dir / "mt" / nick / "manifest.json")


def test_skim_reads_the_shifted_columns_of_each_kind(config):
    run_skim(config, _analysis())
    emb = read_skims(config.skim_dir, "mt", ["EMB_A"], None)
    data = read_skims(config.skim_dir, "mt", ["DATA_A"], None)
    assert {"m_vis__tesUp", "m_vis__tesDown", "fake_factor__ffStatUp"} <= set(emb.columns)
    assert "fake_factor__ffStatUp" in data and "m_vis__tesUp" not in data  # tes does not apply to data
    assert _manifest(config, "EMB_A")["contract"]["column_variations"] == {"CMS_tesUp": "__tesUp", "CMS_tesDown": "__tesDown", "CMS_ffStatUp": "__ffStatUp", "CMS_ffStatDown": "__ffStatDown"}
    assert _manifest(config, "DATA_A")["contract"]["column_variations"] == {"CMS_ffStatUp": "__ffStatUp", "CMS_ffStatDown": "__ffStatDown"}


def test_an_event_that_passes_only_shifted_is_kept_and_filled_only_shifted(config):
    analysis = _analysis(skim_cut="m_vis > 100")
    run_skim(config, analysis)
    emb = read_skims(config.skim_dir, "mt", ["EMB_A"], None)
    only_shifted = (emb["m_vis"] <= 100) & (emb["m_vis__tesUp"] > 100)
    assert only_shifted.any() and ((emb["m_vis"] > 100) | only_shifted).all()
    hset = run_hist(config, analysis, ["mt"], True, None, True, ["EMB"], config.output_dir / "control.root")
    nominal = hset[HistKey("mt", INCLUSIVE, "EMB", "nominal", "Nominal", "m_vis")]
    up = hset[HistKey("mt", INCLUSIVE, "EMB", "nominal", "CMS_tesUp", "m_vis")]
    base = (emb["veto"] < 0.5) & ((emb["q_1"] * emb["q_2"]) < 0) & (emb["id_tight"] > 0.5) & (emb["gen_match"] == 5)
    weights = emb["emb_genweight"].astype(float) * emb["trg_wgt"].astype(float)
    assert nominal.sum() == pytest.approx(weights[base & (emb["m_vis"] > 100)].sum())  # the skim cut is re-applied nominally
    assert up.sum() == pytest.approx(weights[base & (emb["m_vis__tesUp"] > 100) & (emb["m_vis__tesUp"] <= 200)].sum())  # 200: last edge


def test_a_file_without_events_and_its_friend_without_branches_are_skimmed(config, tmp_path):
    """CROWN writes the friend of a main file without events as a tree without branches: no columns and no shifted
    branches to check, so the file adds an empty skim."""
    data = tmp_path / "data"
    first = uproot.open(data / "CROWNRun" / "2018" / "DATA_A" / "mt" / "DATA_A_0.root")
    columns = {name: array[:0] for name, array in first["ntuple"].arrays(library="np").items()}
    make_ntuple(data / "CROWNRun" / "2018" / "DATA_A" / "mt" / "DATA_A_2.root", columns, json.loads(str(first["metadata"])))
    shutil.copy(TREE_WITHOUT_BRANCHES, data / "CROWNFriends" / "nn" / "2018" / "DATA_A" / "mt" / "DATA_A_2.root")
    run_skim(config, _analysis())
    assert "DATA_A_2.root" in _manifest(config, "DATA_A")["completed"]
    assert "fake_factor__ffStatUp" in read_skims(config.skim_dir, "mt", ["DATA_A"], None)


def test_a_declared_shift_without_branches_fails_the_skim(config):
    analysis = build_embedding()
    channel = analysis.channel("mt")
    jes = tuple(ColumnVariation(f"CMS_jes{d}", f"__jes{d}", applies_to=("mc",)) for d in ("Up", "Down"))
    analysis = dataclasses.replace(analysis, channels={"mt": dataclasses.replace(channel, variations=channel.variations + jes)})
    with pytest.raises(RuntimeError, match=r"declared shift CMS_jesUp \(__jesUp\) has no shifted branch"):
        run_skim(config, analysis)


def test_a_shift_restricted_to_groups_belongs_to_their_samples_only(config):
    analysis = build_embedding()
    jes = tuple(ColumnVariation(f"CMS_jes{d}", f"__jes{d}", groups=("HH",)) for d in ("Up", "Down"))
    channel = dataclasses.replace(analysis.channel("mt"), variations=analysis.channel("mt").variations + jes)
    assert [b.process.name for b in bookings(channel, True) if jes[0] in b.variations] == ["HH"]
    with pytest.raises(RuntimeError, match=r"declared shift CMS_jesUp \(__jesUp\) has no shifted branch") as excinfo:
        run_skim(config, dataclasses.replace(analysis, channels={"mt": channel}))
    assert "SIG_1" in str(excinfo.value) and "ZTT_1" not in str(excinfo.value)


def test_an_undeclared_shift_branch_fails_the_skim_of_kinds_with_declared_shifts(config):
    with pytest.raises(RuntimeError, match=r"ZTT_1_0.root: undeclared shift __ffStatDown; undeclared shift __ffStatUp") as excinfo:
        run_skim(config, _analysis(keep=TES))
    assert "DATA_A" not in str(excinfo.value)  # no shift declared for data: its fake_factor__ffStat branches are not checked


def test_a_superset_skim_serves_fewer_variations(config):
    run_skim(config, _analysis())
    assert all(r.skipped for r in run_skim(config, _analysis(keep=FF)))
    assert all(r.skipped for r in run_skim(config, _analysis(keep=())))


def test_a_missing_variation_makes_the_skim_incompatible(config):
    run_skim(config, _analysis(keep=()))  # without declared shifts, the shifted branches are not checked
    with pytest.raises(ValueError, match=r"mt/EMB_A: column variations missing: CMS_ffStatDown, CMS_ffStatUp, CMS_tesDown, CMS_tesUp"):
        run_skim(config, _analysis())


def test_an_incomplete_superset_skim_is_skimmed_again(config):
    run_skim(config, _analysis())
    (config.skim_dir / "mt" / "EMB_A" / "EMB_A_1.parquet").unlink()
    results = run_skim(config, _analysis(keep=()))
    assert sorted(r.basename for r in results if not r.skipped) == ["EMB_A_0.root", "EMB_A_1.root"]
    assert "column_variations" not in _manifest(config, "EMB_A")["contract"]


def test_friends_are_part_of_the_contract(config, tmp_path):
    run_skim(config, _analysis())
    assert _manifest(config, "ZTT_1")["contract"]["friends"] == [str(tmp_path / "data" / "CROWNFriends" / "nn")]
    shutil.copytree(tmp_path / "data" / "CROWNFriends" / "nn", tmp_path / "data" / "CROWNFriends" / "nn_v2")
    mc_only = FriendConfig(base=str(tmp_path / "data" / "CROWNFriends" / "nn_v2"), applies_to=("mc",))
    changed = config.model_copy(update={"ntuples": config.ntuples.model_copy(update={"friends": [*config.ntuples.friends, mc_only]})})
    with pytest.raises(ValueError, match=r"incompatible") as excinfo:
        run_skim(changed, _analysis())
    assert "mt/ZTT_1: friends missing" in str(excinfo.value) and "DATA_A" not in str(excinfo.value) and "EMB_A" not in str(excinfo.value)


def test_the_auxiliary_process_is_booked_nominal_only_without_variations():
    by_process = {b.process.name: b for b in bookings(_analysis().channel("mt"), systematics=True)}
    assert (by_process["ZTT"].regions, by_process["ZTT"].variations) == (("nominal",), ())
    assert by_process["EMB"].regions == ("nominal", "anti_iso") and {v.name for v in by_process["EMB"].variations} == set(TES + FF)


def test_column_variations_are_filled_only_where_they_change_an_expression(config):
    analysis = _analysis()
    run_skim(config, analysis)
    hset = run_hist(config, analysis, ["mt"], True, None, True, None, config.output_dir / "control.root")
    categories = run_hist(config, analysis, ["mt"], False, None, True, None, config.output_dir / "shapes.root")

    def variations(hs, process, region):
        return {k.variation for k in hs.select(process=process, region=region)} - {"Nominal"}

    assert variations(hset, "EMB", "nominal") == set(TES)  # the ff weight exists only in anti_iso, where ffStat is filled
    assert variations(hset, "EMB", "anti_iso") == set(TES + FF)
    assert variations(hset, "data", "anti_iso") == set(FF) and variations(hset, "data", "nominal") == set()
    assert variations(hset, "ZL", "nominal") == {"CMS_puUp", "CMS_puDown", *TES}
    assert variations(categories, "EMB", "nominal") == set()  # score and cls carry no tes shift
    emb = read_skims(config.skim_dir, "mt", ["EMB_A"], None)
    selected = emb[(emb["veto"] < 0.5) & ((emb["q_1"] * emb["q_2"]) < 0) & (emb["id_tight"] > 0.5) & (emb["gen_match"] == 5)]
    expected, _ = np.histogram(selected["m_vis__tesUp"], bins=[0.0, 50.0, 100.0, 200.0], weights=selected["emb_genweight"].astype(float) * selected["trg_wgt"].astype(float))
    assert np.allclose(hset[HistKey("mt", INCLUSIVE, "EMB", "nominal", "CMS_tesUp", "m_vis")].values, expected)


def test_a_process_cut_on_a_shifted_column_is_rewritten(config):
    analysis = _analysis()
    channel = analysis.channel("mt")
    zl = channel.process("ZL")
    zl = dataclasses.replace(zl, selection=dataclasses.replace(zl.selection, cuts={**zl.selection.cuts, "mass": "m_vis > 60"}))
    analysis = dataclasses.replace(analysis, channels={"mt": dataclasses.replace(channel, processes=tuple(zl if p.name == "ZL" else p for p in channel.processes))})
    run_skim(config, analysis)
    hset = run_hist(config, analysis, ["mt"], False, None, True, ["ZL"], config.output_dir / "shapes.root")
    mc = read_skims(config.skim_dir, "mt", ["ZTT_1"], None)
    base = (mc["veto"] < 0.5) & ((mc["q_1"] * mc["q_2"]) < 0) & (mc["id_tight"] > 0.5) & (mc["gen_match"] != 5) & (mc["cls"] == 0)
    weights = mc["norm_weight"] * 10.0 * mc["trg_wgt"].astype(float) * mc["puweight"].astype(float)
    assert hset[HistKey("mt", "sig", "ZL", "nominal", "CMS_tesUp", "score")].sum() == pytest.approx(weights[base & (mc["m_vis__tesUp"] > 60)].sum())
    assert hset[HistKey("mt", "sig", "ZL", "nominal", "Nominal", "score")].sum() == pytest.approx(weights[base & (mc["m_vis"] > 60)].sum())


def test_jet_fakes_carry_the_column_variations_of_their_inputs(config):
    analysis = _analysis()
    run_skim(config, analysis)
    hset = run_hist(config, analysis, ["mt"], True, None, True, None, config.output_dir / "control.root")
    run_estimates(hset, analysis, ["mt"])
    assert {k.variation for k in hset.select(process="jetFakes")} == {"Nominal", *TES, *FF}
    key = HistKey("mt", INCLUSIVE, "jetFakes", "nominal", "CMS_tesUp", "m_vis")
    data = hset[HistKey("mt", INCLUSIVE, "data", "anti_iso", "Nominal", "m_vis")].values
    emb = hset[HistKey("mt", INCLUSIVE, "EMB", "anti_iso", "CMS_tesUp", "m_vis")].values
    zl = hset[HistKey("mt", INCLUSIVE, "ZL", "anti_iso", "CMS_tesUp", "m_vis")].values
    assert np.allclose(hset[key].values, data - emb - zl)  # data has no tes shift: its nominal enters
    assert {k.variation for k in hset.select(process="EMB", region="nominal")} >= {"CMS_htt_emb_ttbar_2018Up", "CMS_htt_emb_ttbar_2018Down"}

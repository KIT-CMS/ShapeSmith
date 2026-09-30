"""The measurement against the predecessor's 2018 outputs (jvoss's smhtt_ul area); skipped where they are absent."""
import filecmp
import json
from pathlib import Path

import pytest
import uproot

from shapesmith import cmssw, payloads
from shapesmith.config import CombineConfig
from shapesmith.histogram import on_template
from shapesmith.measurements.tau_id_es import combine
from shapesmith.measurements.tau_id_es.grid import CATEGORIES, grid, grid_name
from shapesmith.measurements.tau_id_es.payload import correction_set
from shapesmith.measurements.tau_id_es.synced import shapes_path, signal_name

AREA = Path("/work/jvoss/smhtt_ul_SFs_v15")
CMSSW = AREA / "CMSSW_14_1_0_pre4"
WPS = {"M_VVL": ("Medium", "VVLoose"), "M_T": ("Medium", "Tight"), "T_VVL": ("Tight", "VVLoose"), "T_T": ("Tight", "Tight")}
GRID = grid(-200, 200, 2)
pytestmark = pytest.mark.skipif(not AREA.exists(), reason="needs the predecessor's area /work/jvoss/smhtt_ul_SFs_v15")


def _cards(tag, category):
    vsjet, vsele = WPS[tag]
    return AREA / "output" / "datacards" / "SFs_EMB_Run2_04_08_26__full" / f"{tag}_18_full" / "2018" / vsjet / vsele / f"htt_mt_{category}"


def _synced(tag):
    vsjet, vsele = WPS[tag]
    return AREA / "output" / "shapes_synced" / "SFs_EMB_Run2_04_08_26__full" / f"2018-{tag}_18_full" / vsjet / vsele


def test_the_payload_from_the_predecessor_fits_is_the_predecessor_payload():
    results = {WPS[tag]: {c: combine.read_singles(_cards(tag, c) / f"higgsCombine.comb_sep_fit_{tag}_18_full_{c}.MultiDimFit.mH125.root", c) for c in CATEGORIES} for tag in WPS}
    ours = json.loads(payloads.dumps(correction_set(results, {})))
    reference = json.loads((AREA / "Tau_SFs_DT2p5" / "DeepTau2018v2p5_id_es_embedding2018UL.json").read_text())
    assert ours["corrections"] == reference["corrections"]


def test_signal_names_are_those_of_the_predecessor_shapes():
    with uproot.open(shapes_path(_synced("T_VVL"), "mt", "2018")) as f:
        keys = {key.split("/", 1)[1] for key in f.keys(cycle=False) if key.startswith("mt_DM0/EMB_")}
    grid_points = [grid_name(shift) for shift in GRID]
    ours = {signal_name("DM0", variation) for variation in ("Nominal", *grid_points)}
    for direction in ("Up", "Down"):
        systematic = f"CMS_emb_ttbar_contamination_Run2018{direction}"
        ours |= {signal_name("DM0", systematic)} | {signal_name("DM0", on_template(systematic, point)) for point in grid_points}
    assert ours == keys


@pytest.mark.skipif(not CMSSW.exists(), reason="needs the predecessor's CMSSW area")
def test_morphing_datacards_equal_the_predecessor_cards(tmp_path):
    category = "DM1_PT20_40"
    cmssw.run([combine.morphing_command(category, _synced("T_T"), "2018", GRID) + " > morphing.log 2>&1"], CombineConfig(cmssw_dir=str(CMSSW)), tmp_path)
    cards = combine.card_dir(tmp_path, category)
    combine.add_rate_parameters(cards, category)
    reference = sorted(_cards("T_T", category).glob("htt_*_Run2018.txt"))
    assert len(reference) == 2 and all(filecmp.cmp(cards / card.name, card, shallow=False) for card in reference)

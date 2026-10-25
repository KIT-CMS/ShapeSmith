"""G2: the mini-dataset outputs must stay bitwise identical to the ones frozen with core aa5daa5.

The reference files in tests/golden/ were written once by make_golden_aa5daa5.py against that core; the script is
kept for the record and does not run against the current API.
"""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import uproot

from shapesmith.config import FriendConfig, NtupleConfig, RunConfig
from shapesmith.datacards import run_datacards
from shapesmith.estimates import run_estimates
from shapesmith.fill import run_hist
from shapesmith.ml_export import run_ml_export
from shapesmith.model import ABCD, Region
from shapesmith.skim import run_skim
from shapesmith.store import read_manifest, write_manifest
from shapesmith.testing import make_mini_dataset
from tests.mini_analysis import build

GOLDEN = Path(__file__).parent / "golden"


def _digest(hset) -> dict:
    return {key.path: [h.edges.tolist(), h.values.tolist(), h.variances.tolist()] for key, h in hset.items()}


def _abcd_variant(analysis):
    channel = analysis.channel("mt")
    anti = "(id_tight < 0.5) & (id_loose > 0.5)"
    regions = channel.regions + (
        Region("abcd_anti_iso", replace_cuts={"tau_iso": anti}),
        Region("abcd_same_sign_anti_iso", replace_cuts={"os": "(q_1 * q_2) > 0", "tau_iso": anti}),
    )
    estimators = (ABCD("QCD", "abcd_anti_iso", "same_sign", "abcd_same_sign_anti_iso", ("ZTT", "ZL")),)
    return dataclasses.replace(analysis, channels={"mt": dataclasses.replace(channel, regions=regions, estimators=estimators)})


def _produce(tmp: Path) -> tuple[dict, str]:
    make_mini_dataset(tmp / "data", n_files=2, n_events=400)
    config = RunConfig(
        analysis="tests.mini_analysis:build", era="2018", channels=["mt"], switches={"jet_fakes": "ff", "embedding": False},
        ntuples=NtupleConfig(base=str(tmp / "data" / "CROWNRun"), friends=[FriendConfig(base=str(tmp / "data" / "CROWNFriends" / "nn"))]),
        skim_dir=tmp / "skims", output_dir=tmp / "out", ml_dir=tmp / "ml", workers=1,
    )
    analysis = build(config)
    run_skim(config, analysis)
    result = {}
    control = run_hist(config, analysis, ["mt"], True, None, True, None, tmp / "out" / "control_shapes.root")
    result["control_hist"] = _digest(control)
    run_estimates(control, analysis, ["mt"])
    result["control_estimate"] = _digest(control)
    regions = run_hist(config, analysis, ["mt"], True, None, True, None, tmp / "out" / "regions.root", regions=["all"])
    result["control_regions_all"] = _digest(regions)
    shapes = run_hist(config, analysis, ["mt"], False, None, True, None, tmp / "out" / "shapes.root")
    result["shapes_hist"] = _digest(shapes)
    run_estimates(shapes, analysis, ["mt"])
    result["shapes_estimate"] = _digest(shapes)
    card = run_datacards(shapes, analysis, {"mt": ["mt"]}, tmp / "out" / "datacards", True, 1.0)["mt"].read_text()
    with uproot.open(tmp / "out" / "datacards" / "mt" / "common" / "htt_input_2018.root") as f:
        result["datacard_shapes"] = {
            name: [f[name].axis().edges().tolist(), f[name].values().tolist(), f[name].variances().tolist()]
            for name, cls in sorted(f.classnames(recursive=True, cycle=False).items())
            if cls.startswith("TH1")
        }
    abcd = _abcd_variant(analysis)
    abcd_hset = run_hist(config, abcd, ["mt"], True, None, False, None, tmp / "out" / "abcd.root")
    run_estimates(abcd_hset, abcd, ["mt"])
    result["abcd_control_estimate"] = _digest(abcd_hset)
    run_ml_export(config, analysis, ["mt"])
    result["ml_export"] = _legacy_folds(pd.read_feather(config.ml_dir / "mt" / "events.feather"), analysis.ml.classes)
    result["reused_old_contracts"] = _reuse_old_contracts(config, analysis)
    return result, card


def _legacy_folds(frame: pd.DataFrame, classes: tuple[str, ...]) -> dict:
    """The six fold files of the export of aa5daa5 (five-level columns, labels "is_<class>" in alphabetical order,
    Event.id counting the rows of each process, class weights per file), rebuilt from the flat export without its
    subtraction rows: fold k = event % 2 == (k + 1) % 2, its training part = thesis_training."""
    frame = frame[~frame["is_subtraction"]].reset_index(drop=True)
    weight = frame["weight"].to_numpy().astype(np.float32)
    columns = {"Event|event|||": frame["event"].to_numpy(), "Event|id|||": frame.groupby("process", observed=True).cumcount().to_numpy()}
    columns |= {f"Labels|is_{name}|||": (frame["class_index"].to_numpy() == classes.index(name)).astype(np.int32) for name in sorted(classes)}
    columns |= {f"Nominal|variables|{v}||": frame[v].to_numpy() for v in ("m_vis", "score")}
    columns |= {"Nominal|weight|||": weight, "Nominal|cut|||": np.ones(len(frame), dtype=np.float32)}
    folds = {}
    for k in (0, 1):
        in_fold = frame["event"].to_numpy() % 2 == (k + 1) % 2
        for name, selected in ((f"fold{k}", in_fold), (f"fold{k}_training", in_fold & frame["thesis_training"].to_numpy()), (f"fold{k}_validation", in_fold & ~frame["thesis_training"].to_numpy())):
            part = {column: values[selected] for column, values in columns.items()}
            labels = np.stack([part[f"Labels|is_{c}|||"] for c in sorted(classes)], axis=1).argmax(axis=1)
            part["Nominal|class_weight|||"] = class_weights(part["Nominal|weight|||"].astype(np.float64), labels).astype(np.float32)
            folds[f"{name}.feather"] = {column: np.asarray(values).astype(np.float64).tolist() for column, values in part.items()}
    return folds


def class_weights(weights: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The class weights of aa5daa5: weight * (sum of all weights) / (sum of weights of the event's class)."""
    result = np.zeros_like(weights, dtype=np.float64)
    for label in np.unique(labels):
        result[labels == label] = weights.sum() / weights[labels == label].sum()
    return result * weights


def _reuse_old_contracts(config, analysis) -> bool:
    """Skims whose manifests carry the contracts written by aa5daa5 (no friends recorded) are reused without
    --force when no friend is configured, as for the v4 skims; a configured friend makes them incompatible."""
    for nick, contract in json.loads((GOLDEN / "mini_golden.json").read_text())["skim_contracts"].items():
        path = config.skim_dir / "mt" / nick / "manifest.json"
        write_manifest(path, {**read_manifest(path), "contract": contract})
    with pytest.raises(ValueError, match="friends missing"):
        run_skim(config, analysis)
    without_friends = config.model_copy(update={"ntuples": config.ntuples.model_copy(update={"friends": []})})
    return all(r.skipped for r in run_skim(without_friends, analysis))


def _bitwise_equal(a, b) -> bool:
    return np.asarray(a, dtype=np.float64).tobytes() == np.asarray(b, dtype=np.float64).tobytes()


@pytest.fixture(scope="module")
def produced(tmp_path_factory):
    return _produce(tmp_path_factory.mktemp("golden"))


@pytest.mark.parametrize("part", ["control_hist", "control_estimate", "control_regions_all", "shapes_hist", "shapes_estimate", "datacard_shapes", "abcd_control_estimate"])
def test_histograms_match_the_golden_outputs(produced, part):
    golden = json.loads((GOLDEN / "mini_golden.json").read_text())[part]
    result = produced[0][part]
    assert sorted(result) == sorted(golden)
    different = [name for name in golden if not all(_bitwise_equal(r, g) for r, g in zip(result[name], golden[name]))]
    assert not different


def test_skims_with_the_old_contracts_are_reused(produced):
    assert produced[0]["reused_old_contracts"]


def test_datacard_text_matches_the_golden_card(produced):
    assert produced[1] == (GOLDEN / "datacard_mt.txt").read_text()


def test_ml_export_matches_the_golden_frames(produced):
    """The flat export holds the rows, variables and weights of the fold files of aa5daa5 bit by bit."""
    golden = json.loads((GOLDEN / "mini_golden.json").read_text())["ml_export"]
    result = produced[0]["ml_export"]
    assert sorted(result) == sorted(golden)
    for fold, columns in golden.items():
        assert sorted(result[fold]) == sorted(columns)
        assert all(_bitwise_equal(result[fold][name], values) for name, values in columns.items()), fold

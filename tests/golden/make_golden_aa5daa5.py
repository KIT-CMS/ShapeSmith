"""G2: golden outputs of the core mini dataset, frozen with core aa5daa5 (run once; needs that core on PYTHONPATH).

Writes <out>/mini_golden.json (every histogram as exact floats: edges, values, variances), the datacard text
<out>/datacard_mt.txt, the ML-export frames and the skim contracts of the old manifests.
"""
import dataclasses
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from shapesmith.config import FriendConfig, NtupleConfig, RunConfig
from shapesmith.datacards import run_datacards
from shapesmith.estimators import run_estimate
from shapesmith.histograms import run_hist
from shapesmith.ml_export import run_ml_export
from shapesmith.model import Estimator, Region
from shapesmith.skim import run_skim
from shapesmith.testing import make_mini_dataset
from tests.mini_analysis import build


def digest(hset) -> dict:
    return {key.path: [h.edges.tolist(), h.values.tolist(), h.variances.tolist()] for key, h in hset.items()}


def abcd_variant(analysis):
    channel = analysis.channel("mt")
    anti = "(id_tight < 0.5) & (id_loose > 0.5)"
    regions = channel.regions + (
        Region("abcd_anti_iso", replace_cuts={"tau_iso": anti}),
        Region("abcd_same_sign_anti_iso", replace_cuts={"os": "(q_1 * q_2) > 0", "tau_iso": anti}),
    )
    estimator = Estimator("abcd", {"B": "abcd_anti_iso", "C": "same_sign", "D": "abcd_same_sign_anti_iso"}, ("ZTT", "ZL"), "QCD")
    return dataclasses.replace(analysis, channels={"mt": dataclasses.replace(channel, regions=regions)}, estimator=estimator)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    make_mini_dataset(tmp / "data", n_files=2, n_events=400)
    config = RunConfig(
        analysis="tests.mini_analysis:build", era="2018", channels=["mt"], switches={"jet_fakes": "ff", "embedding": False},
        ntuples=NtupleConfig(base=str(tmp / "data" / "CROWNRun"), friends=[FriendConfig(base=str(tmp / "data" / "CROWNFriends" / "nn"))]),
        skim_dir=tmp / "skims", output_dir=tmp / "out", ml_dir=tmp / "ml", workers=1,
    )
    analysis = build(config)
    run_skim(config, analysis)
    golden = {}
    control = run_hist(config, analysis, ["mt"], True, None, True, None, tmp / "out" / "control_shapes.root", force=True)
    golden["control_hist"] = digest(control)
    run_estimate(control, analysis, ["mt"])
    golden["control_estimate"] = digest(control)
    regions = run_hist(config, analysis, ["mt"], True, None, True, None, tmp / "out" / "regions.root", force=True, regions=["all"])
    golden["control_regions_all"] = digest(regions)
    shapes = run_hist(config, analysis, ["mt"], False, None, True, None, tmp / "out" / "shapes.root", force=True)
    golden["shapes_hist"] = digest(shapes)
    run_estimate(shapes, analysis, ["mt"])
    golden["shapes_estimate"] = digest(shapes)
    cards = run_datacards(shapes, analysis, {"mt": ["mt"]}, tmp / "out" / "datacards", True, 1.0)
    (out / "datacard_mt.txt").write_text(cards["mt"].read_text())
    with uproot.open(tmp / "out" / "datacards" / "mt" / "common" / "htt_input_2018.root") as f:
        golden["datacard_shapes"] = {name: [f[name].axis().edges().tolist(), f[name].values().tolist(), f[name].variances().tolist()] for name, cls in sorted(f.classnames(recursive=True, cycle=False).items()) if cls.startswith("TH1")}
    abcd = abcd_variant(analysis)
    abcd_hset = run_hist(config, abcd, ["mt"], True, None, False, None, tmp / "out" / "abcd.root", force=True)
    run_estimate(abcd_hset, abcd, ["mt"])
    golden["abcd_control_estimate"] = digest(abcd_hset)
    ml = {}
    for path in run_ml_export(config, analysis, ["mt"]):
        frame = pd.read_feather(path)
        ml[path.name] = {"|".join(column): np.asarray(frame[column]).astype(np.float64).tolist() for column in frame.columns}
    golden["ml_export"] = ml
    golden["skim_contracts"] = {p.parent.name: json.loads(p.read_text())["contract"] for p in sorted((tmp / "skims" / "mt").glob("*/manifest.json"))}
    (out / "mini_golden.json").write_text(json.dumps(golden, indent=0, sort_keys=True))
    print({name: len(value) for name, value in golden.items()})


if __name__ == "__main__":
    main(Path(sys.argv[1]))

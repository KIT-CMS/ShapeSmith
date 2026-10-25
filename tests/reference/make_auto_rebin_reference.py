"""Write auto_rebin_combineharvester.json: the reference of the AutoRebin replica in tests/test_nn_binning.py, computed
once with CombineHarvester (run manually in a CMSSW area with CombineHarvester, e.g. the one of the thesis cards):

    cd /work/sdaigler/smhtt_ul/CMSSW_14_1_9/src && cmsenv
    python3 tests/reference/make_auto_rebin_reference.py tests/reference/auto_rebin_combineharvester.json

Every case is a TH1F total background (contents float32, Sumw2 double) passed with its edges to
ch::AutoRebin::FindNewBinning(total, edges, 1.0, 0.9, 1), the call of AutoRebin::Rebin with the settings of
HHDatacards/hh_datacards.py (SetBinThreshold(1.0), SetBinUncertFraction(0.9), SetRebinMode(1)); the contents are stored
as ROOT holds them. Random cases (seed 11) of 3-25 bins mix large, small, zero, negative and poorly known bins; a few
hand-made cases add the thesis start edges.
"""
import array
import json
import sys

import numpy as np
import ROOT

ROOT.gSystem.Load("libCombineHarvesterCombineTools")
ROOT.gInterpreter.Declare('#include "CombineHarvester/CombineTools/interface/AutoRebin.h"')
ROOT.TH1.AddDirectory(False)
THESIS_HH_EDGES = [0.0, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]


def find_new_binning(edges, values, sumw2, threshold=1.0, fraction=0.9):
    total = ROOT.TH1F("total", "", len(edges) - 1, array.array("d", edges))
    total.Sumw2()
    for i, (value, w2) in enumerate(zip(values, sumw2), start=1):
        total.SetBinContent(i, float(value))
        total.GetSumw2()[i] = float(w2)
    contents = [total.GetBinContent(i) for i in range(1, len(edges))]
    new = ROOT.std.vector("double")()
    for edge in edges:
        new.push_back(float(edge))
    ROOT.ch.AutoRebin().FindNewBinning(total, new, threshold, fraction, 1)
    return {"edges": list(edges), "values": contents, "sumw2": [float(w) for w in sumw2], "result": list(new)}


def random_cases(n_cases=300):
    rng = np.random.default_rng(11)
    cases = []
    for _ in range(n_cases):
        n = int(rng.integers(3, 26))
        edges = [round(float(x), 6) for x in np.linspace(0.0, 1.0, n + 1)]
        scale = float(rng.choice([0.3, 2.0, 20.0]))
        values = rng.exponential(scale, n) * rng.choice([1.0, 0.0, -0.05], n, p=[0.85, 0.1, 0.05])
        weights = rng.choice([0.05, 0.5, 3.0], n)
        sumw2 = np.abs(values) * weights + rng.uniform(0.0, 0.2, n) * (values != 0)
        cases.append(find_new_binning(edges, values.tolist(), sumw2.tolist()))
    return cases


def thesis_cases():
    rng = np.random.default_rng(12)
    cases = []
    for _ in range(20):
        n = len(THESIS_HH_EDGES) - 1
        values = 50.0 * np.exp(-6.0 * np.asarray(THESIS_HH_EDGES[:-1])) * rng.uniform(0.2, 1.5, n) * np.diff(THESIS_HH_EDGES) * 10
        sumw2 = values * rng.choice([0.1, 1.0, 5.0], n)
        cases.append(find_new_binning(THESIS_HH_EDGES, values.tolist(), sumw2.tolist()))
    return cases


if __name__ == "__main__":
    output = sys.argv[1]
    with open(output, "w") as handle:
        json.dump({"random": random_cases(), "thesis": thesis_cases()}, handle, indent=0)
    print(f"wrote {output}")

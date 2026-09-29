"""Write smoothing_root.json: the reference of tests/test_smoothing.py, computed once with ROOT (run manually).

Needs pyROOT (LCG_108: `source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh`) and jvoss's
TauFakeFactors copy /work/jvoss/FF_Updated, whose `_smooth_function` produced the SM 2018 fake-factor payload:

    python tests/reference/make_smoothing_reference.py

- `kernel`: ROOT TGraphSmooth::SmoothKern(graph, "normal", bandwidth, nout, xout) directly, including gaps, points at
  the cutoff and unsorted input. The TGraphSmooth object owns the output graph, so it is kept alive while reading.
- `smooth`: FF_Updated `_smooth_function` on histograms with centre-of-mass bin positions, for every fit option.
"""
import array
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import ROOT

FF_UPDATED = Path("/work/jvoss/FF_Updated")
CUTOFF_1 = 4 * 0.3706506  # the kernel cutoff at bandwidth 1


def root_smooth_kern(x, y, x_out, bandwidth, nout):
    graph = ROOT.TGraph(len(x), array.array("d", x), array.array("d", y))
    smoother = ROOT.TGraphSmooth("normal")
    out = smoother.SmoothKern(graph, "normal", float(bandwidth), int(nout), array.array("d", x_out))
    return [out.GetPointX(i) for i in range(out.GetN())], [out.GetPointY(i) for i in range(out.GetN())]


def kernel_cases():
    rng = np.random.default_rng(7)
    cases = []
    for _ in range(12):  # the TauFakeFactors use: 100 n grid points on [x_0, x_n-1], the last one dropped
        n = int(rng.integers(2, 9))
        x = np.sort(rng.uniform(0, 150, n))
        cases.append((x, rng.uniform(0, 1, n), np.linspace(x[0], x[-1], 100 * n + 1), float(rng.choice([0.5, 3, 12, 20, 50])), 100 * n))
    cases += [
        ([0.0, 10.0], [1.0, 2.0], [0.0, 2.0, 5.0, 8.0, 9.9], 1.0, 5),  # a gap starting at a data point
        ([0.0, 10.0], [1.0, 2.0], [5.0], 1.0, 1),  # an isolated output point in the gap
        ([0.0, 100.0], [1.0, 2.0], np.linspace(0.0, 100.0, 11), 0.5, 11),  # exp underflow in a wide gap
        ([0.0, 1.0, 2.0, 40.0, 41.0, 42.0], [1.0, 2.0, 3.0, 2.0, 1.0, 2.0], np.linspace(20.0, 42.0, 23), 2.0, 22),  # a gap in the first window
        ([0.0, 100.0], [3.0, 5.0], [CUTOFF_1], 1.0, 1),  # a point exactly at x0 - cutoff
        ([0.0, 100.0], [3.0, 5.0], [-CUTOFF_1], 1.0, 1),  # a point exactly at x0 + cutoff
        ([35.0, 10.0, 60.0, 20.0], [1.1, 1.3, 0.9, 1.2], np.linspace(10.0, 60.0, 201), 12.0, 200),  # unsorted input
    ]
    result = []
    for x, y, x_out, bandwidth, nout in cases:
        x_root, y_root = root_smooth_kern(x, y, x_out, bandwidth, nout)
        result.append({"x": list(map(float, x)), "y": list(map(float, y)), "bandwidth": bandwidth, "x_out": x_root, "y_out": y_root})
    return result


def smooth_cases():
    wurlitzer = types.ModuleType("wurlitzer")  # imported by ff_functions for log capturing only
    wurlitzer.STDOUT, wurlitzer.pipes = None, None
    sys.modules["wurlitzer"] = wurlitzer
    sys.path.insert(0, str(FF_UPDATED))
    from helper.ff_functions import _smooth_function
    from helper.hooks_and_patches import _EXTRA_PARAM_FLAG, _EXTRA_PARAM_MEANS

    histograms = [  # (bin edges, centres of mass, values, errors, bandwidth)
        ([30.0, 33.53, 38.0, 44.89, 58.39, 150.0], [31.7, 35.6, 41.2, 50.9, 72.5], [0.21, 0.19, 0.17, 0.15, 0.13], [0.01, 0.01, 0.012, 0.015, 0.02], 20.0),
        ([30.0, 40.0, 60.0, 90.0, 150.0], [34.8, 48.1, 71.3, 105.0], [1.1, 0.95, 1.02, 0.7], [0.05, 0.06, 0.1, 0.4], 12.0),
        ([0.0, 1.0, 2.0, 5.0, 10.0, 11.0], [0.5, 1.4, 3.2, 6.8, 10.5], [1.2, 0.9, 1.05, 0.02, 1.4], [0.3, 0.2, 0.25, 0.3, 0.9], 1.0),  # gaps, negative samples
    ]
    options = ["smoothed", "binwise", "binwise#[0]+smoothed", "binwise#[-1]+smoothed", "binwise#[0,]#[-1,]+smoothed"]
    result = []
    for edges, x, y, errors, bandwidth in histograms:
        for option in options:
            hist = ROOT.TH1D(f"h{len(result)}", "", len(y), array.array("d", edges))
            for i, (value, error) in enumerate(zip(y, errors)):
                hist.SetBinContent(i + 1, value)
                hist.SetBinError(i + 1, error)
            setattr(hist, _EXTRA_PARAM_FLAG, True)
            setattr(hist, _EXTRA_PARAM_MEANS, list(x))
            _, curve = _smooth_function(hist, edges, option, bandwidth, stat_sigma=1.0)
            default = curve["default"]
            result.append({
                "bin_edges": edges, "x": x, "y": y, "errors": errors, "option": option, "bandwidth": bandwidth,
                "edges": list(map(float, default["edges"])), "nominal": list(map(float, default["nominal"])),
                "up": list(map(float, default["variations"]["StatUp"])), "down": list(map(float, default["variations"]["StatDown"])),
            })
    return result


def main() -> None:
    reference = {
        "root_version": ROOT.gROOT.GetVersion(),
        "ff_functions_sha256": hashlib.sha256((FF_UPDATED / "helper" / "ff_functions.py").read_bytes()).hexdigest(),
        "kernel": kernel_cases(),
        "smooth": smooth_cases(),
    }
    path = Path(__file__).with_name("smoothing_root.json")
    path.write_text(json.dumps(reference, indent=0))
    print(f"wrote {path}: {len(reference['kernel'])} kernel and {len(reference['smooth'])} smoothing cases")


if __name__ == "__main__":
    main()

"""Write fake_factor_root.json: the reference of tests/test_fake_factor_hist.py and tests/test_fake_factor_fit.py,
computed once with ROOT and jvoss's TauFakeFactors copy /work/jvoss/FF_Updated (run manually, like
make_smoothing_reference.py):

    python tests/reference/make_fake_factor_reference.py

- `hists`: events of data and three MC samples in an SR-like and an AR-like region, filled with FF_Updated's patched
  Histo1D (centre of mass); the data minus MC yields (also with the MC shifted by +-1 sigma, AddError), their ratio,
  a scaled ratio (unpatched Scale), QCD_SS_estimate and calculate_non_closure_correction (bins <= 0 set to 1 +- 1).
- `fits`: FF_Updated smooth_function on such ratios for every fit option, with and without MC-shifted ratios,
  including ratios compatible with 1 (reset by statistical_check); and fit_to_constant.
"""
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import ROOT

FF_UPDATED = Path("/work/jvoss/FF_Updated")
EDGES = [0.0, 1.0, 2.0, 3.5, 5.0, 8.0]
SAMPLES = ("data", "mc_a", "mc_b", "mc_c")


def import_tau_fake_factors():
    wurlitzer = types.ModuleType("wurlitzer")  # imported by ff_functions for log capturing only
    wurlitzer.STDOUT, wurlitzer.pipes = None, None
    sys.modules["wurlitzer"] = wurlitzer
    sys.path.insert(0, str(FF_UPDATED))
    import helper.ff_functions as ff_functions
    import helper.hooks_and_patches as patches

    return ff_functions, patches


def events(rng, n: int, data: bool) -> dict:
    x = rng.uniform(-0.5, 8.5, n).astype(np.float32)
    x[:3] = np.array([EDGES[1], EDGES[-1], np.float32(3.5)], dtype=np.float32)  # on bin edges, the last one included by numpy only
    weights = np.ones(n) if data else rng.normal(0.3, 0.25, n)  # MC with some negative weights
    return {"x": x, "w": weights}


def fill(patches, sample: dict, name: str):
    frame = ROOT.RDF.FromNumpy({"x": sample["x"], "w": sample["w"].astype(np.float64)})
    return patches.Histo1DPatchedRDataFrame(frame).Histo1D(("x", name, len(EDGES) - 1, np.array(EDGES)), "x", "w").GetValue()


def dump(patches, hist) -> dict:
    n = hist.GetNbinsX()
    return {
        "values": [hist.GetBinContent(i) for i in range(1, n + 1)],
        "errors": [hist.GetBinError(i) for i in range(1, n + 1)],
        "counts": list(map(float, getattr(hist, patches._EXTRA_PARAM_COUNTS))),
        "means": list(map(float, getattr(hist, patches._EXTRA_PARAM_MEANS))),
        "base_values": getattr(hist, patches._EXTRA_PARAM_BASE_VALUES).tolist(),
        "base_errors": getattr(hist, patches._EXTRA_PARAM_BASE_ERRORS_STD).tolist(),
        "suppressed_errors": getattr(hist, patches._EXTRA_PARAM_BASE_ERRORS_MC_SUPPRESSED).tolist(),
    }


def subtracted(hists: dict, shift: float | None):
    result = hists["data"].Clone()
    for name in SAMPLES[1:]:
        result.Add(hists[name].Clone() if shift is None else hists[name].Clone().AddError(shift), -1)
    return result


def hist_cases(ff_functions, patches) -> list[dict]:
    rng = np.random.default_rng(11)
    cases = []
    for n_data in (400, 60):  # the small one has bins <= 0 after the subtraction
        inputs = {region: {s: events(rng, n_data if s == "data" else 150, s == "data") for s in SAMPLES} for region in ("sr", "ar")}
        hists = {region: {s: fill(patches, sample, f"{region}_{s}_{n_data}") for s, sample in samples.items()} for region, samples in inputs.items()}
        yields = {region: {label: subtracted(h, shift) for label, shift in (("nominal", None), ("up", 1.0), ("down", -1.0))} for region, h in hists.items()}
        ratio = yields["sr"]["nominal"].Clone()
        ratio.Divide(yields["ar"]["nominal"])
        scaled = ratio.Clone()
        scaled.Scale(1.7)
        closure, _ = ff_functions.calculate_non_closure_correction({"data_subtracted": yields["sr"]["nominal"]}, {"data_ff": yields["ar"]["nominal"]}, skip_frac=True)
        cases.append({
            "edges": EDGES,
            "events": {region: {s: {"x": v["x"].tolist(), "w": v["w"].tolist()} for s, v in samples.items()} for region, samples in inputs.items()},
            "filled": {region: {s: dump(patches, h) for s, h in samples.items()} for region, samples in hists.items()},
            "yields": {region: {label: dump(patches, h) for label, h in values.items()} for region, values in yields.items()},
            "ratio": dump(patches, ratio),
            "scaled": dump(patches, scaled),
            "qcd": dump(patches, ff_functions.QCD_SS_estimate(dict(hists["ar"]))),
            "closure": dump(patches, closure),
        })
    return cases


def ratio_hist(patches, values, errors, means, suppressed, name: str):
    hist = ROOT.TH1D(name, "", len(values), np.array(EDGES[: len(values) + 1]))
    for i, (value, error) in enumerate(zip(values, errors)):
        hist.SetBinContent(i + 1, value)
        hist.SetBinError(i + 1, error)
    setattr(hist, patches._EXTRA_PARAM_FLAG, True)
    setattr(hist, patches._EXTRA_PARAM_MEANS, list(means))
    setattr(hist, patches._EXTRA_PARAM_COUNTS, [1.0] * len(values))
    setattr(hist, patches._EXTRA_PARAM_BASE_VALUES, np.array(values, dtype=float))
    setattr(hist, patches._EXTRA_PARAM_BASE_ERRORS_STD, np.array(errors, dtype=float))
    setattr(hist, patches._EXTRA_PARAM_BASE_ERRORS_MC_SUPPRESSED, np.array(suppressed, dtype=float))
    return hist


def dump_result(result: dict) -> dict:
    """The stored (sparsified) steps; they sample the full-resolution curves."""
    default, stored = result["default"], result["downsampled"]
    return {
        "p_value": default.get("p_value"), "reset": default.get("_auto_skipped", False),
        "stored": {"edges": list(map(float, stored["nominal"]["edges"])), "values": list(map(float, stored["nominal"]["content"]))},
        "stored_variations": {k: {"edges": list(map(float, v["edges"])), "values": list(map(float, v["content"]))} for k, v in stored["variations"].items()},
    }


def fit_cases(ff_functions, patches) -> list[dict]:
    rng = np.random.default_rng(5)
    means = [0.4, 1.6, 2.9, 4.1, 6.2]
    shapes = {
        "falling": ([0.30, 0.26, 0.21, 0.18, 0.12], [0.01, 0.012, 0.015, 0.02, 0.03]),
        "compatible_with_one": ([1.03, 0.97, 1.05, 0.99, 1.01], [0.08, 0.09, 0.1, 0.12, 0.2]),
        "non_closure": ([1.25, 1.1, 0.95, 0.9, 0.7], [0.03, 0.04, 0.05, 0.06, 0.1]),
    }
    options = ["binwise", "smoothed", "binwise#[0]+smoothed", "binwise#[-1]+smoothed", "binwise#[2]+smoothed"]
    cases = []
    for shape, (values, errors) in shapes.items():
        suppressed = [e * 0.8 for e in errors]
        for option in options:
            for with_mc in (False, True):
                name = f"{shape}_{option}_{with_mc}"
                hist = ratio_hist(patches, values, errors, means, suppressed, name)
                shifted = [[v * (1 + s * 0.05) + rng.normal(0, 0.002) for v in values] for s in (1, -1)]
                mc = {label: ratio_hist(patches, v, errors, means, suppressed, f"{name}_{label}") for label, v in zip(("MCShiftUp", "MCShiftDown"), shifted)} if with_mc else None
                _, result = ff_functions.smooth_function(hist.Clone(), EDGES, option, 2.0, mc_shifted_hist=mc)
                cases.append({
                    "option": option, "bandwidth": 2.0, "values": values, "errors": errors, "means": means, "suppressed": suppressed,
                    "mc_shifted": shifted if with_mc else None, "result": dump_result(result),
                    "constants": [ff_functions.fit_to_constant(mc[k])[0] for k in ("MCShiftUp", "MCShiftDown")] if with_mc else None,
                })
    return cases


def main() -> None:
    ff_functions, patches = import_tau_fake_factors()
    reference = {
        "root_version": ROOT.gROOT.GetVersion(),
        "ff_functions_sha256": hashlib.sha256((FF_UPDATED / "helper" / "ff_functions.py").read_bytes()).hexdigest(),
        "hooks_and_patches_sha256": hashlib.sha256((FF_UPDATED / "helper" / "hooks_and_patches.py").read_bytes()).hexdigest(),
        "hists": hist_cases(ff_functions, patches),
        "fits": fit_cases(ff_functions, patches),
    }
    path = Path(__file__).with_name("fake_factor_root.json")
    path.write_text(json.dumps(reference))
    print(f"wrote {path}: {len(reference['hists'])} histogram and {len(reference['fits'])} fit cases")


if __name__ == "__main__":
    main()

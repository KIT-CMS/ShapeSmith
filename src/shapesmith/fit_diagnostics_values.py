"""The values of a combine FitDiagnostics fit as JSON, for `shapesmith fit --background-only`.

Run by the python3 of CMSSW (PyROOT and the standard library only, Python 3.9), never imported by ShapeSmith:

    python3 fit_diagnostics_values.py workspace.root fitDiagnostics.<name>.root <output>.json

It writes the status and covariance quality of the background-only fit (fit_b); [value, error] of every floating
parameter after the fit (fit_b) and, for the Gaussian-constrained ones, [mean, width] of their constraint term
`<name>_Pdf` in the workspace (combine 10 stores no prefit widths in nuisances_prefit); [value, error] of every
normalisation `<bin>/<process>` and of the totals of each bin before (norm_prefit) and after the fit (norm_fit_b); and
the data events per bin (shapes_prefit/<bin>/data, a density, times the bin widths).

An undefined postfit error is written as null, never as a number: a parameter whose error is NaN or 0 (robustHesse
drops a parameter of a negative eigenvalue from the covariance and leaves it the error 0), and a normalisation whose
error is NaN or whose postfit template (shapes_fit_b/<bin>/<process>) has a NaN bin error (combine samples the
covariance for --saveWithUncertainties; a dropped parameter makes the samples NaN and the stored error 0). Their names
are listed under "undefined".
"""
import json
import math
import sys


def defined(error):
    """The error, or None where it is undefined (NaN, or 0 for a floating parameter)."""
    return None if math.isnan(error) or error == 0.0 else error


def pairs(collection):
    return {arg.GetName(): [arg.getVal(), arg.getError()] for arg in collection}


def parameters(collection):
    """[value, error] of the floating parameters after the fit, an undefined error as None."""
    return {arg.GetName(): [arg.getVal(), defined(arg.getError())] for arg in collection}


def template_error_is_nan(shapes, key):
    """A bin error of the postfit template <bin>/<process> is NaN (False without the template)."""
    histogram = shapes.Get(key)
    return bool(histogram) and histogram.InheritsFrom("TH1") and any(math.isnan(histogram.GetBinError(i)) for i in range(1, histogram.GetNbinsX() + 1))


def normalisations(collection, shapes):
    """[value, error] of the postfit normalisations, an undefined error (NaN, or a NaN template error) as None."""
    result = {}
    for arg in collection:
        error = arg.getError()
        undefined = math.isnan(error) or template_error_is_nan(shapes, arg.GetName())
        result[arg.GetName()] = [arg.getVal(), None if undefined else error]
    return result


def constraints(workspace):
    result = {}
    for nuisance in workspace.genobj("ModelConfig").GetNuisanceParameters():
        term = workspace.pdf(nuisance.GetName() + "_Pdf")
        if term and term.InheritsFrom("RooGaussian"):
            result[nuisance.GetName()] = [term.getMean().getVal(), term.getSigma().getVal()]
    return result


def data(directory):
    result = {}
    for key in directory.GetListOfKeys():
        graph = key.ReadObj().Get("data")
        result[key.GetName()] = sum(graph.GetPointY(i) * (graph.GetErrorXlow(i) + graph.GetErrorXhigh(i)) for i in range(graph.GetN()))
    return result


def main(workspace_path, path, output):
    import ROOT

    workspace, handle = ROOT.TFile.Open(workspace_path), ROOT.TFile.Open(path)
    fit = handle.Get("fit_b")
    if not fit:
        sys.exit(f"{path}: no fit_b, the background-only fit failed")
    values = {
        "status": fit.status(),
        "covariance_quality": fit.covQual(),
        "prefit": constraints(workspace.Get("w")),
        "postfit": parameters(fit.floatParsFinal()),
        "norm_prefit": pairs(handle.Get("norm_prefit")),
        "norm_fit_b": normalisations(handle.Get("norm_fit_b"), handle.Get("shapes_fit_b")),
        "data": data(handle.Get("shapes_prefit")),
    }
    values["undefined"] = {
        "postfit": sorted(name for name, (_, error) in values["postfit"].items() if error is None),
        "norm_fit_b": sorted(name for name, (_, error) in values["norm_fit_b"].items() if error is None),
    }
    with open(output, "w") as out:
        json.dump(values, out, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])

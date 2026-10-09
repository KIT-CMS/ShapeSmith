"""The values of a combine FitDiagnostics fit as JSON, for `shapesmith fit --background-only`.

Run by the python3 of CMSSW (PyROOT and the standard library only, Python 3.9), never imported by ShapeSmith:

    python3 fit_diagnostics_values.py workspace.root fitDiagnostics.<name>.root <output>.json

It writes the status and covariance quality of the background-only fit (fit_b); [value, error] of every floating
parameter after the fit (fit_b) and, for the Gaussian-constrained ones, [mean, width] of their constraint term
`<name>_Pdf` in the workspace (combine 10 stores no prefit widths in nuisances_prefit); [value, error] of every
normalisation `<bin>/<process>` and of the totals of each bin before (norm_prefit) and after the fit (norm_fit_b); and
the data events per bin (shapes_prefit/<bin>/data, a density, times the bin widths).
"""
import json
import sys


def pairs(collection):
    return {arg.GetName(): [arg.getVal(), arg.getError()] for arg in collection}


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
        "postfit": pairs(fit.floatParsFinal()),
        "norm_prefit": pairs(handle.Get("norm_prefit")),
        "norm_fit_b": pairs(handle.Get("norm_fit_b")),
        "data": data(handle.Get("shapes_prefit")),
    }
    with open(output, "w") as out:
        json.dump(values, out, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])

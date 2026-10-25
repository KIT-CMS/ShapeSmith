"""The background-only fit to the data: combine FitDiagnostics with r frozen to 0 on the datacard of a final state,
and its summary (nuisance pulls, prefit and postfit yields per channel, category and process).

The fit runs in the card directory <datacards>/<final state>/ and writes fitDiagnostics.BackgroundOnly.root there;
fit_diagnostics_values.py, run by the python3 of CMSSW right after it, writes its values to
fitDiagnostics.BackgroundOnly.json. The summary of all fitted final states: <datacards>/background_fit.json and .md;
per final state the plots <card dir>/plots/pulls.pdf and plots/{prefit,postfit}/<bin>.pdf/.png.

It fits real data: the blinded channels of the card's blinding.json are masked (mask_<bin>=1, workspace with
--channel-masks; masking.py) unless `unblind`; their plots show no data.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from shapesmith import cmssw, combine_options, masking
from shapesmith.cards import data_histogram
from shapesmith.config import CombineConfig, RunConfig
from shapesmith.datacards import bin_categories
from shapesmith.model import Analysis

NAME = ".BackgroundOnly"
OUTPUT = f"fitDiagnostics{NAME}"  # .root from combine, .json from the values script
VALUES_SCRIPT = Path(__file__).with_name("fit_diagnostics_values.py")
LARGEST_PULLS = 10  # nuisances listed in the markdown summary

logger = logging.getLogger(__name__)


def workspace_command(mass: str = "125", masks: list[str] | tuple[str, ...] = (), card: str = "combined.txt") -> str:
    """text2workspace of the card; with masks the channel masks (mask_<bin>) are built into the workspace."""
    return f"text2workspace.py {card} -o workspace.root -m {mass}" + (" --channel-masks" if masks else "")


def fit_commands(mass: str = "125", combine: CombineConfig | None = None, masks: list[str] | tuple[str, ...] = ()) -> list[str]:
    """Workspace, fit and values; `masks` (mask_<bin>=1) join r=0 in --setParameters, the minimizer options of the
    run replace the built-in ones."""
    return [
        workspace_command(mass, masks),
        # robustHesse: with many normalisation nuisances on one bin (em) the Minuit covariance is forced positive
        # definite and its errors are far too small; the covariance quality is then -1
        combine_options.command(
            f"combine -M FitDiagnostics -d workspace.root -m {mass} {combine_options.parameters(['r=0', *masks])} --freezeParameters r --skipSBFit "
            f"--cminDefaultMinimizerStrategy 0 --robustHesse 1 --saveNormalizations --saveShapes --saveWithUncertainties -n {NAME}", combine, with_range=False),
        f"python3 {VALUES_SCRIPT} workspace.root {OUTPUT}.root {OUTPUT}.json",
    ]


def summarise(values: dict, analysis: Analysis, masked: list[str] | tuple[str, ...] = ()) -> dict:
    """The fit status; per nuisance its prefit [mean, width] (of its Gaussian constraint), postfit [value, error], pull
    (value - mean) / width and constraint error / width (None for a parameter without a Gaussian constraint, e.g. a
    free rate); and per channel and category (the parts of a blinded category as `<category> lo` and `<category> hi`,
    datacards.bin_categories) the data and the prefit and postfit [value, error] of every process and of the totals;
    the entries of the `masked` card bins are marked, their data are the Asimov prediction of the card."""
    return summarise_bins(values, bin_categories(analysis), masked)


def summarise_bins(values: dict, categories: dict[str, tuple[str, str]], masked: list[str] | tuple[str, ...] = ()) -> dict:
    """summarise with the map card bin -> (channel label, category) given (e.g. of a combination of eras). A postfit
    error the values script found undefined stays None (null in the JSON, "undefined" in the markdown), and so does
    the constraint of its parameter; the summary lists them under "undefined" and logs them."""
    nuisances = {}
    for name in sorted(values["postfit"]):
        prefit, postfit = values["prefit"].get(name), values["postfit"][name]
        nuisances[name] = {
            "prefit": prefit,
            "postfit": postfit,
            "pull": (postfit[0] - prefit[0]) / prefit[1] if prefit else None,
            "constraint": postfit[1] / prefit[1] if prefit and postfit[1] is not None else None,
        }
    yields: dict = {}
    for stage, norms in (("prefit", values["norm_prefit"]), ("postfit", values["norm_fit_b"])):
        for key, value in norms.items():
            card_bin, _, process = key.rpartition("/")
            channel, category = categories.get(card_bin, (card_bin, "-"))  # a bin the map lacks under its own name
            entry = yields.setdefault(channel, {}).setdefault(category, {"data": values["data"][card_bin], **({"masked": True} if card_bin in masked else {})})
            entry.setdefault(stage, {})[process] = value
    undefined = undefined_errors(values)
    if undefined["postfit"] or undefined["norm_fit_b"]:
        logger.warning(f"undefined postfit errors (reported as undefined, not 0): {len(undefined['postfit'])} parameters ({', '.join(undefined['postfit'])}), "
                       f"{len(undefined['norm_fit_b'])} normalisations ({', '.join(undefined['norm_fit_b'])})")
    return {"fit_status": values["status"], "covariance_quality": values["covariance_quality"], "nuisances": nuisances, "yields": yields, "undefined": undefined}


def undefined_errors(values: dict) -> dict[str, list[str]]:
    """The parameters and normalisations whose postfit error is undefined (None, or NaN in values of an older values
    script, which wrote NaN as NaN)."""
    def missing(error) -> bool:
        return error is None or (isinstance(error, float) and math.isnan(error))

    return {stage: sorted(name for name, (_, error) in values[stage].items() if missing(error)) for stage in ("postfit", "norm_fit_b")}


def _error(error) -> str:
    return "undefined" if error is None or (isinstance(error, float) and math.isnan(error)) else f"{error:.1f}"


def summary_lines(name: str, result: dict) -> list[str]:
    """Data over the total background per bin before and after the fit, and the largest pulls."""
    lines = [f"## {name}: fit status {result['fit_status']}, covariance quality {result['covariance_quality']}", ""]
    if result.get("masked"):
        lines += [f"Masked (blinded) channels, not in the fit: {', '.join(result['masked'])}", ""]
    if result.get("unblinded"):
        lines += ["**UNBLINDED**: fitted without the masks of blinding.json", ""]
    lines += ["| channel | category | data | prefit | postfit | data/prefit | data/postfit |", "|---|---|---|---|---|---|---|"]
    for channel, categories in result["yields"].items():
        for category, entry in categories.items():
            prefit, postfit = entry["prefit"]["total_background"], entry["postfit"]["total_background"]
            # a masked bin is not fitted and its data_obs is the Asimov prediction, not data
            data, ratios = ("masked", "- | -") if entry.get("masked") else (f"{entry['data']:.0f}", f"{entry['data'] / prefit[0]:.3f} | {entry['data'] / postfit[0]:.3f}")
            lines.append(f"| {channel} | {category} | {data} | {prefit[0]:.1f} ± {_error(prefit[1])} | {postfit[0]:.1f} ± {_error(postfit[1])} | {ratios} |")
    pulled = sorted((item for item in result["nuisances"].items() if item[1]["pull"] is not None), key=lambda item: -abs(item[1]["pull"]))
    lines += ["", "| nuisance | pull | constraint |", "|---|---|---|"]
    lines += [f"| {nuisance} | {entry['pull']:+.2f} | {'undefined' if entry['constraint'] is None else format(entry['constraint'], '.2f')} |" for nuisance, entry in pulled[:LARGEST_PULLS]]
    undefined = result.get("undefined", {})
    if undefined.get("postfit") or undefined.get("norm_fit_b"):
        lines += ["", "Undefined postfit errors (NaN in the fit, e.g. a parameter robustHesse dropped from the covariance):", ""]
        lines += [f"- parameter {name}" for name in undefined.get("postfit", [])] + [f"- normalisation {name}" for name in undefined.get("norm_fit_b", [])]
    return lines + [""]


def write_summary(results: dict[str, dict], output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    json_path = output_dir / "background_fit.json"
    json_path.write_text(json.dumps(results, indent=2))
    md_path = output_dir / "background_fit.md"
    md_path.write_text("\n".join(line for name, result in results.items() for line in summary_lines(name, result)))
    logger.info(f"summary of {len(results)} background-only fits: {json_path}, {md_path.name}")
    return json_path, md_path


def plot_fit(values: dict, result: dict, analysis: Analysis, card_dir: Path, masked: list[str] | tuple[str, ...], title: str) -> list[Path]:
    """<card dir>/plots/pulls.pdf (of the summary `result`) and the prefit and postfit stack of every card bin of the
    values (no data in a masked bin)."""
    from shapesmith.plotting.fit_plots import fit_stack, pull_plot

    card_dir = Path(card_dir)
    written = [path for path in (pull_plot(result["nuisances"], card_dir / "plots" / "pulls.pdf", title),) if path is not None]
    shapes = values.get("shapes")
    if not shapes:
        logger.warning(f"{title}: {OUTPUT}.json has no templates (written by an older values script), no prefit/postfit plots")
        return written
    for stage, key in (("prefit", "prefit"), ("postfit", "fit_b")):
        for card_bin, entries in shapes[key].items():
            written += fit_stack(entries, analysis, card_bin, card_dir / "plots" / stage / card_bin, card_bin in masked, bin_edges(card_dir / "combined.txt", card_bin), stage)
    logger.info(f"{title}: {len(written)} fit plots in {card_dir / 'plots'}")
    return written


def bin_edges(card: Path, card_bin: str):
    """The bin edges of the data histogram of a card bin (None if the card or its shape file is not readable)."""
    import uproot

    located = data_histogram(card, card_bin) if Path(card).exists() else None
    if located is None or not located[0].exists():
        return None
    try:
        with uproot.open(located[0]) as handle:
            return handle[located[1]].axis().edges()
    except (KeyError, OSError, ValueError) as error:
        logger.debug(f"{card_bin}: no bin edges from {located[0]}:{located[1]} ({error})")
        return None


def run_background_fit(config: RunConfig, analysis: Analysis, datacard_dir: Path, final_states: list[str], skip_combine: bool = False, unblind: bool = False, plots: bool = True) -> dict:
    """The background-only fit of every final state (or, with `skip_combine`, its stored values), with the blinded
    channels masked unless `unblind`, the summary and the plots."""
    datacard_dir = Path(datacard_dir)
    results = {}
    for name in final_states:
        card_dir = datacard_dir / name
        masks = masking.data_masks(card_dir, unblind)
        if not skip_combine:
            logger.info(f"{name}: background-only fit in {card_dir}")
            cmssw.run(fit_commands(combine=config.combine, masks=masks), config.combine, card_dir)
        values = json.loads((card_dir / f"{OUTPUT}.json").read_text())
        blinded = masking.read(card_dir).masked
        results[name] = summarise(values, analysis, [] if unblind else blinded)
        results[name].update({"masked": [] if unblind else list(blinded), **({"unblinded": True} if unblind else {})})
        if plots:
            plot_fit(values, results[name], analysis, card_dir, [] if unblind else blinded, name)
    write_summary(results, datacard_dir)
    return results

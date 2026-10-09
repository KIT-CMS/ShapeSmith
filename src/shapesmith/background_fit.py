"""The background-only fit to the data: combine FitDiagnostics with r frozen to 0 on the datacard of a final state,
and its summary (nuisance pulls, prefit and postfit yields per channel, category and process).

The fit runs in the card directory <datacards>/<final state>/ and writes fitDiagnostics.BackgroundOnly.root there;
fit_diagnostics_values.py, run by the python3 of CMSSW right after it, writes its values to
fitDiagnostics.BackgroundOnly.json. The summary of all fitted final states: <datacards>/background_fit.json and .md.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from shapesmith import cmssw
from shapesmith.config import RunConfig
from shapesmith.datacards import bin_name
from shapesmith.model import Analysis

NAME = ".BackgroundOnly"
OUTPUT = f"fitDiagnostics{NAME}"  # .root from combine, .json from the values script
VALUES_SCRIPT = Path(__file__).with_name("fit_diagnostics_values.py")
LARGEST_PULLS = 10  # nuisances listed in the markdown summary

logger = logging.getLogger(__name__)


def fit_commands(mass: str = "125") -> list[str]:
    return [
        f"text2workspace.py combined.txt -o workspace.root -m {mass}",
        f"combine -M FitDiagnostics -d workspace.root -m {mass} --setParameters r=0 --freezeParameters r --skipSBFit "
        f"--saveNormalizations --saveShapes --saveWithUncertainties -n {NAME}",
        f"python3 {VALUES_SCRIPT} workspace.root {OUTPUT}.root {OUTPUT}.json",
    ]


def summarise(values: dict, analysis: Analysis) -> dict:
    """The fit status; per nuisance its prefit [mean, width] (of its Gaussian constraint), postfit [value, error], pull
    (value - mean) / width and constraint error / width (None for a parameter without a Gaussian constraint, e.g. a
    free rate); and per channel and category the data and the prefit and postfit [value, error] of every process and
    of the totals."""
    nuisances = {}
    for name in sorted(values["postfit"]):
        prefit, postfit = values["prefit"].get(name), values["postfit"][name]
        nuisances[name] = {
            "prefit": prefit,
            "postfit": postfit,
            "pull": (postfit[0] - prefit[0]) / prefit[1] if prefit else None,
            "constraint": postfit[1] / prefit[1] if prefit else None,
        }
    categories = {bin_name(analysis, channel, index): (channel, category.name) for channel in analysis.channels for index, category in enumerate(analysis.channel(channel).categories)}
    yields: dict = {}
    for stage, norms in (("prefit", values["norm_prefit"]), ("postfit", values["norm_fit_b"])):
        for key, value in norms.items():
            card_bin, _, process = key.rpartition("/")
            channel, category = categories[card_bin]
            entry = yields.setdefault(channel, {}).setdefault(category, {"data": values["data"][card_bin]})
            entry.setdefault(stage, {})[process] = value
    return {"fit_status": values["status"], "covariance_quality": values["covariance_quality"], "nuisances": nuisances, "yields": yields}


def summary_lines(name: str, result: dict) -> list[str]:
    """Data over the total background per bin before and after the fit, and the largest pulls."""
    lines = [f"## {name}: fit status {result['fit_status']}, covariance quality {result['covariance_quality']}", ""]
    lines += ["| channel | category | data | prefit | postfit | data/prefit | data/postfit |", "|---|---|---|---|---|---|---|"]
    for channel, categories in result["yields"].items():
        for category, entry in categories.items():
            prefit, postfit = entry["prefit"]["total_background"][0], entry["postfit"]["total_background"][0]
            lines.append(f"| {channel} | {category} | {entry['data']:.0f} | {prefit:.1f} | {postfit:.1f} | {entry['data'] / prefit:.3f} | {entry['data'] / postfit:.3f} |")
    pulled = sorted((item for item in result["nuisances"].items() if item[1]["pull"] is not None), key=lambda item: -abs(item[1]["pull"]))
    lines += ["", "| nuisance | pull | constraint |", "|---|---|---|"]
    lines += [f"| {nuisance} | {entry['pull']:+.2f} | {entry['constraint']:.2f} |" for nuisance, entry in pulled[:LARGEST_PULLS]]
    return lines + [""]


def write_summary(results: dict[str, dict], output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    json_path = output_dir / "background_fit.json"
    json_path.write_text(json.dumps(results, indent=2))
    md_path = output_dir / "background_fit.md"
    md_path.write_text("\n".join(line for name, result in results.items() for line in summary_lines(name, result)))
    logger.info(f"summary of {len(results)} background-only fits: {json_path}, {md_path.name}")
    return json_path, md_path


def run_background_fit(config: RunConfig, analysis: Analysis, datacard_dir: Path, final_states: list[str], skip_combine: bool = False) -> dict:
    """The background-only fit of every final state (or, with `skip_combine`, its stored values) and the summary."""
    datacard_dir = Path(datacard_dir)
    results = {}
    for name in final_states:
        card_dir = datacard_dir / name
        if not skip_combine:
            logger.info(f"{name}: background-only fit in {card_dir}")
            cmssw.run(fit_commands(), config.combine, card_dir)
        results[name] = summarise(json.loads((card_dir / f"{OUTPUT}.json").read_text()), analysis)
    write_summary(results, datacard_dir)
    return results

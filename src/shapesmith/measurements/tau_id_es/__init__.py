"""Tau-ID scale factor and energy scale of embedded taus, per decay-mode (and pT) category.

The chain of the predecessor (smhtt_ul tauID_SFs_dev) on ShapeSmith histograms. One run per working-point
combination fills the mt channel and its mm control region (the analysis builds them for `vsjet_wp` and `vsele_wp`),
adds the estimates, writes the shapes in the MorphingTauID2017 format and runs per category in CMSSW (combine.py);
the result is the singles fit. `--merge` combines the runs into the correctionlib payload.

Output, below <output_dir>/tau_id_es/<era>/:
- <vsjet>_<vsele>/: shapes.root (histograms and estimates), synced/ (MorphingTauID2017 inputs), fits/ (per
  category the cards, workspace, scan and singles fit), results.json, plots/;
- DeepTau2018v2p5_id_es_embedding<era>UL.json.gz, written by --merge.
A result with a problem (an interval at the fit range, a scan region at the scan boundary, a failed crossing) is an
error: the run raises after writing results.json, and --merge refuses it.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from shapesmith import cmssw, payloads
from shapesmith.config import CombineConfig
from shapesmith.estimates import run_estimates
from shapesmith.fill import run_hist
from shapesmith.measurements import MeasureContext
from shapesmith.measurements.tau_id_es import combine, plots
from shapesmith.measurements.tau_id_es.combine import Interval
from shapesmith.measurements.tau_id_es.grid import CATEGORIES, CHANNEL, CONTROL_CATEGORY, CONTROL_CHANNEL, SIGNAL, grid_name
from shapesmith.measurements.tau_id_es.payload import correction_set, payload_name
from shapesmith.measurements.tau_id_es.synced import write_synced
from shapesmith.model import Analysis

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TauIdEsMeasurement:
    vsjet_wp: str
    vsele_wp: str
    grid: tuple[int, ...]  # the ES shifts in 0.1 % (grid.grid); each is a template variation of EMB in mt
    name: str = "tau_id_es"

    def run(self, context: MeasureContext) -> None:
        if context.suggest_binning:
            raise ValueError("tau_id_es: --suggest-binning is not supported, the m_vis bins are fixed in the analysis")
        if context.merge:
            merge(context)
        else:
            measure(self, context)


def check(analysis: Analysis, measurement: TauIdEsMeasurement) -> None:
    """The analysis has the channels, categories, signal and grid variations the chain expects."""
    problems = [f"channel {name} is missing" for name in (CHANNEL, CONTROL_CHANNEL) if name not in analysis.channels]
    if not problems:
        channel = analysis.channel(CHANNEL)
        problems += [f"unknown mt category {c.name} (known: {', '.join(CATEGORIES)})" for c in channel.categories if c.name not in CATEGORIES]
        if [c.name for c in analysis.channel(CONTROL_CHANNEL).categories] != [CONTROL_CATEGORY]:
            problems.append(f"the {CONTROL_CHANNEL} channel needs exactly the category {CONTROL_CATEGORY}")
        if SIGNAL not in channel.backgrounds():
            problems.append(f"process {SIGNAL} is missing in {CHANNEL}")
        declared = {v.name for v in channel.variations}
        problems += [f"grid variation {grid_name(shift)} is missing in {CHANNEL}" for shift in measurement.grid if grid_name(shift) not in declared]
    if problems:
        raise ValueError("tau_id_es: " + "; ".join(problems))


def fit_category(combine_config: CombineConfig | None, category: str, synced: Path, directory: Path, era: str, grid: tuple[int, ...]) -> dict:
    """Datacards, workspace, 2D scan, singles fit and postfit shapes of one category below `directory`/fits; its
    result and problems."""
    work_dir = Path(directory) / "fits"
    work_dir.mkdir(parents=True, exist_ok=True)
    cmssw.run([f"{combine.morphing_command(category, synced, era, grid)} > morphing_{category}.log 2>&1"], combine_config, work_dir)
    cards = combine.card_dir(work_dir, category)
    combine.add_rate_parameters(cards, category)
    cmssw.run([f"{combine.workspace_command(category)} > workspace.log 2>&1", f"{combine.scan_command(category, grid)} > scan.log 2>&1"], combine_config, cards)
    scan = combine.read_scan(combine.output_path(cards, f"scan_2D_{category}"), category)
    start, r_range, es_range, problems = combine.singles_inputs(scan, grid)
    cmssw.run([f"{combine.singles_command(category, start, r_range, es_range)} > singles.log 2>&1", f"{combine.postfit_command(category)} > postfit.log 2>&1"], combine_config, cards)
    sf, es = combine.read_singles(combine.output_path(cards, f"singles_{category}"), category)
    problems += combine.interval_problems("r", sf, r_range) + combine.interval_problems("ES", es, es_range)
    return {"sf": asdict(sf), "es": asdict(es), "start": start, "ranges": {"r": r_range, "ES": es_range}, "problems": problems}


def plot_category(directory: Path, category: str, fit: dict) -> None:
    """The 2D scan with the singles result and the postfit shapes of one fitted category."""
    cards = combine.card_dir(Path(directory) / "fits", category)
    scan = combine.read_scan(combine.output_path(cards, f"scan_2D_{category}"), category)
    plots.plot_scan(scan, Interval(**fit["sf"]), Interval(**fit["es"]), category, Path(directory) / "plots" / "scans" / category)
    plots.plot_postfit(cards / "postfit_shapes.root", Path(directory) / "plots" / "postfit" / category)


def measure(measurement: TauIdEsMeasurement, context: MeasureContext) -> Path:
    analysis, config = context.analysis, context.config
    check(analysis, measurement)
    directory = context.output / f"{measurement.vsjet_wp}_{measurement.vsele_wp}"
    shapes = directory / "shapes.root"
    hset = run_hist(config, analysis, [CHANNEL, CONTROL_CHANNEL], False, None, True, None, shapes)
    run_estimates(hset, analysis, [CHANNEL, CONTROL_CHANNEL])
    hset.save(shapes)
    synced = write_synced(hset, analysis, directory / "synced")
    categories = [c.name for c in analysis.channel(CHANNEL).categories]
    for category in categories:
        plots.plot_control(hset, analysis, category, directory / "plots" / "control" / category)
    with ThreadPoolExecutor(max_workers=max(1, config.workers)) as pool:  # each category runs in its own CMSSW process
        fits = list(pool.map(lambda category: fit_category(config.combine, category, synced, directory, analysis.era, measurement.grid), categories))
    for category, fit in zip(categories, fits):
        plot_category(directory, category, fit)
    record = {
        "vsjet_wp": measurement.vsjet_wp,
        "vsele_wp": measurement.vsele_wp,
        "categories": dict(zip(categories, fits)),
        "provenance": context.provenance(vsjet_wp=measurement.vsjet_wp, vsele_wp=measurement.vsele_wp),
    }
    (directory / "results.json").write_text(json.dumps(record, indent=2))
    problems = [f"{category}: {problem}" for category, fit in record["categories"].items() for problem in fit["problems"]]
    if problems:
        raise ValueError(f"tau_id_es {measurement.vsjet_wp}/{measurement.vsele_wp}, see {directory / 'results.json'}:\n" + "\n".join(problems))
    return directory


def merge(context: MeasureContext) -> Path:
    """The payload of every working-point combination measured below the output directory."""
    results, sources, problems = {}, {}, []
    for path in sorted(Path(context.output).glob("*/results.json")):
        record = json.loads(path.read_text())
        pair = (record["vsjet_wp"], record["vsele_wp"])
        results[pair] = {category: (Interval(**fit["sf"]), Interval(**fit["es"])) for category, fit in record["categories"].items()}
        sources["_".join(pair)] = record["provenance"]
        problems += [f"{'_'.join(pair)}/{category}: {problem}" for category, fit in record["categories"].items() for problem in fit["problems"]]
    if not results:
        raise ValueError(f"tau_id_es: no results.json below {context.output}; run the measurement per working point first")
    if problems:
        raise ValueError("tau_id_es: results with problems, not merged:\n" + "\n".join(problems))
    cset = correction_set(results, context.provenance(working_points=sorted(sources), measurements=sources))
    path = payloads.write(cset, Path(context.output) / payload_name(context.analysis.era))
    logger.info(f"payload of {len(results)} working-point combinations: {path}")
    return path

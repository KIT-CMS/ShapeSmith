"""Tau-ID scale factor and energy scale of embedded taus, per decay-mode (and pT) category.

The chain of the predecessor (smhtt_ul tauID_SFs_dev) on ShapeSmith histograms. One run per working-point
combination fills the mt channel and its mm control region (the analysis builds them for `vsjet_wp` and `vsele_wp`),
adds the estimates, writes the shapes in the MorphingTauID2017 format and runs per category in CMSSW (combine.py):
the result is the best value and 1 sigma interval of each POI from its profile (v7 spec 5.4, U11; r from the envelope
of a 2D grid, see combine.py); the singles fit of the predecessor runs as a cross-check. `--refit` repeats the CMSSW part from the synced/ shapes of an
earlier run, without histogramming. `--merge` combines the runs into the correctionlib payload.

Output, below <output_dir>/tau_id_es/<era>/:
- <vsjet>_<vsele>/: shapes.root (histograms and estimates), synced/ (MorphingTauID2017 inputs), fits/ (per
  category the cards, workspace, scans, the fit at the profile minima and the singles fit), results.json, report.md
  (profile and singles intervals side by side), plots/;
- DeepTau2018v2p5_id_es_embedding<era>UL.json.gz and report.md, written by --merge.
A result with a problem (a side of an interval without a crossing after the widened rescans, a best value at the scan
range, a profiled POI at its range, a CMSSW step of the result that failed or ran longer than STEP_TIMEOUT) is an
error: the run raises after writing results.json, and --merge refuses it, naming category and side. A singles fit
that fails or differs from the profile interval by more than combine.SINGLES_DEVIATION of its width is a warning,
flagged in the reports.
"""
from __future__ import annotations

import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from shapesmith import cmssw, payloads
from shapesmith.config import CombineConfig
from shapesmith.estimates import run_estimates
from shapesmith.fill import run_hist
from shapesmith.measurements import MeasureContext
from shapesmith.measurements.tau_id_es import combine, plots
from shapesmith.measurements.tau_id_es.combine import Interval, Profile
from shapesmith.measurements.tau_id_es.grid import CATEGORIES, CHANNEL, CONTROL_CATEGORY, CONTROL_CHANNEL, SIGNAL, grid_name
from shapesmith.measurements.tau_id_es.payload import correction_set, payload_name
from shapesmith.measurements.tau_id_es.synced import shapes_path, write_synced
from shapesmith.model import Analysis

logger = logging.getLogger(__name__)
STEP_TIMEOUT = 3600.0  # seconds per CMSSW step of a category
SINGLES_TIMEOUT = 900.0  # seconds of the singles cross-check, which can iterate without end; beyond it a warning
CMSSW_FAILURES = (subprocess.CalledProcessError, subprocess.TimeoutExpired)


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


def scan_profile(combine_config: CombineConfig | None, category: str, poi: str, start: tuple[float, float], ranges: tuple[tuple[float, float], tuple[float, float]], grid: tuple[int, ...], cards: Path) -> tuple[Profile, tuple, int]:
    """The profile scan of `poi` (combine.profile_command), rescanned on a widened range while a side of its 1 sigma
    interval has no crossing (combine.widened), at most MAX_WIDENINGS times; the last profile, the (r, ES) ranges of
    its scan and the number of rescans."""
    index = 0 if poi == combine.pois(category)[0] else 1
    limits = combine.bounds(grid)[index]
    ranges = list(ranges)
    for attempt in range(combine.MAX_WIDENINGS + 1):
        name = combine.profile_name(poi, attempt)
        cmssw.run([f"{combine.profile_command(category, poi, start, ranges[0], ranges[1], attempt)} > {name}.log 2>&1"], combine_config, cards, STEP_TIMEOUT)
        profile = combine.read_poi_profile(cards, category, poi, attempt)
        new_range = combine.widened(profile, limits)
        if new_range is None or attempt == combine.MAX_WIDENINGS:
            return profile, tuple(ranges), attempt
        logger.info(f"{category}: {poi} has no 1 sigma crossing on a side of {profile.range}, rescanning on {new_range}")
        ranges[index] = new_range
    raise AssertionError("unreachable")


def fit_category(combine_config: CombineConfig | None, category: str, synced: Path, directory: Path, era: str, grid: tuple[int, ...]) -> dict:
    """Datacards, workspace, 2D scan, the profiles (the result: ES from its 1D scan, r from the envelope of a 2D grid
    in the ES window of the ES profile), the fit at the profile minima with its postfit shapes and the singles
    cross-check of one category below `directory`/fits; its result, problems and warnings."""
    work_dir = Path(directory) / "fits"
    work_dir.mkdir(parents=True, exist_ok=True)
    cmssw.run([f"{combine.morphing_command(category, synced, era, grid)} > morphing_{category}.log 2>&1"], combine_config, work_dir, STEP_TIMEOUT)
    cards = combine.card_dir(work_dir, category)
    combine.add_rate_parameters(cards, category)
    cmssw.run([f"{combine.workspace_command(category)} > workspace.log 2>&1", f"{combine.scan_command(category, grid)} > scan.log 2>&1"], combine_config, cards, STEP_TIMEOUT)
    scan = combine.read_scan(combine.output_path(cards, f"scan_2D_{category}"), category)
    start, r_window, es_window = combine.profile_inputs(scan, grid)
    r_poi, es_poi = combine.pois(category)
    r_bounds, es_bounds = combine.bounds(grid)
    result: dict = {"scan_2d": _inputs(start, r_window, es_window), "profiles": {}, "problems": [], "warnings": []}
    es_profile, es_ranges, es_widenings = scan_profile(combine_config, category, es_poi, start, (r_window, es_window), grid, cards)
    es_interval, problems = combine.profile_interval("ES", es_profile, es_bounds, "r", es_ranges[0])
    result["problems"] += problems
    result["profiles"]["ES"] = {"range": list(es_ranges[1]), "profiled_range": list(es_ranges[0]), "widenings": es_widenings, "points": len(es_profile.values)}
    envelope_start = (es_profile.profiled_at_minimum or start[0], es_profile.best)
    es_around = es_profile.window(combine.ENVELOPE_MARGIN, es_bounds, combine.ENVELOPE_LEVEL)
    r_profile, r_ranges, r_widenings = scan_profile(combine_config, category, r_poi, envelope_start, (r_window, es_around), grid, cards)
    r_interval, problems = combine.profile_interval("r", r_profile, r_bounds, "ES", r_ranges[1])
    result["problems"] += problems
    result["profiles"]["r"] = {"range": list(r_ranges[0]), "profiled_range": list(r_ranges[1]), "widenings": r_widenings, "points": len(r_profile.values)}
    intervals = {"sf": r_interval, "es": es_interval}
    result.update({key: None if interval is None else asdict(interval) for key, interval in intervals.items()})
    result["ranges"] = {name: entry["range"] for name, entry in result["profiles"].items()}
    if all(intervals.values()):
        best = (r_interval.best, es_interval.best)
        try:
            cmssw.run([f"{combine.bestfit_command(category, best, r_bounds, es_bounds)} > bestfit.log 2>&1", f"{combine.postfit_command(category)} > postfit.log 2>&1"], combine_config, cards, STEP_TIMEOUT)
        except CMSSW_FAILURES as error:
            result["warnings"].append(f"postfit at the profile minima failed: {error}")
    result["singles"] = singles_cross_check(combine_config, category, (r_profile, es_profile), grid, cards, intervals)
    result["warnings"] += result["singles"].get("deviations", []) + ([result["singles"]["error"]] if "error" in result["singles"] else [])
    return result


def singles_cross_check(combine_config: CombineConfig | None, category: str, profiles: tuple[Profile, Profile], grid: tuple[int, ...], cards: Path, intervals: dict) -> dict:
    """The predecessor's singles fit from the 1D profiles, compared with the profile intervals."""
    start, r_range, es_range = combine.singles_inputs(profiles, grid)
    record = _inputs(start, r_range, es_range)
    try:
        cmssw.run([f"{combine.singles_command(category, start, r_range, es_range)} > singles.log 2>&1"], combine_config, cards, SINGLES_TIMEOUT)
        sf, es = combine.read_singles(combine.output_path(cards, f"singles_{category}"), category)
    except CMSSW_FAILURES as error:
        return {**record, "error": f"singles cross-check failed: {error}"}
    record.update(sf=asdict(sf), es=asdict(es), problems=combine.interval_problems("r", sf, r_range) + combine.interval_problems("ES", es, es_range))
    record["deviations"] = [d for key, name, singles in (("sf", "r", sf), ("es", "ES", es)) if intervals[key] is not None for d in combine.deviations(name, intervals[key], singles)]
    return record


def _inputs(start: tuple[float, float], r_range: tuple[float, float], es_range: tuple[float, float]) -> dict:
    return {"start": list(start), "ranges": {"r": list(r_range), "ES": list(es_range)}}


def fit_or_failure(combine_config: CombineConfig | None, category: str, synced: Path, directory: Path, era: str, grid: tuple[int, ...]) -> dict:
    """fit_category, or a result without values whose problem is the failed CMSSW step (see its log)."""
    try:
        return fit_category(combine_config, category, synced, directory, era, grid)
    except CMSSW_FAILURES as error:
        logger.error(f"{category}: CMSSW step failed: {error}")
        return {"sf": None, "es": None, "problems": [f"CMSSW step failed: {error}"], "warnings": []}


def plot_category(directory: Path, category: str, fit: dict) -> None:
    """The 2D scan with the profile result (and the singles result), the 1D profiles and the postfit shapes of one
    fitted category."""
    cards = combine.card_dir(Path(directory) / "fits", category)
    output = Path(directory) / "plots"
    scan = combine.read_scan(combine.output_path(cards, f"scan_2D_{category}"), category)
    singles = fit.get("singles", {})
    plots.plot_scan(scan, Interval(**fit["sf"]), Interval(**fit["es"]), category, output / "scans" / category,
                    Interval(**singles["sf"]) if "sf" in singles else None, Interval(**singles["es"]) if "es" in singles else None)
    for poi, name, key in zip(combine.pois(category), ("r", "ES"), ("sf", "es")):
        profile = combine.read_poi_profile(cards, category, poi, fit["profiles"][name]["widenings"])
        plots.plot_profile(profile, Interval(**fit[key]), name, category, output / "profiles" / f"{category}_{name}")
    if (cards / "postfit_shapes.root").exists():
        plots.plot_postfit(cards / "postfit_shapes.root", output / "postfit" / category)


def measure(measurement: TauIdEsMeasurement, context: MeasureContext) -> Path:
    analysis, config = context.analysis, context.config
    check(analysis, measurement)
    directory = context.output / f"{measurement.vsjet_wp}_{measurement.vsele_wp}"
    categories = [c.name for c in analysis.channel(CHANNEL).categories]
    if context.refit:
        synced = directory / "synced"
        missing = [path for path in (shapes_path(synced, channel, analysis.era) for channel in (CHANNEL, CONTROL_CHANNEL)) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"tau_id_es --refit: the synced shapes of an earlier run are missing: {', '.join(map(str, missing))}")
        logger.info(f"tau_id_es {measurement.vsjet_wp}/{measurement.vsele_wp}: refit from {synced}")
    else:
        shapes = directory / "shapes.root"
        logger.info(f"tau_id_es {measurement.vsjet_wp}/{measurement.vsele_wp}: histograms of {CHANNEL} and {CONTROL_CHANNEL} with {len(measurement.grid)} ES shifts -> {directory}")
        hset = run_hist(config, analysis, [CHANNEL, CONTROL_CHANNEL], False, None, True, None, shapes)
        run_estimates(hset, analysis, [CHANNEL, CONTROL_CHANNEL])
        hset.save(shapes)
        synced = write_synced(hset, analysis, directory / "synced")
        logger.info(f"synced shapes: {synced}")
        for category in categories:
            plots.plot_control(hset, analysis, category, directory / "plots" / "control" / category)
    logger.info(f"fitting {len(categories)} categories, {max(1, config.workers)} at a time: {', '.join(categories)}")
    with ThreadPoolExecutor(max_workers=max(1, config.workers)) as pool:  # each category runs in its own CMSSW process
        fits = list(pool.map(lambda category: fit_or_failure(config.combine, category, synced, directory, analysis.era, measurement.grid), categories))
    for category, fit in zip(categories, fits):
        if fit["sf"] is not None and fit["es"] is not None:
            logger.info(f"{category}: SF {_interval(fit['sf'])}, ES {_interval(fit['es'])}" + (f", problems: {'; '.join(fit['problems'])}" if fit["problems"] else ""))
            plot_category(directory, category, fit)
        for warning in fit.get("warnings", []):
            logger.warning(f"{category}: {warning}")
    record = {
        "vsjet_wp": measurement.vsjet_wp,
        "vsele_wp": measurement.vsele_wp,
        "categories": dict(zip(categories, fits)),
        "provenance": context.provenance(vsjet_wp=measurement.vsjet_wp, vsele_wp=measurement.vsele_wp, refit=context.refit),
    }
    (directory / "results.json").write_text(json.dumps(record, indent=2))
    (directory / "report.md").write_text(report([record]))
    problems = [f"{category}: {problem}" for category, fit in record["categories"].items() for problem in fit["problems"]]
    if problems:
        raise ValueError(f"tau_id_es {measurement.vsjet_wp}/{measurement.vsele_wp}, see {directory / 'results.json'}:\n" + "\n".join(problems))
    return directory


def _interval(interval: dict) -> str:
    return " ".join(f"{name}={value:.4g}" for name, value in interval.items() if isinstance(value, (int, float)))


def _text(interval: dict | None) -> str:
    return "-" if not interval else f"{interval['best']:.4g} [{interval['low']:.4g}, {interval['high']:.4g}]"


def report(records: list[dict]) -> str:
    """Markdown: per working-point combination and category the profile and the singles intervals of SF and ES side
    by side, the rescans, and the problems and warnings (a singles interval that differs by more than
    combine.SINGLES_DEVIATION of the profile interval width is flagged)."""
    lines = ["| WP | category | SF (profile) | SF (singles) | ES % (profile) | ES % (singles) | rescans r/ES | flags |", "|---|---|---|---|---|---|---|---|"]
    for record in records:
        for category, fit in record["categories"].items():
            singles = fit.get("singles", {})
            rescans = "/".join(str(fit.get("profiles", {}).get(name, {}).get("widenings", "-")) for name in ("r", "ES"))
            flags = "; ".join([f"PROBLEM {p}" for p in fit["problems"]] + [f"check: {w}" for w in fit.get("warnings", [])]) or "ok"
            lines.append(f"| {record['vsjet_wp']}_{record['vsele_wp']} | {category} | {_text(fit['sf'])} | {_text(singles.get('sf'))} | {_text(fit['es'])} | {_text(singles.get('es'))} | {rescans} | {flags} |")
    return "\n".join(lines) + "\n"


def merge_problems(record: dict) -> list[str]:
    """Why a run cannot enter the payload: its problems, and an interval equal to its fit range on a side (checked
    again on the stored values)."""
    label = f"{record['vsjet_wp']}_{record['vsele_wp']}"
    problems = []
    for category, fit in record["categories"].items():
        problems += [f"{label}/{category}: {problem}" for problem in fit["problems"]]
        if fit["problems"] or fit.get("sf") is None or fit.get("es") is None:
            if not fit["problems"]:
                problems.append(f"{label}/{category}: no result")
            continue
        for key, name in (("sf", "r"), ("es", "ES")):
            fit_range = fit.get("ranges", {}).get(name)
            if fit_range is not None:
                problems += [f"{label}/{category}: {problem}" for problem in combine.interval_problems(name, Interval(**fit[key]), tuple(fit_range))]
    return problems


def merge(context: MeasureContext) -> Path:
    """The payload of every working-point combination measured below the output directory."""
    records = [json.loads(path.read_text()) for path in sorted(Path(context.output).glob("*/results.json"))]
    logger.info(f"tau_id_es: merging {len(records)} working-point combinations from {context.output}")
    if not records:
        raise ValueError(f"tau_id_es: no results.json below {context.output}; run the measurement per working point first")
    (Path(context.output) / "report.md").write_text(report(records))
    problems = [problem for record in records for problem in merge_problems(record)]
    if problems:
        raise ValueError("tau_id_es: results with problems, not merged:\n" + "\n".join(problems))
    results = {(r["vsjet_wp"], r["vsele_wp"]): {category: (Interval(**fit["sf"]), Interval(**fit["es"])) for category, fit in r["categories"].items()} for r in records}
    sources = {f"{r['vsjet_wp']}_{r['vsele_wp']}": r["provenance"] for r in records}
    cset = correction_set(results, context.provenance(working_points=sorted(sources), measurements=sources))
    path = payloads.write(cset, Path(context.output) / payload_name(context.analysis.era))
    logger.info(f"payload of {len(results)} working-point combinations: {path}")
    return path

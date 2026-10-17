"""The efficiency of the tau leg of the di-tau trigger in data (and MC), measured with mu tau_h tag and probe for the
tt embedding (v7 spec E33, v7_tt_emb_trigger_2026-10-16 section 12; variant A: the tt embedding is weighted with
eps_data(tau1) * eps_data(tau2)).

One run per isolation class of the probe (`wp_class`: Medium, VVVLoose_not_Medium) and selection variant (`variant`:
nominal, or a variation whose change is a systematic uncertainty) fills the mt channel: per (decay mode, pT) bin a pass
and a fail category, m_vis, the probe passing or failing the monitoring leg. Per bin a datacard (cards.py), the fit in
CMSSW (fit.py: a 1D profile scan of the efficiency, the 1 sigma interval from its delta NLL = 0.5 crossings, rescans
on a widened range; the singles fit as a cross-check), the counting cross-check in a counting region and m_vis window,
eps_MC (genuine MC taus) and eps_EMB^mon (the embedded taus with the monitoring flags) by counting. `--refit` repeats
the fits from the stored shapes.root; `--suggest-binning` proposes per decay mode the pT edges where the relative error
of eps_data (counting) stays below `max_relative_error`, merging bins from the top; `--merge` writes the payload
(payload.py) of every class: eff_data with up/down the quadratic sum of the fit interval, of the variants (each change
times its weight in `systematic_variants`) and of the fit-counting difference where it exceeds their combined error;
eff_mc from the MC count.

Output, below <output_dir>/tau_trigger/<era>/:
- <wp_class>/<variant>/: shapes.root, fits/<bin>/ (card, shapes, scans, fits), results.json, report.md, plots/;
- ditau_trigger_DeepTau2018v2p5_embedding<era>UL.json.gz, report.md and plots/, written by --merge.
A bin with a problem (no interval inside [0, 1], a failed CMSSW step) is an error; --merge refuses it.
"""
from __future__ import annotations

import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from shapesmith import payloads
from shapesmith.estimates import run_estimates
from shapesmith.events import Query
from shapesmith.expressions import columns_of, evaluate, mask
from shapesmith.fill import run_hist
from shapesmith.histogram import NOMINAL_VARIATION, HistKey, Histogram, HistogramSet, on_template
from shapesmith.measurements import MeasureContext
from shapesmith.measurements.tau_trigger import cards, plots
from shapesmith.measurements.tau_trigger.counting import Count, data_efficiency, efficiency, relative_error
from shapesmith.measurements.tau_trigger.fit import fit_or_failure
from shapesmith.measurements.tau_trigger.payload import check as check_payload
from shapesmith.measurements.tau_trigger.payload import correction_set, payload_name
from shapesmith.model import NOMINAL, Analysis

logger = logging.getLogger(__name__)
SIDES = ("pass", "fail")
EFFICIENCY_FLOOR = 1e-4  # the down variation of eff_data is not set below this


@dataclass(frozen=True)
class Bin:
    """A (decay mode, pT) bin: its categories `<name>_pass` and `<name>_fail` in the channel. decay_modes (-1,) is
    every decay mode (payload dm -1); (10, 11) gives both payload decay modes the result of one fit."""

    name: str
    decay_modes: tuple[int, ...]
    pt: tuple[float, float | None]  # GeV, None: open above

    def category(self, side: str) -> str:
        return f"{self.name}_{side}"


@dataclass(frozen=True)
class TauTriggerMeasurement:
    wp_class: str  # the payload wp: Medium | VVVLoose_not_Medium
    bins: tuple[Bin, ...]
    signal: str  # the process whose efficiency is the POI: the embedded genuine taus (or the MC ones in a variant)
    channel: str = "mt"
    variant: str = "nominal"
    systematic_variants: Mapping[str, float] = field(default_factory=dict)  # variant -> the fraction of its change taken as uncertainty
    signal_template: str | None = None  # the variation of the signal that is its nominal (the embedded taus with their ES)
    signal_shapes: tuple[str, ...] = ()  # Up/Down pairs of the signal written as shape nuisances (without the direction)
    normalisations: Mapping[str, tuple[tuple[str, ...], float]] = field(default_factory=dict)  # lnN -> (processes, kappa)
    fake_processes: tuple[str, ...] = ()  # the pass/fail anti-correlated lnN of the fake trigger efficiency
    fake_efficiency: float = 0.10
    mc_process: str | None = None  # genuine MC taus: eps_MC
    monitor_process: str | None = None  # the embedded taus with their monitoring flags: eps_EMB^mon
    counting_regions: tuple[str, str] | None = None  # the counting cross-check: (region, its same-sign version)
    counting_window: tuple[float, float] = (40.0, 90.0)  # of the category variable (m_vis) in the counting
    qcd_subtract: tuple[str, ...] = ()  # the simulated processes subtracted from the same-sign data in a counting
    suggest_regions: tuple[str, str] = (NOMINAL, "same_sign")
    max_relative_error: float = 0.03  # --suggest-binning: of eps_data per bin
    reference: str | None = None  # a payload with tau_trigger (the POG file) drawn in the plots
    trigtype: str = "ditau"
    name: str = "tau_trigger"

    def run(self, context: MeasureContext) -> None:
        if context.suggest_binning:
            suggest(self, context)
        elif context.merge:
            merge(self, context)
        else:
            measure(self, context)


def check(analysis: Analysis, measurement: TauTriggerMeasurement) -> None:
    """The analysis has the channel, the pass and fail categories of every bin, the processes, variations and regions
    the measurement names."""
    if measurement.channel not in analysis.channels:
        raise ValueError(f"tau_trigger: channel {measurement.channel} is missing")
    channel = analysis.channel(measurement.channel)
    categories = {c.name for c in channel.categories}
    processes = {p.name for p in channel.processes}
    variations = {v.name for v in channel.variations}
    problems = [f"category {b.category(side)} is missing" for b in measurement.bins for side in SIDES if b.category(side) not in categories]
    problems += [f"process {name} is missing" for name in (measurement.signal, measurement.mc_process, measurement.monitor_process, *measurement.qcd_subtract) if name and name not in processes]
    if measurement.signal_template and measurement.signal_template not in variations:
        problems.append(f"the signal template {measurement.signal_template} is not a variation of {measurement.channel}")
    problems += [f"shape variation {name}{d} is missing" for name in measurement.signal_shapes for d in ("Up", "Down") if f"{name}{d}" not in variations]
    regions = {NOMINAL, *(r.name for r in channel.regions)}
    problems += [f"region {name} is missing" for name in (*(measurement.counting_regions or ()), *measurement.suggest_regions) if name not in regions]
    names = [b.name for b in measurement.bins]
    problems += [f"bin {name} is defined twice" for name in sorted({n for n in names if names.count(n) > 1})]
    if problems:
        raise ValueError("tau_trigger: " + "; ".join(problems))


def _variable(analysis: Analysis, measurement: TauTriggerMeasurement, category: str) -> str:
    return next(c.variable.name for c in analysis.channel(measurement.channel).categories if c.name == category)


def collect(hset: HistogramSet, analysis: Analysis, measurement: TauTriggerMeasurement, trigger_bin: Bin) -> cards.TriggerBin:
    """The histograms of a bin for its card: data, the signal template (pass + fail, its template variation if
    any) with its shape variations, the backgrounds (the estimator outputs included) of pass and fail."""
    channel = analysis.channel(measurement.channel)

    def get(process: str, side: str, variation: str = NOMINAL_VARIATION) -> Histogram | None:
        category = trigger_bin.category(side)
        return hset.get(HistKey(measurement.channel, category, process, NOMINAL, variation, _variable(analysis, measurement, category)))

    def signal(variation: str) -> Histogram:
        parts = [get(measurement.signal, side, variation) for side in SIDES]
        if any(h is None for h in parts):
            raise KeyError(f"{trigger_bin.name}: the {measurement.signal} histograms of variation {variation} are missing")
        return parts[0].copy().add(parts[1])

    template = measurement.signal_template or NOMINAL_VARIATION
    shapes = {}
    for name in measurement.signal_shapes:
        for direction in ("Up", "Down"):
            variation = f"{name}{direction}"
            on_top = on_template(variation, template) if measurement.signal_template else None
            shapes[variation] = signal(on_top if on_top and get(measurement.signal, "pass", on_top) is not None else variation)
    data = {side: get(channel.data(), side) for side in SIDES}
    if any(h is None for h in data.values()):
        raise KeyError(f"{trigger_bin.name}: data histograms missing")
    backgrounds = {side: {name: h for name in channel.backgrounds() if name != measurement.signal and (h := get(name, side)) is not None} for side in SIDES}
    return cards.TriggerBin(trigger_bin.name, data, signal(template), backgrounds, shapes)


def _histogram_count(hset: HistogramSet, analysis: Analysis, measurement: TauTriggerMeasurement, trigger_bin: Bin, process: str, variation: str) -> dict | None:
    """The efficiency of one process from the sums of its pass and fail histograms."""
    counts = {}
    for side in SIDES:
        category = trigger_bin.category(side)
        h = hset.get(HistKey(measurement.channel, category, process, NOMINAL, variation, _variable(analysis, measurement, category)))
        if h is None:
            return None
        counts[side] = Count(h.sum(), float(h.variances.sum()))
    return efficiency(counts["pass"], counts["fail"])


def event_counts(context: MeasureContext, measurement: TauTriggerMeasurement, processes: tuple[str, ...], regions: tuple[str, str], window: tuple[float, float] | None) -> dict:
    """(process, region, bin, side) -> Count from the events of the skims: the category cut of the side and, with
    `window`, the category variable inside it."""
    channel = context.analysis.channel(measurement.channel)
    categories = {c.name: c for c in channel.categories}
    needed = [categories[b.category(side)] for b in measurement.bins for side in SIDES]
    columns = columns_of([c.cut for c in needed] + [c.variable.expr for c in needed])
    result = {}
    for process in processes:
        for region in regions:
            events = context.events(Query(measurement.channel, process, region), columns)
            frame, weights = events.frame, events.weights
            for b in measurement.bins:
                for side in SIDES:
                    category = categories[b.category(side)]
                    selected = mask(frame, [category.cut]) if len(frame) else np.zeros(0, dtype=bool)
                    if window is not None and len(frame):
                        values = evaluate(frame, category.variable.expr).astype(np.float64)
                        selected &= (values > window[0]) & (values < window[1])
                    w = weights[selected]
                    result[process, region, b.name, side] = Count(float(w.sum()), float((w * w).sum()))
    return result


def _background_processes(analysis: Analysis, measurement: TauTriggerMeasurement) -> tuple[str, ...]:
    return tuple(p.name for p in analysis.channel(measurement.channel).processes if p.role == "background" and p.name != measurement.signal)


def counting(context: MeasureContext, measurement: TauTriggerMeasurement, regions: tuple[str, str], window: tuple[float, float] | None) -> dict[str, dict | None]:
    """bin -> the counting efficiency of data in `regions` (and the m_vis `window`)."""
    analysis = context.analysis
    data = analysis.channel(measurement.channel).data()
    backgrounds = _background_processes(analysis, measurement)
    processes = tuple(dict.fromkeys((data, *backgrounds, *measurement.qcd_subtract)))
    counts = event_counts(context, measurement, processes, regions, window)
    return {
        b.name: data_efficiency(lambda process, region, side, name=b.name: counts[process, region, name, side], data, backgrounds, measurement.qcd_subtract, *regions)
        for b in measurement.bins
    }


def measure(measurement: TauTriggerMeasurement, context: MeasureContext) -> Path:
    analysis, config = context.analysis, context.config
    check(analysis, measurement)
    directory = context.output / measurement.wp_class / measurement.variant
    shapes = directory / "shapes.root"
    previous = json.loads((directory / "results.json").read_text()) if context.refit and (directory / "results.json").exists() else None
    if context.refit:
        if not shapes.exists():
            raise FileNotFoundError(f"tau_trigger --refit: {shapes} of an earlier run is missing")
        hset = HistogramSet.load(shapes)
        logger.info(f"tau_trigger {measurement.wp_class}/{measurement.variant}: refit from {shapes}")
    else:
        logger.info(f"tau_trigger {measurement.wp_class}/{measurement.variant}: histograms of {len(measurement.bins)} bins -> {directory}")
        hset = run_hist(config, analysis, [measurement.channel], False, None, True, None, shapes)
        run_estimates(hset, analysis, [measurement.channel])
        hset.save(shapes)
    work = {}
    for b in measurement.bins:
        work[b.name] = directory / "fits" / b.name
        work[b.name].mkdir(parents=True, exist_ok=True)
        cards.write(collect(hset, analysis, measurement, b), work[b.name], measurement.signal, measurement.normalisations, measurement.fake_processes, measurement.fake_efficiency)
    logger.info(f"fitting {len(measurement.bins)} bins, {max(1, config.workers)} at a time")
    with ThreadPoolExecutor(max_workers=max(1, config.workers)) as pool:
        fits = dict(zip(work, pool.map(lambda path: fit_or_failure(config.combine, path), work.values())))
    if measurement.counting_regions is None:
        counted = {}
    elif previous is not None:
        counted = {name: entry.get("counting") for name, entry in previous["bins"].items()}
    else:
        counted = counting(context, measurement, measurement.counting_regions, measurement.counting_window)
    template = measurement.signal_template or NOMINAL_VARIATION
    record = {"wp_class": measurement.wp_class, "variant": measurement.variant, "bins": {}, "provenance": context.provenance(wp_class=measurement.wp_class, variant=measurement.variant, refit=context.refit)}
    for b in measurement.bins:
        entry = {"decay_modes": list(b.decay_modes), "pt": list(b.pt), **fits[b.name], "counting": counted.get(b.name)}
        entry["mc"] = _histogram_count(hset, analysis, measurement, b, measurement.mc_process, NOMINAL_VARIATION) if measurement.mc_process else None
        entry["emb_monitor"] = _histogram_count(hset, analysis, measurement, b, measurement.monitor_process, template if measurement.monitor_process == measurement.signal else NOMINAL_VARIATION) if measurement.monitor_process else None
        entry.setdefault("warnings", [])
        if entry["fit"] and entry["counting"]:
            fit, count = entry["fit"], entry["counting"]
            if abs(fit["best"] - count["value"]) > math.hypot(0.5 * (fit["high"] - fit["low"]), count["error"]):
                entry["warnings"].append(f"fit {fit['best']:.4f} and counting {count['value']:.4f} differ by more than 1 sigma")
        record["bins"][b.name] = entry
        logger.info(f"{b.name}: eff {_text(entry['fit'])}" + (f", counting {entry['counting']['value']:.4f}" if entry["counting"] else "") + (f", problems: {'; '.join(entry['problems'])}" if entry["problems"] else ""))
    (directory / "results.json").write_text(json.dumps(record, indent=2))
    (directory / "report.md").write_text(report({measurement.wp_class: {measurement.variant: record}}))
    plot_record(record, directory / "plots", measurement)
    problems = [f"{name}: {problem}" for name, entry in record["bins"].items() for problem in entry["problems"]]
    if problems:
        raise ValueError(f"tau_trigger {measurement.wp_class}/{measurement.variant}, see {directory / 'results.json'}:\n" + "\n".join(problems))
    return directory


def _text(interval: dict | None) -> str:
    return "-" if not interval else f"{interval['best']:.4f} [{interval['low']:.4f}, {interval['high']:.4f}]"


def _value(entry: dict | None) -> str:
    return "-" if not entry else f"{entry['value']:.4f} +- {entry['error']:.4f}"


def report(records: Mapping[str, Mapping[str, dict]]) -> str:
    """Markdown: per class, variant and bin the fit, the counting, MC and EMB monitoring efficiencies and the flags."""
    lines = ["| class | variant | bin | eff (fit) | counting | MC | EMB monitor | flags |", "|---|---|---|---|---|---|---|---|"]
    for wp, variants in records.items():
        for variant, record in variants.items():
            for name, entry in record["bins"].items():
                flags = "; ".join([f"PROBLEM {p}" for p in entry["problems"]] + [f"check: {w}" for w in entry.get("warnings", [])]) or "ok"
                lines.append(f"| {wp} | {variant} | {name} | {_text(entry['fit'])} | {_value(entry.get('counting'))} | {_value(entry.get('mc'))} | {_value(entry.get('emb_monitor'))} | {flags} |")
    return "\n".join(lines) + "\n"


def _by_decay_mode(bins: Mapping[str, dict]) -> dict[tuple[int, ...], list[dict]]:
    groups: dict[tuple[int, ...], list[dict]] = {}
    for name, entry in bins.items():
        groups.setdefault(tuple(entry["decay_modes"]), []).append({**entry, "name": name})
    return {modes: sorted(entries, key=lambda e: e["pt"][0]) for modes, entries in groups.items()}


def _reference(measurement: TauTriggerMeasurement, wp: str, dm: int):
    if measurement.reference is None:
        return None
    import correctionlib

    evaluator = correctionlib.CorrectionSet.from_file(measurement.reference)["tau_trigger"]
    try:
        evaluator.evaluate(60.0, dm, measurement.trigtype, wp, "eff_data", "nom")
    except Exception:  # noqa: BLE001 - the reference lacks this class or decay mode
        return None
    return lambda pt: evaluator.evaluate(max(pt, 40.0), dm, measurement.trigtype, wp, "eff_data", "nom")


def plot_record(record: dict, directory: Path, measurement: TauTriggerMeasurement) -> list[Path]:
    paths = []
    for modes, entries in _by_decay_mode(record["bins"]).items():
        label = "all decay modes" if modes == (-1,) else "DM " + "+".join(map(str, modes))
        tag = "DMall" if modes == (-1,) else "DM" + "".join(map(str, modes))
        paths += plots.plot_efficiencies(entries, f"{record['wp_class']}, {record['variant']}, {label}", Path(directory) / f"eff_{tag}", _reference(measurement, record["wp_class"], modes[0]))
    return paths


def suggest(measurement: TauTriggerMeasurement, context: MeasureContext) -> dict:
    """Per decay-mode group the pT edges where every bin keeps the relative counting error of eps_data at most
    `max_relative_error`: from the highest-pT bin down, a bin is merged with the next lower one until it qualifies
    (a remainder at the low end joins the bin above it). Logs the edges, writes nothing."""
    check(context.analysis, measurement)
    data = context.analysis.channel(measurement.channel).data()
    backgrounds = _background_processes(context.analysis, measurement)
    processes = tuple(dict.fromkeys((data, *backgrounds, *measurement.qcd_subtract)))
    counts = event_counts(context, measurement, processes, measurement.suggest_regions, None)
    suggestions = {}
    for modes, entries in _by_decay_mode({b.name: {"decay_modes": b.decay_modes, "pt": b.pt} for b in measurement.bins}).items():
        groups, current = [], []
        for entry in reversed(entries):
            current.insert(0, entry["name"])
            eff = data_efficiency(lambda process, region, side, names=tuple(current): _sum(counts, process, region, names, side), data, backgrounds, measurement.qcd_subtract, *measurement.suggest_regions)
            if relative_error(eff) <= measurement.max_relative_error:
                groups.insert(0, current)
                current = []
        if current:
            if groups:
                groups[0] = current + groups[0]
            else:
                groups = [current]
        pts = {e["name"]: e["pt"] for e in entries}
        edges = [pts[group[0]][0] for group in groups] + [pts[groups[-1][-1]][1]]
        suggestions["DMall" if modes == (-1,) else "DM" + "".join(map(str, modes))] = edges
        logger.info(f"tau_trigger {measurement.wp_class} {modes}: suggested pT edges {edges} (max relative error {measurement.max_relative_error:g})")
    logger.info(f"tau_trigger {measurement.wp_class} suggested edges (JSON): {json.dumps(suggestions)}")
    return suggestions


def _sum(counts: dict, process: str, region: str, names: tuple[str, ...], side: str) -> Count:
    total = Count(0.0, 0.0)
    for name in names:
        total = total + counts[process, region, name, side]
    return total


def _payload_keys(decay_modes: list[int]) -> list[int]:
    return [-1] if list(decay_modes) == [-1] else list(decay_modes)


def merged_values(measurement: TauTriggerMeasurement, variants: Mapping[str, dict]) -> tuple[dict, dict, list[str]]:
    """The eff_data and eff_mc values of one class per payload decay mode (bins in pT order), and the warnings: the
    fit interval, plus the changes of the variants and the fit-counting difference beyond its error."""
    nominal = variants["nominal"]
    warnings = [f"variant {name} missing: its uncertainty is not included" for name in measurement.systematic_variants if name not in variants]
    data, mc = {}, {}
    for name, entry in nominal["bins"].items():
        fit = entry["fit"]
        best = fit["best"]
        changes = {}
        for variant, weight in measurement.systematic_variants.items():
            other = variants.get(variant, {}).get("bins", {}).get(name, {}).get("fit")
            if other is not None:
                changes[variant] = weight * (other["best"] - best)
        count = entry.get("counting")
        if count and abs(best - count["value"]) > math.hypot(0.5 * (fit["high"] - fit["low"]), count["error"]):
            changes["method"] = best - count["value"]
        syst = math.sqrt(sum(change * change for change in changes.values()))
        values = {"nom": best, "up": min(1.0, best + math.hypot(fit["high"] - best, syst)), "down": max(EFFICIENCY_FLOOR, best - math.hypot(best - fit["low"], syst))}
        entry["systematics"] = changes
        entry["payload"] = values
        for dm in _payload_keys(entry["decay_modes"]):
            data.setdefault(dm, []).append((entry["pt"][0], entry["pt"][1], values))
        if entry.get("mc"):
            m = entry["mc"]
            mc_values = {"nom": m["value"], "up": min(1.0, m["value"] + m["error"]), "down": max(EFFICIENCY_FLOOR, m["value"] - m["error"])}
            for dm in _payload_keys(entry["decay_modes"]):
                mc.setdefault(dm, []).append((entry["pt"][0], entry["pt"][1], mc_values))
    order = {dm: sorted(bins, key=lambda b: b[0]) for dm, bins in data.items()}
    return order, {dm: sorted(bins, key=lambda b: b[0]) for dm, bins in mc.items()}, warnings


def merge(measurement: TauTriggerMeasurement, context: MeasureContext) -> Path:
    """The payload of every class measured below the output directory (its nominal variant and the others)."""
    output = Path(context.output)
    records: dict[str, dict[str, dict]] = {}
    for path in sorted(output.glob("*/*/results.json")):
        record = json.loads(path.read_text())
        records.setdefault(record["wp_class"], {})[record["variant"]] = record
    if not records:
        raise ValueError(f"tau_trigger: no results.json below {output}; run the measurement per class first")
    problems = [f"{wp}: the nominal variant is missing" for wp, variants in records.items() if "nominal" not in variants]
    problems += [f"{wp}/nominal/{name}: {problem}" for wp, variants in records.items() for name, entry in variants.get("nominal", {}).get("bins", {}).items() for problem in entry["problems"]]
    problems += [f"{wp}/nominal/{name}: no fit" for wp, variants in records.items() for name, entry in variants.get("nominal", {}).get("bins", {}).items() if not entry["problems"] and entry.get("fit") is None]
    if problems:
        raise ValueError("tau_trigger: results with problems, not merged:\n" + "\n".join(problems))
    results, warnings = {}, []
    for wp, variants in records.items():
        data, mc, notes = merged_values(measurement, variants)
        results[wp] = {"eff_data": data, **({"eff_mc": mc} if mc else {})}
        warnings += [f"{wp}: {note}" for note in notes]
    sources = {f"{wp}/{variant}": record["provenance"] for wp, variants in records.items() for variant, record in variants.items()}
    cset = correction_set(results, context.provenance(classes=sorted(records), measurements=sources), measurement.trigtype)
    problems = check_payload(cset, results, measurement.trigtype)
    (output / "report.md").write_text(report(records) + "".join(f"\n- {warning}" for warning in warnings) + "\n")
    if problems:
        raise ValueError("tau_trigger: the payload fails its checks:\n" + "\n".join(problems))
    for warning in warnings:
        logger.warning(warning)
    for wp, variants in records.items():
        plot_record(variants["nominal"], output / "plots" / wp, measurement)
    path = payloads.write(cset, output / payload_name(context.analysis.era))
    logger.info(f"payload of the classes {', '.join(records)}: {path}")
    return path

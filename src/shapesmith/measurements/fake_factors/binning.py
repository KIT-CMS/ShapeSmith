"""`measure --suggest-binning`: equipopulated bin edges from data, as TauFakeFactors' adjust_binning.py
(FF_Updated _equipopulated_binned_variable and get_binning).

Per category: the |weight|-weighted quantiles (inverted CDF) of the variable in [low, high], the outer edges set to
low and high, rounded, the `add_left` edges put in front. The events are the data of the quantity's region: the
SR-like region of the process for its fake factors and non-closures (the data/MC scale region for an MC fake factor),
the orthogonal SR-like region for the DR->SR correction and its non-closures, the fraction region for the fractions;
a DR->SR correction from MC (W+jets) takes its MC target, in the SR for the correction and in the orthogonal
SR-like region for its non-closures. The last log line holds
every suggestion as JSON (quantity -> edges per category), to be entered into the analysis tables of an era.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    from shapesmith.measurements.fake_factors.measure import EventSource
    from shapesmith.measurements.fake_factors.model import Binned, Equipopulated, FakeFactorMeasurement, Split

logger = logging.getLogger(__name__)


def equipopulated(x: np.ndarray, weights: np.ndarray, options: Equipopulated, n_bins: int) -> list[float]:
    inside = (x >= options.low) & (x <= options.high)
    edges = np.quantile(x[inside], np.linspace(0, 1, n_bins + 1), weights=np.abs(weights[inside]), method="inverted_cdf") if inside.any() else np.array([options.low, options.high])
    edges[0], edges[-1] = options.low, options.high
    return [*options.add_left, *(round(edge, options.rounding) for edge in np.round(edges, options.rounding).tolist())]


def quantities(measurement: FakeFactorMeasurement, channel: str, data: str = "data") -> Iterator[tuple[str, Binned, Split, str, str]]:
    """(name, quantity, split, region, process) of every quantity: the events of `process` (the data, or the MC
    target of a DR->SR correction from MC) in `region` give its equipopulated edges."""
    for leg in measurement.legs[channel]:
        for process in leg.processes:
            name = process.name + leg.suffix
            region = process.scale.sr_like if process.scale is not None else process.sr_like
            yield f"{name}_fake_factors", process.fake_factors, process.split, region, data
            yield from ((f"{name}_non_closure_{b.variable}", b, process.split, region, data) for b in process.non_closures)
            if process.dr_sr is not None:
                dr_sr = process.dr_sr
                if dr_sr.target is None:
                    yield f"{name}_DR_SR", dr_sr.correction, process.split, dr_sr.sr_like, data
                    yield from ((f"{name}_for_DRtoSR_non_closure_{b.variable}", b, process.split, dr_sr.sr_like, data) for b in dr_sr.non_closures)
                else:
                    yield f"{name}_DR_SR", dr_sr.correction, process.split, dr_sr.sr, dr_sr.target
                    yield from ((f"{name}_for_DRtoSR_non_closure_{b.variable}", b, process.split, dr_sr.sr_like, dr_sr.target) for b in dr_sr.non_closures)
        yield f"process_fractions{leg.suffix}", leg.fractions.binned, leg.fractions.split, leg.fractions.region, data


def suggestions(measurement: FakeFactorMeasurement, source: EventSource) -> list[tuple[str, int, list[float], list[float]]]:
    """(quantity, category, suggested edges, current edges)."""
    result = []
    for name, binned, split, region, process in quantities(measurement, source.channel, source.data):
        if binned.equipopulated is None:
            continue
        events = source.events(process, region)
        split_values = events.frame[split.variable].to_numpy()
        for category in split.categories:
            selected = (split_values >= split.edges[category]) & (split_values < split.edges[category + 1])
            x = events.frame[binned.variable].to_numpy()[selected]
            suggested = equipopulated(x, events.weights[selected], binned.equipopulated, binned.equipopulated.n_bins[category])
            result.append((name, category, suggested, list(binned.edges[category])))
    return result


def log_suggestions(measurement: FakeFactorMeasurement, source: EventSource) -> dict[str, list[list[float]]]:
    """Log every suggestion and, last, all of them as JSON; returns quantity -> suggested edges per category."""
    collected: dict[str, list[list[float]]] = {}
    for name, category, suggested, current in suggestions(measurement, source):
        logger.info(f"{source.channel} {name} category {category}: suggested {suggested}, current {current}")
        collected.setdefault(name, []).append(suggested)
    logger.info(f"{source.channel} suggested edges (JSON): {json.dumps(collected)}")
    return collected

"""`measure --suggest-binning`: equipopulated bin edges from data, as TauFakeFactors' adjust_binning.py
(FF_Updated _equipopulated_binned_variable and get_binning).

Per category: the |weight|-weighted quantiles (inverted CDF) of the variable in [low, high], the outer edges set to
low and high, rounded, the `add_left` edges put in front. The data are those of the quantity's region: the SR-like
region of the process for its fake factors and non-closures (the data/MC scale region for an MC fake factor), the
orthogonal SR-like region for the DR->SR correction and its non-closures, the fraction region for the fractions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    from shapesmith.measurements.fake_factors.measure import EventSource
    from shapesmith.measurements.fake_factors.model import Binned, Equipopulated, FakeFactorMeasurement, Split


def equipopulated(x: np.ndarray, weights: np.ndarray, options: Equipopulated, n_bins: int) -> list[float]:
    inside = (x >= options.low) & (x <= options.high)
    edges = np.quantile(x[inside], np.linspace(0, 1, n_bins + 1), weights=np.abs(weights[inside]), method="inverted_cdf") if inside.any() else np.array([options.low, options.high])
    edges[0], edges[-1] = options.low, options.high
    return [*options.add_left, *(round(edge, options.rounding) for edge in np.round(edges, options.rounding).tolist())]


def quantities(measurement: FakeFactorMeasurement, channel: str) -> Iterator[tuple[str, Binned, Split, str]]:
    """(name, quantity, split, region) of every quantity with equipopulated options."""
    for leg in measurement.legs[channel]:
        for process in leg.processes:
            name = process.name + leg.suffix
            region = process.scale.sr_like if process.scale is not None else process.sr_like
            yield f"{name}_fake_factors", process.fake_factors, process.split, region
            yield from ((f"{name}_non_closure_{b.variable}", b, process.split, region) for b in process.non_closures)
            if process.dr_sr is not None:
                dr_sr = process.dr_sr
                yield f"{name}_DR_SR", dr_sr.correction, process.split, dr_sr.sr_like
                yield from ((f"{name}_for_DRtoSR_non_closure_{b.variable}", b, process.split, dr_sr.sr_like) for b in dr_sr.non_closures)
        yield f"process_fractions{leg.suffix}", leg.fractions.binned, leg.fractions.split, leg.fractions.region


def suggestions(measurement: FakeFactorMeasurement, source: EventSource) -> list[tuple[str, int, list[float], list[float]]]:
    """(quantity, category, suggested edges, current edges)."""
    result = []
    for name, binned, split, region in quantities(measurement, source.channel):
        if binned.equipopulated is None:
            continue
        events = source.events(source.data, region)
        split_values = events.frame[split.variable].to_numpy()
        for category in split.categories:
            selected = (split_values >= split.edges[category]) & (split_values < split.edges[category + 1])
            x = events.frame[binned.variable].to_numpy()[selected]
            suggested = equipopulated(x, events.weights[selected], binned.equipopulated, binned.equipopulated.n_bins[category])
            result.append((name, category, suggested, list(binned.edges[category])))
    return result


def print_suggestions(measurement: FakeFactorMeasurement, source: EventSource) -> None:
    for name, category, suggested, current in suggestions(measurement, source):
        print(f"{source.channel} {name} category {category}: suggested {suggested}, current {current}")

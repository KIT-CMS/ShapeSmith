"""The fake-factor measurement of one channel, in TauFakeFactors' order (FF_Updated ff_calculation.py and
ff_corrections.py):

1. per leg the QCD and ttbar fake factors and the process fractions -> fake_factors_<ch>.json.gz;
2. per QCD leg the orthogonal fake factors and their non-closures in the DR->SR regions, then the DR->SR correction
   (the intermediate payloads -> FF_for_DRtoSR_<ch>.json.gz);
3. per process the non-closures in table order, each measured with the fake factors, the DR->SR correction (QCD) and
   the earlier non-closures applied per event -> FF_corrections_<ch>.json.gz.

Fake factors and corrections are applied per event with correctionlib, from the payload models of this run, as the
CROWN friend applies them. A "yield" is the histogram of the target process minus the subtracted processes, nominal
and with every subtracted histogram shifted by +-1 sigma (SystMCShift).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable

import correctionlib.schemav2 as cs
import numpy as np

from shapesmith import payloads
from shapesmith.events import Events, Query
from shapesmith.measurements.fake_factors import payload, plots
from shapesmith.measurements.fake_factors.binning import log_suggestions
from shapesmith.measurements.fake_factors.fit import Fitted, fit
from shapesmith.measurements.fake_factors.hist import Hist

if TYPE_CHECKING:
    from shapesmith.measurements import MeasureContext
    from shapesmith.measurements.fake_factors.model import Binned, FakeFactorMeasurement, Fractions, Leg, ProcessFF, Split

logger = logging.getLogger(__name__)

Yields = tuple[Hist, Hist, Hist]  # nominal, subtracted shifted up, subtracted shifted down


@dataclass(frozen=True)
class Ratio:
    """A measured ratio of the nominal yields, the ratios with shifted subtraction (None without subtraction) and the
    nominal yields themselves (the closure plots)."""

    nominal: Hist
    mc_shifted: tuple[Hist, Hist] | None
    numerator: Hist
    denominator: Hist


class EventSource:
    """The selected events of the processes of one channel per region, each read once."""

    def __init__(self, context: MeasureContext, channel: str, columns: set[str]):
        self.channel = channel
        self.data = context.analysis.channel(channel).data()
        self._context, self._columns, self._events = context, columns, {}

    def events(self, process: str, region: str) -> Events:
        if (process, region) not in self._events:
            self._events[process, region] = self._context.events(Query(self.channel, process, region), self._columns)
            logger.debug(f"{self.channel}: {process} in {region}, {len(self._events[process, region].weights)} events, sum of weights {self._events[process, region].weights.sum():.4g}")
        return self._events[process, region]

    def hist(self, process: str, region: str, variable: str, edges, split: Split, category: int, weights: np.ndarray | None = None) -> Hist:
        """`variable` in one category of the split, with the event weights or the given per-event weights."""
        events = self.events(process, region)
        values = events.frame[split.variable].to_numpy()
        selected = (values >= split.edges[category]) & (values < split.edges[category + 1])
        weights = events.weights if weights is None else weights
        return Hist.fill(edges, events.frame[variable].to_numpy()[selected], weights[selected])

    def weighted(self, process: str, region: str, corrections: list[Callable]) -> np.ndarray:
        """The event weights times the corrections evaluated per event, multiplied in order."""
        events = self.events(process, region)
        weights = events.weights
        for correction in corrections:
            weights = weights * correction(events.frame)
        return weights


def evaluator(correction: cs.Correction) -> Callable:
    """The nominal of a payload correction per event, with the inputs as float (as TauFakeFactors and the friend)."""
    evaluate = payloads.correction_set([correction], {}).to_evaluator()[correction.name].evaluate
    columns = [variable.name for variable in correction.inputs[:-1]]  # the last input is syst
    return lambda frame: evaluate(*(np.asarray(frame[c], dtype=np.float32).astype(np.float64) for c in columns), "nominal")


class ChannelMeasurement:
    """The measurement of one channel: the payload corrections and a record of every measured ratio and fit."""

    def __init__(self, measurement: FakeFactorMeasurement, source: EventSource):
        self.measurement, self.source = measurement, source
        self.fake_factors: list[cs.Correction] = []
        self.intermediate: list[cs.Correction] = []  # the orthogonal fake factors and their non-closures
        self.corrections: list[cs.Correction] = []
        self.compounds: list[cs.CompoundCorrection] = []
        self.record: dict[str, list[dict]] = {}

    def measure(self, legs: tuple[Leg, ...]) -> None:
        for leg in legs:
            self.fake_factors += [self.process_fake_factors(p.name + leg.suffix, p, p.sr_like, p.ar_like, p.subtract) for p in leg.processes]
            self.fake_factors.append(self.fractions(f"process_fractions{leg.suffix}", leg.fractions))
        by_name = {c.name: c for c in self.fake_factors}
        for leg in legs:
            for process in leg.processes:
                name = process.name + leg.suffix
                applied = [by_name[f"{name}_fake_factors"]]
                if process.dr_sr is not None:
                    applied.append(self.dr_sr(name, process))
                closures = self.non_closures(name, process, process.non_closures, process.sr_like, process.ar_like, process.subtract, applied)
                self.corrections += closures
                if closures:
                    self.compounds.append(payload.compound(name, process.non_closures, process.split))

    def dr_sr(self, name: str, process: ProcessFF) -> cs.Correction:
        """The DR->SR correction, measured with the orthogonal fake factors and their non-closures."""
        dr_sr = process.dr_sr
        orthogonal = self.process_fake_factors(name, process, dr_sr.sr_like, dr_sr.ar_like, dr_sr.subtract, stage="for_DRtoSR/")
        closures = self.non_closures(name, process, dr_sr.non_closures, dr_sr.sr_like, dr_sr.ar_like, dr_sr.subtract, [orthogonal], stage="for_DRtoSR/")
        self.intermediate += [orthogonal, *closures]
        correction = self.closure(name, "DR_SR", process, dr_sr.correction, dr_sr.sr, dr_sr.ar, dr_sr.subtract, [orthogonal, *closures])
        self.corrections.append(correction)
        return correction

    def process_fake_factors(self, name: str, process: ProcessFF, sr_like: str, ar_like: str, subtract: tuple[str, ...], stage: str = "") -> cs.Correction:
        """{name}_fake_factors, the target yield in `sr_like` over that in `ar_like`, times the data/MC factor of an
        MC fake factor."""
        scale = None if process.scale is None else self.data_scale(process)
        if scale is not None:
            logger.info(f"{self.source.channel} {stage}{name}: data/MC scale factor {scale:.4f}")
            self.record[f"{stage}{name}_data_scale"] = [{"factor": scale}]

        def measure(category: int) -> Ratio:
            ratio = self.ratio(process, process.fake_factors, category, (sr_like, ar_like), subtract, from_data=scale is None)
            return ratio if scale is None else replace(ratio, nominal=ratio.nominal.scaled(scale))

        fitted = self.fitted(f"{stage}{name}_fake_factors", process.fake_factors, process.split, measure)
        return payload.fake_factors(name, process.fake_factors, process.split, fitted, self.measurement.version)

    def closure(self, name: str, correction_name: str, process: ProcessFF, binned: Binned, sr_like: str, ar_like: str, subtract: tuple[str, ...], applied: list[cs.Correction], stage: str = "") -> cs.Correction:
        """{name}_{correction_name}_correction: the target yield in `sr_like` over that in `ar_like` weighted per event
        with the `applied` fake factors and corrections. From data, bins <= 0 are set to 1 +- 1."""
        from_data = process.scale is None
        evaluators = [evaluator(correction) for correction in applied]
        weights = {p: self.source.weighted(p, ar_like, evaluators) for p in (process.target, *subtract)}

        def measure(category: int) -> Ratio:
            ratio = self.ratio(process, binned, category, (sr_like, ar_like), subtract, from_data, weights)
            if not from_data:
                return ratio
            return replace(ratio, nominal=ratio.nominal.with_empty_bins_set_to_one(), mc_shifted=tuple(h.with_empty_bins_set_to_one() for h in ratio.mc_shifted))

        fitted = self.fitted(f"{stage}{name}_{correction_name}", binned, process.split, measure)
        return payload.correction(name, correction_name, binned, process.split, fitted, self.measurement.version)

    def non_closures(self, name: str, process: ProcessFF, non_closures: tuple[Binned, ...], sr_like: str, ar_like: str, subtract: tuple[str, ...], applied: list[cs.Correction], stage: str = "") -> list[cs.Correction]:
        """The non-closures in table order, each on top of `applied` and the earlier ones."""
        result = []
        for binned in non_closures:
            result.append(self.closure(name, f"non_closure_{binned.variable}", process, binned, sr_like, ar_like, subtract, applied + result, stage))
        return result

    def ratio(self, process: ProcessFF, binned: Binned, category: int, regions: tuple[str, str], subtract: tuple[str, ...], from_data: bool, ar_weights: dict[str, np.ndarray] | None = None) -> Ratio:
        """The yield in the SR-like over the (weighted) yield in the AR-like region; from data also the ratios of the
        shifted yields."""
        sr_like, ar_like = regions
        numerator = self.yields(process.target, subtract, sr_like, binned, process.split, category)
        denominator = self.yields(process.target, subtract, ar_like, binned, process.split, category, ar_weights)
        nominal, up, down = (n.divide(d) for n, d in zip(numerator, denominator))
        return Ratio(nominal, (up, down) if from_data else None, numerator[0], denominator[0])

    def yields(self, target: str, subtract: tuple[str, ...], region: str, binned: Binned, split: Split, category: int, weights: dict[str, np.ndarray] | None = None) -> Yields:
        edges = binned.edges[category]

        def hist(process: str) -> Hist:
            return self.source.hist(process, region, binned.variable, edges, split, category, weights[process] if weights else None)

        nominal = up = down = hist(target)
        for process in subtract:
            h = hist(process)
            nominal, up, down = nominal.add(h, -1.0), up.add(h.shifted(1.0), -1.0), down.add(h.shifted(-1.0), -1.0)
        return nominal, up, down

    def data_scale(self, process: ProcessFF) -> float:
        """The global data/MC factor of an MC fake factor, from the event totals of its four scale regions (summed and
        subtracted in order, as TauFakeFactors' one-bin histograms)."""
        scale, data = process.scale, self.source.data

        def total(name: str, region: str) -> float:
            weights = self.source.events(name, region).weights
            return float(np.cumsum(weights)[-1]) if len(weights) else 0.0

        def minus(value: float, names, region: str) -> float:
            for name in names:
                value -= total(name, region)
            return value

        def data_minus_mc(region: str, same_sign: str) -> float:
            qcd = max(0.0, minus(total(data, same_sign), scale.subtract, same_sign))
            return minus(total(data, region), (p for p in scale.subtract if p != process.target), region) - qcd

        ratio = data_minus_mc(scale.sr_like, scale.sr_like_same_sign) / data_minus_mc(scale.ar_like, scale.ar_like_same_sign)
        return ratio / (total(process.target, scale.sr_like) / total(process.target, scale.ar_like))

    def fractions(self, name: str, fractions: Fractions) -> cs.Correction:
        """process_fractions{s}: QCD and ttbar over their sum, nominal and with each of them shifted by +-1 sigma."""
        values: dict[str, dict[str, list[np.ndarray]]] = {}
        for category in fractions.split.categories:
            hists = self.fraction_hists(fractions, category)
            variations = {"nominal": hists}
            for varied in hists:
                variations[f"frac_{varied}_up"] = {**hists, varied: hists[varied].shifted(1.0)}
                variations[f"frac_{varied}_down"] = {**hists, varied: hists[varied].shifted(-1.0)}
            for variation, shifted in variations.items():
                for process, h in shifted.items():
                    total = h
                    for other in (p for p in shifted if p != process):
                        total = total.add(shifted[other])
                    values.setdefault(variation, {}).setdefault(process, []).append(h.divide(total).values)
            self.record.setdefault(name, []).append({
                "category": category, "edges": list(fractions.binned.edges[category]),
                "yields": {p: h.values.tolist() for p, h in hists.items()}, "fractions": {p: values["nominal"][p][-1].tolist() for p in hists},
            })
        return payload.fractions(name, fractions.binned, fractions.split, values, self.measurement.version)

    def fraction_hists(self, fractions: Fractions, category: int) -> dict[str, Hist]:
        """QCD = data - all MC with negative bins 0, and ttbar, in one category."""
        def hist(process: str) -> Hist:
            return self.source.hist(process, fractions.region, fractions.binned.variable, fractions.binned.edges[category], fractions.split, category)

        qcd = hist(self.source.data)
        for process in fractions.subtract:
            qcd = qcd.add(hist(process), -1.0)
        return {"QCD": qcd.with_negative_bins_zeroed(), "ttbar_J": hist(fractions.ttbar)}

    def fitted(self, name: str, binned: Binned, split: Split, measure: Callable[[int], Ratio]) -> list[Fitted]:
        """The fit per category of the ratio `measure(category)`, recorded under `name`."""
        result = []
        for category in split.categories:
            ratio = measure(category)
            fitted = fit(ratio.nominal, binned.fits[category], ratio.mc_shifted, self.measurement.stat_sigma)
            logger.debug(
                f"{self.source.channel} {name} category {category} ({split.variable} {split.edges[category]}-{split.edges[category + 1]}): {binned.variable} fit {binned.fits[category]}, "
                f"ratio {', '.join(f'{v:.3g}' for v in ratio.nominal.values)}" + (f", p-value {fitted.p_value:.3g}" if fitted.p_value is not None else "")
            )
            if fitted.reset:
                logger.info(f"{self.source.channel} {name} category {category}: compatible with 1 (p-value {fitted.p_value:.3g}), set to 1")
            self.record.setdefault(name, []).append(_record(binned.variable, split, category, ratio, fitted))
            result.append(fitted)
        return result


def run(measurement: FakeFactorMeasurement, context: MeasureContext) -> None:
    for channel in context.channels:
        source = EventSource(context, channel, measurement.columns(channel))
        if context.suggest_binning:
            log_suggestions(measurement, source)
            continue
        logger.info(f"{channel}: measuring the fake factors of {len(measurement.legs[channel])} legs")
        result = ChannelMeasurement(measurement, source)
        result.measure(measurement.legs[channel])
        logger.info(f"{channel}: {len(result.fake_factors)} fake factors and fractions, {len(result.corrections)} corrections, {len(result.intermediate)} intermediate corrections")
        provenance, output = context.provenance(channel=channel), context.output
        payloads.write(payloads.correction_set(result.fake_factors, provenance), output / f"fake_factors_{channel}.json.gz")
        payloads.write(payloads.correction_set(result.corrections, provenance, result.compounds), output / f"FF_corrections_{channel}.json.gz")
        payloads.write(payloads.correction_set(result.intermediate, provenance), output / f"FF_for_DRtoSR_{channel}.json.gz")
        (output / f"measurement_{channel}.json").write_text(json.dumps(result.record, indent=1))
        plots.plot_record(result.record, output / "plots" / channel)
        logger.info(f"{channel}: payloads written to {output}")


def _points(h: Hist) -> dict:
    return {"x": h.means.tolist(), "y": h.values.tolist(), "errors": h.errors.tolist()}


def _record(variable: str, split: Split, category: int, ratio: Ratio, fitted: Fitted) -> dict:
    """The fit inputs (the ratio and the shifted ratios as points at their centres of mass, the yields) and results."""
    return {
        "variable": variable, "category": category, "split": [split.edges[category], split.edges[category + 1]],
        "edges": ratio.nominal.edges.tolist(), **_points(ratio.nominal),
        "mc_shifted": [_points(h) for h in ratio.mc_shifted] if ratio.mc_shifted else None,
        "numerator": ratio.numerator.values.tolist(), "denominator": ratio.denominator.values.tolist(),
        "p_value": fitted.p_value, "reset": fitted.reset,
        "curve": {"edges": fitted.curve.edges.tolist(), "nominal": fitted.curve.values.tolist(), **{k: v.tolist() for k, v in fitted.variations.items()}},
    }

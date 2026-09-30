"""Stage 3: processes estimated from histograms, run per channel in the order of `Channel.estimators`.

DataMinus: data minus the subtracted processes in its region, for the nominal and every column variation filled on
data or a subtracted process there; an input without that variation contributes its nominal (under the fill rule it
is unchanged). Weight variations are not propagated. Optionally negative bins are clipped.
ABCD: (data - MC)(B) * (data - MC)(C).sum() / (data - MC)(D).sum(), nominal only, negative bins clipped.
TemplateShift: `process` +/- `fraction` * `template`, e.g. the ttbar contamination of the embedded sample; also on top of
every template variation of `process`.
VariationSum: `name`Up/Down = nominal + the sum of (part - nominal) over its parts, for every histogram with a part;
place it after the estimators whose outputs should get the summed variation (DataMinus builds its output for the
parts as for any column variation).
"""
from __future__ import annotations

import logging

from dataclasses import replace

from shapesmith.histogram import NOMINAL_VARIATION, HistKey, Histogram, HistogramSet, is_template, on_template, part_of
from shapesmith.model import ABCD, NOMINAL, Analysis, Channel, ColumnVariation, DataMinus, TemplateShift, VariationSum

logger = logging.getLogger(__name__)


def data_minus(hset: HistogramSet, channel: Channel, category: str, variable: str, region: str, subtract: tuple[str, ...], variation: str = NOMINAL_VARIATION) -> Histogram | None:
    """data[region] - sum of the given processes[region] under `variation` (nominal where an input lacks it);
    None if the data histogram is missing."""

    def get(process: str) -> Histogram | None:
        key = HistKey(channel.name, category, process, region, variation, variable)
        if key not in hset:
            key = HistKey(channel.name, category, process, region, NOMINAL_VARIATION, variable)
        return hset.get(key)

    data = get(channel.data())
    if data is None:
        return None
    result = data.copy()
    for process in subtract:
        h = get(process)
        if h is None:
            logger.warning(f"{channel.name}_{category}/{process}#{region}#{variable} missing, not subtracted")
            continue
        result.add(h, -1.0)
    return result


def categories_and_variables(hset: HistogramSet, channel: Channel, region: str) -> list[tuple[str, str]]:
    return sorted({(k.category, k.variable) for k in hset.select(channel=channel.name, process=channel.data(), region=region, variation=NOMINAL_VARIATION)})


def estimate_data_minus(hset: HistogramSet, channel: Channel, estimator: DataMinus) -> list[HistKey]:
    column_variations = {v.name for v in channel.variations if isinstance(v, ColumnVariation)}
    inputs = (channel.data(), *estimator.subtract)
    added = []
    for category, variable in categories_and_variables(hset, channel, estimator.region):
        present = {k.variation for process in inputs for k in hset.select(channel=channel.name, category=category, process=process, region=estimator.region, variable=variable)}
        variations = (NOMINAL_VARIATION, *sorted(present & column_variations))
        for variation in variations:
            h = data_minus(hset, channel, category, variable, estimator.region, estimator.subtract, variation).scale(estimator.scale)
            key = HistKey(channel.name, category, estimator.output, NOMINAL, variation, variable)
            hset[key] = clip_negative_bins(h) if estimator.clip_negative else h
            added.append(key)
        nominal = hset[HistKey(channel.name, category, estimator.output, NOMINAL, NOMINAL_VARIATION, variable)]
        logger.debug(f"{channel.name}/{category}/{variable}: {estimator.output} = data - {' - '.join(estimator.subtract)} in {estimator.region}, yield {nominal.sum():.2f}, {len(variations) - 1} variations")
    return added


def clip_negative_bins(h: Histogram) -> Histogram:
    """Set negative bins to zero and rescale to the original integral (zero everything if the integral is <= 0)."""
    total = h.sum()
    positive = float(h.values[h.values > 0].sum())
    h.values[h.values < 0] = 0.0
    if total <= 0.0:
        logger.warning(f"integral {total:.3f} <= 0 after subtraction, histogram set to zero")
        h.values[:] = 0.0
        h.variances[:] = 0.0
    elif positive > 0.0 and positive != total:
        h.scale(total / positive)
    return h


def estimate_abcd(hset: HistogramSet, channel: Channel, estimator: ABCD) -> list[HistKey]:
    added = []
    for category, variable in categories_and_variables(hset, channel, estimator.b):
        b, c, d = (data_minus(hset, channel, category, variable, region, estimator.subtract) for region in (estimator.b, estimator.c, estimator.d))
        if b is None or c is None or d is None:
            logger.warning(f"{channel.name}/{category}/{variable}: ABCD control region missing, skipped")
            continue
        yield_c, yield_d = c.sum(), d.sum()
        if yield_c <= 0.0 or yield_d <= 0.0:
            logger.warning(f"{channel.name}/{category}/{variable}: C={yield_c:.2f}, D={yield_d:.2f}, {estimator.output} set to zero")
            factor = 0.0
        else:
            factor = yield_c / yield_d
        logger.debug(f"{channel.name}/{category}/{variable}: ABCD {estimator.output}, B={b.sum():.2f}, C={yield_c:.2f}, D={yield_d:.2f}, C/D={factor:.4f}")
        key = HistKey(channel.name, category, estimator.output, NOMINAL, NOMINAL_VARIATION, variable)
        hset[key] = clip_negative_bins(b.scale(factor))
        added.append(key)
    return added


def estimate_template_shift(hset: HistogramSet, channel: Channel, estimator: TemplateShift) -> list[HistKey]:
    added = []
    for key in hset.select(channel=channel.name, process=estimator.process, region=NOMINAL):
        if key.variation != NOMINAL_VARIATION and not is_template(key.variation):
            continue
        template = HistKey(channel.name, key.category, estimator.template, NOMINAL, NOMINAL_VARIATION, key.variable)
        if template not in hset:
            logger.warning(f"{template.path} missing, no {estimator.name} variation")
            continue
        for direction, sign in (("Up", 1.0), ("Down", -1.0)):
            name = f"{estimator.name}{direction}"
            variation = name if key.variation == NOMINAL_VARIATION else on_template(name, key.variation)
            varied = HistKey(channel.name, key.category, estimator.process, NOMINAL, variation, key.variable)
            hset[varied] = hset[key].copy().add(hset[template], sign * estimator.fraction)
            added.append(varied)
    return added


def estimate_variation_sum(hset: HistogramSet, channel: Channel, estimator: VariationSum) -> list[HistKey]:
    """The summed variation of every nominal histogram of the channel that has a part; the variances stay the
    nominal ones."""
    added = []
    for key in hset.select(channel=channel.name, variation=NOMINAL_VARIATION):
        for direction in ("Up", "Down"):
            name = f"{estimator.name}{direction}"
            parts = [replace(key, variation=part_of(name, part)) for part in estimator.parts]
            parts = [part for part in parts if part in hset]
            if not parts:
                continue
            nominal = hset[key]
            varied = nominal.copy()
            for part in parts:
                varied.values += hset[part].values - nominal.values
            hset[replace(key, variation=name)] = varied
            added.append(replace(key, variation=name))
    return added


ESTIMATES = {DataMinus: estimate_data_minus, ABCD: estimate_abcd, TemplateShift: estimate_template_shift, VariationSum: estimate_variation_sum}


def run_estimates(hset: HistogramSet, analysis: Analysis, channels: list[str]) -> list[HistKey]:
    added: list[HistKey] = []
    for name in channels:
        channel = analysis.channel(name)
        for estimator in channel.estimators:
            keys = ESTIMATES[type(estimator)](hset, channel, estimator)
            logger.info(f"{name}: {type(estimator).__name__} {getattr(estimator, 'output', None) or estimator.name}, {len(keys)} histograms")
            added += keys
    return added

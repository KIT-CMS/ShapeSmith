"""The categories of a ScoreBinning variable on a histogram set: their edges resolved from the nominal templates after
`estimate` (nn_binning.py, never from data), the negative bins left after the rebinning clipped, and the blinding
split (N14). `datacards` writes the cards from it, `plot` and the gallery yields show the same bins.

Card processes: the signal and the backgrounds of the channel whose nominal template has a positive sum, as in
datacards. Data-blind rules: every card background with its unweighted counts (histogram.raw_variable; an estimator
output has those of its data); a raw-count group keeps the names among the card backgrounds and is dropped (logged)
when none is left. Thesis binning: the fine bins rebinned to the start edges, then AutoRebin on the total of the card
backgrounds.

Clipping (SH6): the processes of ScoreBinning.clip and the outputs of the estimators that clip (ABCD, DataMinus with
clip_negative) get their negative bins set to zero after the rebinning, nominal and every variation, keeping the
integral of each (estimates.clip_negative_bins); every clipped bin is recorded.

Blinding: with ScoreBinning.blinding, the bins from the first one with S/B(mu=1) above it (nominal signal over the
total of the card backgrounds, after clipping) form the `hi` part; the bins below the `lo` part (none if the first bin
is above).
"""
from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass

import numpy as np

from shapesmith.estimates import clip_negative_bins
from shapesmith.histogram import NOMINAL_VARIATION, HistKey, Histogram, HistogramSet, is_part, raw_variable
from shapesmith.model import ABCD, NOMINAL, Analysis, Category, Channel, DataMinus, ScoreBinning
from shapesmith.nn_binning import auto_rebin, blinding_index, resolve, snap

logger = logging.getLogger(__name__)


@dataclass
class ScoreView:
    """A resolved ScoreBinning category: the final edges, the index of the first `hi` bin (None: no split), the
    nominal templates of the card processes on the final edges (clipped), the processes to clip, the clipped bins and
    the record of nn_binning.json."""

    edges: list[float]
    split: int | None
    templates: dict[str, Histogram]
    clip: frozenset[str]
    clipped: list[dict]
    record: dict


def score_binning(channel: Channel, category: str, variable: str) -> ScoreBinning | None:
    """The ScoreBinning of a category of the channel and its variable, None for any other binning."""
    for c in channel.categories:
        if c.name == category and c.variable.name == variable and isinstance(c.variable.edges, ScoreBinning):
            return c.variable.edges
    return None


def clipped_processes(channel: Channel, binning: ScoreBinning) -> frozenset[str]:
    """ScoreBinning.clip and the outputs of the estimators that clip their negative bins."""
    outputs = {e.output for e in channel.estimators if isinstance(e, ABCD) or (isinstance(e, DataMinus) and e.clip_negative)}
    return frozenset(binning.clip) | outputs


def clip(h: Histogram, process: str, variation: str) -> tuple[Histogram, dict | None]:
    """The histogram with its negative bins clipped (keeping the integral) and the record of what was clipped."""
    negative = np.flatnonzero(h.values < 0.0)
    if not negative.size:
        return h, None
    entry = {"process": process, "variation": variation, "bins": negative.tolist(), "values": h.values[negative].tolist(), "integral": h.sum()}
    return clip_negative_bins(h.copy()), entry


def _required(rules, names: set[str], where: str) -> tuple[dict[str, tuple[str, ...]], list[str]]:
    """The raw-count groups restricted to the card backgrounds; the labels of the groups left without one."""
    groups, dropped = {}, []
    for label, members in rules.required.items():
        present = tuple(n for n in members if n in names)
        if present:
            groups[label] = present
        else:
            dropped.append(label)
    if dropped:
        logger.debug(f"{where}: raw-count groups {', '.join(dropped)} have no card background, their rule is dropped")
    return groups, dropped


def _finite(entry: dict) -> dict:
    """A record entry with None for its infinite numbers (strict JSON)."""
    return {key: None if isinstance(value, float) and not math.isfinite(value) else value for key, value in entry.items()}


def _raw_counts(hset: HistogramSet, key: HistKey, where: str) -> np.ndarray:
    raw = dataclasses.replace(key, variable=raw_variable(key.variable))
    if raw not in hset:
        raise ValueError(f"{where}: no unweighted counts {raw.path}; fill the histograms (and estimate) with this ShapeSmith version")
    return hset[raw].values


def resolve_category(hset: HistogramSet, analysis: Analysis, channel_name: str, category: Category) -> ScoreView | None:
    """The resolved ScoreBinning category (module docstring); None without card processes. It logs at DEBUG only
    (plots resolve it again); datacards log the record (log_record)."""
    channel = analysis.channel(channel_name)
    binning = category.variable.edges
    variable = category.variable.name
    where = f"{channel_name}/{category.name}/{variable}"

    def key_of(process: str) -> HistKey:
        return HistKey(channel_name, category.name, process, NOMINAL, NOMINAL_VARIATION, variable)

    fine = {p: hset[key_of(p)] for p in (analysis.signal, *channel.backgrounds()) if key_of(p) in hset}
    card = {p: h for p, h in fine.items() if h.sum() > 0.0}
    backgrounds = {p: h for p, h in card.items() if p != analysis.signal}
    if not card:
        return None
    fine_edges = np.asarray(binning.fine, dtype=np.float64)
    signal = fine[analysis.signal].values if analysis.signal in fine else np.zeros(len(fine_edges) - 1)
    record: dict = {"variable": variable, "fine": {"low": binning.fine[0], "high": binning.fine[-1], "n_bins": len(binning.fine) - 1}}
    if binning.rules is not None:
        required, dropped = _required(binning.rules, set(backgrounds), where)
        rules = dataclasses.replace(binning.rules, required=required)
        templates = {p: (h.values, h.variances, _raw_counts(hset, key_of(p), where)) for p, h in backgrounds.items() if any(p in names for names in required.values())}
        templates.update({p: (h.values, h.variances, np.zeros(len(h.values))) for p, h in backgrounds.items() if p not in templates})
        resolution = resolve(fine_edges, signal, templates, rules)
        edges = list(resolution.edges)
        record.update(method="data_blind", rules=dataclasses.asdict(rules), dropped_groups=dropped, start=list(resolution.start), capped=resolution.capped,
                      leftover=list(resolution.leftover) if resolution.leftover else None, passed=resolution.passed, bins=[_finite(dataclasses.asdict(b)) for b in resolution.bins])
    else:
        start = [float(fine_edges[i]) for i in snap(binning.thesis, fine_edges)]
        total = Histogram.empty(start)
        for h in backgrounds.values():
            total.add(h.rebin(start))
        edges = list(auto_rebin(start, total.values, total.variances))
        record.update(method="thesis", start=start)
    record["edges"] = edges
    clip_names = clipped_processes(channel, binning)
    templates, clipped = {}, []
    for process, h in card.items():
        templates[process], entry = clip(h.rebin(edges), process, NOMINAL_VARIATION) if process in clip_names else (h.rebin(edges), None)
        if entry is not None:
            clipped.append(entry)
    split = None
    if binning.blinding is not None:
        total = np.sum([h.values for p, h in templates.items() if p != analysis.signal], axis=0) if backgrounds else np.zeros(len(edges) - 1)
        s = templates[analysis.signal].values if analysis.signal in templates else np.zeros(len(edges) - 1)
        split = blinding_index(s, total, binning.blinding)
        ratios = [float(signal_ / b) if b > 0.0 else None for signal_, b in zip(s, total)]
        record["blinding"] = {"s_over_b": binning.blinding, "first_hi_bin": split, "edge": edges[split] if split is not None else None, "s_over_b_per_bin": ratios}
    logger.debug(f"{where}: {record['method']} binning, {len(binning.fine) - 1} -> {len(edges) - 1} bins" + (f", blinded from {edges[split]:g}" if split is not None else ""))
    return ScoreView(edges, split, templates, clip_names, clipped, record)


def log_record(where: str, record: dict) -> None:
    """The binning of a category at INFO, the rules its bins fail and the dropped raw-count groups as warnings."""
    blinding = record.get("blinding") or {}
    edge = f", blinded from {blinding['edge']:g}" if blinding.get("edge") is not None else (", nothing blinded" if blinding else "")
    logger.info(f"{where}: {record['method']} binning, {record['fine']['n_bins']} -> {len(record['edges']) - 1} bins, edges {', '.join(f'{e:g}' for e in record['edges'])}{edge}")
    if blinding and len(record["edges"]) < 4:
        logger.warning(f"{where}: only {len(record['edges']) - 1} bins in a category with blinding")
    if record.get("dropped_groups"):
        logger.warning(f"{where}: the raw-count groups {', '.join(record['dropped_groups'])} have no card background, their rule is dropped")
    failing = [b for b in record.get("bins", []) if b["failed"]]
    if failing:
        logger.warning(f"{where}: {len(failing)} of {len(record['bins'])} bins fail a rule of the data-blind binning: "
                       + "; ".join(f"[{b['low']:g}, {b['high']:g}) {', '.join(b['failed'])}" for b in failing))


def rebinned_variations(hset: HistogramSet, view: ScoreView, channel_name: str, category: Category, process: str) -> dict[str, Histogram]:
    """The variations of a card process on the final edges (clipped like its nominal), without the parts of summed
    variations; the clipped bins join view.clipped."""
    result = {}
    for key in hset.select(channel=channel_name, category=category.name, process=process, region=NOMINAL, variable=category.variable.name):
        if key.variation == NOMINAL_VARIATION or is_part(key.variation):
            continue
        h = hset[key].rebin(view.edges)
        if process in view.clip:
            h, entry = clip(h, process, key.variation)
            if entry is not None:
                view.clipped.append(entry)
        result[key.variation] = h
    return result


def part(h: Histogram, start: int, stop: int | None = None) -> Histogram:
    """The bins start..stop - 1 of a histogram."""
    stop = len(h.values) if stop is None else stop
    return Histogram(h.edges[start:stop + 1].copy(), h.values[start:stop].copy(), h.variances[start:stop].copy())


def asimov(templates: dict[str, Histogram], signal: str | None, edges) -> Histogram:
    """The sum of the background templates on `edges`, as data (variances = values)."""
    values = np.sum([h.values for p, h in templates.items() if p != signal] or [np.zeros(len(edges) - 1)], axis=0)
    return Histogram(np.asarray(edges, dtype=np.float64).copy(), values, np.clip(values, 0.0, None))

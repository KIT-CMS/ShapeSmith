"""Stage 2: skims -> histograms per process, region, category, variable and variation.

Bookings: the signal in the nominal region, auxiliary processes in the nominal region without variations, every
other process in the nominal region and the estimator regions (or the explicitly requested regions). A variation is
filled for the processes whose sample kind it applies to, in its regions; a column variation only where it changes
an expression of the histogram (elsewhere it equals the nominal, and consumers fall back to that).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from shapesmith.config import RunConfig
from shapesmith.events import event_weights, lumi, select
from shapesmith.expressions import columns_of, evaluate, mask, shift
from shapesmith.histogram import INCLUSIVE, NOMINAL_VARIATION, HistKey, Histogram, HistogramSet
from shapesmith.model import ABCD, NOMINAL, Analysis, Channel, ColumnVariation, DataMinus, Process, Variable, Variation, applies
from shapesmith.parallel import run_jobs
from shapesmith.store import SKIM_COLUMNS, read_skims, schema

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Booking:
    process: Process
    regions: tuple[str, ...]
    variations: tuple[Variation, ...]


@dataclass(frozen=True)
class Target:
    category: str
    cut: str | None
    variable: Variable


def estimator_regions(channel: Channel) -> tuple[str, ...]:
    regions = []
    for estimator in channel.estimators:
        if isinstance(estimator, DataMinus):
            regions.append(estimator.region)
        elif isinstance(estimator, ABCD):
            regions += [estimator.b, estimator.c, estimator.d]
    return tuple(dict.fromkeys(regions))


def resolve_regions(channel: Channel, regions: list[str] | None) -> tuple[str, ...] | None:
    """An explicit region request; `all` expands to the nominal and every region of the channel."""
    if regions is None:
        return None
    names = (NOMINAL, *(region.name for region in channel.regions)) if "all" in regions else tuple(regions)
    resolved = tuple(dict.fromkeys(names))
    for name in resolved:
        channel.region(name)  # validates with a channel-specific error
    return resolved


def bookings(channel: Channel, systematics: bool, regions: list[str] | None = None) -> list[Booking]:
    requested = resolve_regions(channel, regions)
    result = []
    for process in channel.processes:
        if process.role == "auxiliary":
            if requested is None or NOMINAL in requested:
                result.append(Booking(process, (NOMINAL,), ()))
            continue
        if requested is not None:
            process_regions = requested
        elif process.role == "signal":
            process_regions = (NOMINAL,)
        else:
            process_regions = tuple(dict.fromkeys((NOMINAL, *estimator_regions(channel))))
        kind = channel.kind_of(process)
        variations = tuple(v for v in channel.variations if applies(v, kind, process.group)) if systematics else ()
        result.append(Booking(process, process_regions, variations))
    return result


def targets(channel: Channel, control: bool, variables: list[str] | None) -> list[Target]:
    if control:
        names = variables or list(channel.variables)
        return [Target(INCLUSIVE, None, channel.variables[name]) for name in names]  # KeyError for unknown variables
    return [Target(c.name, c.cut, c.variable) for c in channel.categories]


class Fill(NamedTuple):
    """The histograms of one region and variation of a booking: its cuts and weights, and per target the category
    cut and the variable expression."""

    region: str
    variation: Variation | None
    cuts: list[str]
    weights: list[str]
    targets: list[tuple[Target, str | None, str]]


def _target_exprs(target: Target, variation: Variation | None, available: set[str]) -> tuple[str | None, str]:
    if not isinstance(variation, ColumnVariation):
        return target.cut, target.variable.expr
    return (shift(target.cut, variation, available) if target.cut else None), shift(target.variable.expr, variation, available)


def plan_booking(channel: Channel, booking: Booking, targets_: list[Target], available: set[str]) -> list[Fill]:
    """What to fill: every region with its applicable variations; a column variation only for the targets whose
    cuts, weights, category or variable it changes."""
    fills = []
    for region_name in booking.regions:
        region = channel.region(region_name)
        nominal = select(channel, booking.process, region)
        for variation in (None, *(v for v in booking.variations if v.regions is None or region_name in v.regions)):
            selection = select(channel, booking.process, region, variation, available)
            if selection is None:
                logger.debug(f"{booking.process.name}: variation {variation.name} not applicable, skipped")
                continue
            exprs = [(target, *_target_exprs(target, variation, available)) for target in targets_]
            if isinstance(variation, ColumnVariation) and selection == nominal:
                exprs = [(target, cut, expr) for target, cut, expr in exprs if (cut, expr) != (target.cut, target.variable.expr)]
            if exprs:
                fills.append(Fill(region_name, variation, *selection, exprs))
    return fills


def fill_booking(config: RunConfig, analysis: Analysis, channel_name: str, booking: Booking, targets_: list[Target]) -> dict[HistKey, Histogram]:
    """All histograms of one booking, from one frame of the process's skims."""
    channel = analysis.channel(channel_name)
    process = booking.process
    nicks = [s.nick for s in channel.samples_of(process.group)]
    fills = plan_booking(channel, booking, targets_, schema(config.skim_dir, channel_name, nicks))
    exprs = []
    for f in fills:
        exprs += f.cuts + f.weights + [e for _, cut, var in f.targets for e in (cut, var) if e]
    frame = read_skims(config.skim_dir, channel_name, nicks, set(SKIM_COLUMNS) | columns_of(exprs))
    lumi_pb = lumi(analysis, channel, process)
    result: dict[HistKey, Histogram] = {}
    for f in fills:
        selected = frame[mask(frame, f.cuts)]
        w = event_weights(selected, f.weights, lumi_pb)
        for target, cut, expr in f.targets:
            in_category = mask(selected, [cut]) if cut else np.ones(len(selected), dtype=bool)
            values = evaluate(selected, expr).astype(np.float64)
            finite = np.isfinite(values) & np.isfinite(w)
            if (in_category & ~finite).any():
                logger.warning(f"{process.name}/{f.region}/{target.variable.name}: {(in_category & ~finite).sum()} events with NaN/inf skipped")
            keep = in_category & finite
            key = HistKey(channel_name, target.category, process.name, f.region, f.variation.name if f.variation else NOMINAL_VARIATION, target.variable.name)
            result[key] = Histogram.fill(target.variable.edges, values[keep], w[keep])
    return result


def _fill_job(args) -> dict[HistKey, Histogram]:
    return fill_booking(*args)


def run_hist(config: RunConfig, analysis: Analysis, channels: list[str] | None, control: bool, variables: list[str] | None, systematics: bool, processes: list[str] | None, output: Path, regions: list[str] | None = None) -> HistogramSet:
    """Fill every booked histogram of the requested channels into `output` (+ .json index).

    The requested scopes (channel, process, region, category, variable) are always filled; histograms of other
    scopes already in `output` are kept."""
    output = Path(output)
    jobs, scopes = [], set()
    for channel_name in channels or config.channels:
        channel = analysis.channel(channel_name)
        targets_ = targets(channel, control, variables)
        for booking in bookings(channel, systematics, regions):
            if processes and booking.process.name not in processes:
                continue
            jobs.append((config, analysis, channel_name, booking, targets_))
            scopes |= {(channel_name, booking.process.name, region, t.category, t.variable.name) for region in booking.regions for t in targets_}
    hset = HistogramSet()
    if output.exists():
        previous = HistogramSet.load(output)
        hset.update((key, h) for key, h in previous.items() if (key.channel, key.process, key.region, key.category, key.variable) not in scopes)
        logger.info(f"{output}: filling {len(scopes)} histogram scopes, keeping {len(hset)} other histograms")
    logger.info(f"filling {len(jobs)} process bookings with {config.workers} workers")
    for job, result, error in run_jobs(_fill_job, jobs, config.workers):
        if error is not None:
            raise RuntimeError(f"{job[2]}/{job[3].process.name}: {error}") from error
        hset.update(result)
    hset.save(output)
    return hset

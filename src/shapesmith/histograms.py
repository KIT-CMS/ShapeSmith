"""Stage 2: skims -> histograms per process, region, category, variable and variation (Spec §8).

Weight rules:
  * MC:        norm_weight * lumi_pb * baseline weights * process weights * region weights
  * embedding: norm_weight (=1) * process weights * region weights
  * data:      region weights only (e.g. the fake factor in the anti-isolated region)
A weight variation naming a weight the process does not carry is skipped for that process.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import uproot

from shapesmith.config import RunConfig
from shapesmith.expressions import apply_region, columns_in, evaluate, mask, weight
from shapesmith.histogram import Histogram
from shapesmith.io.skims import read_skims
from shapesmith.parallel import run_jobs
from shapesmith.model import Analysis, Process, Variable, WeightVariation
from shapesmith.skim import SKIM_COLUMNS

logger = logging.getLogger(__name__)

NOMINAL_REGION = "nominal"
NOMINAL_VARIATION = "Nominal"
INCLUSIVE = "inclusive"


@dataclass(frozen=True, order=True)
class HistKey:
    channel: str
    category: str
    process: str
    region: str
    variation: str
    variable: str

    @property
    def directory(self) -> str:
        return f"{self.channel}_{self.category}"

    @property
    def object_name(self) -> str:
        return f"{self.process}#{self.region}#{self.variation}#{self.variable}"

    @property
    def path(self) -> str:
        return f"{self.directory}/{self.object_name}"

    @classmethod
    def parse(cls, path: str) -> "HistKey":
        directory, name = path.split("/", 1)
        channel, category = directory.split("_", 1)
        process, region, variation, variable = name.split("#")
        return cls(channel, category, process, region, variation, variable)


class HistogramSet:
    """HistKey -> Histogram with ROOT persistence (uproot) and a JSON index."""

    def __init__(self):
        self._hists: dict[HistKey, Histogram] = {}

    def add(self, key: HistKey, h: Histogram) -> None:
        self._hists[key] = h

    def get(self, key: HistKey) -> Histogram:
        return self._hists[key]

    def has(self, key: HistKey) -> bool:
        return key in self._hists

    def keys(self, **filters) -> list[HistKey]:
        return sorted(k for k in self._hists if all(getattr(k, name) == value for name, value in filters.items()))

    def items(self) -> Iterator[tuple[HistKey, Histogram]]:
        return iter(sorted(self._hists.items(), key=lambda kv: kv[0]))

    def update(self, other: "HistogramSet") -> None:
        self._hists.update(other._hists)

    def __len__(self) -> int:
        return len(self._hists)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with uproot.recreate(path) as f:
            for key, h in self.items():
                f[key.path] = h.to_root(key.object_name)
        index = [{**asdict(key), "sum": h.sum(), "bins": len(h.values)} for key, h in self.items()]
        path.with_suffix(".json").write_text(json.dumps(index, indent=1))

    @classmethod
    def load(cls, path: Path) -> "HistogramSet":
        hset = cls()
        with uproot.open(path) as f:
            for name, classname in f.classnames(recursive=True, cycle=False).items():
                if classname.startswith("TH1"):
                    hset.add(HistKey.parse(name), Histogram.from_root(f[name]))
        return hset


@dataclass(frozen=True)
class Booking:
    process: Process
    regions: tuple[str, ...]
    variations: tuple[WeightVariation, ...]


@dataclass(frozen=True)
class Target:
    category: str
    cut: str | None
    variable: Variable


def bookings(analysis: Analysis, channel_name: str, systematics: bool) -> list[Booking]:
    estimator_regions = tuple(analysis.estimator.regions.values()) if analysis.estimator else ()
    result = []
    for process in analysis.processes:
        if not analysis.samples_for(process.group, channel_name):
            continue
        regions = (NOMINAL_REGION,) if process.kind == "signal" else (NOMINAL_REGION, *estimator_regions)
        variations = analysis.weight_variations if systematics and process.kind not in ("data", "embedding") else ()
        result.append(Booking(process, regions, tuple(variations)))
    return result


def targets(analysis: Analysis, control: bool, variables: list[str] | None) -> list[Target]:
    if control:
        names = variables or list(analysis.control_variables)
        return [Target(INCLUSIVE, None, analysis.control_variables[name]) for name in names]  # KeyError for unknown variables
    return [Target(c.name, c.cut, c.variable) for c in analysis.categories]


def _needed_columns(analysis: Analysis, channel_name: str, booking: Booking, targets_: list[Target], kinds: set[str]) -> set[str]:
    """Columns to read for this process: cuts always, weights only for the sample kinds that use them."""
    channel = analysis.channel(channel_name)
    process_selection = booking.process.selection_for(channel_name)
    columns = set(SKIM_COLUMNS)
    for expr in list(channel.baseline.cuts.values()) + list(process_selection.cuts.values()):
        columns |= columns_in(expr)
    if kinds == {"mc"}:
        for expr in channel.baseline.weights.values():
            columns |= columns_in(expr)
        for variation in booking.variations:
            for expr in variation.replace_weights.values():
                columns |= columns_in(expr)
    if kinds <= {"mc", "embedding"}:
        for expr in process_selection.weights.values():
            columns |= columns_in(expr)
    for region_name in booking.regions:
        region = channel.region(region_name)
        for expr in list(region.replace_cuts.values()) + list(region.add_weights.values()):
            columns |= columns_in(expr)
    for target in targets_:
        columns |= columns_in(target.variable.expr) | (columns_in(target.cut) if target.cut else set())
    return columns


def _varied(weights: dict[str, str], variation: WeightVariation | None) -> dict[str, str] | None:
    """Weights with the variation applied, or None if the variation names a weight not present."""
    if variation is None:
        return weights
    if any(name not in weights for name in variation.replace_weights):
        return None
    return {**weights, **variation.replace_weights}


def fill_booking(config: RunConfig, analysis: Analysis, channel_name: str, booking: Booking, targets_: list[Target]) -> dict[HistKey, Histogram]:
    channel = analysis.channel(channel_name)
    process = booking.process
    samples = analysis.samples_for(process.group, channel_name)
    kinds = {s.kind for s in samples}
    frame = read_skims(config.skim_dir, channel_name, [s.nick for s in samples], _needed_columns(analysis, channel_name, booking, targets_, kinds))
    lumi = analysis.lumi_pb if kinds == {"mc"} else 1.0
    process_selection = process.selection_for(channel_name)
    process_mask = mask(frame, process_selection.cuts)
    result: dict[HistKey, Histogram] = {}
    for region_name in booking.regions:
        region = channel.region(region_name)
        cuts = apply_region(channel.baseline, region).cuts
        selected = frame[process_mask & mask(frame, cuts)]
        weights: dict[str, str] = {}
        if kinds == {"mc"}:
            weights.update(channel.baseline.weights)
        if kinds <= {"mc", "embedding"}:
            weights.update(process_selection.weights)
        weights.update(region.add_weights)
        for variation in (None, *booking.variations):
            varied = _varied(weights, variation)
            if varied is None:
                logger.debug(f"{process.name}: variation {variation.name} not applicable, skipped")
                continue
            w = selected["norm_weight"].to_numpy() * lumi * weight(selected, varied)
            for target in targets_:
                in_category = mask(selected, {"category": target.cut}) if target.cut else np.ones(len(selected), dtype=bool)
                values = evaluate(selected, target.variable.expr).astype(np.float64)
                finite = np.isfinite(values) & np.isfinite(w)
                if (in_category & ~finite).any():
                    logger.warning(f"{process.name}/{region_name}/{target.variable.name}: {(in_category & ~finite).sum()} events with NaN/inf skipped")
                keep = in_category & finite
                key = HistKey(channel_name, target.category, process.name, region_name, variation.name if variation else NOMINAL_VARIATION, target.variable.name)
                result[key] = Histogram.fill(target.variable.edges, values[keep], w[keep])
    return result


def _fill_job(args) -> dict[HistKey, Histogram]:
    return fill_booking(*args)


def run_hist(config: RunConfig, analysis: Analysis, channels: list[str] | None, control: bool, variables: list[str] | None, systematics: bool, processes: list[str] | None, output: Path, force: bool = False) -> HistogramSet:
    """Fill every booked histogram of the requested channels and save them to `output` (+ .json index)."""
    output = Path(output)
    selected_channels = list(channels or config.channels)
    hset = HistogramSet()
    if output.exists():
        if not force:
            logger.info(f"{output} exists, loading (use --force to refill)")
            return HistogramSet.load(output)
        refilled = {p.name for p in analysis.processes if processes is None or p.key in processes}
        for key, h in HistogramSet.load(output).items():  # keep what this call does not refill
            if key.channel not in selected_channels or key.process not in refilled:
                hset.add(key, h)
        logger.info(f"{output}: refilling {sorted(selected_channels)}, keeping {len(hset)} histograms of other channels/processes")
    targets_ = targets(analysis, control, variables)
    jobs = []
    for channel_name in selected_channels:
        for booking in bookings(analysis, channel_name, systematics):
            if processes and booking.process.key not in processes:
                continue
            jobs.append((config, analysis, channel_name, booking, targets_))
    logger.info(f"filling {len(jobs)} process bookings with {config.workers} workers")
    for job, result, error in run_jobs(_fill_job, jobs, config.workers):
        if error is not None:
            raise RuntimeError(f"{job[2]}/{job[3].process.name}: {error}") from error
        for key, h in result.items():
            hset.add(key, h)
    hset.save(output)
    return hset

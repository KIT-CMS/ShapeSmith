"""The one rule that turns a process in a region under a variation into cuts and weights, and loads its events.

Cuts: the channel cuts with the region's replacements, the process cuts and the skim cuts. The skim cuts are a no-op
on nominal rows; under a variation they keep out the events a wider skim stored for other variations.
Weights, in order: the process weights, with the region's replacements where the process carries the name, then the
region's added weights. A weight variation replaces weights in place and does not apply to a process lacking one of
them; a column variation rewrites every cut and weight (expressions.shift).
Event weight: (norm_weight * lumi) * product of the weights, with lumi only for simulated (mc) processes.
Repeated events: with Analysis.event_key, every data and embedding sample (UNIQUE_KINDS) is read without the rows that
repeat the event key of an earlier row of the sample, in file order (read_events, store.read_skims), so that `hist`,
`measure`, the equal-data binning and `ml-export` count such an event once; repeated_record lists what is dropped, and
`hist`, `measure` and `ml-export` write it to repeated_events.json (write_repeated). The skims keep every row.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from shapesmith.config import RunConfig
from shapesmith.expressions import columns_of, mask, product, shift
from shapesmith.model import NOMINAL, Analysis, Channel, ColumnVariation, Process, Region, Variation, WeightVariation
from shapesmith.store import SKIM_COLUMNS, read_skims, repeated_events, schema

logger = logging.getLogger(__name__)
UNIQUE_KINDS = ("data", "embedding")  # the sample kinds read without repeated events (Analysis.event_key)
REPEATED = "repeated_events.json"


@dataclass(frozen=True)
class Query:
    channel: str
    process: str
    region: str = NOMINAL
    variation: Variation | None = None


@dataclass
class Events:
    frame: pd.DataFrame
    weights: np.ndarray


def select(channel: Channel, process: Process, region: Region, variation: Variation | None = None, available: set[str] = frozenset()) -> tuple[list[str], list[str]] | None:
    """Cut and weight expressions of `process` in `region` under `variation`; None if the variation does not apply."""
    weights = {name: region.replace_weights.get(name, expr) for name, expr in process.selection.weights.items()}
    weights.update(region.add_weights)
    if isinstance(variation, WeightVariation):
        if any(name not in weights for name in variation.replace_weights):
            return None
        weights.update(variation.replace_weights)
    cuts = [*{**channel.cuts, **region.replace_cuts}.values(), *process.selection.cuts.values(), *channel.skim.values()]
    weights = list(weights.values())
    if isinstance(variation, ColumnVariation):
        cuts = [shift(expr, variation, available) for expr in cuts]
        weights = [shift(expr, variation, available) for expr in weights]
    return cuts, weights


def lumi(analysis: Analysis, channel: Channel, process: Process) -> float:
    return analysis.lumi_pb if channel.kind_of(process) == "mc" else 1.0


def event_weights(frame: pd.DataFrame, weights: Iterable[str], lumi_pb: float) -> np.ndarray:
    return frame["norm_weight"].to_numpy() * lumi_pb * product(frame, weights)


def unique_nicks(analysis: Analysis, channel: Channel, nicks: Iterable[str]) -> set[str]:
    """The nicks among `nicks` of data and embedding samples, which are read without their repeated events; none
    without Analysis.event_key."""
    if not analysis.event_key:
        return set()
    kinds = {sample.nick: sample.kind for sample in channel.samples}
    return {nick for nick in nicks if kinds.get(nick) in UNIQUE_KINDS}


def read_events(config: RunConfig, analysis: Analysis, channel: Channel, nicks: Iterable[str], columns: Iterable[str] | None) -> pd.DataFrame:
    """The skim rows of the nicks (store.read_skims), the data and embedding samples without their repeated events."""
    nicks = list(nicks)
    return read_skims(config.skim_dir, channel.name, nicks, columns, analysis.event_key, unique_nicks(analysis, channel, nicks))


def repeated_record(config: RunConfig, analysis: Analysis, channels: Iterable[str]) -> dict:
    """What read_events drops in the channels: per channel the data and embedding samples checked and, of those with
    repeated events, the skim rows, the repeated ones and their skim files (store.repeated_events); each such sample is
    logged as a warning. Empty without Analysis.event_key."""
    if not analysis.event_key:
        return {}
    record = {}
    for name in channels:
        channel = analysis.channel(name)
        nicks = sorted(unique_nicks(analysis, channel, [s.nick for s in channel.samples]))
        repeated = {}
        for nick in nicks:
            entry = repeated_events(config.skim_dir, name, nick, analysis.event_key)
            if entry["repeated"]:
                repeated[nick] = entry
                files = ", ".join(f"{file} {n}" for file, n in entry["files"].items())
                logger.warning(f"{name}/{nick}: {entry['repeated']} of {entry['rows']} skim rows repeat the event ({', '.join(analysis.event_key)}) of an earlier row of the sample and are dropped (in {files})")
        record[name] = {"checked": nicks, "repeated": repeated}
        logger.info(f"{name}: {len(nicks)} data and embedding samples checked for repeated events, {sum(e['repeated'] for e in repeated.values())} rows dropped")
    return record


def write_repeated(path: Path, analysis: Analysis, record: dict) -> Path | None:
    """Merge the repeated_record of the run into the JSON at `path` ({"event_key", "kinds", "eras": {era: {channel:
    entry}}}): its channels replace the stored ones of the era, the others are kept. Nothing without Analysis.event_key."""
    if not analysis.event_key:
        return None
    path = Path(path)
    stored = json.loads(path.read_text()) if path.exists() else {}
    stored.update(event_key=list(analysis.event_key), kinds=list(UNIQUE_KINDS))
    stored.setdefault("eras", {}).setdefault(analysis.era, {}).update(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stored, indent=1))
    return path


def load(config: RunConfig, analysis: Analysis, query: Query, columns: Iterable[str] = ()) -> Events:
    """The selected events of a query (one frame, in sample order) with their weights; `columns` are read in addition."""
    channel = analysis.channel(query.channel)
    process = channel.process(query.process)
    nicks = [s.nick for s in channel.samples_of(process.group)]
    available = schema(config.skim_dir, channel.name, nicks) if isinstance(query.variation, ColumnVariation) else set()
    selection = select(channel, process, channel.region(query.region), query.variation, available)
    if selection is None:
        raise ValueError(f"{query.channel}/{query.process}: variation {query.variation.name} does not apply")
    cuts, weights = selection
    frame = read_events(config, analysis, channel, nicks, set(SKIM_COLUMNS) | columns_of(cuts + weights) | set(columns))
    selected = frame[mask(frame, cuts)].reset_index(drop=True)
    variation = f" ({query.variation.name})" if query.variation is not None else ""
    logger.debug(f"{query.channel}/{query.process}#{query.region}{variation}: {len(selected)} of {len(frame)} events selected")
    return Events(selected, event_weights(selected, weights, lumi(analysis, channel, process)))

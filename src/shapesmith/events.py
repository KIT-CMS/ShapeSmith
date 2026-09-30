"""The one rule that turns a process in a region under a variation into cuts and weights, and loads its events.

Cuts: the channel cuts with the region's replacements, the process cuts and the skim cuts. The skim cuts are a no-op
on nominal rows; under a variation they keep out the events a wider skim stored for other variations.
Weights, in order: the process weights, with the region's replacements where the process carries the name, then the
region's added weights. A weight variation replaces weights in place and does not apply to a process lacking one of
them; a column variation rewrites every cut and weight (expressions.shift).
Event weight: (norm_weight * lumi) * product of the weights, with lumi only for simulated (mc) processes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from shapesmith.config import RunConfig
from shapesmith.expressions import columns_of, mask, product, shift
from shapesmith.model import NOMINAL, Analysis, Channel, ColumnVariation, Process, Region, Variation, WeightVariation
from shapesmith.store import SKIM_COLUMNS, read_skims, schema

logger = logging.getLogger(__name__)


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
    frame = read_skims(config.skim_dir, channel.name, nicks, set(SKIM_COLUMNS) | columns_of(cuts + weights) | set(columns))
    selected = frame[mask(frame, cuts)].reset_index(drop=True)
    variation = f" ({query.variation.name})" if query.variation is not None else ""
    logger.debug(f"{query.channel}/{query.process}#{query.region}{variation}: {len(selected)} of {len(frame)} events selected")
    return Events(selected, event_weights(selected, weights, lumi(analysis, channel, process)))

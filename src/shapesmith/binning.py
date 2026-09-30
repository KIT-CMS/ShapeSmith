"""Equal-data binning: the edges of an `EqualData` variable from the data of the run, as smhtt_ul gof/build_binning.py
(get_data_selection and get_1d_binning).

Per category: the data process of the channel in the nominal region, its events in the category, the values of the
variable in (low, high); the unweighted percentiles 0, 100 / n_bins, ..., 100 (linear interpolation), duplicates
removed, every edge moved by -1e-4 and the last one by +2e-4 on top, so that the smallest and the largest value are
inside. `run_hist` resolves the edges whenever it fills, so every region and variation of a category shares them, and
records them in `binning.json` next to its output. smhtt_ul's mask of default values (-11, -999, -10, -1) is not
applied.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from shapesmith.config import RunConfig
from shapesmith.events import Query, load
from shapesmith.expressions import columns_of, evaluate, mask
from shapesmith.model import NOMINAL, Analysis, EqualData

if TYPE_CHECKING:
    from shapesmith.fill import Target

EPSILON = 1e-4
RECORD = "binning.json"

logger = logging.getLogger(__name__)


def inside(values: np.ndarray, rule: EqualData) -> np.ndarray:
    """The finite values in (low, high), as float64."""
    x = np.asarray(values, dtype=np.float64)
    return x[np.isfinite(x) & (x > rule.low) & (x < rule.high)]


def equal_data_edges(values: np.ndarray, rule: EqualData) -> tuple[float, ...]:
    x = inside(values, rule)
    if not x.size:
        raise ValueError(f"no data in ({rule.low}, {rule.high})")
    edges = sorted({float(edge) for edge in np.percentile(x, np.linspace(0.0, 100.0, rule.n_bins + 1))})
    if len(edges) < 2:
        raise ValueError(f"fewer than two distinct edges from {x.size} values")
    edges = [edge - EPSILON for edge in edges]
    edges[-1] += 2 * EPSILON
    return tuple(edges)


def resolve(config: RunConfig, analysis: Analysis, channel_name: str, targets: list[Target]) -> tuple[list[Target], dict]:
    """The targets with the edges of their EqualData variables, and the record of those edges
    ({category: {variable: {"rule", "n_data", "edges"}}}). The data are read only if a target needs them."""
    pending = [t for t in targets if isinstance(t.variable.edges, EqualData)]
    if not pending:
        return list(targets), {}
    channel = analysis.channel(channel_name)
    exprs = [t.variable.expr for t in pending] + [t.cut for t in pending if t.cut]
    frame = load(config, analysis, Query(channel_name, channel.data(), NOMINAL), columns_of(exprs)).frame
    resolved, record = [], {}
    for target in targets:
        rule = target.variable.edges
        if not isinstance(rule, EqualData):
            resolved.append(target)
            continue
        values = evaluate(frame[mask(frame, [target.cut])] if target.cut else frame, target.variable.expr)
        try:
            edges = equal_data_edges(values, rule)
        except ValueError as error:
            raise ValueError(f"{channel_name}/{target.category}/{target.variable.name}: equal-data binning: {error}") from None
        resolved.append(dataclasses.replace(target, variable=dataclasses.replace(target.variable, edges=edges)))
        n_data = int(inside(values, rule).size)
        record.setdefault(target.category, {})[target.variable.name] = {"rule": dataclasses.asdict(rule), "n_data": n_data, "edges": list(edges)}
        logger.info(f"{channel_name}/{target.category}/{target.variable.name}: {len(edges) - 1} equal-data bins from {n_data} data events, edges {', '.join(f'{edge:.6g}' for edge in edges)}")
    return resolved, record


def write_record(path: Path, record: dict) -> Path:
    """Merge `record` ({channel: {category: {variable: entry}}}) into the JSON at `path`: its entries replace the stored
    ones, the others are kept (as the histograms of other scopes in the shapes file)."""
    path = Path(path)
    stored = json.loads(path.read_text()) if path.exists() else {}
    for channel, categories in record.items():
        for category, variables in categories.items():
            stored.setdefault(channel, {}).setdefault(category, {}).update(variables)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stored, indent=1))
    return path

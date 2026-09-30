"""Training folds from the skims, in the Feather layout of the previous smhtt_ul export.

Columns are 5-level tuples: ("Event","id"), ("Event","event"), ("Labels",<label>),
("Nominal","variables",<var>), ("Nominal","weight"), ("Nominal","cut"), ("Nominal","class_weight").
Fold k holds the events with `event % folds == (k + 1) % folds` (fold0 = odd events for two folds);
the training/validation split repeats the row pattern [True, True, False, False] per process.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

from shapesmith.config import RunConfig
from shapesmith.events import Query, load
from shapesmith.expressions import columns_of, evaluate
from shapesmith.model import NOMINAL, Analysis, Channel

logger = logging.getLogger(__name__)
PATTERN = np.array([True, True, False, False])


def tuple_column(*parts: str, length: int = 5) -> tuple[str, ...]:
    return tuple(list(parts) + [""] * (length - len(parts)))


def fold_masks(event: np.ndarray, folds: int) -> dict[str, np.ndarray]:
    tiled = np.resize(PATTERN, len(event)).astype(bool)
    masks = {}
    for k in range(folds):
        in_fold = (np.asarray(event).astype(np.int64) % folds) == ((k + 1) % folds)
        masks[f"fold{k}"] = in_fold
        masks[f"fold{k}_training"] = in_fold & tiled
        masks[f"fold{k}_validation"] = in_fold & ~tiled
    return masks


def class_weights(weights: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """weight * (sum of all weights) / (sum of weights of the event's class)."""
    result = np.zeros_like(weights, dtype=np.float64)
    total = weights.sum()
    for label in np.unique(labels):
        selected = labels == label
        result[selected] = total / weights[selected].sum()
    return result * weights


def _variable_expr(channel: Channel, name: str) -> str:
    if name in channel.variables:
        return channel.variables[name].expr
    for category in channel.categories:
        if category.variable.name == name:
            return category.variable.expr
    return name


def export_process(config: RunConfig, analysis: Analysis, channel_name: str, name: str) -> pd.DataFrame:
    """The events of one exported process; an estimator output is exported as the data of its region."""
    ml = analysis.ml
    channel = analysis.channel(channel_name)
    process = name if name in {p.name for p in channel.processes} else channel.data()  # an estimator output: data in its region
    exprs = {v: _variable_expr(channel, v) for v in ml.variables}
    events = load(config, analysis, Query(channel_name, process, ml.region_of.get(name, NOMINAL)), {ml.fold_column} | columns_of(exprs.values()))
    selected = events.frame
    out = pd.DataFrame(index=selected.index)
    out[tuple_column("Event", "id")] = np.arange(len(selected))
    out[tuple_column("Event", ml.fold_column)] = selected[ml.fold_column].to_numpy()
    for label in sorted(set(ml.label_of.values())):
        out[tuple_column("Labels", label)] = np.int32(label == ml.label_of[name])
    for variable, expr in exprs.items():
        out[tuple_column("Nominal", "variables", variable)] = evaluate(selected, expr).astype(np.float32)
    out[tuple_column("Nominal", "weight")] = events.weights.astype(np.float32)
    out[tuple_column("Nominal", "cut")] = np.float32(1.0)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out


def run_ml_export(config: RunConfig, analysis: Analysis, channels: list[str]) -> list[Path]:
    if analysis.ml is None:
        raise ValueError("Analysis.ml is not set")
    if config.ml_dir is None:
        raise ValueError("RunConfig.ml_dir is not set")
    written = []
    for channel in channels:
        parts: dict[str, list[pd.DataFrame]] = {}
        for name in analysis.ml.processes:
            frame = export_process(config, analysis, channel, name)
            logger.info(f"{channel}/{name}: {len(frame)} events")
            for fold, selected in fold_masks(frame[tuple_column("Event", analysis.ml.fold_column)].to_numpy(), analysis.ml.folds).items():
                parts.setdefault(fold, []).append(frame[selected])
        directory = Path(config.ml_dir) / channel
        directory.mkdir(parents=True, exist_ok=True)
        for fold, frames in parts.items():
            combined = pd.concat(frames, ignore_index=True)
            labels = combined["Labels"].to_numpy().argmax(axis=1)
            combined[tuple_column("Nominal", "class_weight")] = class_weights(combined[tuple_column("Nominal", "weight")].to_numpy().astype(np.float64), labels).astype(np.float32)
            path = directory / f"{fold}.feather"
            feather.write_feather(pa.Table.from_pandas(combined), path)  # pyarrow keeps the MultiIndex columns in its pandas metadata
            written.append(path)
    return written

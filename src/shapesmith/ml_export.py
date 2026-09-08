"""Training folds from the skims, in the Feather layout of the previous smhtt_ul export (Spec §12).

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
from shapesmith.expressions import apply_region, columns_in, evaluate, mask, weight
from shapesmith.io.skims import read_skims
from shapesmith.model import Analysis
from shapesmith.skim import SKIM_COLUMNS

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


def _variable_expr(analysis: Analysis, name: str) -> str:
    if name in analysis.control_variables:
        return analysis.control_variables[name].expr
    for category in analysis.categories:
        if category.variable.name == name:
            return category.variable.expr
    return name


def export_process(config: RunConfig, analysis: Analysis, channel_name: str, name: str) -> pd.DataFrame:
    ml = analysis.ml
    channel = analysis.channel(channel_name)
    region = channel.region(ml.region_of.get(name, "nominal"))
    if analysis.estimator is not None and name == analysis.estimator.output:
        samples = analysis.samples_for("data", channel_name)
        cuts = apply_region(channel.baseline, region).cuts
        weights = dict(region.add_weights)
        lumi = 1.0
    else:
        process = analysis.process(name)
        process_selection = process.selection_for(channel_name)
        samples = analysis.samples_for(process.group, channel_name)
        kinds = {s.kind for s in samples}
        cuts = {**apply_region(channel.baseline, region).cuts, **process_selection.cuts}
        weights = {}
        if kinds == {"mc"}:
            weights.update(channel.baseline.weights)
        if kinds <= {"mc", "embedding"}:
            weights.update(process_selection.weights)
        weights.update(region.add_weights)
        lumi = analysis.lumi_pb if kinds == {"mc"} else 1.0
    exprs = {v: _variable_expr(analysis, v) for v in ml.variables}
    columns = set(SKIM_COLUMNS) | {ml.fold_column} | set().union(*(columns_in(e) for e in list(cuts.values()) + list(weights.values()) + list(exprs.values())))
    frame = read_skims(config.skim_dir, channel_name, [s.nick for s in samples], columns)
    selected = frame[mask(frame, cuts)].reset_index(drop=True)
    out = pd.DataFrame(index=selected.index)
    out[tuple_column("Event", "id")] = np.arange(len(selected))
    out[tuple_column("Event", ml.fold_column)] = selected[ml.fold_column].to_numpy()
    for label in sorted(set(ml.label_of.values())):
        out[tuple_column("Labels", label)] = np.int32(label == ml.label_of[name])
    for variable, expr in exprs.items():
        out[tuple_column("Nominal", "variables", variable)] = evaluate(selected, expr).astype(np.float32)
    out[tuple_column("Nominal", "weight")] = (selected["norm_weight"].to_numpy() * lumi * weight(selected, weights)).astype(np.float32)
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

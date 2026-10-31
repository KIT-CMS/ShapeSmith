"""Training data from the skims: per channel one flat Feather file `<ml_dir>/<channel>/events.feather` with a row per
exported event, and the class scheme in `<ml_dir>/classes.json`.

Rows: the events of every exported process in the nominal region; an estimator output is the data of its region_of
region (with the region's weights, e.g. the fake factor, times the scale of its DataMinus). With
MLExportConfig.subtraction the processes a DataMinus output subtracts follow in its region with negated weights, in the
class of the output and flagged is_subtraction.
Columns, in this order:
- the identifiers (MLExportConfig.identifiers, e.g. run, lumi, event) with their n-tuple types;
- era, channel, sample_nick, process: dictionary-encoded strings (pandas category); (era, channel, sample_nick, run,
  lumi, event) is the key of an event, process the exported process (a subtraction row: the subtracted one);
- row (int64): the position in the file, so (era, channel, row) is unique; class_index (int8): the position of the
  class in classes.json;
- weight (float64): the event weight with its sign, luminosity and normalisation included;
- is_subtraction (bool); thesis_training (bool): the split within a parity fold of the previous smhtt_ul export, the row
  pattern [True, True, False, False] over the rows of each process (its fold<k>_training of K folds was
  `event % K == (k + 1) % K & thesis_training`);
- the variables (float32), NaN where a process has no such column (MLExportConfig.mc_only in data and embedding).
No folds and no class weights: the trainer derives its folds from `event` and its class weights from `weight`.
Rows of data and embedding come without their repeated events (Analysis.event_key, events.read_events), listed in
`<ml_dir>/repeated_events.json`. `merge_eras` concatenates the exports of several eras and merges their records.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from shapesmith.config import RunConfig
from shapesmith.events import REPEATED, Query, load, repeated_record, write_repeated
from shapesmith.expressions import columns_of, evaluate
from shapesmith.model import NOMINAL, Analysis, Channel, DataMinus, MLExportConfig
from shapesmith.parallel import run_jobs

logger = logging.getLogger(__name__)
EVENTS = "events.feather"
CLASSES = "classes.json"
THESIS_PATTERN = np.array([True, True, False, False])
KEY_COLUMNS = ("era", "channel", "sample_nick", "process")  # dictionary-encoded strings
BOOKKEEPING = (*KEY_COLUMNS, "row", "class_index", "weight", "is_subtraction", "thesis_training")
SCHEME_FIELDS = ("profile", "classes", "variables", "mc_only", "identifiers", "regions")  # equal in every era


def reserved_columns(ml: MLExportConfig) -> set[str]:
    """Column names of the export that a variable cannot take."""
    return set(BOOKKEEPING) | set(ml.identifiers)


def _variable_expr(channel: Channel, name: str) -> str:
    if name in channel.variables:
        return channel.variables[name].expr
    for category in channel.categories:
        if category.variable.name == name:
            return category.variable.expr
    return name


def variable_columns(channel: Channel, ml: MLExportConfig, kind: str) -> set[str]:
    """The skim columns the export reads for a process of a sample kind."""
    return set(ml.identifiers) | columns_of(_variable_expr(channel, v) for v in ml.columns(kind))


def subtraction_of(channel: Channel, ml: MLExportConfig, name: str) -> DataMinus | None:
    """The DataMinus estimate of an exported estimator output in its region_of region (None: not estimated so)."""
    region = ml.region_of.get(name, NOMINAL)
    estimates = [e for e in channel.estimators if isinstance(e, DataMinus) and e.output == name and e.target == NOMINAL]
    return next((e for e in estimates if e.region == region), None)


@dataclass(frozen=True)
class Rows:
    """A block of export rows: the events of `process` in `region` under the name `name` (the process column) in the
    class `class_index`, their weights times `scale`, negated for a subtraction."""

    channel: str
    name: str
    process: str
    region: str
    class_index: int
    scale: float = 1.0
    subtraction: bool = False

    def __str__(self) -> str:
        return f"{self.channel}/{self.name}#{self.region}" + (" (subtraction)" if self.subtraction else "")


def _constant(value: str, n: int) -> pd.Categorical:
    return pd.Categorical.from_codes(np.zeros(n, dtype=np.int8), categories=[value])


def export_rows(config: RunConfig, analysis: Analysis, rows: Rows) -> pd.DataFrame:
    """The rows of a block in the column order of the export; `row` is set when the blocks of a channel are joined."""
    ml, channel = analysis.ml, analysis.channel(rows.channel)
    exprs = {v: _variable_expr(channel, v) for v in ml.columns(channel.kind_of(channel.process(rows.process)))}
    query = Query(rows.channel, rows.process, rows.region)
    events = load(config, analysis, query, set(ml.identifiers) | columns_of(exprs.values()))
    frame, n = events.frame, len(events.frame)
    columns = {identifier: frame[identifier].to_numpy() for identifier in ml.identifiers}
    columns |= {
        "era": _constant(analysis.era, n),
        "channel": _constant(rows.channel, n),
        "sample_nick": pd.Categorical(frame["sample_nick"]),
        "process": _constant(rows.name, n),
        "row": np.zeros(n, dtype=np.int64),
        "class_index": np.full(n, rows.class_index, dtype=np.int8),
        "weight": events.weights.astype(np.float64) * (-rows.scale if rows.subtraction else rows.scale),
        "is_subtraction": np.full(n, rows.subtraction),
        "thesis_training": np.resize(THESIS_PATTERN, n).astype(bool),
    }
    for variable in ml.variables:
        missing = np.full(n, np.nan, dtype=np.float32)
        columns[variable] = evaluate(frame, exprs[variable]).astype(np.float32) if variable in exprs else missing
    return pd.DataFrame(columns)


def _main_rows(analysis: Analysis, channel_name: str, name: str) -> Rows:
    ml, channel = analysis.ml, analysis.channel(channel_name)
    if name not in ml.exported(channel):
        raise ValueError(f"channel {channel_name}: {name} is no signal or background process or estimator output")
    if name in {p.name for p in channel.processes}:
        return Rows(channel_name, name, name, NOMINAL, ml.class_index(name))
    estimate = subtraction_of(channel, ml, name)  # an estimator output: the data of its region
    scale = estimate.scale if estimate else 1.0
    return Rows(channel_name, name, channel.data(), ml.region_of.get(name, NOMINAL), ml.class_index(name), scale)


def export_process(config: RunConfig, analysis: Analysis, channel_name: str, name: str) -> pd.DataFrame:
    """The rows of one process the channel exports (MLExportConfig.exported); an estimator output is the data of its
    region_of region, times the scale of its DataMinus."""
    return export_rows(config, analysis, _main_rows(analysis, channel_name, name))


def export_blocks(analysis: Analysis, channel_name: str) -> list[Rows]:
    """The row blocks of a channel in file order: every exported process, then the subtraction rows."""
    ml, channel = analysis.ml, analysis.channel(channel_name)
    result = [_main_rows(analysis, channel_name, name) for name in ml.exported(channel)]
    for name in ml.exported(channel) if ml.subtraction else ():
        estimate = subtraction_of(channel, ml, name)
        if estimate is not None:
            index, region = ml.class_index(name), estimate.region
            result += [Rows(channel_name, p, p, region, index, estimate.scale, True) for p in estimate.subtract]
    return result


def class_scheme(analysis: Analysis, channels: list[str]) -> dict:
    """classes.json: the class order and per channel the processes of every class and the subtracted processes of an
    estimator output."""
    ml = analysis.ml
    scheme = {
        "profile": ml.profile, "classes": list(ml.classes), "variables": list(ml.variables),
        "mc_only": list(ml.mc_only), "identifiers": list(ml.identifiers), "regions": dict(ml.region_of),
        "eras": [analysis.era], "channels": {},
    }
    for name in channels:
        channel = analysis.channel(name)
        exported = ml.exported(channel)
        estimates = {output: subtraction_of(channel, ml, output) for output in exported} if ml.subtraction else {}
        scheme["channels"][name] = {
            "processes": {c: [p for p in exported if ml.class_of[p] == c] for c in ml.classes},
            "subtraction": {output: list(e.subtract) for output, e in estimates.items() if e is not None},
        }
    return scheme


def _check_scheme(stored: dict, scheme: dict, where: str) -> None:
    different = [name for name in SCHEME_FIELDS if stored.get(name) != scheme.get(name)]
    if different:
        raise ValueError(f"{where}: another class scheme ({', '.join(different)} differ); use another ml_dir")


def write_scheme(directory: Path, scheme: dict) -> Path:
    """classes.json of an export directory; a stored one of the same scheme keeps its other channels and eras."""
    path = Path(directory) / CLASSES
    if path.exists():
        stored = json.loads(path.read_text())
        _check_scheme(stored, scheme, str(path))
        eras = sorted(set(stored["eras"]) | set(scheme["eras"]))
        scheme = {**scheme, "eras": eras, "channels": {**stored["channels"], **scheme["channels"]}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scheme, indent=2))
    return path


def finish(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """The rows of a channel in one frame: `row` numbered, the key strings dictionary-encoded."""
    combined = pd.concat(frames, ignore_index=True)
    combined["row"] = np.arange(len(combined), dtype=np.int64)
    for column in KEY_COLUMNS:
        combined[column] = combined[column].astype("category")
    return combined


def summary(frame: pd.DataFrame, classes: tuple[str, ...]) -> list[str]:
    """Rows and signed weight per class, the subtraction rows apart."""
    lines = []
    for index, name in enumerate(classes):
        for subtraction in (False, True):
            selected = (frame["class_index"] == index) & (frame["is_subtraction"] == subtraction)
            if selected.any() or not subtraction:
                label = f"{name} (subtraction)" if subtraction else name
                lines.append(f"{label}: {selected.sum()} rows, weight {frame.loc[selected, 'weight'].sum():.6g}")
    return lines


def repeated_keys(frame: pd.DataFrame, identifiers: tuple[str, ...]) -> int:
    """The rows whose event key (era, channel, sample_nick, identifiers) an earlier row has, e.g. an event that two
    n-tuple files of a sample both hold."""
    return int(frame.duplicated(["era", "channel", "sample_nick", *identifiers]).sum())


def _rows_job(job: tuple[int, RunConfig, Analysis, Rows]) -> pd.DataFrame:
    return export_rows(*job[1:])


def run_ml_export(config: RunConfig, analysis: Analysis, channels: list[str]) -> list[Path]:
    if analysis.ml is None:
        raise ValueError("Analysis.ml is not set")
    if config.ml_dir is None:
        raise ValueError("RunConfig.ml_dir is not set")
    ml, exported, pending = analysis.ml, [], []
    for channel in channels:
        if channel in analysis.control_channels:
            logger.info(f"{channel}: control channel, not exported")
            continue
        names = ml.exported(analysis.channel(channel))
        skipped = [name for name in ml.processes if name not in names]
        if skipped:
            logger.info(f"{channel}: {', '.join(skipped)} not exported, no process or estimator output of the channel")
        if names:
            exported.append(channel)
            pending += export_blocks(analysis, channel)
    dropped = repeated_record(config, analysis, exported)
    frames: dict[int, pd.DataFrame] = {}
    failures = []
    jobs = [(index, config, analysis, rows) for index, rows in enumerate(pending)]
    for (index, *_, rows), frame, error in run_jobs(_rows_job, jobs, config.workers, label="ml-export"):
        if error is not None:
            failures.append(f"{rows}: {error}")
            continue
        logger.debug(f"{rows}: {len(frame)} rows")
        frames[index] = frame
    if failures:
        raise RuntimeError(f"{len(failures)} export failures:\n" + "\n".join(failures))
    written = []
    for channel in exported:
        combined = finish([frames[index] for index, rows in enumerate(pending) if rows.channel == channel])
        path = Path(config.ml_dir) / channel / EVENTS
        path.parent.mkdir(parents=True, exist_ok=True)
        feather.write_feather(combined, path)
        for line in summary(combined, ml.classes):
            logger.info(f"{channel}: {line}")
        repeated = repeated_keys(combined, ml.identifiers)
        if repeated:
            key = ", ".join(("era", "channel", "sample_nick", *ml.identifiers))
            logger.warning(f"{channel}: {repeated} rows repeat the event key ({key}) of another row")
        logger.info(f"{channel}: {len(combined)} rows, {len(combined.columns)} columns -> {path}")
        written.append(path)
    if exported:
        written.append(write_scheme(config.ml_dir, class_scheme(analysis, exported)))
        if (path := write_repeated(Path(config.ml_dir) / REPEATED, analysis, dropped)) is not None:
            written.append(path)
    return written


def merge_eras(directories: list[Path], output: Path, channels: list[str] | None = None) -> list[Path]:
    """The exports of several eras (one ml_dir each) as one: per channel the rows of every era one after the other (the
    era column tells them apart, `row` stays that of its era), classes.json with the eras and, per channel and class,
    the processes of any era, the repeated events of every era (repeated_events.json) and the provenance records of every
    era in versions/<era>/. Every era needs the same class scheme and the channels; the weights stay those of their era,
    the class weights are the trainer's."""
    schemes = [json.loads((Path(d) / CLASSES).read_text()) for d in directories]
    for directory, scheme in zip(directories[1:], schemes[1:]):
        _check_scheme(schemes[0], scheme, str(Path(directory) / CLASSES))
    eras = [era for scheme in schemes for era in scheme["eras"]]
    if len(set(eras)) < len(eras):
        raise ValueError(f"an era is in more than one export: {eras}")
    names = channels or list(schemes[0]["channels"])
    missing = [f"{d}: {n}" for d, scheme in zip(directories, schemes) for n in names if n not in scheme["channels"]]
    if missing:
        raise ValueError("channels missing in an export:\n" + "\n".join(missing))
    merged = {**schemes[0], "eras": eras, "channels": {}}
    written = []
    for name in names:
        entries = [scheme["channels"][name] for scheme in schemes]
        merged["channels"][name] = {}
        for field in ("processes", "subtraction"):  # class or estimator output -> the processes of any era
            keys = dict.fromkeys(key for e in entries for key in e[field])
            union = {k: list(dict.fromkeys(p for e in entries for p in e[field].get(k, []))) for k in keys}
            merged["channels"][name][field] = union
        frames = [pd.read_feather(Path(d) / name / EVENTS) for d in directories]
        combined = pd.concat(frames, ignore_index=True)
        for column in KEY_COLUMNS:
            combined[column] = combined[column].astype(str).astype("category")
        path = Path(output) / name / EVENTS
        path.parent.mkdir(parents=True, exist_ok=True)
        feather.write_feather(combined, path)
        counts = " + ".join(str(len(f)) for f in frames)
        logger.info(f"{name}: {counts} = {len(combined)} rows of {', '.join(eras)} -> {path}")
        written.append(path)
    path = Path(output) / CLASSES
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2))
    written.append(path)
    records = [json.loads((Path(d) / REPEATED).read_text()) for d in directories if (Path(d) / REPEATED).exists()]
    if records:  # the repeated events of every era (events.write_repeated)
        repeated = {**records[0], "eras": {era: entry for r in records for era, entry in r["eras"].items()}}
        (Path(output) / REPEATED).write_text(json.dumps(repeated, indent=1))
        written.append(Path(output) / REPEATED)
    for directory, scheme in zip(directories, schemes):
        versions = Path(directory) / "versions"
        if versions.is_dir():
            shutil.copytree(versions, Path(output) / "versions" / "+".join(scheme["eras"]), dirs_exist_ok=True)
    return written

"""Stage 1: ntuples (+ friends) -> Parquet skims with the loose baseline selection (Spec §7)."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from shapesmith import __version__
from shapesmith.config import RunConfig
from shapesmith.expressions import columns_in, mask
from shapesmith.io.discovery import NtupleFile, discover
from shapesmith.io.ntuples import read_ntuple
from shapesmith.io.skims import manifest_path, read_manifest, skim_path, write_manifest, write_skim
from shapesmith.parallel import run_jobs
from shapesmith.model import Analysis, Channel, Sample

logger = logging.getLogger(__name__)

SKIM_COLUMNS = ("sample_nick", "norm_weight", "is_data", "is_mc", "is_embedding")
EVENT_COLUMNS = ("event",)
MC_COLUMNS = ("genWeight",)


@dataclass(frozen=True)
class SkimResult:
    nick: str
    basename: str
    n_in: int
    n_out: int
    path: Path
    metadata: dict = field(default_factory=dict)  # CROWN metadata block of the file
    skipped: bool = False  # reused from a previous run; the counts come from its manifest record


def _columns(expressions) -> set[str]:
    columns: set[str] = set()
    for expr in expressions:
        columns |= columns_in(expr)
    return columns


def columns_for_sample(analysis: Analysis, channel_name: str, sample: Sample) -> set[str]:
    """Columns this sample must carry: the common cuts, regions, categories and variables, the cuts (and for
    MC/embedding the weights) of the processes of its own group; MC additionally the baseline weights, the
    weight variations and the generator columns."""
    channel = analysis.channel(channel_name)
    columns = _columns(channel.skim.cuts.values()) | _columns(channel.baseline.cuts.values()) | set(channel.keep_columns) | set(EVENT_COLUMNS)
    for region in channel.regions:
        columns |= _columns(region.replace_cuts.values())
        region_weights = [expr for name, expr in region.add_weights.items() if sample.kind == "mc" or name not in channel.baseline.weights]
        columns |= _columns(region_weights)
    for category in analysis.categories:
        columns |= columns_in(category.cut) | columns_in(category.variable.expr)
    for variable in analysis.control_variables.values():
        columns |= columns_in(variable.expr)
    for process in analysis.processes:
        if process.group != sample.group:
            continue
        selection = process.selection_for(channel_name)
        columns |= _columns(selection.cuts.values())
        if sample.kind != "data":
            columns |= _columns(selection.weights.values())
    if sample.kind == "mc":
        columns |= _columns(channel.skim.weights.values()) | _columns(channel.baseline.weights.values()) | set(MC_COLUMNS)
        for variation in analysis.weight_variations:
            columns |= _columns(variation.replace_weights.values())
    return columns


def required_columns(analysis: Analysis, channel_name: str) -> set[str]:
    """Every column any sample of the channel needs (the skim reads this union from each file)."""
    columns: set[str] = set()
    for sample in analysis.samples:
        if sample.channels is None or channel_name in sample.channels:
            columns |= columns_for_sample(analysis, channel_name, sample)
    return columns


def skim_one(ntuple: NtupleFile, sample: Sample, channel: Channel, columns: set[str], out_path: Path, optional: set[str] = frozenset()) -> SkimResult:
    """Read one file, apply the skim selection, add the bookkeeping columns, write Parquet.

    Every requested column must exist, except the `optional` ones (MC weights in data/embedding files),
    which are left out and reported at debug level. A missing column is a configuration error, not a warning.
    """
    frame, metadata = read_ntuple(ntuple, columns, optional)
    if len(frame.columns) < len(columns):
        logger.debug(f"{ntuple.basename} ({sample.nick}): optional columns not in this sample: {sorted(set(columns) - set(frame.columns))}")
    n_in = len(frame)
    frame = frame[mask(frame, channel.skim.cuts)].reset_index(drop=True)
    sign = np.sign(frame["genWeight"].to_numpy()) if sample.kind == "mc" else np.ones(len(frame))
    sign[sign == 0] = 1.0
    frame["sample_nick"] = pd.array([sample.nick] * len(frame), dtype="string")  # typed even for empty files
    frame["norm_weight"] = (sample.norm_weight * sign).astype(np.float64)
    for column, kind in (("is_data", "data"), ("is_mc", "mc"), ("is_embedding", "embedding")):
        frame[column] = np.full(len(frame), sample.kind == kind, dtype=bool)
    write_skim(frame, out_path)
    return SkimResult(sample.nick, ntuple.basename, n_in, len(frame), out_path, metadata)


def _skim_job(args) -> SkimResult:
    return skim_one(*args)


def select_samples(analysis: Analysis, samples: list[str] | None) -> tuple[Sample, ...]:
    """All samples, or those selected by name: a sample group (exact) or, for names that are not a group, a nick prefix."""
    if not samples:
        return analysis.samples
    groups = {s.group for s in analysis.samples}
    prefixes = [name for name in samples if name not in groups]
    chosen = tuple(s for s in analysis.samples if s.group in samples or any(s.nick.startswith(prefix) for prefix in prefixes))
    if not chosen:
        raise ValueError(f"no sample matches {samples}; groups: {sorted({s.group for s in analysis.samples})}")
    return chosen


def _normalisation(sample: Sample) -> dict:
    return {"kind": sample.kind, "xsec": sample.xsec, "nevents": sample.nevents, "generator_weight": sample.generator_weight}


def _skim_contract(channel: Channel, columns: set[str], sample: Sample) -> dict:
    return {
        "selection": dict(channel.skim.cuts),
        "required_columns": sorted(columns),
        "normalisation": _normalisation(sample),
    }


def _validate_reuse(path: Path, channel: Channel, columns: set[str], sample: Sample) -> dict:
    """Reject a stale skim before its Parquet files are silently reused; returns the compatible manifest.

    Legacy manifests predate selection contracts.  Their columns and
    normalisation can be inspected, but reuse is unsafe because the selection
    that produced their rows cannot be verified.
    """
    if not path.exists():
        raise ValueError(f"skim {channel.name}/{sample.nick} is incompatible: its manifest is missing; re-run `shapesmith skim --force`")
    manifest = read_manifest(path)
    expected = _skim_contract(channel, columns, sample)
    contract = manifest.get("contract")
    problems = []
    if contract is None:
        problems.append("selection was not recorded in the legacy manifest")
        stored_columns = set(manifest.get("columns", ()))
        if not set(expected["required_columns"]) <= stored_columns:
            problems.append("required columns changed")
        if manifest.get("normalisation") != expected["normalisation"]:
            problems.append("normalisation changed")
        if manifest.get("nick") != sample.nick or manifest.get("channel") != channel.name:
            problems.append("sample or channel changed")
    else:
        if contract.get("selection") != expected["selection"]:
            problems.append("selection changed")
        if contract.get("required_columns") != expected["required_columns"]:
            problems.append("required columns changed")
        if contract.get("normalisation") != expected["normalisation"]:
            problems.append("normalisation changed")
    if problems:
        raise ValueError(f"skim {channel.name}/{sample.nick} is incompatible ({', '.join(problems)}); re-run `shapesmith skim --force`")
    return manifest


def _start_manifest(path: Path, channel: Channel, columns: set[str], sample: Sample, files: list[NtupleFile], reuse: bool) -> dict:
    """The manifest this run of the sample starts from: with `reuse`, a compatible previous manifest keeps its
    metadata and the records of the files it finished; otherwise nothing counts as finished."""
    manifest = {
        "nick": sample.nick,
        "channel": channel.name,
        "files": [f.path for f in files],
        "metadata": {},
        "normalisation": _normalisation(sample),
        "contract": _skim_contract(channel, columns, sample),
        "completed": {},
    }
    if reuse:
        previous = _validate_reuse(path, channel, columns, sample)
        basenames = {f.basename for f in files}
        manifest["metadata"] = previous.get("metadata", {})
        manifest["completed"] = {name: record for name, record in previous.get("completed", {}).items() if name in basenames}
    return manifest


def _write_manifest(path: Path, manifest: dict, columns: set[str]) -> None:
    completed = manifest["completed"].values()
    manifest.update(
        n_in=sum(record["n_in"] for record in completed),
        n_out=sum(record["n_out"] for record in completed),
        columns=sorted(columns | set(SKIM_COLUMNS)),
        shapesmith_version=__version__,
    )
    write_manifest(path, manifest)


def run_skim(config: RunConfig, analysis: Analysis, channels: list[str] | None = None, force: bool = False, samples: list[str] | None = None) -> list[SkimResult]:
    """Skim every sample (or the ones selected by group/nick prefix) of every requested channel.

    A file is reused when its Parquet exists and the sample's manifest records it as completed under the current
    contract; `force` recomputes everything. The manifest is written before the first file of a sample and after
    every finished file, so an interrupted run resumes with only the unfinished files.
    """
    results: list[SkimResult] = []
    for channel_name in channels or config.channels:
        channel = analysis.channel(channel_name)
        columns = required_columns(analysis, channel_name)
        jobs, manifests = [], {}
        chosen = [s for s in select_samples(analysis, samples) if s.channels is None or channel_name in s.channels]
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(chosen)))) as listing:  # directory listings are latency bound
            discovered = list(listing.map(lambda s: discover(config.ntuples, config.era, s.nick, channel_name, s.kind), chosen))
        for sample, files in zip(chosen, discovered):
            path = manifest_path(config.skim_dir, channel_name, sample.nick)
            outputs = [(ntuple, skim_path(config.skim_dir, channel_name, sample.nick, ntuple.basename)) for ntuple in files]
            reuse = not force and any(out_path.exists() for _, out_path in outputs)
            manifest = manifests[sample.nick] = _start_manifest(path, channel, columns, sample, files, reuse)
            pending = 0
            for ntuple, out_path in outputs:
                record = manifest["completed"].get(ntuple.basename)
                if record is not None and out_path.exists():
                    results.append(SkimResult(sample.nick, ntuple.basename, record["n_in"], record["n_out"], out_path, skipped=True))
                    continue
                manifest["completed"].pop(ntuple.basename, None)  # its Parquet is gone: the record no longer counts
                jobs.append((ntuple, sample, channel, columns, out_path, columns - columns_for_sample(analysis, channel_name, sample)))
                pending += 1
            if pending:  # record the contract before the first file is written
                _write_manifest(path, manifest, columns)
        logger.info(f"{channel_name}: skimming {len(jobs)} files with {config.workers} workers")
        failures = []
        for job, result, error in run_jobs(_skim_job, jobs, config.workers):
            if error is not None:  # collect, report all at the end
                failures.append(f"{job[0].path}: {error}")
                continue
            results.append(result)
            manifest = manifests[result.nick]
            manifest["completed"][result.basename] = {"n_in": result.n_in, "n_out": result.n_out}
            if not manifest["metadata"]:
                manifest["metadata"] = result.metadata
            _write_manifest(manifest_path(config.skim_dir, channel_name, result.nick), manifest, columns)
        if failures:
            raise RuntimeError("skim failures:\n" + "\n".join(failures))
    return results

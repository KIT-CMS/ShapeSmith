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
        columns |= _columns(region.replace_cuts.values()) | _columns(region.add_weights.values())
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


def run_skim(config: RunConfig, analysis: Analysis, channels: list[str] | None = None, force: bool = False, samples: list[str] | None = None) -> list[SkimResult]:
    """Skim every sample (or the ones selected by group/nick prefix) of every requested channel; existing Parquet files are skipped unless `force`."""
    results: list[SkimResult] = []
    for channel_name in channels or config.channels:
        channel = analysis.channel(channel_name)
        columns = required_columns(analysis, channel_name)
        jobs, manifests = [], {}
        chosen = [s for s in select_samples(analysis, samples) if s.channels is None or channel_name in s.channels]
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(chosen)))) as listing:  # directory listings are latency bound
            discovered = list(listing.map(lambda s: discover(config.ntuples, config.era, s.nick, channel_name, s.kind), chosen))
        for sample, files in zip(chosen, discovered):
            manifests[sample.nick] = {"nick": sample.nick, "channel": channel_name, "files": [f.path for f in files], "metadata": {}, "normalisation": {"kind": sample.kind, "xsec": sample.xsec, "nevents": sample.nevents, "generator_weight": sample.generator_weight}}
            for ntuple in files:
                out_path = skim_path(config.skim_dir, channel_name, sample.nick, ntuple.basename)
                if out_path.exists() and not force:
                    results.append(SkimResult(sample.nick, ntuple.basename, 0, 0, out_path))
                    continue
                jobs.append((ntuple, sample, channel, columns, out_path, columns - columns_for_sample(analysis, channel_name, sample)))
        logger.info(f"{channel_name}: skimming {len(jobs)} files with {config.workers} workers")
        failures = []
        for job, result, error in run_jobs(_skim_job, jobs, config.workers):
            if error is not None:  # collect, report all at the end
                failures.append(f"{job[0].path}: {error}")
            else:
                results.append(result)
        if failures:
            raise RuntimeError("skim failures:\n" + "\n".join(failures))
        for nick, manifest in manifests.items():
            done = [r for r in results if r.nick == nick and r.path.parent == Path(config.skim_dir) / channel_name / nick]
            path = manifest_path(config.skim_dir, channel_name, nick)
            processed = [r for r in done if r.n_in]
            if processed:
                manifest["metadata"] = processed[0].metadata
            elif path.exists():  # nothing re-skimmed: keep the metadata of the previous run
                manifest["metadata"] = read_manifest(path).get("metadata", {})
            manifest.update(n_in=sum(r.n_in for r in done), n_out=sum(r.n_out for r in done), columns=sorted(columns | set(SKIM_COLUMNS)), shapesmith_version=__version__)
            write_manifest(path, manifest)
    return results

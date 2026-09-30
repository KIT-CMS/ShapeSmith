"""Stage 1: ntuples (+ friends) -> Parquet skims with the loose skim selection and every column the analysis needs.

Columns: per sample, the columns of every expression its processes can use (all regions, categories, variables and
the variations of its kind), the skim cuts, keep_columns, `event`, `genWeight` for MC and those of its own cut. For
every CROWN shift of its kind (a ColumnVariation with a suffix) the shifted branches `c + suffix` are read where
they exist.
Rows: an event is kept if the skim cuts pass nominally or under a column variation of its kind; then the sample cut
applies (the normalisation stays that of the whole sample).
Checks, per file, where the channel declares CROWN shifts for the sample (its kind and group): every declared shift
has a shifted branch, and every shifted branch `c__X` of a needed column has a declared suffix `__X`.
Reuse: a stored skim serves when its skim cuts, normalisation and sample cut are equal, its column variations and
friends contain the current ones and its Parquet schema holds the needed columns; see reuse_problems.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from shapesmith import __version__, logs
from shapesmith.config import RunConfig
from shapesmith.events import select
from shapesmith.expressions import columns_in, columns_of, mask, shift
from shapesmith.model import NOMINAL_REGION, Analysis, Channel, ColumnVariation, Sample, WeightVariation, applies
from shapesmith.ntuples import NtupleFile, discover, friend_bases, join_url, read_ntuple
from shapesmith.parallel import run_jobs
from shapesmith.store import SKIM_COLUMNS, manifest_path, read_manifest, schema, skim_path, write_manifest, write_skim

logger = logging.getLogger(__name__)

ROW_FIELDS = ("selection", "normalisation", "sample_cut", "column_variations", "friends")  # the contract fields that define the rows


@dataclass(frozen=True)
class SkimResult:
    nick: str
    basename: str
    n_in: int
    n_out: int
    path: Path
    metadata: dict = field(default_factory=dict)  # CROWN metadata block of the file
    skipped: bool = False  # reused from a previous run; the counts come from its manifest record


def column_variations(channel: Channel, sample: Sample) -> list[ColumnVariation]:
    return [v for v in channel.variations if isinstance(v, ColumnVariation) and applies(v, sample.kind, sample.group)]


def needed_columns(channel: Channel, sample: Sample) -> set[str]:
    """Every nominal column the sample's histograms, estimates and exports can read."""
    weight_variations = [v for v in channel.variations if isinstance(v, WeightVariation) and sample.kind in v.applies_to]
    exprs = [*channel.skim.values(), *(c.cut for c in channel.categories), *(c.variable.expr for c in channel.categories)]
    exprs += [variable.expr for variable in channel.variables.values()]
    exprs += [expr for variation in column_variations(channel, sample) for expr in variation.derived.values()]
    for process in channel.processes:
        if process.group != sample.group:
            continue
        for region in (NOMINAL_REGION, *channel.regions):
            for variation in (None, *weight_variations):
                selection = select(channel, process, region, variation)
                if selection is not None:
                    exprs += selection[0] + selection[1]
    columns = columns_of(exprs) | set(channel.keep_columns) | {"event"}
    if sample.kind == "mc":
        columns.add("genWeight")
    if sample.cut is not None:
        columns |= columns_in(sample.cut)
    return columns


def check_shifts(channel: Channel, sample: Sample, columns: set[str], branches: set[str]) -> None:
    """Both directions of the declared CROWN shifts, on one file; nothing is checked for a sample without any."""
    declared = {v.suffix: v.name for v in column_variations(channel, sample) if v.suffix}
    if not declared:
        return
    problems = [f"declared shift {name} ({suffix}) has no shifted branch" for suffix, name in declared.items() if not any(c + suffix in branches for c in columns)]
    found = {"__" + branch.split("__", 1)[1] for branch in branches if "__" in branch and branch.split("__", 1)[0] in columns}
    problems += [f"undeclared shift {suffix}" for suffix in sorted(found - set(declared))]
    if problems:
        raise ValueError("; ".join(problems))


def skim_one(ntuple: NtupleFile, sample: Sample, channel: Channel, columns: set[str], out_path: Path, optional: set[str] = frozenset()) -> SkimResult:
    """Read one file, keep the events that pass the skim cuts under any variation and the sample cut, add the
    bookkeeping columns and write Parquet. Every column in `columns` must exist; `optional` ones may be absent."""
    start = time.monotonic()
    variations = column_variations(channel, sample)
    shifted = {column + v.suffix for v in variations if v.suffix for column in columns}
    frame, metadata, branches = read_ntuple(ntuple, columns | shifted, set(optional) | shifted)
    read = time.monotonic() - start
    check_shifts(channel, sample, columns, branches)
    n_in = len(frame)
    nominal = list(channel.skim.values())
    selected = mask(frame, nominal)
    for variation in variations:
        cuts = [shift(expr, variation, set(frame.columns)) for expr in nominal]
        if cuts != nominal:
            selected |= mask(frame, cuts)
    if sample.cut is not None:
        selected &= mask(frame, [sample.cut])
    frame = frame[selected].reset_index(drop=True)
    sign = np.sign(frame["genWeight"].to_numpy()) if sample.kind == "mc" else np.ones(len(frame))
    sign[sign == 0] = 1.0
    frame["sample_nick"] = pd.array([sample.nick] * len(frame), dtype="string")  # typed even for empty files
    frame["norm_weight"] = (sample.norm_weight * sign).astype(np.float64)
    for column, kind in (("is_data", "data"), ("is_mc", "mc"), ("is_embedding", "embedding")):
        frame[column] = np.full(len(frame), sample.kind == kind, dtype=bool)
    write_skim(frame, out_path)
    logger.debug(
        f"{channel.name}/{sample.nick}/{ntuple.basename}: {n_in} -> {len(frame)} events, {len(frame.columns)} columns "
        f"({len(shifted & set(frame.columns))} shifted), read in {logs.duration(read)}, done in {logs.duration(time.monotonic() - start)}"
    )
    return SkimResult(sample.nick, ntuple.basename, n_in, len(frame), out_path, metadata)


def _skim_job(args) -> SkimResult:
    return skim_one(*args)


def select_samples(channel: Channel, names: list[str] | None) -> tuple[Sample, ...]:
    """All samples, or those selected by name: a sample group (exact) or, for names that are not a group, a nick prefix."""
    if not names:
        return channel.samples
    groups = {s.group for s in channel.samples}
    prefixes = [name for name in names if name not in groups]
    return tuple(s for s in channel.samples if s.group in names or any(s.nick.startswith(prefix) for prefix in prefixes))


def normalisation(sample: Sample) -> dict:
    return {"kind": sample.kind, "xsec": sample.xsec, "nevents": sample.nevents, "generator_weight": sample.generator_weight}


def contract(config: RunConfig, channel: Channel, sample: Sample, columns: set[str]) -> dict:
    """What the stored rows and columns depend on; optional fields are recorded only when set, so that skims
    without them stay reusable."""
    result = {"selection": dict(channel.skim), "required_columns": sorted(columns), "normalisation": normalisation(sample)}
    if sample.cut is not None:
        result["sample_cut"] = sample.cut
    variations = {v.name: v.suffix or dict(v.derived) for v in column_variations(channel, sample)}
    if variations:
        result["column_variations"] = variations
    friends = friend_bases(config.ntuples, sample.kind)
    if friends:
        result["friends"] = friends
    return result


def reuse_problems(stored: dict, expected: dict, stored_columns: set[str]) -> list[str]:
    """Why a stored skim cannot serve: its rows must be a superset of the needed rows, its columns of the needed columns."""
    problems = [f"{name} changed" for name in ("selection", "normalisation", "sample_cut") if stored.get(name) != expected.get(name)]
    recorded = stored.get("column_variations", {})
    missing = sorted(name for name, spec in expected.get("column_variations", {}).items() if recorded.get(name) != spec)
    if missing:
        problems.append(f"column variations missing: {', '.join(missing)}")
    friends = sorted(set(expected.get("friends", [])) - set(stored.get("friends", [])))
    if friends:
        problems.append(f"friends missing: {', '.join(friends)}")
    columns = sorted(set(expected["required_columns"]) - stored_columns)
    if columns:
        problems.append(f"columns missing: {', '.join(columns[:10])}{' ...' if len(columns) > 10 else ''}")
    return problems


def _plan(config: RunConfig, channel: Channel, sample: Sample, files: list[NtupleFile], force: bool) -> tuple[dict, list[tuple], str | None]:
    """The manifest and the pending jobs of one sample, or the reason why its stored skim is incompatible.

    A complete compatible skim is reused as it is. An incomplete one resumes if it has the same rows: only its
    unfinished files are skimmed, with the stored columns too, so that the files of a sample share one schema. An
    incomplete stored superset is skimmed again."""
    columns = needed_columns(channel, sample)
    outputs = [(ntuple, skim_path(config.skim_dir, channel.name, sample.nick, ntuple.basename)) for ntuple in files]
    manifest = {"nick": sample.nick, "channel": channel.name, "files": [f.path for f in files], "metadata": {}, "completed": {}, "normalisation": normalisation(sample)}
    manifest["contract"] = contract(config, channel, sample, columns)
    stored_columns: set[str] = set()
    path = manifest_path(config.skim_dir, channel.name, sample.nick)
    if not force and any(out.exists() for _, out in outputs):
        if not path.exists():
            return manifest, [], "its manifest is missing"
        previous = read_manifest(path)
        stored_columns = schema(config.skim_dir, channel.name, [sample.nick]) - set(SKIM_COLUMNS)
        problems = reuse_problems(previous.get("contract", {}), manifest["contract"], stored_columns)
        if problems:
            return manifest, [], ", ".join(problems)
        completed = {name: record for name, record in previous.get("completed", {}).items() if name in {f.basename for f in files}}
        if all(ntuple.basename in completed and out.exists() for ntuple, out in outputs):
            logger.debug(f"{channel.name}/{sample.nick}: stored skim of {len(files)} files reused")
            return {**previous, "completed": completed}, [], None
        if all(previous["contract"].get(name) == manifest["contract"].get(name) for name in ROW_FIELDS):
            logger.debug(f"{channel.name}/{sample.nick}: resuming the stored skim, {len(completed)} of {len(files)} files done")
            manifest.update(metadata=previous.get("metadata", {}), completed=completed)
            columns |= stored_columns
            manifest["contract"]["required_columns"] = sorted(columns)
        else:
            logger.info(f"{channel.name}/{sample.nick}: the stored skim holds more variations but is incomplete, skimming it again")
            stored_columns = set()
    jobs = [(ntuple, sample, channel, columns, out, stored_columns) for ntuple, out in outputs if not (ntuple.basename in manifest["completed"] and out.exists())]
    logger.debug(f"{channel.name}/{sample.nick}: {len(jobs)} of {len(files)} files to skim, {len(columns)} columns" + (" (--force)" if force else ""))
    for ntuple, *_ in jobs:
        manifest["completed"].pop(ntuple.basename, None)  # its Parquet is gone: the record no longer counts
    return manifest, jobs, None


def _write_manifest(config: RunConfig, manifest: dict) -> None:
    completed = manifest["completed"].values()
    manifest.update(
        n_in=sum(record["n_in"] for record in completed),
        n_out=sum(record["n_out"] for record in completed),
        columns=sorted(set(manifest["contract"]["required_columns"]) | set(SKIM_COLUMNS)),
        shapesmith_version=__version__,
    )
    write_manifest(manifest_path(config.skim_dir, manifest["channel"], manifest["nick"]), manifest)


def run_skim(config: RunConfig, analysis: Analysis, channels: list[str] | None = None, force: bool = False, samples: list[str] | None = None) -> list[SkimResult]:
    """Skim every sample (or the ones selected by group/nick prefix) of every requested channel.

    Every incompatible stored skim is reported before anything is written; `force` recomputes everything. The
    manifest is written before the first file of a sample and after every finished file, so an interrupted run
    resumes with only the unfinished files.
    """
    results, jobs, manifests, incompatible = [], [], {}, []
    for channel_name in channels or config.channels:
        channel = analysis.channel(channel_name)
        chosen = select_samples(channel, samples)
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(chosen)))) as listing:  # directory listings are latency bound
            discovered = list(listing.map(lambda s: discover(config.ntuples, config.era, s.nick, channel_name, s.kind), chosen))
        logger.info(f"{channel_name}: {len(chosen)} samples, {sum(len(files) for files in discovered)} files in {join_url(config.ntuples.server, config.ntuples.base)}")
        counts = dict.fromkeys(("reused", "resumed", "new", "empty", "pending"), 0)
        for sample, files in zip(chosen, discovered):
            if not files:
                logger.warning(f"{channel_name}/{sample.nick}: no ntuple files")
            manifest, pending, problem = _plan(config, channel, sample, files, force)
            if problem is not None:
                incompatible.append(f"{channel_name}/{sample.nick}: {problem}")
                continue
            manifests[channel_name, sample.nick] = manifest
            results += [SkimResult(sample.nick, name, record["n_in"], record["n_out"], skim_path(config.skim_dir, channel_name, sample.nick, name), skipped=True) for name, record in manifest["completed"].items()]
            jobs += pending
            counts["empty" if not files else "reused" if not pending else "resumed" if manifest["completed"] else "new"] += 1
            counts["pending"] += len(pending)
        logger.info(f"{channel_name}: samples {counts['reused']} reused, {counts['resumed']} resumed, {counts['new']} new, {counts['empty']} without files; {counts['pending']} files to skim")
    if samples and not manifests and not incompatible:
        raise ValueError(f"no sample matches {samples}")
    if incompatible:
        raise ValueError("incompatible skims (re-run `shapesmith skim --force --samples ...` for them):\n" + "\n".join(incompatible))
    for channel_name, nick in {(job[2].name, job[1].nick) for job in jobs}:
        _write_manifest(config, manifests[channel_name, nick])  # the contract is recorded before the first file is written
    logger.info(f"skimming {len(jobs)} files with {config.workers} workers into {config.skim_dir}")
    failures = []
    for job, result, error in run_jobs(_skim_job, jobs, config.workers, label="skim"):
        if error is not None:  # collect, report all at the end
            failures.append(f"{job[0].path}: {error}")
            logger.error(f"{job[2].name}/{job[1].nick}/{job[0].basename}: {error}")
            continue
        results.append(result)
        manifest = manifests[job[2].name, result.nick]
        manifest["completed"][result.basename] = {"n_in": result.n_in, "n_out": result.n_out}
        if not manifest["metadata"]:
            manifest["metadata"] = result.metadata
        _write_manifest(config, manifest)
    for channel_name in channels or config.channels:
        done = [r for r in results if r.path.parent.parent.name == channel_name]  # <skim_dir>/<channel>/<nick>/<file>
        logger.info(f"{channel_name}: {len(done)} files, {sum(r.n_in for r in done)} -> {sum(r.n_out for r in done)} events")
    if failures:
        raise RuntimeError(f"{len(failures)} skim failures:\n" + "\n".join(failures))
    return results

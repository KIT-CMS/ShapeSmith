"""Consistency checks of an Analysis; every problem is collected and reported at once."""
from __future__ import annotations

from shapesmith.model import NOMINAL, ROLES, SAMPLE_KINDS, Analysis, AnalysisError, Channel, ColumnVariation, DataMinus, TemplateShift, WeightVariation


def _duplicates(what: str, names: list[str]) -> list[str]:
    return [f"duplicate {what} {name}" for name in sorted({n for n in names if names.count(n) > 1})]


def _partner(name: str) -> str | None:
    """The other side of an Up/Down pair, None for a name without a direction (a template)."""
    for direction, other in (("Up", "Down"), ("Down", "Up")):
        if name.endswith(direction):
            return name[: -len(direction)] + other
    return None


def _process_problems(channel: Channel) -> list[str]:
    problems = _duplicates("process", [p.name for p in channel.processes])
    problems += _duplicates("sample", [s.nick for s in channel.samples])
    problems += [f"sample {s.nick}: unknown kind {s.kind}" for s in channel.samples if s.kind not in SAMPLE_KINDS]
    for process in channel.processes:
        kinds = {s.kind for s in channel.samples_of(process.group)}
        if process.role not in ROLES:
            problems.append(f"process {process.name}: unknown role {process.role}")
        if not kinds:
            problems.append(f"process {process.name}: no samples for group {process.group}")
        elif len(kinds) > 1:
            problems.append(f"process {process.name}: samples of different kinds {sorted(kinds)}")
    return problems


def _region_problems(channel: Channel) -> list[str]:
    weights = {name for p in channel.processes for name in p.selection.weights}
    problems = _duplicates("region", [NOMINAL, *(r.name for r in channel.regions)])
    for region in channel.regions:
        problems += [f"region {region.name}: replaces unknown cut {cut}" for cut in region.replace_cuts if cut not in channel.cuts]
        problems += [f"region {region.name}: replaces unknown weight {w}" for w in region.replace_weights if w not in weights]
        problems += [f"region {region.name}: adds weight {w}, which a process carries (use replace_weights)" for w in region.add_weights if w in weights]
    return problems


def _variation_problems(channel: Channel, region_names: set[str]) -> list[str]:
    weights = {name for p in channel.processes for name in p.selection.weights}
    groups = {s.group for s in channel.samples}
    names = [v.name for v in channel.variations]
    problems = _duplicates("variation", names)
    for variation in channel.variations:
        partner = _partner(variation.name)
        if partner is not None and partner not in names:
            problems.append(f"variation {variation.name}: its partner {partner} is missing")
        problems += [f"variation {variation.name}: unknown sample kind {k}" for k in variation.applies_to if k not in SAMPLE_KINDS]
        problems += [f"variation {variation.name}: unknown region {r}" for r in variation.regions or () if r not in region_names]
        if isinstance(variation, ColumnVariation):
            problems += [f"variation {variation.name}: unknown sample group {g}" for g in variation.groups or () if g not in groups]
        if isinstance(variation, WeightVariation):
            problems += [f"variation {variation.name}: replaces unknown weight {w}" for w in variation.replace_weights if w not in weights]
        elif bool(variation.suffix) == bool(variation.derived):
            problems.append(f"variation {variation.name}: needs exactly one of suffix and derived")
    return problems


def _estimator_problems(channel: Channel, region_names: set[str]) -> list[str]:
    roles = {p.name: p.role for p in channel.processes}
    problems = []
    for estimator in channel.estimators:
        if isinstance(estimator, TemplateShift):
            problems += [f"template shift {estimator.name}: unknown process {name}" for name in (estimator.process, estimator.template) if name not in roles]
            continue
        regions = (estimator.region,) if isinstance(estimator, DataMinus) else (estimator.b, estimator.c, estimator.d)
        problems += [f"estimator {estimator.output}: region {r} does not exist" for r in regions if r not in region_names]
        problems += [f"estimator {estimator.output}: subtracts unknown process {name}" for name in estimator.subtract if name not in roles]
        problems += [f"estimator {estimator.output}: subtracts the auxiliary process {name}" for name in estimator.subtract if roles.get(name) == "auxiliary"]
        if list(roles.values()).count("data") != 1:
            problems.append(f"estimator {estimator.output}: needs exactly one data process")
        if estimator.output in roles:
            problems.append(f"estimator {estimator.output}: output collides with a process")
    return problems


def channel_problems(channel: Channel) -> list[str]:
    region_names = {NOMINAL, *(r.name for r in channel.regions)}
    problems = _process_problems(channel) + _region_problems(channel)
    problems += _duplicates("category", [c.name for c in channel.categories])
    problems += _variation_problems(channel, region_names) + _estimator_problems(channel, region_names)
    return [f"channel {channel.name}: {problem}" for problem in problems]


def validate(analysis: Analysis) -> None:
    """Raise AnalysisError listing every problem of the analysis."""
    problems = [problem for channel in analysis.channels.values() for problem in channel_problems(channel)]
    processes = {p.name: p.role for channel in analysis.channels.values() for p in channel.processes}
    if analysis.signal is not None and processes.get(analysis.signal) != "signal":
        problems.append(f"signal {analysis.signal} is not a process with the role signal")
    if analysis.ml is not None:
        problems += [f"ML export: {name} is an auxiliary process" for name in analysis.ml.processes if processes.get(name) == "auxiliary"]
    if problems:
        raise AnalysisError("\n".join(sorted(set(problems))))


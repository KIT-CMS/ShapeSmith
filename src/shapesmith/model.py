"""The analysis data model (Spec §5).

An analysis repository builds one `Analysis` object; every ShapeSmith step reads it. Expressions
are strings in pandas.eval syntax (see expressions.py). Cut and weight *names* are the handles that
regions and variations replace, so they must be unique within a selection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

Kind = Literal["data", "signal", "true_tau", "lepton_fake", "jet_fake", "single_higgs", "other", "embedding"]
SampleKind = Literal["data", "mc", "embedding"]
KINDS = ("data", "signal", "true_tau", "lepton_fake", "jet_fake", "single_higgs", "other", "embedding")


class AnalysisError(ValueError):
    """Raised by Analysis.validate() with every problem found, one per line."""


def _frozen(mapping: Mapping | None) -> dict:
    """A private copy of the mapping (plain dict: the objects must stay picklable for the worker pools)."""
    return dict(mapping or {})


@dataclass(frozen=True)
class Sample:
    nick: str
    group: str
    kind: SampleKind
    xsec: float = 1.0
    nevents: int = 1
    generator_weight: float = 1.0
    channels: tuple[str, ...] | None = None  # None = all channels

    @property
    def norm_weight(self) -> float:
        """Per-event normalisation constant (without lumi and without sign(genWeight))."""
        if self.kind != "mc":
            return 1.0
        return self.xsec / (self.nevents * self.generator_weight)


@dataclass(frozen=True)
class Selection:
    cuts: Mapping[str, str] = field(default_factory=dict)
    weights: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "cuts", _frozen(self.cuts))
        object.__setattr__(self, "weights", _frozen(self.weights))


@dataclass(frozen=True)
class Process:
    key: str
    group: str
    name: str
    kind: Kind
    plot_group: str
    selection: Selection | Mapping[str, Selection] = field(default_factory=Selection)  # one Selection or {channel: Selection}

    def selection_for(self, channel: str) -> Selection:
        """The process selection in `channel` (a single Selection applies to every channel)."""
        if isinstance(self.selection, Selection):
            return self.selection
        return self.selection[channel]


@dataclass(frozen=True)
class Region:
    name: str
    replace_cuts: Mapping[str, str] = field(default_factory=dict)
    add_weights: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "replace_cuts", _frozen(self.replace_cuts))
        object.__setattr__(self, "add_weights", _frozen(self.add_weights))


NOMINAL = Region("nominal")


@dataclass(frozen=True)
class Variable:
    name: str
    expr: str
    edges: tuple[float, ...]


@dataclass(frozen=True)
class Category:
    name: str
    cut: str
    variable: Variable


@dataclass(frozen=True)
class WeightVariation:
    name: str
    replace_weights: Mapping[str, str]

    def __post_init__(self):
        object.__setattr__(self, "replace_weights", _frozen(self.replace_weights))


@dataclass(frozen=True)
class ColumnVariation:
    name: str
    suffix: str


@dataclass(frozen=True)
class LnN:
    name: str
    processes: tuple[str, ...]
    value: float | tuple[float, float]
    channels: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Channel:
    name: str
    skim: Selection
    baseline: Selection
    regions: tuple[Region, ...] = ()
    keep_columns: tuple[str, ...] = ()

    def region(self, name: str) -> Region:
        if name == NOMINAL.name:
            return NOMINAL
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(f"channel {self.name}: unknown region {name!r}")


@dataclass(frozen=True)
class Estimator:
    name: Literal["fake_factors", "abcd"]
    regions: Mapping[str, str]
    subtract: tuple[str, ...]
    output: str

    def __post_init__(self):
        object.__setattr__(self, "regions", _frozen(self.regions))


@dataclass(frozen=True)
class Style:
    colors: Mapping[str, str]
    labels: Mapping[str, str]
    group_order: tuple[str, ...]
    signal_label: str
    lumi_label: str
    channel_labels: Mapping[str, str]
    axis_labels: Mapping[str, Mapping[str, str]] = field(default_factory=dict)  # channel -> variable -> label


@dataclass(frozen=True)
class MLExportConfig:
    variables: tuple[str, ...]
    processes: tuple[str, ...]
    label_of: Mapping[str, str]
    region_of: Mapping[str, str] = field(default_factory=dict)
    fold_column: str = "event"
    folds: int = 2


@dataclass(frozen=True)
class Analysis:
    name: str
    era: str
    lumi_pb: float
    channels: Mapping[str, Channel]
    samples: tuple[Sample, ...]
    processes: tuple[Process, ...]
    signal: str
    categories: tuple[Category, ...] = ()
    control_variables: Mapping[str, Variable] = field(default_factory=dict)
    weight_variations: tuple[WeightVariation, ...] = ()
    column_variations: tuple[ColumnVariation, ...] = ()
    lnn: tuple[LnN, ...] = ()
    estimator: Estimator | None = None
    style: Style | None = None
    ml: MLExportConfig | None = None

    def channel(self, name: str) -> Channel:
        return self.channels[name]

    def process(self, name: str) -> Process:
        for process in self.processes:
            if process.name == name:
                return process
        raise KeyError(f"unknown process {name!r}")

    def processes_of_kind(self, *kinds: str) -> tuple[Process, ...]:
        return tuple(p for p in self.processes if p.kind in kinds)

    def backgrounds(self) -> tuple[str, ...]:
        names = [p.name for p in self.processes if p.kind not in ("data", "signal")]
        if self.estimator is not None:
            names.append(self.estimator.output)
        return tuple(names)

    def samples_for(self, group: str, channel: str) -> tuple[Sample, ...]:
        return tuple(s for s in self.samples if s.group == group and (s.channels is None or channel in s.channels))

    def validate(self) -> None:
        problems: list[str] = []
        names = [p.name for p in self.processes]
        problems += [f"duplicate process name {n}" for n in sorted({n for n in names if names.count(n) > 1})]
        keys = [p.key for p in self.processes]
        problems += [f"duplicate process key {k}" for k in sorted({k for k in keys if keys.count(k) > 1})]
        groups = {s.group for s in self.samples}
        problems += [f"process {p.name}: no samples for group {p.group}" for p in self.processes if p.group not in groups]
        if self.signal not in names:
            problems.append(f"signal {self.signal} is not a process")
        for kind in {p.kind for p in self.processes}:
            if kind not in KINDS:
                problems.append(f"unknown process kind {kind}")
        category_names = [c.name for c in self.categories]
        problems += [f"duplicate category {n}" for n in sorted({n for n in category_names if category_names.count(n) > 1})]
        for channel in self.channels.values():
            cut_names, weight_names = set(channel.baseline.cuts), set(channel.baseline.weights)
            for region in channel.regions:
                for cut in region.replace_cuts:
                    if cut not in cut_names:
                        problems.append(f"channel {channel.name}, region {region.name}: replaces unknown cut {cut}")
            if self.estimator is not None:
                region_names = {r.name for r in channel.regions}
                for region in self.estimator.regions.values():
                    if region not in region_names:
                        problems.append(f"channel {channel.name}: estimator region {region} does not exist")
            all_weights = weight_names | {w for p in self.processes for w in p.selection_for(channel.name).weights}
            for variation in self.weight_variations:
                for weight in variation.replace_weights:
                    if weight not in all_weights:
                        problems.append(f"variation {variation.name}: replaces unknown weight {weight} (channel {channel.name})")
        if self.estimator is not None:
            for name in self.estimator.subtract:
                if name not in names:
                    problems.append(f"estimator subtracts unknown process {name}")
        if problems:
            raise AnalysisError("\n".join(sorted(set(problems))))

"""The analysis data model.

An analysis repository builds one `Analysis`; every ShapeSmith stage reads it. Everything that differs by channel
(samples, processes, regions, categories, variables, variations, estimators) lives in its `Channel`. Expressions are
strings in pandas.eval syntax (see expressions.py). Cut and weight *names* are the handles that regions and
variations replace, so they are unique within a selection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Mapping

if TYPE_CHECKING:
    from shapesmith.measurements import Measurement

SampleKind = Literal["data", "mc", "embedding"]
Role = Literal["data", "signal", "background", "auxiliary"]  # auxiliary: booked for an estimator, never a datacard process
SAMPLE_KINDS = ("data", "mc", "embedding")
ROLES = ("data", "signal", "background", "auxiliary")
NOMINAL = "nominal"


class AnalysisError(ValueError):
    """An inconsistent analysis; validate() lists every problem found, one per line."""


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
    cut: str | None = None  # per-sample event selection applied at skim time, e.g. to keep one generator-level part of a sample

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
    """A process owns its whole weight set: the analysis puts the common weights first, then the specific ones."""

    name: str
    group: str  # the sample group it is made of
    role: Role
    plot_group: str
    selection: Selection = field(default_factory=Selection)


@dataclass(frozen=True)
class Region:
    """A selection derived from the nominal one: cuts replaced by name, weights replaced where a process carries
    them, weights added to every process."""

    name: str
    replace_cuts: Mapping[str, str] = field(default_factory=dict)
    add_weights: Mapping[str, str] = field(default_factory=dict)
    replace_weights: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("replace_cuts", "add_weights", "replace_weights"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))


NOMINAL_REGION = Region(NOMINAL)


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
    """Weights replaced by name; a process lacking one of them does not get the variation."""

    name: str
    replace_weights: Mapping[str, str]
    applies_to: tuple[SampleKind, ...] = ("mc",)
    regions: tuple[str, ...] | None = None  # None: every booked region

    def __post_init__(self):
        object.__setattr__(self, "replace_weights", _frozen(self.replace_weights))


@dataclass(frozen=True)
class ColumnVariation:
    """Columns read shifted: `c + suffix` wherever that branch exists (a CROWN shift), or `derived[c]`, an expression
    of nominal columns (a shift computed in ShapeSmith). A name ending in Up/Down is one side of a shape nuisance;
    any other name is a template that datacards never turn into a shape line."""

    name: str
    suffix: str = ""
    derived: Mapping[str, str] = field(default_factory=dict)
    applies_to: tuple[SampleKind, ...] = ("mc",)
    regions: tuple[str, ...] | None = None  # None: every booked region where the rewrite changes an expression

    def __post_init__(self):
        object.__setattr__(self, "derived", _frozen(self.derived))


Variation = WeightVariation | ColumnVariation


@dataclass(frozen=True)
class DataMinus:
    """output = scale * (data - sum of `subtract`) in `region`, per column variation of its inputs."""

    output: str
    region: str
    subtract: tuple[str, ...]
    scale: float = 1.0


@dataclass(frozen=True)
class ABCD:
    """output = (data - MC)(b) * (data - MC)(c).sum() / (data - MC)(d).sum(), nominal only."""

    output: str
    b: str
    c: str
    d: str
    subtract: tuple[str, ...]


@dataclass(frozen=True)
class TemplateShift:
    """Variation `name`Up/Down of `process`: its nominal plus/minus `fraction` times the `template` process."""

    name: str
    process: str
    template: str
    fraction: float


Estimator = DataMinus | ABCD | TemplateShift


@dataclass(frozen=True)
class LnN:
    name: str
    processes: tuple[str, ...]
    value: float | tuple[float, float]
    channels: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Channel:
    name: str
    samples: tuple[Sample, ...]
    skim: Mapping[str, str]  # cuts applied when skimming; re-applied (under each variation) when filling
    cuts: Mapping[str, str]  # the nominal selection
    processes: tuple[Process, ...]
    regions: tuple[Region, ...] = ()
    categories: tuple[Category, ...] = ()
    variables: Mapping[str, Variable] = field(default_factory=dict)  # control variables
    variations: tuple[Variation, ...] = ()
    estimators: tuple[Estimator, ...] = ()
    keep_columns: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("skim", "cuts", "variables"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))

    def region(self, name: str) -> Region:
        if name == NOMINAL:
            return NOMINAL_REGION
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(f"channel {self.name}: unknown region {name!r}")

    def process(self, name: str) -> Process:
        for process in self.processes:
            if process.name == name:
                return process
        raise KeyError(f"channel {self.name}: unknown process {name!r}")

    def samples_of(self, group: str) -> tuple[Sample, ...]:
        return tuple(s for s in self.samples if s.group == group)

    def kind_of(self, process: Process) -> str:
        """The sample kind of a process (validate() makes sure its samples share one)."""
        return self.samples_of(process.group)[0].kind

    def data(self) -> str:
        """Name of the data process."""
        return next(p.name for p in self.processes if p.role == "data")

    def backgrounds(self) -> tuple[str, ...]:
        """The background processes followed by the estimator outputs: plots, sync, datacards and the ML export."""
        names = [p.name for p in self.processes if p.role == "background"]
        names += [e.output for e in self.estimators if isinstance(e, (DataMinus, ABCD))]
        return tuple(names)


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
    signal: str  # the signal process of datacards and plots
    channels: Mapping[str, Channel]
    lnn: tuple[LnN, ...] = ()
    style: Style | None = None
    ml: MLExportConfig | None = None
    measurement: Measurement | None = None  # run by `shapesmith measure`

    def channel(self, name: str) -> Channel:
        return self.channels[name]

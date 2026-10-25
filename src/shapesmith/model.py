"""The analysis data model.

An analysis repository builds one `Analysis`; every ShapeSmith stage reads it. Everything that differs by channel
(samples, processes, regions, categories, variables, variations, estimators) lives in its `Channel`. Expressions are
strings in pandas.eval syntax (see expressions.py). Cut and weight *names* are the handles that regions and
variations replace, so they are unique within a selection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Mapping, get_args

if TYPE_CHECKING:
    from shapesmith.measurements import Measurement

SampleKind = Literal["data", "mc", "embedding"]
Role = Literal["data", "signal", "background", "auxiliary"]  # auxiliary: booked for an estimator, never a datacard process
SAMPLE_KINDS = get_args(SampleKind)
ROLES = get_args(Role)
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
    # the era of its n-tuples when it is borrowed from another era or the analysis spans several eras (None: the era
    # of the run); it is normalised with the luminosity of the analysis that uses it, times lumi_scale
    era: str | None = None
    # the share of the analysis luminosity that its era contributes, for an analysis over several eras (e.g. one
    # fake-factor measurement of 2016preVFP and 2016postVFP: 19.52 / 36.33 and 16.81 / 36.33); 1 otherwise
    lumi_scale: float = 1.0

    @property
    def norm_weight(self) -> float:
        """Per-event normalisation constant (without lumi and without sign(genWeight))."""
        if self.kind != "mc":
            return 1.0
        return self.xsec / (self.nevents * self.generator_weight) * self.lumi_scale


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
class EqualData:
    """Edges from the data of the run (binning.py): `n_bins` bins with equal data counts in (low, high), computed from
    the data of the category in the nominal region whenever histograms are filled; with `cut`, from the data passing
    that cut instead of the category's (e.g. common edges of a pass and a fail category)."""

    n_bins: int
    low: float
    high: float
    cut: str | None = None


@dataclass(frozen=True)
class Variable:
    name: str
    expr: str
    edges: tuple[float, ...] | EqualData


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
    any other name is a template that datacards never turn into a shape line. `groups` restricts it to the samples of
    some groups, e.g. a CROWN shift produced for some samples only."""

    name: str
    suffix: str = ""
    derived: Mapping[str, str] = field(default_factory=dict)
    applies_to: tuple[SampleKind, ...] = ("mc",)
    regions: tuple[str, ...] | None = None  # None: every booked region where the rewrite changes an expression
    groups: tuple[str, ...] | None = None  # None: the samples of every group of those kinds

    def __post_init__(self):
        object.__setattr__(self, "derived", _frozen(self.derived))


Variation = WeightVariation | ColumnVariation


def applies(variation: Variation, kind: str, group: str) -> bool:
    """Whether a variation belongs to the samples of a kind and group."""
    if kind not in variation.applies_to:
        return False
    return not isinstance(variation, ColumnVariation) or variation.groups is None or group in variation.groups


@dataclass(frozen=True)
class DataMinus:
    """output = scale * (data - sum of `subtract`) in `region`, per column variation of its inputs, written to the
    region `target`; with `clip_negative`, negative bins are set to zero keeping the integral."""

    output: str
    region: str
    subtract: tuple[str, ...]
    scale: float = 1.0
    clip_negative: bool = False
    target: str = NOMINAL  # the region of the output, e.g. a diagnostic region


@dataclass(frozen=True)
class ABCD:
    """output = (data - MC)(b) * (data - MC)(c).sum() / (data - MC)(d).sum(), without variations, written to the
    region `target`."""

    output: str
    b: str
    c: str
    d: str
    subtract: tuple[str, ...]
    target: str = NOMINAL  # the region of the output, e.g. a diagnostic region


@dataclass(frozen=True)
class TemplateShift:
    """Variation `name`Up/Down of `process`: its nominal plus/minus `fraction` times the (nominal) `template` process.
    Every template variation of `process` (e.g. an energy-scale grid point) gets the same variation on top of it."""

    name: str
    process: str
    template: str
    fraction: float


@dataclass(frozen=True)
class VariationSum:
    """Variation `name`Up/Down of every process with a histogram of one of its parts: the nominal plus the sum over
    `parts` of the part's difference to the nominal, e.g. one nuisance from CROWN shifts produced per decay mode.
    The part `p` of a direction is the column variation `histogram.part_of(f"{name}{direction}", p)`. Exact where no
    event is changed by two parts; a process without a part histogram gets no variation."""

    name: str
    parts: tuple[str, ...]


Estimator = DataMinus | ABCD | TemplateShift | VariationSum


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
        return tuple(dict.fromkeys(names))  # an output estimated for several target regions once


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
    """What `ml-export` writes (ml_export.py): one row per event of the exported processes with its class.

    `class_of` maps every exported process or estimator output to one of `classes`, whose order is the class index.
    `variables` are columns or the names of channel or category variables (their expressions); `mc_only` among them
    exist for simulated samples only (generator truth) and are NaN in the rows of data and embedded processes.
    `identifiers` are copied with their n-tuple types. With `subtraction`, an exported DataMinus output also brings the
    processes it subtracts, in its region with negative weights (rows flagged is_subtraction)."""

    variables: tuple[str, ...]
    classes: tuple[str, ...]
    class_of: Mapping[str, str]
    region_of: Mapping[str, str] = field(default_factory=dict)
    identifiers: tuple[str, ...] = ("event",)
    mc_only: tuple[str, ...] = ()
    subtraction: bool = True
    profile: str = ""  # the name of the class scheme, recorded in classes.json

    def __post_init__(self):
        for name in ("class_of", "region_of"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))

    @property
    def processes(self) -> tuple[str, ...]:
        """The exported processes and estimator outputs, in the order of `class_of`."""
        return tuple(self.class_of)

    def class_index(self, name: str) -> int:
        return self.classes.index(self.class_of[name])

    def exported(self, channel: Channel) -> tuple[str, ...]:
        """The processes a channel exports: those of `processes` that are its signal or background processes or
        estimator outputs; a name the channel lacks (e.g. a process of another channel) is left out there."""
        own = {p.name for p in channel.processes if p.role == "signal"} | set(channel.backgrounds())
        return tuple(name for name in self.processes if name in own)

    def columns(self, kind: str) -> tuple[str, ...]:
        """The variables a process of a sample kind has: without `mc_only` for data and embedding."""
        return tuple(v for v in self.variables if kind == "mc" or v not in self.mc_only)


@dataclass(frozen=True)
class Analysis:
    name: str
    era: str
    lumi_pb: float
    signal: str | None  # the signal process of datacards and plots (None: a measurement analysis without one)
    channels: Mapping[str, Channel]
    control_channels: tuple[str, ...] = ()  # in every final state of datacards and fits, never one themselves
    lnn: tuple[LnN, ...] = ()
    style: Style | None = None
    ml: MLExportConfig | None = None
    measurement: Measurement | None = None  # run by `shapesmith measure`
    # Up/Down variations (name without the direction) that datacards write as an lnN instead of a shape: per bin and
    # process the ratio of the varied to the nominal yield, e.g. a weight variation whose shape effect is negligible
    lnn_variations: tuple[str, ...] = ()
    # Up/Down variations (name without the direction) that datacards symmetrise around the nominal where their pair is
    # one-sided or strongly asymmetric (datacards.symmetrise), e.g. the fake-factor shifts of a measurement payload
    symmetrised_variations: tuple[str, ...] = ()
    # the asymmetry above which a pair of symmetrised_variations is symmetrised: the larger relative yield effect of Up
    # and Down over the smaller one
    asymmetry_ratio: float = 2.0

    def channel(self, name: str) -> Channel:
        return self.channels[name]

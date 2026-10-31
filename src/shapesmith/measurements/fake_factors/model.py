"""The fake-factor measurement as data: the SM method of TauFakeFactors (QCD, optionally W+jets, and ttbar fake
factors, their fractions, the DR->SR corrections and the non-closure corrections), built by an analysis from its
regions and tables.

Regions and processes are names in the analysis channel; every quantity is measured per category of one split
variable (n_jets), in bins of its own variable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping


@dataclass(frozen=True)
class Fit:
    """How a measured ratio becomes the stored function: `binwise` keeps the bins; `smoothed` is the kernel
    smoothing with `bandwidth`, where the first `binwise_left` and the last `binwise_right` bins keep their values."""

    kind: Literal["binwise", "smoothed"]
    bandwidth: float = 0.0
    binwise_left: int = 0
    binwise_right: int = 0


@dataclass(frozen=True)
class Equipopulated:
    """The --suggest-binning options of a quantity (TauFakeFactors' equipopulated_binning_options): `n_bins` per
    category over [low, high], edges rounded to `rounding` digits, `add_left` edges put in front."""

    n_bins: tuple[int, ...]
    low: float
    high: float
    rounding: int = 2
    add_left: tuple[float, ...] = ()


@dataclass(frozen=True)
class Split:
    """Category i holds the events with edges[i] <= variable < edges[i + 1]; the payload bins on the same edges."""

    variable: str
    edges: tuple[float, ...]

    @property
    def categories(self) -> range:
        return range(len(self.edges) - 1)


@dataclass(frozen=True)
class Binned:
    """A quantity in bins of `variable`: edges[i] and fits[i] in category i of the split (fractions have no fits)."""

    variable: str
    edges: tuple[tuple[float, ...], ...]
    fits: tuple[Fit, ...] = ()
    equipopulated: Equipopulated | None = None


@dataclass(frozen=True)
class DrSr:
    """A DR->SR correction. Orthogonal fake factors (the process variable, bins and fits) in `sr_like`/`ar_like`,
    with their own `non_closures` there; the correction is the SR-like yield over the orthogonal-FF weighted AR-like
    yield in `sr`/`ar`. Every step measures `target` minus `subtract`: by default the target of the process (data,
    the QCD correction), or an MC process (the W+jets correction from W MC, TauFakeFactors' DR_SR with
    compute_orthogonal_fake_factors_using_data false), whose ratios keep empty bins and have no SystMCShift."""

    sr_like: str
    ar_like: str
    sr: str
    ar: str
    subtract: tuple[str, ...]
    correction: Binned
    non_closures: tuple[Binned, ...] = ()
    target: str | None = None  # None: the target of the process


@dataclass(frozen=True)
class SameSignQCD:
    """QCD subtracted from the target of a data-driven fake factor in every region where the process measures
    (TauFakeFactors' QCD_SS_estimate of the W+jets fake factors): data minus `subtract` in the same-sign version
    `regions[region]` of the region, bins below 0 set to 0, then subtracted after the processes of `subtract` and
    shifted with them for SystMCShift. In a closure the same-sign events carry the applied fake factors too."""

    regions: Mapping[str, str]  # region -> its same-sign version
    subtract: tuple[str, ...]  # every MC process, the target's own MC among them

    def region(self, name: str) -> str:
        if name not in self.regions:
            raise KeyError(f"no same-sign region for {name} (known: {', '.join(self.regions)})")
        return self.regions[name]


@dataclass(frozen=True)
class DataScale:
    """The global data/MC factor of an MC fake factor: (data - the MC but the target - QCD)(sr_like) / (...)(ar_like)
    over the same ratio of the target, from event totals; QCD is data minus all MC in the same-sign version of each
    region. `subtract` lists all MC processes, the target among them, in subtraction order."""

    sr_like: str
    ar_like: str
    sr_like_same_sign: str
    ar_like_same_sign: str
    subtract: tuple[str, ...]


@dataclass(frozen=True)
class ProcessFF:
    """The fake factors of one process, `target`(sr_like) / `target`(ar_like) with target = the `target` process
    minus the `subtract` processes (and minus `qcd`): from data (QCD, Wjets), or from MC scaled by `scale` (ttbar).
    The non-closures are measured in the same regions in table order, each on top of the fake factors, the DR->SR
    correction and the earlier non-closures."""

    name: str  # the payload process: QCD, Wjets or ttbar
    target: str
    subtract: tuple[str, ...]
    sr_like: str
    ar_like: str
    split: Split
    fake_factors: Binned
    non_closures: tuple[Binned, ...] = ()
    dr_sr: DrSr | None = None
    scale: DataScale | None = None
    qcd: SameSignQCD | None = None


@dataclass(frozen=True)
class Fractions:
    """The process fractions in `region`: QCD = data - `subtract` (negative bins 0), ttbar = `ttbar` and, with
    W+jets fake factors, Wjets = `wjets` (both MC)."""

    region: str
    subtract: tuple[str, ...]
    ttbar: str
    split: Split
    binned: Binned
    wjets: str | None = None


@dataclass(frozen=True)
class Leg:
    """The fake factors of one hadronic tau: payload names carry `suffix` ("" or "_subleading"); `wjets` adds the
    W+jets fake factors (the variant QCD + W + ttbar)."""

    suffix: str
    qcd: ProcessFF
    ttbar: ProcessFF
    fractions: Fractions
    wjets: ProcessFF | None = None

    @property
    def processes(self) -> tuple[ProcessFF, ...]:
        """In TauFakeFactors' order: QCD, Wjets, ttbar."""
        return tuple(p for p in (self.qcd, self.wjets, self.ttbar) if p is not None)


@dataclass(frozen=True)
class FakeFactorMeasurement:
    """`shapesmith measure` for the fake factors of every channel: the payloads fake_factors_<ch>.json.gz and
    FF_corrections_<ch>.json.gz in <output_dir>/<name>/<era>/ (one name per payload variant, e.g. fake_factors and
    fake_factors_w). With `placeholder_edges` (why the bin edges are not the measurement's own yet, e.g. an era
    without tables) measuring refuses and only --suggest-binning runs."""

    legs: Mapping[str, tuple[Leg, ...]]  # channel -> legs
    version: int = 1  # Correction.version of the payloads
    stat_sigma: float = 1.0
    name: str = "fake_factors"
    placeholder_edges: str | None = None

    def columns(self, channel: str) -> set[str]:
        """The columns every quantity of the channel is measured and evaluated in."""
        columns = set()
        for leg in self.legs[channel]:
            columns |= {leg.fractions.split.variable, leg.fractions.binned.variable}
            for process in leg.processes:
                binned = [process.fake_factors, *process.non_closures]
                if process.dr_sr is not None:
                    binned += [process.dr_sr.correction, *process.dr_sr.non_closures]
                columns |= {process.split.variable, *(b.variable for b in binned)}
        return columns

    def run(self, context) -> None:
        if context.merge or context.refit:
            raise ValueError(f"fake_factors: --{'merge' if context.merge else 'refit'} is not supported, one run measures every channel")
        if self.placeholder_edges is not None and not context.suggest_binning:
            raise ValueError(f"fake_factors: the bin edges are placeholders ({self.placeholder_edges}); run measure --suggest-binning and enter the edges first")
        from shapesmith.measurements.fake_factors.measure import run  # plots and correctionlib load only for a run

        run(self, context)

    def gallery(self, output, channels):
        """The plots, notes and facts of the measurement in `output` for `shapesmith publish`."""
        from shapesmith.measurements.fake_factors.gallery import content

        return content(self, output, channels)

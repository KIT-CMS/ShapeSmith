"""The fake-factor measurement as data: the SM method of TauFakeFactors (QCD and ttbar fake factors, their fractions,
the QCD DR->SR correction and the non-closure corrections), built by an analysis from its regions and tables.

Regions and processes are names in the analysis channel; every quantity is measured per category of one split
variable (n_jets), in bins of its own variable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    """The QCD DR->SR correction. Orthogonal fake factors (the QCD variable, bins and fits) from data in
    `sr_like`/`ar_like`, with their own `non_closures` there; the correction is the SR-like yield over the
    orthogonal-FF weighted AR-like yield in `sr`/`ar`. Every step subtracts `subtract`."""

    sr_like: str
    ar_like: str
    sr: str
    ar: str
    subtract: tuple[str, ...]
    correction: Binned
    non_closures: tuple[Binned, ...] = ()


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
    minus the `subtract` processes: from data (QCD), or from MC scaled by `scale` (ttbar). The non-closures are
    measured in the same regions in table order, each on top of the fake factors, the DR->SR correction and the
    earlier non-closures."""

    name: str  # the payload process: QCD or ttbar
    target: str
    subtract: tuple[str, ...]
    sr_like: str
    ar_like: str
    split: Split
    fake_factors: Binned
    non_closures: tuple[Binned, ...] = ()
    dr_sr: DrSr | None = None
    scale: DataScale | None = None


@dataclass(frozen=True)
class Fractions:
    """The QCD and ttbar fractions in `region`: QCD = data - `subtract` (negative bins 0), ttbar = `ttbar`."""

    region: str
    subtract: tuple[str, ...]
    ttbar: str
    split: Split
    binned: Binned


@dataclass(frozen=True)
class Leg:
    """The fake factors of one hadronic tau: payload names carry `suffix` ("" or "_subleading")."""

    suffix: str
    qcd: ProcessFF
    ttbar: ProcessFF
    fractions: Fractions

    @property
    def processes(self) -> tuple[ProcessFF, ProcessFF]:
        return (self.qcd, self.ttbar)


@dataclass(frozen=True)
class FakeFactorMeasurement:
    """`shapesmith measure` for the fake factors of every channel: the payloads fake_factors_<ch>.json.gz and
    FF_corrections_<ch>.json.gz in <output_dir>/fake_factors/<era>/."""

    legs: Mapping[str, tuple[Leg, ...]]  # channel -> legs
    version: int = 1  # Correction.version of the payloads
    stat_sigma: float = 1.0
    name: str = field(default="fake_factors", init=False)

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
        if context.merge:
            raise ValueError("fake_factors: --merge is not supported, one run measures every channel")
        from shapesmith.measurements.fake_factors.measure import run  # plots and correctionlib load only for a run

        run(self, context)

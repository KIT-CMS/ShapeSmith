"""The CMSSW part of the measurement, per category, with the options of the predecessor (smhtt_ul tauID_SFs_dev,
tau_id_es_measurement/emb_tau_id_sfs_ul.sh): MorphingTauID2017 datacards plus the r_DY_incl rate parameters, the
multiSignalModel workspace, the 2D likelihood scan over (r_EMB, ES), the close-up 1D scans of each POI with the other
profiled and the MultiDimFit singles fit, which is the result. As in the predecessor (plot_2D_scan.py), each step
starts at the lowest grid point of the previous one and its ranges are the 2 sigma intervals of that step, widened
by a margin: the profiles of the 2D scan give the close-up ranges (the predecessor took them from 1D scans over the
full range, some with margins set by hand), the close-up scans give the singles ranges.

The POIs: r_EMB_<category> (the scale factor) and ES_<category> (the energy scale in percent).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import uproot

from shapesmith.measurements.tau_id_es.grid import CHANNEL, CONTROL_CHANNEL, SIGNAL
from shapesmith.measurements.tau_id_es.synced import POSTFIX

MASS = "125"
R_RANGE = (0.1, 2.9)  # of the workspace
SCAN_R_RANGE = (0.11, 2.89)
SCAN_POINTS = 289
CLOSEUP_POINTS = 17
LEVEL = 6.18 / 2  # delta NLL of the predecessor's 2 sigma intervals (2 delta NLL = 6.18, the 2 sigma level of two parameters)
CLOSEUP_MARGIN = (0.2, 2.0)  # (r, ES): widening of the 2D scan's intervals to the close-up ranges
SINGLES_MARGIN = (0.05, 0.5)  # (r, ES): widening of the close-up scans' intervals to the singles ranges
ROBUST_FIT = "--robustFit=1 --setRobustFitAlgo=Minuit2 --X-rtd FITTER_NEW_CROSSING_ALGO --X-rtd FITTER_NEVER_GIVE_UP"
SCAN_FIT = f"{ROBUST_FIT} --cminFallbackAlgo Minuit2,Migrad,0:0.001 --cminFallbackAlgo Minuit2,Migrad,1:0.01 --cminPreScan"
Pair = tuple[float, float]


@dataclass(frozen=True)
class Interval:
    best: float
    low: float
    high: float


@dataclass(frozen=True)
class Scan:
    """The grid points of the 2D scan (r, ES, delta NLL)."""

    r: np.ndarray
    es: np.ndarray
    dnll: np.ndarray

    def profile(self, axis: int) -> Profile:
        """The profile of r (axis 0) or ES (axis 1) on the grid: the lowest delta NLL over the other POI."""
        values = (self.r, self.es)[axis]
        dnll = np.where(np.isfinite(self.dnll), self.dnll, np.inf)
        grid = np.unique(values)
        return Profile.of(grid, np.array([dnll[values == value].min() for value in grid]))


@dataclass(frozen=True)
class Profile:
    """A 1D likelihood scan: sorted grid values and their delta NLL, relative to the fit or, if a grid point lies
    lower (a fit that missed the minimum), to that point."""

    values: np.ndarray
    dnll: np.ndarray

    @classmethod
    def of(cls, values: np.ndarray, dnll: np.ndarray) -> Profile:
        finite = np.isfinite(dnll)
        order = np.argsort(values[finite], kind="stable")
        values, dnll = values[finite][order], dnll[finite][order]
        return cls(values, dnll - min(0.0, dnll.min()))

    @property
    def minimum(self) -> float:
        """The lowest grid point, the start value of the next step (the predecessor's workaround for failed fits)."""
        return float(self.values[np.argmin(self.dnll)])

    def crossings(self, level: float) -> tuple[float | None, float | None]:
        """The outermost crossings of `level` on each side of the minimum, linearly interpolated; None if there is
        none (plot_2D_scan.extract_confidence_interval)."""
        x, y, best = self.values, self.dnll, int(np.argmin(self.dnll))

        def crossing(i: int) -> float:  # on the segment from grid point i to i + 1
            return float(x[i] + (level - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))

        low = [i for i in range(best) if y[i] > level > y[i + 1]]
        high = [i for i in range(best, len(x) - 1) if y[i] < level < y[i + 1]]
        return (crossing(low[0]) if low else None), (crossing(high[-1]) if high else None)

    def window(self, margin: float, bounds: Pair) -> Pair:
        """The 2 sigma interval (an end of the scan where it does not cross), widened by `margin`, within `bounds`."""
        low, high = self.crossings(LEVEL)
        low, high = self.values[0] if low is None else low, self.values[-1] if high is None else high
        return max(float(low) - margin, bounds[0]), min(float(high) + margin, bounds[1])


def pois(category: str) -> tuple[str, str]:
    return f"r_{SIGNAL}_{category}", f"ES_{category}"


def es_range(grid: tuple[int, ...]) -> tuple[float, float]:
    """The ES range of the datacards (percent): the ends of the grid."""
    return min(grid) / 10, max(grid) / 10


def scan_es_range(grid: tuple[int, ...]) -> tuple[float, float]:
    low, high = es_range(grid)
    return low + 0.1, high - 0.1


def morphing_command(category: str, synced_dir: Path, era: str, grid: tuple[int, ...]) -> str:
    """MorphingTauID2017 as in the predecessor; it writes output/<category>/htt_mt_<category>/ below the working
    directory. Its ES loop runs in float from es_min in steps of the precision and skips 0; the accumulated rounding
    drops the last grid point (with the +-20 %, 0.2 % grid of the predecessor: +20.0)."""
    steps = set(np.diff(sorted(grid + (0,))))
    if len(steps) != 1:
        raise ValueError(f"MorphingTauID2017 needs a uniform grid, got steps {sorted(steps)}")
    low, high = es_range(grid)
    options = {
        "base_path": Path(synced_dir).resolve(), "input_folder_mt": CHANNEL, "input_folder_mm": CONTROL_CHANNEL,
        "real_data": "true", "classic_bbb": "false", "binomial_bbb": "false", "jetfakes": 0, "embedding": 1,
        "verbose": "false", "postfix": POSTFIX, "use_control_region": "true", "auto_rebin": "false",
        "rebin_categories": "false", "manual_rebin_for_yields": "false", "categories": category, "era": era,
        "tes_precision": steps.pop() / 10, "es_min": low, "es_max": high, "output": category,
    }
    return "MorphingTauID2017 " + " ".join(f"--{name}={value}" for name, value in options.items())


def card_dir(work_dir: Path, category: str) -> Path:
    return Path(work_dir) / "output" / category / f"htt_{CHANNEL}_{category}"


def add_rate_parameters(cards: Path, category: str) -> None:
    """The common normalisation r_DY_incl_<category> of the Z->ll processes in both regions: EMB, ZL and ZJ in mt,
    the embedded muons in mm."""
    line = f"r_DY_incl_{category} rateParam * {{}} 1.0 [0.5,1.5]\n"
    for card in sorted(Path(cards).glob(f"htt_{CHANNEL}_*.txt")):
        processes = [f"{SIGNAL}_{category}", "ZL", "ZJ"] if "ZL" in card.read_text() else [f"{SIGNAL}_{category}", "ZJ"]
        with card.open("a") as handle:
            handle.writelines(line.format(process) for process in processes)
    for card in sorted(Path(cards).glob(f"htt_{CONTROL_CHANNEL}_*.txt")):
        with card.open("a") as handle:
            handle.write(line.format("MUEMB"))


def workspace_command(category: str) -> str:
    r = pois(category)[0]
    return (f"combineTool.py -M T2W -i . -o workspace.root -m {MASS} -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel"
            f" --PO \"map=^.*/{SIGNAL}_{category}:{r}[1,{R_RANGE[0]},{R_RANGE[1]}]\"")


def _ranges(category: str, r_range: Pair, es: Pair) -> str:
    r, es_poi = pois(category)
    return f"--setParameterRanges {r}={r_range[0]!r},{r_range[1]!r}:{es_poi}={es[0]!r},{es[1]!r}"


def scan_command(category: str, grid: tuple[int, ...]) -> str:
    r, es = pois(category)
    return (f"combineTool.py -M MultiDimFit -n .scan_2D_{category} -d workspace.root --setParameters {r}=1.0,{es}=0.0"
            f" {_ranges(category, SCAN_R_RANGE, scan_es_range(grid))} {SCAN_FIT}"
            f" --redefineSignalPOIs {r},{es} --floatOtherPOIs=1 --points={SCAN_POINTS} --algo grid -m {MASS}"
            " --alignEdges=1 --cminDefaultMinimizerStrategy 0")


def closeup_command(category: str, poi: str, start: Pair, r_range: Pair, es: Pair) -> str:
    """The close-up scan of `poi`, the other POI profiled."""
    r, es_poi = pois(category)
    return (f"combineTool.py -M MultiDimFit -n .closeup_{poi} -d workspace.root --setParameters {es_poi}={start[1]!r},{r}={start[0]!r}"
            f" {_ranges(category, r_range, es)} {SCAN_FIT} --redefineSignalPOIs {poi} --algo grid -m {MASS}"
            f" --cminDefaultMinimizerStrategy 0 --floatOtherPOIs=1 --points={CLOSEUP_POINTS} --alignEdges=1")


def singles_command(category: str, start: Pair, r_range: Pair, es: Pair) -> str:
    r, es_poi = pois(category)
    return (f"combineTool.py -M MultiDimFit -n .singles_{category} -d workspace.root"
            f" --setParameters {es_poi}={start[1]!r},{r}={start[0]!r} {_ranges(category, r_range, es)} {ROBUST_FIT}"
            " --cminFallbackAlgo Minuit2,Migrad,0:0.001,Minuit2,Migrad,0:0.01 --cminPreScan"
            f" --redefineSignalPOIs {r},{es_poi} --floatOtherPOIs=1 --algo singles -m {MASS} --saveFitResult"
            " --cminDefaultMinimizerStrategy=1")


def postfit_command(category: str) -> str:
    return (f"PostFitShapesFromWorkspace -m {MASS} -w workspace.root --output postfit_shapes.root"
            f" -f multidimfit.singles_{category}.root:fit_mdf --postfit")


def output_path(cards: Path, name: str) -> Path:
    return Path(cards) / f"higgsCombine.{name}.MultiDimFit.mH{MASS}.root"


def read_scan(path: Path, category: str) -> Scan:
    """The grid points of the 2D scan (its first entry is the fit)."""
    r, es = pois(category)
    with uproot.open(path) as f:
        rows = f["limit"].arrays([r, es, "deltaNLL"], library="np")
    return Scan(rows[r][1:].astype(float), rows[es][1:].astype(float), rows["deltaNLL"][1:].astype(float))


def read_profile(path: Path, poi: str) -> Profile:
    """A close-up scan (its first entry is the fit)."""
    with uproot.open(path) as f:
        rows = f["limit"].arrays([poi, "deltaNLL"], library="np")
    return Profile.of(rows[poi][1:].astype(float), rows["deltaNLL"][1:].astype(float))


def closeup_inputs(scan: Scan, grid: tuple[int, ...]) -> tuple[Pair, Pair, Pair]:
    """Start values, r range and ES range of the close-up scans, from the profiles of the 2D scan."""
    return _next_inputs((scan.profile(0), scan.profile(1)), CLOSEUP_MARGIN, grid)


def singles_inputs(closeups: tuple[Profile, Profile], grid: tuple[int, ...]) -> tuple[Pair, Pair, Pair]:
    """Start values, r range and ES range of the singles fit, from the close-up scans of r and ES."""
    return _next_inputs(closeups, SINGLES_MARGIN, grid)


def _next_inputs(profiles: tuple[Profile, Profile], margins: Pair, grid: tuple[int, ...]) -> tuple[Pair, Pair, Pair]:
    start = (profiles[0].minimum, profiles[1].minimum)
    r_range, es = (p.window(margin, bounds) for p, margin, bounds in zip(profiles, margins, (SCAN_R_RANGE, scan_es_range(grid))))
    return start, r_range, es


def read_singles(path: Path, category: str) -> tuple[Interval, Interval]:
    """The singles fit: best fit and 1 sigma interval of r and ES (the limit tree holds the best fit, then the low
    and high crossing of each POI with the others at their best fit)."""
    names = pois(category)
    with uproot.open(path) as f:
        rows = f["limit"].arrays(list(names), library="np")
    return tuple(Interval(float(rows[name][0]), float(rows[name][1:].min()), float(rows[name][1:].max())) for name in names)


def interval_problems(name: str, interval: Interval, fit_range: tuple[float, float]) -> list[str]:
    """An interval that failed (an end equals the best fit) or ends at the fit range is a problem."""
    tolerance = 1e-5 * (fit_range[1] - fit_range[0])
    problems = []
    if not interval.low < interval.best < interval.high:
        problems.append(f"{name}: no interval around the best fit {interval.best:g} ({interval.low:g}, {interval.high:g})")
    if interval.low - fit_range[0] < tolerance or fit_range[1] - interval.high < tolerance:
        problems.append(f"{name}: interval [{interval.low:g}, {interval.high:g}] at the fit range [{fit_range[0]:g}, {fit_range[1]:g}]")
    return problems

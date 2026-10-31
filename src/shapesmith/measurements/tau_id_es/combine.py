"""The CMSSW part of the measurement, per category: MorphingTauID2017 datacards plus the r_DY_incl rate parameters, the
multiSignalModel workspace, the 2D likelihood scan over (r_EMB, ES), the profiles of both POIs, which give the result,
a fit with the POIs fixed at the profile minima for the postfit shapes, and the MultiDimFit singles fit of the
predecessor (smhtt_ul tauID_SFs_dev) as a cross-check.

Intervals from profile scans (v7 spec 5.4, U11): in v5 the singles fits landed in wrong minima and intervals ended at
the fit range. Now the best value of a POI is the minimum of its profile (PROFILE_POINTS grid points, the parabola
through the lowest point and its neighbours) and its 1 sigma interval the outermost crossings of delta NLL = 0.5 on
either side, linearly interpolated. A side without a crossing widens the scan range on that side and scans again, up
to MAX_WIDENINGS times within the bounds of the 2D scan; a side that still has no crossing (the interval would end at
the range) is a problem, which stops `--merge` with the category and the side. An interval with fewer than
MIN_INTERVAL_POINTS grid points inside is scanned again on a narrower range around it, up to MAX_REFINEMENTS times; a
best value outside its interval (an unresolved interval) is a problem.

The profiles. ES: a 1D scan with r profiled (r has one minimum). r: the likelihood has several minima in ES (the
morphing between the templates of the ES grid), and a 1D scan of r with ES floating follows whichever minimum its
start fit found (2018 Tight/VVLoose DM1011: ES jumping between 1 and 7 %, an r interval 15 times too narrow). The r
profile is therefore the envelope of a 2D grid with both POIs fixed in every point: PROFILE_POINTS values of r times
ENVELOPE_ES_POINTS values of ES in the window of the ES profile at delta NLL ENVELOPE_LEVEL (widened by
ENVELOPE_MARGIN), the lowest delta NLL over ES at each r. The ES at that minimum must stay inside its window within
the interval.

Ranges: as in the predecessor (plot_2D_scan.py), each step starts at the lowest grid point of the previous one. The
profiles of the 2D scan give the scan ranges (their 2 sigma windows above the lowest grid point, widened by
PROFILE_MARGIN); the r profiled in the
ES scan stays within its window, so that the scan points do not wander into a far minimum. The profiles give the
ranges of the singles cross-check (SINGLES_MARGIN).

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
PROFILE_POINTS = 41  # of each profile (U11)
ENVELOPE_ES_POINTS = 25  # ES values per r of the envelope scan
ENVELOPE_LEVEL = 2.0  # delta NLL of the ES profile that bounds the ES window of the envelope (2 sigma of one parameter)
ENVELOPE_MARGIN = 1.0  # percent: widening of that window
ONE_SIGMA = 0.5  # delta NLL of a 1 sigma interval of one parameter
LEVEL = 6.18 / 2  # delta NLL of the predecessor's 2 sigma windows (2 delta NLL = 6.18, the 2 sigma level of two parameters)
PROFILE_MARGIN = (0.2, 2.0)  # (r, ES): widening of the 2D scan's windows to the 1D scan ranges
SINGLES_MARGIN = (0.05, 0.5)  # (r, ES): widening of the profiles' windows to the singles ranges
MAX_WIDENINGS = 3  # rescans of a 1D profile whose 1 sigma interval leaves its range on a side
WIDENING = 1.0  # a rescan moves the open side of the range out by this times the range width
MIN_INTERVAL_POINTS = 5  # grid points inside a 1 sigma interval below which the profile is rescanned on a narrower range
MAX_REFINEMENTS = 2  # such rescans of a 1D profile
REFINEMENT = 1.0  # a narrower range is the interval widened on both sides by this times its width
SINGLES_DEVIATION = 0.2  # singles and profile intervals differ by more than this fraction of the profile interval width
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
        """The profile of r (axis 0) or ES (axis 1) on the grid: the lowest delta NLL over the other POI, whose value
        there is the profiled one."""
        values, others = ((self.r, self.es), (self.es, self.r))[axis]
        dnll = np.where(np.isfinite(self.dnll), self.dnll, np.inf)
        grid = np.unique(values)
        lowest = [np.flatnonzero(values == value)[np.argmin(dnll[values == value])] for value in grid]
        return Profile.of(grid, dnll[lowest], others[lowest])


@dataclass(frozen=True)
class Profile:
    """A 1D likelihood scan: sorted grid values and their delta NLL, relative to the fit or, if a grid point lies
    lower (a fit that missed the minimum), to that point. `profiled` holds the value of the profiled parameter at
    every grid point, where the scan tracked it."""

    values: np.ndarray
    dnll: np.ndarray
    profiled: np.ndarray | None = None

    @classmethod
    def of(cls, values: np.ndarray, dnll: np.ndarray, profiled: np.ndarray | None = None) -> Profile:
        finite = np.isfinite(dnll)
        order = np.argsort(values[finite], kind="stable")
        values, dnll = values[finite][order], dnll[finite][order]
        profiled = None if profiled is None else np.asarray(profiled)[finite][order]
        return cls(values, dnll - min(0.0, dnll.min()), profiled)

    @property
    def minimum(self) -> float:
        """The lowest grid point, the start value of the next step (the predecessor's workaround for failed fits)."""
        return float(self.values[np.argmin(self.dnll)])

    @property
    def best(self) -> float:
        """The minimum of the parabola through the lowest grid point and its neighbours (the lowest grid point at an
        end of the scan or where the parabola does not open upwards)."""
        i = int(np.argmin(self.dnll))
        if i == 0 or i == len(self.values) - 1:
            return float(self.values[i])
        (x0, x1, x2), (y0, y1, y2) = self.values[i - 1 : i + 2], self.dnll[i - 1 : i + 2]
        denominator = (x1 - x0) * (y1 - y2) - (x1 - x2) * (y1 - y0)
        if denominator >= 0:
            return float(x1)
        best = x1 - 0.5 * ((x1 - x0) ** 2 * (y1 - y2) - (x1 - x2) ** 2 * (y1 - y0)) / denominator
        return float(min(max(best, x0), x2))

    def crossings(self, level: float) -> tuple[float | None, float | None]:
        """The outermost crossings of `level` on each side of the minimum, linearly interpolated; None if there is
        none (plot_2D_scan.extract_confidence_interval). A grid point exactly at the level counts as a crossing."""
        x, y, best = self.values, self.dnll, int(np.argmin(self.dnll))

        def crossing(i: int) -> float:  # on the segment from grid point i to i + 1
            return float(x[i] + (level - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))

        low = [i for i in range(best) if y[i] >= level > y[i + 1]]
        high = [i for i in range(best, len(x) - 1) if y[i] < level <= y[i + 1]]
        return (crossing(low[0]) if low else None), (crossing(high[-1]) if high else None)

    def window(self, margin: float, bounds: Pair, level: float = LEVEL) -> Pair:
        """The window of `level` above the lowest grid point (default the 2 sigma level of two parameters; an end of
        the scan where it does not cross), widened by `margin`, within `bounds`. Above the lowest grid point, not the
        fit: on a grid coarse against the width of the likelihood every grid point can lie above `level` (2016preVFP
        Medium/Tight DM0: the lowest point of the 2D scan 4.7 above the fit), and a window of the fit's level would
        then be the whole scan range."""
        low, high = self.crossings(level + float(self.dnll.min()))
        low, high = self.values[0] if low is None else low, self.values[-1] if high is None else high
        return max(float(low) - margin, bounds[0]), min(float(high) + margin, bounds[1])

    @property
    def range(self) -> Pair:
        return float(self.values[0]), float(self.values[-1])

    @property
    def profiled_at_minimum(self) -> float | None:
        """The profiled parameter at the lowest grid point (None where it was not tracked)."""
        return None if self.profiled is None else float(self.profiled[np.argmin(self.dnll)])


def pois(category: str) -> tuple[str, str]:
    return f"r_{SIGNAL}_{category}", f"ES_{category}"


def es_range(grid: tuple[int, ...]) -> tuple[float, float]:
    """The ES range of the datacards (percent): the ends of the grid."""
    return min(grid) / 10, max(grid) / 10


def scan_es_range(grid: tuple[int, ...]) -> tuple[float, float]:
    low, high = es_range(grid)
    return low + 0.1, high - 0.1


def bounds(grid: tuple[int, ...]) -> tuple[Pair, Pair]:
    """The ranges of r and ES that every scan stays within: those of the 2D scan."""
    return SCAN_R_RANGE, scan_es_range(grid)


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


def profile_name(poi: str, attempt: int = 0) -> str:
    """The combine name (-n) of the profile scan of a POI (r: the envelope scan); a rescan on a widened range counts
    up."""
    kind = "envelope" if poi.startswith("r_") else "profile"
    return f"{kind}_{poi}" + (f"_{attempt}" if attempt else "")


def profile_command(category: str, poi: str, start: Pair, r_range: Pair, es: Pair, attempt: int = 0) -> str:
    """The profile scan of `poi`: for ES the 1D scan, r profiled within its range and tracked in the output; for r the
    envelope scan, a 2D grid of PROFILE_POINTS r values times ENVELOPE_ES_POINTS ES values with both fixed."""
    r, es_poi = pois(category)
    common = f"combineTool.py -M MultiDimFit -n .{profile_name(poi, attempt)} -d workspace.root --setParameters {es_poi}={start[1]!r},{r}={start[0]!r} {_ranges(category, r_range, es)} {SCAN_FIT}"
    if poi == r:
        return (f"{common} --redefineSignalPOIs {r},{es_poi} --floatOtherPOIs=1 --algo grid -m {MASS} --cminDefaultMinimizerStrategy 0"
                f" --gridPoints {PROFILE_POINTS},{ENVELOPE_ES_POINTS} --alignEdges=1")
    return (f"{common} --redefineSignalPOIs {poi} --algo grid -m {MASS}"
            f" --cminDefaultMinimizerStrategy 0 --floatOtherPOIs=1 --points={PROFILE_POINTS} --alignEdges=1 --trackParameters {r}")


def singles_command(category: str, start: Pair, r_range: Pair, es: Pair) -> str:
    r, es_poi = pois(category)
    return (f"combineTool.py -M MultiDimFit -n .singles_{category} -d workspace.root"
            f" --setParameters {es_poi}={start[1]!r},{r}={start[0]!r} {_ranges(category, r_range, es)} {ROBUST_FIT}"
            " --cminFallbackAlgo Minuit2,Migrad,0:0.001,Minuit2,Migrad,0:0.01 --cminPreScan"
            f" --redefineSignalPOIs {r},{es_poi} --floatOtherPOIs=1 --algo singles -m {MASS} --saveFitResult"
            " --cminDefaultMinimizerStrategy=1")


def bestfit_command(category: str, best: Pair, r_range: Pair, es: Pair) -> str:
    """The fit of the nuisances with both POIs fixed at the minima of their profiles (--algo none), whose fit result
    gives the postfit shapes."""
    r, es_poi = pois(category)
    return (f"combineTool.py -M MultiDimFit -n .bestfit_{category} -d workspace.root --setParameters {r}={best[0]!r},{es_poi}={best[1]!r}"
            f" {_ranges(category, r_range, es)} --freezeParameters {r},{es_poi} {ROBUST_FIT} --redefineSignalPOIs {r},{es_poi}"
            f" --algo none -m {MASS} --saveFitResult --cminDefaultMinimizerStrategy=1")


def postfit_command(category: str, best: Pair) -> str:
    """The postfit shapes of the fit at the profile minima. PostFitShapesFromWorkspace takes the floating parameters
    of the fit result only, so the frozen POIs are set with --freeze; without it the shapes had r = 1 and ES = 0
    (2018 Medium/VVLoose DM1: data/postfit 1.07 in the EMB bins at r = 1.09)."""
    r, es = pois(category)
    return (f"PostFitShapesFromWorkspace -m {MASS} -w workspace.root --output postfit_shapes.root"
            f" -f multidimfit.bestfit_{category}.root:fit_mdf --postfit --freeze {r}={best[0]!r},{es}={best[1]!r}")


def output_path(cards: Path, name: str) -> Path:
    return Path(cards) / f"higgsCombine.{name}.MultiDimFit.mH{MASS}.root"


def read_scan(path: Path, category: str) -> Scan:
    """The grid points of the 2D scan (its first entry is the fit)."""
    r, es = pois(category)
    with uproot.open(path) as f:
        rows = f["limit"].arrays([r, es, "deltaNLL"], library="np")
    return Scan(rows[r][1:].astype(float), rows[es][1:].astype(float), rows["deltaNLL"][1:].astype(float))


def read_poi_profile(cards: Path, category: str, poi: str, attempt: int = 0) -> Profile:
    """The profile of a POI from its scan (profile_command): the 1D scan of ES, the envelope of the r scan."""
    r, es = pois(category)
    path = output_path(cards, profile_name(poi, attempt))
    return read_scan(path, category).profile(0) if poi == r else read_profile(path, poi, r)


def read_profile(path: Path, poi: str, profiled: str | None = None) -> Profile:
    """A 1D scan (its first entry is the fit), with the tracked value of the profiled parameter if present."""
    with uproot.open(path) as f:
        tree = f["limit"]
        tracked = f"trackedParam_{profiled}" if profiled else None
        names = [poi, "deltaNLL"] + ([tracked] if tracked in tree.keys() else [])
        rows = tree.arrays(names, library="np")
    other = rows[tracked][1:].astype(float) if tracked in rows else None
    return Profile.of(rows[poi][1:].astype(float), rows["deltaNLL"][1:].astype(float), other)


def profile_inputs(scan: Scan, grid: tuple[int, ...]) -> tuple[Pair, Pair, Pair]:
    """Start values, r range and ES range of the 1D profile scans, from the profiles of the 2D scan."""
    return _next_inputs((scan.profile(0), scan.profile(1)), PROFILE_MARGIN, grid)


def singles_inputs(profiles: tuple[Profile, Profile], grid: tuple[int, ...]) -> tuple[Pair, Pair, Pair]:
    """Start values, r range and ES range of the singles fit, from the 1D profiles of r and ES."""
    return _next_inputs(profiles, SINGLES_MARGIN, grid)


def _next_inputs(profiles: tuple[Profile, Profile], margins: Pair, grid: tuple[int, ...]) -> tuple[Pair, Pair, Pair]:
    start = (profiles[0].minimum, profiles[1].minimum)
    r_range, es = (p.window(margin, limits) for p, margin, limits in zip(profiles, margins, bounds(grid)))
    return start, r_range, es


def _tolerance(limits: Pair) -> float:
    return 1e-5 * (limits[1] - limits[0])


def widened(profile: Profile, limits: Pair) -> Pair | None:
    """The scan range for a rescan: each side without a 1 sigma crossing moved out by WIDENING times the range
    width, within `limits`; None if the profile crosses on both sides or every open side is at its limit already."""
    low, high = profile.crossings(ONE_SIGMA)
    start, end = profile.range
    width, tolerance = end - start, _tolerance(limits)
    new_start = max(limits[0], start - WIDENING * width) if low is None and start - limits[0] > tolerance else start
    new_end = min(limits[1], end + WIDENING * width) if high is None and limits[1] - end > tolerance else end
    return None if (new_start, new_end) == (start, end) else (new_start, new_end)


def refined(profile: Profile, limits: Pair) -> Pair | None:
    """The scan range for a rescan of a profile that resolves its 1 sigma interval with fewer than
    MIN_INTERVAL_POINTS grid points: the interval widened on both sides by REFINEMENT times its width, at least by
    one grid step, within `limits`. Where no grid point lies below delta NLL = 0.5 (the minimum between two grid points far apart), the
    interval is taken at 0.5 above the lowest grid point. None if the interval is resolved, has an open side (that
    is `widened`'s case) or the range would not change.

    2016preVFP Medium/Tight DM0: the r profile on the whole scan range had a spacing of 0.07 against an interval of
    +-0.02; its crossings came from one segment on either side and the parabola put the best value outside them."""
    low, high = profile.crossings(ONE_SIGMA)
    if low is None or high is None:
        if profile.dnll.min() < ONE_SIGMA:
            return None
        low, high = profile.crossings(ONE_SIGMA + float(profile.dnll.min()))
        if low is None or high is None:
            return None
    if np.count_nonzero((profile.values > low) & (profile.values < high)) >= MIN_INTERVAL_POINTS:
        return None
    margin = max(REFINEMENT * (high - low), float(np.max(np.diff(profile.values))))  # at least a grid step: coarse crossings are off by up to one
    new = (max(limits[0], low - margin), min(limits[1], high + margin))
    tolerance = _tolerance(limits)
    return None if abs(new[0] - profile.range[0]) < tolerance and abs(new[1] - profile.range[1]) < tolerance else new


def profile_interval(name: str, profile: Profile, limits: Pair, profiled_name: str | None = None, profiled_range: Pair | None = None) -> tuple[Interval | None, list[str]]:
    """The interval of a 1D profile and its problems: a side without a crossing of delta NLL = 0.5 (the interval
    would end at the scan range; named by the side), a best value at an end of the scan, and a profiled parameter at
    an end of its range anywhere inside the interval (the profile there is no minimum over it)."""
    low, high = profile.crossings(ONE_SIGMA)
    start, end = profile.range
    problems = [f"{name}: no crossing of delta NLL = {ONE_SIGMA:g} on the {side} side within the scan range [{start:g}, {end:g}] (bounds [{limits[0]:g}, {limits[1]:g}])"
                for side, crossing in (("low", low), ("high", high)) if crossing is None]
    best = profile.best
    tolerance = _tolerance(limits)
    if best - start < tolerance or end - best < tolerance:
        problems.append(f"{name}: best value {best:g} at an end of the scan range [{start:g}, {end:g}]")
    if profile.profiled is not None and profiled_range is not None and low is not None and high is not None:
        inside = (profile.values >= low) & (profile.values <= high)
        at_edge = inside & ((profile.profiled - profiled_range[0] < _tolerance(profiled_range)) | (profiled_range[1] - profile.profiled < _tolerance(profiled_range)))
        if at_edge.any():
            problems.append(f"{name}: the profiled {profiled_name} reaches an end of its range [{profiled_range[0]:g}, {profiled_range[1]:g}] inside the interval")
    if low is None or high is None:
        return None, problems
    if not low < best < high:
        problems.append(f"{name}: best value {best:g} outside its interval [{low:g}, {high:g}] (the scan does not resolve the interval)")
        return None, problems
    return Interval(best, low, high), problems


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
        sides = [side for side, at in (("low", interval.low - fit_range[0] < tolerance), ("high", fit_range[1] - interval.high < tolerance)) if at]
        problems.append(f"{name}: interval [{interval.low:g}, {interval.high:g}] at the fit range [{fit_range[0]:g}, {fit_range[1]:g}] ({' and '.join(sides)} side)")
    return problems


def deviations(name: str, profile: Interval, singles: Interval) -> list[str]:
    """Where the singles cross-check differs from the profile interval by more than SINGLES_DEVIATION of its width
    (best value or either end)."""
    width = profile.high - profile.low
    result = []
    for field in ("best", "low", "high"):
        difference = getattr(singles, field) - getattr(profile, field)
        if abs(difference) > SINGLES_DEVIATION * width:
            result.append(f"{name} {field}: singles {getattr(singles, field):g} vs profile {getattr(profile, field):g} ({difference / width:+.0%} of the interval width)")
    return result

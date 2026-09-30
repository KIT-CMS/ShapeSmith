"""The CMSSW part of the measurement, per category, with the options of the predecessor (smhtt_ul tauID_SFs_dev,
tau_id_es_measurement/emb_tau_id_sfs_ul.sh): MorphingTauID2017 datacards plus the r_DY_incl rate parameters, the
multiSignalModel workspace, the 2D likelihood scan over (r_EMB, ES) and the MultiDimFit singles fit, which is the
result. The singles fit starts at the scan's best fit, inside the ranges the scan gives (singles_inputs).

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
SCAN_LEVEL = 4.5  # delta NLL of the scan region that gives the singles ranges: 3 sigma in one parameter
ROBUST_FIT = "--robustFit=1 --setRobustFitAlgo=Minuit2 --X-rtd FITTER_NEW_CROSSING_ALGO --X-rtd FITTER_NEVER_GIVE_UP"


@dataclass(frozen=True)
class Interval:
    best: float
    low: float
    high: float


@dataclass(frozen=True)
class Scan:
    """The 2D scan: its initial best fit and the grid points (r, ES, delta NLL)."""

    best: tuple[float, float]
    r: np.ndarray
    es: np.ndarray
    dnll: np.ndarray


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


def _ranges(category: str, r_range: tuple[float, float], es: tuple[float, float]) -> str:
    r, es_poi = pois(category)
    return f"--setParameterRanges {r}={r_range[0]!r},{r_range[1]!r}:{es_poi}={es[0]!r},{es[1]!r}"


def scan_command(category: str, grid: tuple[int, ...]) -> str:
    r, es = pois(category)
    return (f"combineTool.py -M MultiDimFit -n .scan_2D_{category} -d workspace.root --setParameters {r}=1.0,{es}=0.0"
            f" {_ranges(category, SCAN_R_RANGE, scan_es_range(grid))} {ROBUST_FIT}"
            " --cminFallbackAlgo Minuit2,Migrad,0:0.001 --cminFallbackAlgo Minuit2,Migrad,1:0.01 --cminPreScan"
            f" --redefineSignalPOIs {r},{es} --floatOtherPOIs=1 --points={SCAN_POINTS} --algo grid -m {MASS}"
            " --alignEdges=1 --cminDefaultMinimizerStrategy 0")


def singles_command(category: str, start: tuple[float, float], r_range: tuple[float, float], es: tuple[float, float]) -> str:
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
    r, es = pois(category)
    with uproot.open(path) as f:
        rows = f["limit"].arrays([r, es, "deltaNLL"], library="np")
    return Scan((float(rows[r][0]), float(rows[es][0])), rows[r][1:].astype(float), rows[es][1:].astype(float), rows["deltaNLL"][1:].astype(float))


def _box(values: np.ndarray, inside: np.ndarray, scan_range: tuple[float, float]) -> tuple[tuple[float, float], bool]:
    """The range of `values[inside]` widened by one grid step, clipped to the scan range; True if it was clipped."""
    step = float(np.min(np.diff(np.unique(values))))
    low, high = float(values[inside].min()) - step, float(values[inside].max()) + step
    clipped = low < scan_range[0] or high > scan_range[1]
    return (max(low, scan_range[0]), min(high, scan_range[1])), clipped


def singles_inputs(scan: Scan, grid: tuple[int, ...]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], list[str]]:
    """Start values, r range and ES range of the singles fit, and the problems of the scan.

    The ranges enclose the grid points within SCAN_LEVEL of the lowest one, widened by one grid step: the scan
    replaces the close-up scans the predecessor took them from. The start is the scan's best fit, or the lowest grid
    point if the best fit lies outside. A region that reaches the scan boundary is a problem (extend the scan)."""
    finite = np.isfinite(scan.dnll)
    inside = finite & (scan.dnll <= scan.dnll[finite].min() + SCAN_LEVEL)
    r_range, r_clipped = _box(scan.r[finite], inside[finite], SCAN_R_RANGE)
    es, es_clipped = _box(scan.es[finite], inside[finite], scan_es_range(grid))
    problems = [f"the {name} scan region reaches the scan boundary" for name, clipped in (("r", r_clipped), ("ES", es_clipped)) if clipped]
    start = scan.best
    if not (r_range[0] < start[0] < r_range[1] and es[0] < start[1] < es[1]):
        lowest = int(np.argmin(np.where(finite, scan.dnll, np.inf)))
        start = (float(scan.r[lowest]), float(scan.es[lowest]))
    return start, r_range, es, problems


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

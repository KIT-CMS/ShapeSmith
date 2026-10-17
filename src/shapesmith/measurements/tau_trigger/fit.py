"""The fit of one (decay mode, pT) bin in CMSSW: workspace, a coarse 1D profile scan of the efficiency over [0, 1], a
fine one around its minimum (PROFILE_POINTS points; rescans on a widened range while a side has no crossing of
delta NLL = 0.5, as the tau-ID/ES measurement, v7 spec 5.4), the fit at the scan minimum for the postfit shapes and
the singles fit as a cross-check. The result is the best value and the 1 sigma interval of the profile; an interval at
0 or 1 (no crossing inside [0, 1]) is a problem (v7_tt_emb_trigger 12.2)."""
from __future__ import annotations

import logging
import subprocess
from dataclasses import asdict
from pathlib import Path

import uproot

from shapesmith import cmssw
from shapesmith.config import CombineConfig
from shapesmith.measurements.tau_id_es.combine import Interval, Profile, deviations, profile_interval, read_profile, widened
from shapesmith.measurements.tau_trigger.cards import CARD, POI

logger = logging.getLogger(__name__)
MASS = "125"
BOUNDS = (0.0, 1.0)
COARSE_RANGE = (0.001, 0.999)
PROFILE_POINTS = 41
WINDOW_LEVEL = 2.0  # delta NLL of the coarse profile that bounds the fine scan (2 sigma of one parameter)
WINDOW_MARGIN = 0.02
MAX_WIDENINGS = 3
STEP_TIMEOUT = 1800.0
SINGLES_TIMEOUT = 600.0
ROBUST_FIT = "--robustFit=1 --setRobustFitAlgo=Minuit2 --X-rtd FITTER_NEW_CROSSING_ALGO --cminDefaultMinimizerStrategy=0"
CMSSW_FAILURES = (subprocess.CalledProcessError, subprocess.TimeoutExpired)


def workspace_command() -> str:
    return f"text2workspace.py {CARD} -o workspace.root -m {MASS} --X-allow-no-signal"


def scan_name(attempt: int | None) -> str:
    """coarse, then profile, profile_1, ... for the rescans."""
    return "coarse" if attempt is None else "profile" + (f"_{attempt}" if attempt else "")


def scan_command(scan_range: tuple[float, float], start: float, name: str) -> str:
    return (f"combine -M MultiDimFit -d workspace.root -m {MASS} -n .{name} --algo grid --points {PROFILE_POINTS} --alignEdges 1"
            f" --redefineSignalPOIs {POI} --floatOtherPOIs 1 --setParameters {POI}={start!r} --setParameterRanges {POI}={scan_range[0]!r},{scan_range[1]!r} {ROBUST_FIT}")


def bestfit_command(best: float) -> str:
    """The fit of the other parameters with the efficiency fixed at the profile minimum, for the postfit shapes."""
    return (f"combine -M MultiDimFit -d workspace.root -m {MASS} -n .bestfit --algo none --saveFitResult --redefineSignalPOIs {POI}"
            f" --setParameters {POI}={best!r} --freezeParameters {POI} {ROBUST_FIT}")


def postfit_command() -> str:
    return f"PostFitShapesFromWorkspace -m {MASS} -w workspace.root --output postfit_shapes.root -f multidimfit.bestfit.root:fit_mdf --postfit"


def singles_command(start: float) -> str:
    return (f"combine -M MultiDimFit -d workspace.root -m {MASS} -n .singles --algo singles --redefineSignalPOIs {POI}"
            f" --setParameters {POI}={start!r} --setParameterRanges {POI}={BOUNDS[0]!r},{BOUNDS[1]!r} {ROBUST_FIT}")


def output_path(directory: Path, name: str) -> Path:
    return Path(directory) / f"higgsCombine.{name}.MultiDimFit.mH{MASS}.root"


def read_singles(path: Path) -> Interval:
    with uproot.open(path) as f:
        values = f["limit"].arrays([POI], library="np")[POI].astype(float)
    return Interval(float(values[0]), float(values[1:].min()), float(values[1:].max()))


def fine_range(coarse: Profile) -> tuple[float, float]:
    """The window of the coarse profile at WINDOW_LEVEL widened by WINDOW_MARGIN, within [0, 1]."""
    return coarse.window(WINDOW_MARGIN, BOUNDS, WINDOW_LEVEL)


def fit_bin(combine: CombineConfig | None, directory: Path) -> dict:
    """The fit chain in `directory` (card.txt and shapes.root written): the profile interval of the efficiency, the
    scan ranges and rescans, the singles cross-check; problems and warnings."""
    directory = Path(directory)
    cmssw.run([f"{workspace_command()} > workspace.log 2>&1", f"{scan_command(COARSE_RANGE, 0.8, scan_name(None))} > coarse.log 2>&1"], combine, directory, STEP_TIMEOUT)
    coarse = read_profile(output_path(directory, scan_name(None)), POI)
    scan_range, start = fine_range(coarse), coarse.minimum
    for attempt in range(MAX_WIDENINGS + 1):
        cmssw.run([f"{scan_command(scan_range, start, scan_name(attempt))} > {scan_name(attempt)}.log 2>&1"], combine, directory, STEP_TIMEOUT)
        profile = read_profile(output_path(directory, scan_name(attempt)), POI)
        new_range = widened(profile, BOUNDS)
        if new_range is None or attempt == MAX_WIDENINGS:
            break
        logger.info(f"{directory.name}: no 1 sigma crossing on a side of {profile.range}, rescanning on {new_range}")
        scan_range = new_range
    interval, problems = profile_interval(POI, profile, BOUNDS)
    result = {"fit": None if interval is None else asdict(interval), "range": list(profile.range), "widenings": attempt, "points": len(profile.values), "problems": problems, "warnings": []}
    if interval is not None:
        try:
            cmssw.run([f"{bestfit_command(interval.best)} > bestfit.log 2>&1", f"{postfit_command()} > postfit.log 2>&1"], combine, directory, STEP_TIMEOUT)
        except CMSSW_FAILURES as error:
            result["warnings"].append(f"postfit at the profile minimum failed: {cmssw.failure(error)}")
    try:
        cmssw.run([f"{singles_command(profile.minimum)} > singles.log 2>&1"], combine, directory, SINGLES_TIMEOUT)
        singles = read_singles(output_path(directory, "singles"))
        result["singles"] = asdict(singles)
        if interval is not None:
            result["warnings"] += deviations(POI, interval, singles)
    except CMSSW_FAILURES as error:
        result["warnings"].append(f"singles cross-check failed: {cmssw.failure(error)}")
    return result


def fit_or_failure(combine: CombineConfig | None, directory: Path) -> dict:
    try:
        return fit_bin(combine, directory)
    except CMSSW_FAILURES as error:
        logger.error(f"{Path(directory).name}: CMSSW step failed: {error}")
        return {"fit": None, "problems": [f"CMSSW step failed: {error}"], "warnings": []}


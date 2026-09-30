"""From a measured ratio to the stored step functions with their uncertainty variations (TauFakeFactors'
smooth_function, FF_Updated helper/ff_functions.py sha256 31f89add...).

The variations are StatShift (the statistical band), SystMCShift (the ratio with every subtracted process shifted by
+-1 sigma, fitted the same way: both sides smoothed at their own centres of mass, user decision U13; TauFakeFactors
smooths the Down side of the corrections at the bin centres, because a deepcopy drops the centres of mass),
SystBandHigh/Low (the curve with bandwidth x1.5/x0.5, Down its mirror around the
nominal) and SystBandAsym (Up the x1.5 and Down the x0.5 curve). A smoothed ratio that is compatible with 1
(statistical_check) is reset to 1; smoothed curves are then sparsified. The spec's "Algorithm reference" states
every rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from shapesmith.measurements import smoothing
from shapesmith.measurements.fake_factors.hist import Hist
from shapesmith.measurements.fake_factors.model import Fit

STAT, MC, BAND_HIGH, BAND_LOW, BAND_ASYM = "StatShift", "SystMCShift", "SystBandHigh", "SystBandLow", "SystBandAsym"
BANDWIDTH_FACTORS = {BAND_HIGH: 1.5, BAND_LOW: 0.5}
RESET_P_VALUE = 0.05  # a smoothed ratio with a larger p-value of the compatibility with 1 is set to 1
SPARSIFY_THRESHOLD = 0.01  # a stored bin edge per 1 % of the y range of the variation


@dataclass(frozen=True)
class Step:
    """values[i] on [edges[i], edges[i + 1])."""

    edges: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class Fitted:
    """The fitted ratio: the curve at full resolution (plots) and the stored steps (payload), with variations."""

    curve: Step
    variations: dict[str, np.ndarray]  # on curve.edges
    stored: Step
    stored_variations: dict[str, Step]
    p_value: float | None = None  # of the compatibility with 1 (smoothed fits only)
    reset: bool = False  # compatible with 1, set to 1


def fit(ratio: Hist, fit: Fit, mc_shifted: tuple[Hist, Hist] | None = None, stat_sigma: float = 1.0) -> Fitted:
    """The ratio as binwise or smoothed steps with all variations; `mc_shifted` are the ratios with the subtracted
    processes shifted up and down (None: nothing is subtracted, SystMCShift is the nominal)."""
    measured = curve(ratio, fit, stat_sigma)
    nominal = measured.nominal
    variations = {f"{STAT}Up": measured.up, f"{STAT}Down": measured.down}
    shifted = [curve(h, fit, stat_sigma).nominal for h in mc_shifted] if mc_shifted else [nominal, nominal]
    variations.update({f"{MC}Up": shifted[0], f"{MC}Down": shifted[1]})
    if fit.kind == "binwise":
        variations.update({f"{band}{direction}": nominal for band in (BAND_HIGH, BAND_LOW, BAND_ASYM) for direction in ("Up", "Down")})
        steps = {name: Step(measured.edges, values) for name, values in variations.items()}
        return Fitted(Step(measured.edges, nominal), variations, Step(measured.edges, nominal), steps)
    bands = {band: curve(ratio, replace(fit, bandwidth=fit.bandwidth * factor), stat_sigma).nominal for band, factor in BANDWIDTH_FACTORS.items()}
    for band, values in bands.items():
        variations[f"{band}Up"] = (values - nominal) + nominal
        variations[f"{band}Down"] = (nominal - values) + nominal
    variations.update({f"{BAND_ASYM}Up": bands[BAND_HIGH], f"{BAND_ASYM}Down": bands[BAND_LOW]})
    p_value = compatibility_with_one(ratio)
    reset = p_value > RESET_P_VALUE
    if reset:
        nominal, variations = _reset(ratio, nominal, variations, mc_shifted, stat_sigma)
    stored = Step(*sparsify(measured.edges, nominal))
    stored_variations = {name: Step(*sparsify(measured.edges, values)) for name, values in variations.items()}
    return Fitted(Step(measured.edges, nominal), variations, stored, stored_variations, p_value, reset)


def curve(ratio: Hist, fit: Fit, stat_sigma: float) -> smoothing.Curve:
    """The full-resolution curve of a ratio with its statistical band, at the centres of mass."""
    if fit.kind == "binwise":
        return smoothing.binwise(ratio.values, ratio.errors, ratio.edges, stat_sigma)
    return smoothing.smooth(ratio.means, ratio.values, ratio.errors, ratio.edges, fit.bandwidth, fit.binwise_left, fit.binwise_right, stat_sigma)


def compatibility_with_one(ratio: Hist) -> float:
    """TauFakeFactors' statistical_check: the pulls (y - 1) / sigma with the data-only (suppressed) errors; the
    smallest two-sided p-value of the pull sum over all windows of adjacent bins, corrected for the number of bins."""
    errors = ratio.suppressed_errors.clip(min=1e-6)
    pulls = (ratio.values - 1.0) / errors
    n = len(pulls)
    p_min = 1.0
    for width in range(1, n + 1):
        for start in range(n - width + 1):
            z = abs(np.sum(pulls[start : start + width]) / np.sqrt(width))
            p_min = min(p_min, math.erfc(z / math.sqrt(2.0)))  # 2 * (1 - Phi(z))
    return 1 - (1 - p_min) ** n


def _reset(ratio: Hist, nominal: np.ndarray, variations: dict[str, np.ndarray], mc_shifted: tuple[Hist, Hist] | None, stat_sigma: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """A ratio compatible with 1: nominal and band variations 1, the statistical band that of the inverse-variance
    mean, SystMCShift the constant fitted to each shifted ratio (1 without shifted ratios)."""
    errors = ratio.suppressed_errors.clip(min=1e-6)
    inclusive = 1.0 / np.sqrt(np.sum(1.0 / errors**2))
    ones = np.ones_like(nominal)
    result = {}
    for name, values in variations.items():
        if name.startswith((BAND_ASYM, BAND_HIGH, BAND_LOW)):
            result[name] = ones
        elif name.startswith(STAT):
            result[name] = np.full_like(values, 1.0 + (stat_sigma if name.endswith("Up") else -stat_sigma) * inclusive)
        else:
            result[name] = values
    constants = [constant_fit(h) for h in mc_shifted] if mc_shifted else [1.0, 1.0]
    result[f"{MC}Up"], result[f"{MC}Down"] = (np.full_like(nominal, constant) for constant in constants)
    return ones, result


def constant_fit(hist: Hist) -> float:
    """ROOT's pol0 chi2 fit of the bin values: the inverse-variance weighted mean of the bins with an error; 1 if
    there is none (the fit fails)."""
    errors = hist.errors
    used = errors > 0
    if not used.any():
        return 1.0
    weights = 1.0 / errors[used] ** 2
    return float(np.sum(weights * hist.values[used]) / np.sum(weights))


def sparsify(edges: np.ndarray, values: np.ndarray, threshold: float = SPARSIFY_THRESHOLD) -> tuple[np.ndarray, np.ndarray]:
    """TauFakeFactors' sparsify: keep a bin edge wherever the cumulative |change| of the values passes another
    `threshold` times their range, and the first and the last bin; constant values become one bin."""
    cumulative = np.concatenate(([0], np.cumsum(np.abs(np.diff(values)))))
    step = (values.max() - values.min()) * threshold
    if step == 0:
        return np.array([edges[0], edges[-1]]), np.array([values[0]])
    kept = np.unique(np.searchsorted(cumulative, np.arange(0, cumulative[-1], step)))
    kept = kept[kept < len(values)]
    if 0 not in kept:
        kept = np.insert(kept, 0, 0)
    if len(values) - 1 not in kept:
        kept = np.append(kept, len(values) - 1)
    return np.append(edges[kept], edges[-1]), values[kept]

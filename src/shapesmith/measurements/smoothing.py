"""Kernel smoothing of binned ratios as TauFakeFactors does it, without ROOT.

The reference is FF_Updated `helper/ff_functions.py` (`_smooth_function`, `_get_index_and_slices`; sha256 31f89add...,
the state that produced the SM 2018 payload): 100 samples per bin from Normal(y, stat_sigma * sigma) with seed 19,
each smoothed with ROOT's TGraphSmooth::SmoothKern (normal kernel) on the grid linspace(x[start], x[end], 100 n + 1)
without its last point; the nominal is the mean and the statistical band the standard deviation over the samples.
A hybrid keeps the first or last bins binwise (TauFakeFactors' `binwise#[0]+smoothed`: the first bin).
tests/reference/make_smoothing_reference.py checks the port against ROOT and FF_Updated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

KERNEL_SCALE = 0.3706506  # ROOT's normal kernel: quartiles at +-0.25 * bandwidth
SAMPLES = 100
SEED = 19


@dataclass(frozen=True)
class Curve:
    """A piecewise constant function: `nominal[i]` on [edges[i], edges[i + 1]), with its statistical band."""

    edges: np.ndarray
    nominal: np.ndarray
    up: np.ndarray
    down: np.ndarray


def smooth_kern(x: np.ndarray, y: np.ndarray, x_out: np.ndarray, bandwidth: float) -> np.ndarray:
    """ROOT's TGraphSmooth::SmoothKern with the normal kernel (BDRksmooth), at the sorted points `x_out`.

    A transcription of ROOT's loop, not a textbook kernel smoother: the window of an output point starts at a running
    index that only moves forward and may still include the last point below x0 - cutoff (ROOT sums it), and ends at
    the last point <= x0 + cutoff; where no point contributes, the result is 0. The sums run in ROOT's order with
    libm's exp, so that the result is bitwise ROOT's. `y` may have a second axis (samples), smoothed in one go.
    """
    x_out = np.asarray(x_out, dtype=np.float64)
    if np.any(np.diff(x_out) < 0):
        raise ValueError("x_out must be sorted")
    order = np.argsort(x, kind="stable")
    x, y = np.asarray(x, dtype=np.float64)[order], np.asarray(y, dtype=np.float64)[order]
    width = KERNEL_SCALE * bandwidth
    cutoff = 4 * width
    result = np.zeros((len(x_out), *y.shape[1:]))
    start = int(np.searchsorted(x, x_out[0] - cutoff, side="left"))
    points = x.tolist()
    for j, x0 in enumerate(x_out.tolist()):
        end = int(np.searchsorted(x, x0 + cutoff, side="right"))
        numerator, denominator = 0.0, 0.0
        for i in range(start, end):
            distance = abs(points[i] - x0) / width
            weight = math.exp(-0.5 * distance * distance)
            numerator = numerator + weight * y[i]
            denominator += weight
        if denominator > 0:
            result[j] = numerator / denominator
        start = max(start, int(np.searchsorted(x, x0 - cutoff, side="left")) - 1)
    return result


def binwise(y: np.ndarray, errors: np.ndarray, bin_edges: np.ndarray, stat_sigma: float = 1.0) -> Curve:
    """The bins themselves as the curve, with their statistical errors; negative values are set to 0."""
    y, errors, bin_edges = (np.asarray(a, dtype=np.float64) for a in (y, errors, bin_edges))
    return _clipped(bin_edges, y, y + stat_sigma * errors, y - stat_sigma * errors, bin_edges)


def smooth(x: np.ndarray, y: np.ndarray, errors: np.ndarray, bin_edges: np.ndarray, bandwidth: float, binwise_left: int = 0, binwise_right: int = 0, stat_sigma: float = 1.0) -> Curve:
    """The smoothed curve of a histogram with values `y`, symmetric `errors`, bin `x` positions (the centres of mass)
    and `bin_edges`; the first `binwise_left` and the last `binwise_right` bins keep their bin values (a hybrid).
    Negative values are set to 0, the curve covers the full bin range."""
    x, y, errors, bin_edges = (np.asarray(a, dtype=np.float64) for a in (x, y, errors, bin_edges))
    start, end = binwise_left, len(y) - 1 - binwise_right  # the first and the last smoothed bin
    rng = np.random.RandomState(SEED)  # the stream of TauFakeFactors' np.random.seed(19), without global state
    samples = np.array([rng.normal(mu, stat_sigma * sigma, SAMPLES) for mu, sigma in zip(y, errors)])
    samples[samples < 0.0] = 0.0
    n = 100 * len(y)
    grid = np.linspace(x[start], x[end], n + 1)
    smoothed = smooth_kern(x, samples, grid[:n], bandwidth)
    mean, std = smoothed.mean(axis=1), smoothed.std(axis=1)
    if not binwise_left and not binwise_right:
        return _clipped(grid, mean, mean + std, mean - std, bin_edges)
    left = slice(None, start)
    right = slice(-binwise_right, None) if binwise_right else slice(0, 0)  # counted from the end, as in TauFakeFactors
    overlap = slice(None, -1) if binwise_right else slice(None, None)  # the grid ends where the right bins begin
    parts = [(bin_edges[left], y[left], stat_sigma * errors[left])]
    if binwise_left:  # the smoothed part begins at the lower edge of bin `start`, with its first value
        parts.append((bin_edges[start : start + 1], mean[:1], std[:1]))
    parts.append((grid[overlap], mean[overlap], std[overlap]))
    if binwise_right:  # and ends at the lower edge of bin `end + 1`, with its last value
        parts.append((bin_edges[end + 1 : end + 2], mean[-1:], std[-1:]))
    parts.append((bin_edges[right], y[right], stat_sigma * errors[right]))
    edges, nominal, band = (np.concatenate(column) for column in zip(*parts))
    return _clipped(edges, nominal, nominal + band, nominal - band, bin_edges)


def _clipped(edges: np.ndarray, nominal: np.ndarray, up: np.ndarray, down: np.ndarray, bin_edges: np.ndarray) -> Curve:
    """Negative values set to 0; the first and last value extended to the outer bin edges where the curve stops short."""
    nominal, up, down = (np.where(values < 0, 0.0, values) for values in (nominal, up, down))
    if edges[0] != bin_edges[0]:
        edges, nominal, up, down = np.insert(edges, 0, bin_edges[0]), *(np.insert(v, 0, v[0]) for v in (nominal, up, down))
    if edges[-1] != bin_edges[-1]:
        edges, nominal, up, down = np.append(edges, bin_edges[-1]), *(np.append(v, v[-1]) for v in (nominal, up, down))
    return Curve(edges, nominal, up, down)

"""The histogram of the fake-factor measurement: TauFakeFactors' patched ROOT TH1D, without ROOT.

Besides ROOT's bin contents and Sumw2 (`values`, `variances`) a histogram carries per bin the weighted event count
and mean of x (`counts`, `means`: the centre of mass, the x position of the bin in the fits) and TauFakeFactors'
"base" values with their errors, of which `suppressed_errors` keep only the errors of the minuend of every
subtraction (the data errors, the input of the compatibility check). Every operation follows ROOT and FF_Updated
helper/hooks_and_patches.py (sha256 79acd651..., the state of the SM 2018 payload): `add` and `divide` are its
patched TH1.Add/Divide, `shifted` is its AddError, `scaled` is ROOT's unpatched TH1.Scale (contents and Sumw2
only). tests/reference/make_fake_factor_reference.py checks the port against ROOT with the patches.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Hist:
    edges: np.ndarray
    values: np.ndarray
    variances: np.ndarray
    counts: np.ndarray
    means: np.ndarray
    base_values: np.ndarray
    base_errors: np.ndarray
    suppressed_errors: np.ndarray

    @classmethod
    def fill(cls, edges, x: np.ndarray, weights: np.ndarray) -> Hist:
        """ROOT's TH1D::Fill per event (bins [low, high), in event order) and TauFakeFactors' centre of mass
        (np.histogram, whose last bin includes its upper edge)."""
        edges = np.asarray(edges, dtype=np.float64)
        x, weights = np.asarray(x), np.asarray(weights, dtype=np.float64)
        n = len(edges) - 1
        index = np.searchsorted(edges, x, side="right") - 1
        inside = (index >= 0) & (index < n)
        values = np.bincount(index[inside], weights=weights[inside], minlength=n)
        variances = np.bincount(index[inside], weights=weights[inside] ** 2, minlength=n)
        counts, _ = np.histogram(x, bins=edges, weights=weights)
        sums, _ = np.histogram(x, bins=edges, weights=x * weights)
        centres = 0.5 * (edges[:-1] + edges[1:])
        means = np.array([s / c if c != 0 else centre for s, c, centre in zip(sums, counts, centres)])
        errors = np.sqrt(variances)
        return cls(edges, values, variances, counts, means, values.copy(), errors, errors.copy())

    @property
    def errors(self) -> np.ndarray:
        return np.sqrt(self.variances)

    @property
    def centres(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    def add(self, other: Hist, factor: float = 1.0) -> Hist:
        """self + factor * other; a subtraction (factor < 0) keeps the suppressed errors of self."""
        counts, means = self._centre_of_mass(other, factor, self.counts + factor * other.counts)
        base_errors = np.sqrt(self.base_errors**2 + (factor * other.base_errors) ** 2)
        suppressed = self.suppressed_errors.copy() if factor < 0 else np.sqrt(self.suppressed_errors**2 + (factor * other.suppressed_errors) ** 2)
        return Hist(
            self.edges, self.values + factor * other.values, self.variances + factor * factor * other.variances,
            counts, means, self.base_values + factor * other.base_values, base_errors, suppressed,
        )

    def divide(self, other: Hist) -> Hist:
        """self / other bin by bin, 0 where other is 0 (ROOT's TH1::Divide); the centre of mass of both combined."""
        c0, c1 = self.values, other.values
        nonzero = c1 != 0
        values = np.divide(c0, c1, out=np.zeros_like(c0), where=nonzero)
        c1sq = c1 * c1
        variances = np.divide(self.variances * c1sq + other.variances * c0 * c0, c1sq * c1sq, out=np.zeros_like(c0), where=nonzero)
        ratio_counts = np.divide(self.counts, other.counts, out=np.zeros_like(self.counts), where=other.counts != 0)
        counts, means = self._centre_of_mass(other, 1.0, ratio_counts)
        base_values = np.divide(self.base_values, other.base_values, out=np.zeros_like(self.base_values), where=other.base_values != 0)
        return Hist(
            self.edges, values, variances, counts, means, base_values,
            _relative_sum(base_values, (self.base_errors, self.base_values), (other.base_errors, other.base_values)),
            _relative_sum(base_values, (self.suppressed_errors, self.base_values), (other.suppressed_errors, other.base_values)),
        )

    def scaled(self, factor: float) -> Hist:
        """ROOT's TH1::Scale: contents and Sumw2 only (TauFakeFactors does not patch it)."""
        return replace(self, values=self.values * factor, variances=self.variances * (factor * factor))

    def shifted(self, sigmas: float) -> Hist:
        """Every bin moved by `sigmas` times its error and clipped at 0 (TauFakeFactors' AddError); the error of a
        clipped bin is 0, and so is its count, whose mean moves to the bin centre. The base values are not clipped."""
        errors = self.errors
        values = np.maximum(0.0, self.values + sigmas * errors)
        variances = np.where(values > 0, errors, 0.0) ** 2
        moved = sigmas * (abs(sigmas) * errors)
        counts = self.counts + moved
        means = np.divide(self.counts * self.means + moved * self.means, counts, out=self.centres.copy(), where=counts != 0)
        means = np.where(counts < 0, self.centres, _clip_means(means, self.edges))
        return Hist(
            self.edges, values, variances, np.maximum(0.0, counts), means,
            self.base_values + sigmas * self.base_errors, self.base_errors.copy(), self.suppressed_errors.copy(),
        )

    def with_negative_bins_zeroed(self) -> Hist:
        """Negative contents set to 0 (errors, centre of mass and base values unchanged)."""
        return replace(self, values=np.maximum(self.values, 0.0))

    def with_empty_bins_set_to_one(self) -> Hist:
        """Bins <= 0 become 1 +- 1 with the bin centre as their mean and count 1 (TauFakeFactors'
        calculate_non_closure_correction); the base values are unchanged."""
        empty = self.values <= 0.0
        return replace(
            self, values=np.where(empty, 1.0, self.values), variances=np.where(empty, 1.0, self.variances),
            counts=np.where(empty, 1.0, self.counts), means=np.where(empty, self.centres, self.means),
        )

    def _centre_of_mass(self, other: Hist, factor: float, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """TauFakeFactors' calc_center_of_mass: the means weighted with the counts of both, inside the bin."""
        total = self.counts + factor * other.counts
        weighted = self.counts * self.means + factor * other.counts * other.means
        means = np.divide(weighted, total, out=self.centres.copy(), where=total != 0)
        return counts, np.where(total != 0, _clip_means(means, self.edges), means)


def _clip_means(means: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Means kept strictly inside their bins."""
    low, high = edges[:-1], edges[1:]
    return np.maximum(np.minimum(means, np.nextafter(high, low)), np.nextafter(low, high))


def _relative_sum(values: np.ndarray, *parts: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """|values| times the quadratic sum of the relative errors error / value of the parts (0 where value is 0)."""
    squares = sum(np.divide(error, value, out=np.zeros_like(error), where=value != 0) ** 2 for error, value in parts)
    return np.abs(values) * np.sqrt(squares)

"""Efficiencies by counting: the cross-check of the fit, eps_MC and eps_EMB^mon (v7_tt_emb_trigger 12.2).

A count is the sum of event weights with its variance in the pass and the fail category of a bin. From data:
eps = N_P / (N_P + N_F) with N = data - backgrounds - QCD, QCD = same-sign data - MC (each count below 0 set to 0),
in a counting region and m_vis window; for a single process (MC genuine taus, the embedded taus with their monitoring
flags) N is its own count. The error propagates the variances of N_P and N_F, which are independent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class Count:
    value: float
    variance: float

    def __add__(self, other: Count) -> Count:
        return Count(self.value + other.value, self.variance + other.variance)

    def __sub__(self, other: Count) -> Count:
        return Count(self.value - other.value, self.variance + other.variance)

    def clipped(self) -> Count:
        """A count below 0 set to 0 (its variance kept), as the QCD estimate."""
        return Count(max(self.value, 0.0), self.variance)


ZERO = Count(0.0, 0.0)


def count(weights: np.ndarray) -> Count:
    weights = np.asarray(weights, dtype=np.float64)
    return Count(float(weights.sum()), float((weights * weights).sum()))


def total(counts: Iterable[Count]) -> Count:
    result = ZERO
    for c in counts:
        result = result + c
    return result


def efficiency(passing: Count, failing: Count) -> dict | None:
    """eps = P / (P + F) and its error; None if P + F <= 0."""
    denominator = passing.value + failing.value
    if denominator <= 0:
        return None
    value = passing.value / denominator
    variance = (failing.value ** 2 * passing.variance + passing.value ** 2 * failing.variance) / denominator ** 4
    return {"value": value, "error": math.sqrt(variance), "pass": passing.value, "fail": failing.value}


def data_efficiency(counts: Callable[[str, str, str], Count], data: str, backgrounds: Iterable[str], mc: Iterable[str], region: str, same_sign: str) -> dict | None:
    """The counting cross-check: counts(process, region, side) of data minus the backgrounds minus QCD (same-sign
    data minus `mc`, clipped at 0) in pass and fail."""
    backgrounds, mc = tuple(backgrounds), tuple(mc)
    sides = {}
    for side in ("pass", "fail"):
        qcd = (counts(data, same_sign, side) - total(counts(p, same_sign, side) for p in mc)).clipped()
        sides[side] = counts(data, region, side) - total(counts(p, region, side) for p in backgrounds) - qcd
    return efficiency(sides["pass"], sides["fail"])


def relative_error(entry: Mapping | None) -> float:
    return math.inf if not entry or entry["value"] <= 0 else entry["error"] / entry["value"]

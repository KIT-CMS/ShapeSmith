"""The binning of NN-score categories as pure array functions (numpy in, edges out): the data-blind algorithm of the
NN spec 6.3, the binning of the master's thesis (its start edges plus CombineHarvester's AutoRebin) and the blinding
edge of N14. `datacards` resolves the edges of a ScoreBinning category with them after `estimate`; the proxy limit of
bbtautau-nn imports the same functions. Nothing here reads data.

Inputs are per fine bin (the `hist` binning, e.g. 1000 equal bins on [0, 1]): `fine_edges` (n + 1 values), the
signal sum of weights (n) and per background name a template `(sumw, sumw2, raw_count)`, the sums of weights and of
squared weights and the unweighted event count per fine bin. Edges are always a subset of the fine edges and span
them from the first to the last.

Data-blind algorithm (resolve, resolve_edges), with BinningRules `r`:
1. Start bins in [r.low, last fine edge] (r.low = 1/K for an argmax category of K classes; lower scores cannot occur):
   r.n_start bins of equal signal (start "signal"; the HH category, 50) or of equal total background (start
   "background"; the background categories, 5), each edge the upper edge of the fine bin where the running sum first
   reaches k / n_start of the total; edges that coincide are dropped. Without a positive total, equal numbers of fine
   bins.
2. From the top, start bins are merged downwards until the merged bin passes every rule; then it is accepted and the
   next bin starts below it. A bin passes when
   - the total background is >= r.min_background,
   - the relative statistical uncertainty of the total background, sqrt(sum sumw2) / sum sumw, is <= r.max_rel_uncertainty,
   - every group of r.required (label -> background names) has >= r.min_raw unweighted events (summed over its names),
   - with r.non_negative, the signal and every background template are >= 0 (above -NEGATIVE_TOLERANCE).
3. After r.max_bins accepted bins, or at the bottom, the rest (every lower start bin and the fine bins below r.low)
   joins the lowest accepted bin; where no bin passes, the category is one bin. The leftover can make the lowest bin
   fail a rule again (e.g. a negative template); the Resolution lists the rules each final bin fails.

Thesis binning (auto_rebin): the custom edges of the HH category (0, 0.2, 0.3, 0.35, ..., 0.98, 1) and 20 equal bins
elsewhere, rebinned by CombineHarvester's AutoRebin with SetBinThreshold(1.0), SetBinUncertFraction(0.9) and
SetRebinMode(1) on the total background, as HHDatacards/hh_datacards.py: auto_rebin replicates
AutoRebin::FindNewBinning of CombineHarvester (CombineTools/src/AutoRebin.cc) step by step, in float64 where ROOT
keeps the total in a TH1F (float32), which can only matter where a merged bin lies within float32 rounding of a
threshold.

Blinding (blinding_index): the first bin, from the lowest score, with S/B(mu=1) > threshold (0.005, N14), counted
with the nominal templates of the final binning; the bins from it upwards form the `hi` card channel.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

NEGATIVE_TOLERANCE = 1e-9  # a template sum above -1e-9 counts as non-negative (rounding of cancelling weights)
THESIS_HH_EDGES = (0.0, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0)
THESIS_OTHER_EDGES = tuple(round(0.05 * i, 2) for i in range(21))  # 20 equal bins on [0, 1]
AUTO_REBIN_THRESHOLD = 1.0  # SetBinThreshold of hh_datacards.py
AUTO_REBIN_UNCERTAINTY = 0.9  # SetBinUncertFraction of hh_datacards.py
BLINDING_S_OVER_B = 0.005  # N14

Template = tuple[np.ndarray, np.ndarray, np.ndarray]  # sumw, sumw2, raw count per fine bin


@dataclass(frozen=True)
class BinningRules:
    """The rules of the data-blind algorithm (module docstring). Defaults: the HH category of NN spec 6.3."""

    min_background: float = 1.0
    max_rel_uncertainty: float = 0.30
    min_raw: float = 1.0
    required: Mapping[str, tuple[str, ...]] = field(default_factory=dict)  # group label -> background names
    non_negative: bool = True
    max_bins: int = 15
    start: Literal["signal", "background"] = "signal"
    n_start: int = 50
    low: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "required", {label: tuple(names) for label, names in self.required.items()})


SIGNAL_RULES = BinningRules()
BACKGROUND_RULES = BinningRules(max_rel_uncertainty=0.10, max_bins=5, start="background", n_start=5)


def category_rules(n_classes: int, signal: bool, required: Mapping[str, tuple[str, ...]] | None = None) -> BinningRules:
    """The rules of an argmax category of `n_classes` classes: SIGNAL_RULES for the HH category, BACKGROUND_RULES for
    a background category, with low = 1 / n_classes and the raw-count groups `required`."""
    rules = SIGNAL_RULES if signal else BACKGROUND_RULES
    return dataclasses.replace(rules, low=1.0 / n_classes, required=dict(required or {}))


@dataclass(frozen=True)
class FinalBin:
    """One bin of a Resolution: its range, the signal and total background, the relative uncertainty of the
    background, the raw count of every required group, the smallest template sum (and its name) and the rules it
    fails (background, uncertainty, raw:<group>, negative:<name>)."""

    low: float
    high: float
    signal: float
    background: float
    rel_uncertainty: float
    raw: dict[str, float]
    smallest: tuple[str, float]
    failed: tuple[str, ...]


@dataclass(frozen=True)
class Resolution:
    """The edges of the data-blind algorithm and how they came about: the start edges, the final bins, whether the
    bin cap stopped the merging (capped) and the range the lowest accepted bin took over (leftover; None if none)."""

    edges: tuple[float, ...]
    start: tuple[float, ...]
    bins: tuple[FinalBin, ...]
    capped: bool
    leftover: tuple[float, float] | None

    @property
    def passed(self) -> bool:
        """Every final bin passes every rule."""
        return not any(b.failed for b in self.bins)


def _check(fine_edges: np.ndarray, signal_sumw: np.ndarray, backgrounds: Mapping[str, Template], rules: BinningRules) -> None:
    n = len(fine_edges) - 1
    if n < 1 or not np.all(np.diff(fine_edges) > 0):
        raise ValueError("fine_edges must hold at least two strictly increasing values")
    if len(signal_sumw) != n:
        raise ValueError(f"signal_sumw has {len(signal_sumw)} values for {n} fine bins")
    for name, template in backgrounds.items():
        if len(template) != 3 or any(len(np.asarray(part)) != n for part in template):
            raise ValueError(f"background {name}: (sumw, sumw2, raw_count) need {n} values each")
    unknown = sorted({name for names in rules.required.values() for name in names} - set(backgrounds))
    if unknown:
        raise ValueError(f"required names without a background template: {', '.join(unknown)}")
    empty = sorted(label for label, names in rules.required.items() if not names)
    if empty:
        raise ValueError(f"required groups without names: {', '.join(empty)}")
    if rules.max_bins < 1 or rules.n_start < 1:
        raise ValueError("max_bins and n_start must be >= 1")
    if not rules.low < fine_edges[-1]:
        raise ValueError(f"low {rules.low} is not below the last fine edge {fine_edges[-1]}")


def _quantile_indices(weights: np.ndarray, first: int, n_start: int) -> list[int]:
    """Fine-edge indices of `n_start` bins with equal sums of `weights` (the fine bins from `first` on); equal numbers
    of fine bins without a positive total."""
    n = len(weights) + first
    running = np.maximum.accumulate(np.cumsum(weights))
    total = running[-1] if running.size else 0.0
    if total > 0.0:
        inner = first + 1 + np.searchsorted(running, total * np.arange(1, n_start) / n_start, side="left")
    else:
        inner = np.rint(np.linspace(first, n, n_start + 1)[1:-1]).astype(int)
    return sorted({first, n, *(int(i) for i in inner if first < i < n)})


def start_indices(fine_edges, signal_sumw, backgrounds: Mapping[str, Template], rules: BinningRules) -> list[int]:
    """The fine-edge indices of the start bins (step 1): the first fine edge at or above rules.low, the quantile
    edges, the last fine edge."""
    fine_edges = np.asarray(fine_edges, dtype=np.float64)
    first = int(np.searchsorted(fine_edges, rules.low - 1e-12, side="left"))
    first = min(first, len(fine_edges) - 2)
    if rules.start == "signal":
        weights = np.asarray(signal_sumw, dtype=np.float64)
    else:
        weights = np.sum([np.asarray(t[0], dtype=np.float64) for t in backgrounds.values()], axis=0) if backgrounds else np.zeros(len(fine_edges) - 1)
    return _quantile_indices(weights[first:], first, rules.n_start)


class _Sums:
    """Cumulative sums per fine bin, so that the sums of any range of fine bins cost O(1)."""

    def __init__(self, signal_sumw: np.ndarray, backgrounds: Mapping[str, Template], rules: BinningRules):
        def cumulative(values) -> np.ndarray:
            return np.concatenate([[0.0], np.cumsum(np.asarray(values, dtype=np.float64))])

        self.signal = cumulative(signal_sumw)
        self.templates = {name: cumulative(t[0]) for name, t in backgrounds.items()}
        self.background = cumulative(np.sum([np.asarray(t[0], dtype=np.float64) for t in backgrounds.values()], axis=0)) if backgrounds else np.zeros(len(self.signal))
        self.variance = cumulative(np.sum([np.asarray(t[1], dtype=np.float64) for t in backgrounds.values()], axis=0)) if backgrounds else np.zeros(len(self.signal))
        self.raw = {label: cumulative(np.sum([np.asarray(backgrounds[n][2], dtype=np.float64) for n in names], axis=0)) for label, names in rules.required.items()}
        self.rules = rules

    @staticmethod
    def _range(cumulative: np.ndarray, low: int, high: int) -> float:
        return float(cumulative[high] - cumulative[low])

    def final_bin(self, fine_edges: np.ndarray, low: int, high: int) -> FinalBin:
        rules = self.rules
        background = self._range(self.background, low, high)
        rel = math.sqrt(max(self._range(self.variance, low, high), 0.0)) / background if background > 0.0 else math.inf
        raw = {label: self._range(c, low, high) for label, c in self.raw.items()}
        sums = {"signal": self._range(self.signal, low, high), **{name: self._range(c, low, high) for name, c in self.templates.items()}}
        smallest = min(sums.items(), key=lambda item: item[1])
        failed = []
        if not background >= rules.min_background:
            failed.append("background")
        if not rel <= rules.max_rel_uncertainty:
            failed.append("uncertainty")
        failed += [f"raw:{label}" for label, count in raw.items() if not count >= rules.min_raw]
        if rules.non_negative:
            failed += [f"negative:{name}" for name, value in sums.items() if value < -NEGATIVE_TOLERANCE]
        return FinalBin(float(fine_edges[low]), float(fine_edges[high]), sums["signal"], background, rel, raw, smallest, tuple(failed))


def resolve(fine_edges, signal_sumw, backgrounds: Mapping[str, Template], rules: BinningRules = SIGNAL_RULES) -> Resolution:
    """The data-blind algorithm (module docstring) with its record."""
    fine_edges = np.asarray(fine_edges, dtype=np.float64)
    signal_sumw = np.asarray(signal_sumw, dtype=np.float64)
    _check(fine_edges, signal_sumw, backgrounds, rules)
    starts = start_indices(fine_edges, signal_sumw, backgrounds, rules)
    sums = _Sums(signal_sumw, backgrounds, rules)
    accepted: list[tuple[int, int]] = []  # (low, high) fine-edge indices, from the top
    upper, k = starts[-1], len(starts) - 2
    while k >= 0 and len(accepted) < rules.max_bins:
        if not sums.final_bin(fine_edges, starts[k], upper).failed:
            accepted.append((starts[k], upper))
            upper = starts[k]
        k -= 1
    capped = len(accepted) == rules.max_bins and upper > starts[0]
    leftover = (float(fine_edges[0]), float(fine_edges[upper])) if accepted and upper > 0 else None
    if accepted:
        accepted[-1] = (0, accepted[-1][1])
    else:
        accepted = [(0, len(fine_edges) - 1)]
    indices = [0] + [high for _, high in reversed(accepted)]
    bins = tuple(sums.final_bin(fine_edges, low, high) for low, high in reversed(accepted))
    return Resolution(tuple(float(fine_edges[i]) for i in indices), tuple(float(fine_edges[i]) for i in starts), bins, capped, leftover)


def resolve_edges(fine_edges, signal_sumw, backgrounds: Mapping[str, Template], rules: BinningRules = SIGNAL_RULES) -> tuple[float, ...]:
    """The edges of the data-blind algorithm (module docstring).

    fine_edges: n + 1 increasing values; signal_sumw: n signal sums of weights; backgrounds: name -> (sumw, sumw2,
    raw_count), n values each; rules: BinningRules (SIGNAL_RULES for the HH category, BACKGROUND_RULES for a background
    category, category_rules for both with low = 1/K and the raw-count groups). Returns a subset of the fine edges with
    the first and the last."""
    return resolve(fine_edges, signal_sumw, backgrounds, rules).edges


def _ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator with the IEEE result of C++ (inf, nan) instead of an exception."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(numerator) / np.float64(denominator))


def _find_new_binning(edges: list[float], values: np.ndarray, sumw2: np.ndarray, threshold: float, uncert_fraction: float) -> list[float]:
    """One call of AutoRebin::FindNewBinning in mode 1 (bins 1-based as in ROOT, `values[i - 1]` is bin i), repeated
    until no bin fails or a pass changes nothing."""
    while True:
        n = len(values)
        content = lambda i: float(values[i - 1])  # noqa: E731
        error = lambda i: math.sqrt(max(float(sumw2[i - 1]), 0.0))  # noqa: E731
        low_edge = lambda i: edges[i - 1]  # noqa: E731
        hbin = int(np.argmax(values)) + 1  # GetMaximumBin: the first maximum
        lbin = int(np.argmin(values)) + 1  # GetMinimumBin: the first minimum
        herr, maximum = 1, 0.0  # GetMaximumFracUncertBin
        for i in range(1, n + 1):
            if _ratio(error(i), content(i)) > maximum:
                maximum, herr = _ratio(error(i), content(i)), i
        tot_flag = content(lbin) <= threshold
        err_flag = _ratio(error(herr), content(herr)) >= uncert_fraction
        new = list(edges)
        if tot_flag or err_flag:
            lbin = lbin if tot_flag else herr

            def failing(total: float, err: float) -> bool:
                return (tot_flag and total <= threshold) or (not tot_flag and _ratio(err, total) >= uncert_fraction)

            left_tot, left_err, idx = content(lbin), error(lbin), lbin - 1
            while failing(left_tot, left_err) and idx > 0:
                left_err = math.sqrt(left_err**2 + error(idx) ** 2)
                left_tot += content(idx)
                idx -= 1
            left_bins = lbin - idx
            right_tot, right_err, idx = content(lbin), error(lbin), lbin + 1
            while failing(right_tot, right_err) and idx <= n:
                right_err = math.sqrt(right_err**2 + error(idx) ** 2)
                right_tot += content(idx)
                idx += 1
            right_bins = idx - lbin
            left_pass = left_tot > threshold if tot_flag else _ratio(left_err, left_tot) < uncert_fraction
            right_pass = right_tot > threshold if tot_flag else _ratio(right_err, right_tot) < uncert_fraction
            merge_left = [low_edge(i) for i in range(lbin, lbin - left_bins + 1, -1)]
            merge_right = [low_edge(i + 1) for i in range(lbin, lbin + right_bins - 1)]
            if left_pass and not right_pass:
                removed = merge_left
            elif right_pass and not left_pass:
                removed = merge_right
            elif left_pass and right_pass and left_bins != right_bins:
                removed = merge_left if left_bins < right_bins else merge_right
            elif left_pass and right_pass and lbin != hbin:
                removed = merge_right if lbin < hbin else merge_left  # equally good: towards the peak
            elif left_pass and right_pass:
                removed = []  # equally good at the peak: AutoRebin changes nothing
            else:
                removed = [low_edge(i) for i in range(2, n + 1)]  # no solution: one bin
            new = [edge for edge in edges if edge not in removed]
        if len(new) == len(edges):
            return edges
        indices = [edges.index(edge) for edge in new]
        values = np.add.reduceat(values, indices[:-1])
        sumw2 = np.add.reduceat(sumw2, indices[:-1])
        edges = new
        if all(not (v <= threshold or _ratio(math.sqrt(max(e, 0.0)), v) >= uncert_fraction) for v, e in zip(values, sumw2)):
            return edges


def auto_rebin(edges, total_sumw, total_sumw2, threshold: float = AUTO_REBIN_THRESHOLD, uncert_fraction: float = AUTO_REBIN_UNCERTAINTY) -> tuple[float, ...]:
    """CombineHarvester's AutoRebin in mode 1 (module docstring) on the total background `total_sumw` (sum of weights)
    and `total_sumw2` (sum of squared weights) in the bins of `edges`: merges the bins with a total <= threshold or a
    relative error >= uncert_fraction into their neighbours; returns the new edges (a subset of `edges`)."""
    edges = [float(e) for e in edges]
    values = np.asarray(total_sumw, dtype=np.float64)
    sumw2 = np.asarray(total_sumw2, dtype=np.float64)
    if len(values) != len(edges) - 1 or len(sumw2) != len(values):
        raise ValueError("total_sumw and total_sumw2 need len(edges) - 1 values")
    return tuple(_find_new_binning(edges, values.copy(), sumw2.copy(), threshold, uncert_fraction))


def snap(edges, fine_edges) -> list[int]:
    """The indices of the fine edges closest to `edges` (each within 1e-9, else ValueError)."""
    fine_edges = np.asarray(fine_edges, dtype=np.float64)
    indices = [int(np.argmin(np.abs(fine_edges - edge))) for edge in edges]
    if not np.allclose(fine_edges[indices], edges, rtol=0.0, atol=1e-9):
        raise ValueError(f"edges {list(edges)} are not a subset of the fine edges")
    return indices


def blinding_index(signal, background, threshold: float = BLINDING_S_OVER_B) -> int | None:
    """The index of the first bin (lowest score first) with S/B(mu=1) > threshold, None if no bin has it; a bin with
    a positive signal and no positive background counts as above."""
    signal = np.asarray(signal, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    for i, (s, b) in enumerate(zip(signal, background)):
        if (b > 0.0 and s / b > threshold) or (b <= 0.0 and s > 0.0):
            return i
    return None

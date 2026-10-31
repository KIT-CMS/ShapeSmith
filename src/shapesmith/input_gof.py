"""The goodness of fit of single variables (NN spec 7.1, NV1: every NN input per channel x era): the data against the
prediction of one control variable in one region, as a datacard of its own with every nuisance of the run, tested with
the saturated GoF of checks.py (frequentist toys); the cards of every era, channel and variable run in parallel.

Inputs: the control histograms of `hist --control` with the variations (`--regions` for a control region, which books
the signal there too) and of `estimate --control`, the variables filled in fine bins that the cards merge.

Cards: <output>/<region>/<era>/<channel>/<variable>/combined.txt, one bin htt_<channel>_1_<era> with the signal and every
background of the channel (datacards.collect_bin and write_bins: the lnN, the shapes, no-op nuisances left out, the
symmetrisation of the analysis, autoMCStats). The fine bins are merged to bins of about equal total background
(equal_prediction_edges): at most `n_bins` bins with at least `min_prediction` background events each; an integer
variable (fine bins of width 1 around the integers) keeps one bin per value, its sparse tails merged. The edges come from
the nominal prediction, never from the data: edges at the data quantiles make the data flat by construction and bias
the observed statistic low. `exclude` (regular expressions on the nuisance names) leaves nuisances out of the cards,
e.g. the preliminary pseudo-shifts for the variant without them.

GoF: checks.gof_commands with every toy in one job per card, `combine.parallel` cards at a time: the workspace, the
observed saturated statistic of the background-only fit (r frozen at 0) and the frequentist toys, with a seed per card
(card_seed) so that the toys of different cards are independent. The minimizer options of the run (combine.minimizer)
apply to every fit. Set a fallback (e.g. --cminFallbackAlgo Minuit2,Migrad,0:0.1): with combine's defaults a failed
Migrad (status -1) of the observed fit is reported as it is, a statistic of 10^2-10^3 on an unremarkable card, and the
frequentist toys are generated around the failed fit. Neither setting converges everywhere: the analytic minimization of
the bin parameters of autoMCStats (combine's default) fails on some cards where the plain one (--X-rtd
MINIMIZER_no_analytic) converges, and the reverse. With `alt_minimizer` the observed statistic of every card is also
computed with these options added (alt_jobs); since a failed background-only fit leaves its minimum too high and the
saturated fit converges in both, the lower statistic is the better fit: a card whose alternative statistic is lower by
more than ALT_TOLERANCE runs again (observed and toys) with the alternative options, recorded in
<card>/gof/minimizer.json, which its tail toys use too.

p-values: toys resolve a p-value only down to about MIN_TOYS_ABOVE / toys, far above the threshold alpha / N of the
failure rule (about 5e-5 for 1000 tests). The p-value is the fraction of toys at least as large as the observed
statistic (p_toys) when at least MIN_TOYS_ABOVE toys are; below, the tail of the scaled chi-square s * chi2(k) with the
median and the 90 % quantile of the toys (the saturated statistic of populated Poisson bins is chi-square distributed
asymptotically; quantiles, because a few toys of a failed fit lie far in the tail): p_fit = Q(k / 2, observed / (2 s)).
The summary compares p_fit with p_toys wherever the toys resolve both (calibration of the extrapolation). With few
toys the quantiles, and with them the extrapolation, are noisy: from 300 toys p_fit erred by up to a factor 14 at
p ~ 1e-4 against 20 000-100 000 toys (NV report 2026-10-25). Tail toys (`tail_toys`, run_tail_toys): every card whose
p-value lies below `tail_below` gets further toy jobs (TAIL_JOB_TOYS each, seeds apart from its first toys) until it
holds `tail_toys` toys, and its p-value comes from all of them.

Failure rule (spec 7.1): a variable fails with p < alpha / N in any test (N: every test of the summary) or with
p < alpha in at least `min_eras` eras of one channel and region. A Kolmogorov-Smirnov test compares all p-values with
the uniform distribution; tests of correlated variables (the same events) make it approximate.

Summaries: <output>/input_gof.json (every card, kept across runs into the same output, so that the failure rule counts
the tests of every region and run) and input_gof.md, the p-value distribution pvalues.pdf.

Building blocks of other validations of single variables: flatness (data / prediction flat within ranges of the
variable, e.g. the working-point bins of a b-tag score, NV2), shape_compatibility (two shapes alike within their
statistics and shifts, e.g. MC jet fakes in AR x FF and SR) and merge_low_statistics.
"""
from __future__ import annotations

import json
import logging
import math
import re
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from shapesmith import checks, cmssw, masking
from shapesmith.config import CombineConfig
from shapesmith.datacards import Bin, collect_bin, write_bins
from shapesmith.histogram import INCLUSIVE, NOMINAL_VARIATION, HistKey, Histogram, HistogramSet
from shapesmith.model import Analysis, Category

logger = logging.getLogger(__name__)
MIN_TOYS_ABOVE = 10  # toys at least as large as the observed statistic for p_toys to count
QUANTILE_LOW, QUANTILE_HIGH = 0.5, 0.9  # the quantiles of the toys the scaled chi-square matches
SUMMARY = "input_gof"
TAIL_BELOW = 1e-3  # the p-value below which a card gets the tail toys
TAIL_JOB_TOYS = 500  # toys per job of the tail toys
TAIL_SEED_OFFSET = 10_000_000  # the tail toy job i of a card has the seed card_seed + TAIL_SEED_OFFSET + i
MINIMIZER = "minimizer.json"  # the minimizer options of a card that ran with alt_minimizer, in its gof directory
ALT_TOLERANCE = 0.1  # the alternative statistic must be lower by more than this for a card to run again with it
RECORD = "record.json"  # of a card: target, edges, data, prediction, nuisances
SEED_RANGE = 1_000_000  # card_seed: GOF_SEED + crc32(card name) % SEED_RANGE


@dataclass(frozen=True)
class GofBinning:
    """The merging of the fine bins: at most `n_bins` bins of about equal total background, each with at least
    `min_prediction` events (integer variables: one bin per value with at least `min_prediction` events)."""

    n_bins: int = 20
    min_prediction: float = 20.0


@dataclass(frozen=True)
class CardTarget:
    era: str
    channel: str
    region: str
    variable: str

    @property
    def name(self) -> str:
        return f"{self.region}/{self.era}/{self.channel}/{self.variable}"


def is_integer_binning(edges) -> bool:
    """Fine bins of width 1 centred on integers (the binning of an integer variable, e.g. n_jets)."""
    edges = np.asarray(edges, dtype=float)
    return len(edges) > 1 and bool(np.allclose(np.diff(edges), 1.0) and np.isclose((edges[0] - 0.5) % 1.0, 0.0))


def equal_prediction_edges(edges, prediction, binning: GofBinning = GofBinning(), integer: bool = False) -> list[float]:
    """A subset of the fine `edges`: from the left, a bin closes as soon as it holds max(total / n_bins,
    min_prediction) of the `prediction` (min_prediction alone for an integer variable); the last bin joins its left
    neighbours while it holds less than min_prediction. One bin for a prediction without a positive sum."""
    edges = [float(edge) for edge in edges]
    values = np.asarray(prediction, dtype=float)
    total = float(values.sum())
    if total <= 0.0:
        return [edges[0], edges[-1]]
    target = binning.min_prediction if integer else max(total / binning.n_bins, binning.min_prediction)
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    indices, start = [0], 0
    for i in range(1, len(values)):
        if cumulative[i] - cumulative[start] >= target:
            indices.append(i)
            start = i
    indices.append(len(values))
    while len(indices) > 2 and cumulative[indices[-1]] - cumulative[indices[-2]] < binning.min_prediction:
        del indices[-2]
    return [edges[i] for i in indices]


def card_seed(name: str) -> int:
    """The seed of the toys of a card: checks.GOF_SEED plus a stable hash of its name."""
    return checks.GOF_SEED + zlib.crc32(name.encode()) % SEED_RANGE


def _background(hset: HistogramSet, analysis: Analysis, channel: str, region: str, variable: str) -> Histogram | None:
    total = None
    for process in analysis.channel(channel).backgrounds():
        h = hset.get(HistKey(channel, INCLUSIVE, process, region, NOMINAL_VARIATION, variable))
        if h is not None:
            total = h.copy() if total is None else total.add(h)
    return total


def excluded(name: str, exclude: tuple[str, ...]) -> bool:
    """A variation (with or without its direction) whose nuisance matches one of the regular expressions."""
    return any(re.search(pattern, name) for pattern in exclude)


def card_bin(hset: HistogramSet, analysis: Analysis, target: CardTarget, binning: GofBinning = GofBinning(), exclude: tuple[str, ...] = ()) -> tuple[Bin, list[float]] | None:
    """The card bin of a control variable in a region, its fine bins merged to equal prediction, the variations of
    `exclude` left out; None without data or a positive prediction."""
    channel = analysis.channel(target.channel)
    variable = channel.variables[target.variable]
    total = _background(hset, analysis, target.channel, target.region, target.variable)
    if total is None or total.sum() <= 0.0:
        logger.warning(f"{target.name}: no positive prediction, no card")
        return None
    edges = equal_prediction_edges(total.edges, total.values, binning, is_integer_binning(total.edges))
    b = collect_bin(hset, analysis, target.channel, 0, Category(INCLUSIVE, "True", variable), None, region=target.region, edges=edges)
    if b is None:
        return None
    if exclude:
        b.variations = {process: {name: h for name, h in variations.items() if not excluded(name, exclude)} for process, variations in b.variations.items()}
    return b, edges


def prediction_uncertainty(analysis: Analysis, b: Bin) -> np.ndarray:
    """The prefit uncertainty of the total background per bin: its statistics, every shape pair ((Up - Down) / 2,
    summed over the processes) and every lnN ((kappa - 1) x yield, the processes of one name summed), in quadrature."""
    backgrounds = {p: h for p, h in b.processes.items() if p != analysis.signal}
    squares = np.sum([h.variances for h in backgrounds.values()], axis=0) if backgrounds else np.zeros(len(b.data.values))
    shifts: dict[str, np.ndarray] = {}
    for process, nominal in backgrounds.items():
        variations = b.variations.get(process, {})
        for name, up in variations.items():
            if name.endswith("Up") and (down := variations.get(f"{name[:-2]}Down")) is not None:
                shifts[name[:-2]] = shifts.get(name[:-2], 0.0) + (up.values - down.values) / 2
        for lnn in analysis.lnn:
            if (lnn.channels is None or b.channel in lnn.channels) and ("*" in lnn.processes or process in lnn.processes):
                down, up = lnn.value if isinstance(lnn.value, tuple) else (1 / lnn.value, lnn.value)
                key = lnn.name.replace("$CHANNEL", b.channel).replace("$ERA", analysis.era)
                shifts[key] = shifts.get(key, 0.0) + (up - down) / 2 * nominal.values
    return np.sqrt(squares + sum(np.square(shift) for shift in shifts.values()))


def card_plot(analysis: Analysis, b: Bin, title: str, path: Path, xlabel: str) -> list[Path]:
    """The prefit stack of a card bin with the data and the prefit uncertainty of the prediction, against the bin index
    (bins of equal prediction can be very narrow, e.g. of a b-tag score near 1; the edges are in record.json)."""
    from shapesmith.plotting.fit_plots import fit_stack

    backgrounds = [h for p, h in b.processes.items() if p != analysis.signal]
    total = np.sum([h.values for h in backgrounds], axis=0)
    error = np.sqrt(np.clip(b.data.values, 0.0, None))
    shapes = {process: [h.values, np.sqrt(h.variances)] for process, h in b.processes.items()}
    shapes.update(total_background=[total, prediction_uncertainty(analysis, b)], data=[b.data.values, error, error])
    return fit_stack(shapes, analysis, title, path, False, None, "prefit", f"{xlabel}: bin of equal prediction")


def write_card(hset: HistogramSet, analysis: Analysis, target: CardTarget, output: Path, binning: GofBinning = GofBinning(), exclude: tuple[str, ...] = (), plot: bool = True) -> dict | None:
    """The card of a target in <output>/<region>/<era>/<channel>/<variable>/ (with plot.pdf/png), and its record:
    edges, data and prediction; None without a card. A variable whose prediction fits into one bin (constant or
    undefined in the region, e.g. the b pair under a b veto) gets no card: one bin tests the normalisation only."""
    collected = card_bin(hset, analysis, target, binning, exclude)
    if collected is None:
        return None
    b, edges = collected
    if len(edges) < 3:
        logger.info(f"{target.name}: one bin, no card")
        return None
    directory = Path(output) / target.name
    card = write_bins(analysis, [b], directory)
    # the pairs write_bins left out (a side whose sum is not positive), as <process>: <nuisance>
    dropped = [entry.split("/", 1)[1] for entry in json.loads((directory / "template_changes.json").read_text()).get("empty_pairs", [])]
    if plot:
        from shapesmith.plotting.style import axis_label

        card_plot(analysis, b, f"{target.channel} {target.era}, {target.region}", directory / "plot", axis_label(analysis, target.channel, target.variable))
    prediction = sum(h.sum() for p, h in b.processes.items() if p != analysis.signal)
    nuisances = sum(1 for line in card.read_text().splitlines() if " lnN " in line or " shape " in line)
    record = {**asdict(target), "card": str(card), "edges": edges, "bins": len(edges) - 1, "data": b.data.sum(), "prediction": prediction, "nuisances": nuisances,
              "dropped_pairs": dropped}
    (directory / RECORD).write_text(json.dumps(record, indent=1))
    return record


def stored_records(output: Path, targets: list[CardTarget]) -> list[dict]:
    """The records of the cards an earlier run wrote for the targets (those without a card left out)."""
    paths = [Path(output) / target.name / RECORD for target in targets]
    return [json.loads(path.read_text()) for path in paths if path.exists()]


def gof_job(card_dir: Path, combine: CombineConfig | None, toys: int, seed: int) -> list[str]:
    """The commands of the GoF of one card, run in <card>/gof/: workspace, observed statistic, every toy in one job.
    Earlier toys and minimizer options of the card are removed."""
    directory = Path(card_dir) / "gof"
    directory.mkdir(parents=True, exist_ok=True)
    for stale in [*directory.glob(f"higgsCombine.toys.GoodnessOfFit.mH{checks.MASS}.*.root"), directory / MINIMIZER]:
        stale.unlink(missing_ok=True)
    first, jobs = checks.gof_commands(combine, masking.data_masks(card_dir), toys, toys_per_job=toys, seed=seed)
    return [f"cd {directory.resolve()}", *first, *(command for job in jobs for command in job)]


def alt_job(card_dir: Path, combine: CombineConfig | None) -> list[str]:
    """The observed statistic of a card with the options of `combine` (higgsCombine.alt..., alt.log) beside the one of
    its GoF; the workspace of the GoF is used."""
    directory = (Path(card_dir) / "gof").resolve()
    first, _ = checks.gof_commands(combine, masking.data_masks(card_dir), 0)
    return [f"cd {directory}", first[1].replace(" -n .obs", " -n .alt").replace("> obs.log", "> alt.log")]


def card_combine(card_dir: Path, combine: CombineConfig | None) -> CombineConfig | None:
    """The combine options of a card: those of the run, or the ones it ran again with (MINIMIZER)."""
    path = Path(card_dir) / "gof" / MINIMIZER
    if combine is None or not path.exists():
        return combine
    return combine.model_copy(update={"minimizer": json.loads(path.read_text())["minimizer"]})


# -- p-values --------------------------------------------------------------------------------------------------------


def gamma_q(a: float, x: float) -> float:
    """The regularized upper incomplete gamma function Q(a, x): the series of P for x < a + 1, else the continued
    fraction of Q (Lentz)."""
    if x <= 0.0:
        return 1.0
    log_prefactor = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:
        term = total = 1.0 / a
        n = a
        for _ in range(100000):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return min(1.0, max(0.0, 1.0 - total * math.exp(log_prefactor)))
    tiny = 1e-300
    b = x + 1.0 - a
    c, d = 1.0 / tiny, 1.0 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return min(1.0, math.exp(log_prefactor) * h)


def chi2_sf(x: float, k: float) -> float:
    """P(chi2 with k degrees of freedom >= x)."""
    return gamma_q(k / 2.0, x / 2.0)


def chi2_isf(p: float, k: float) -> float:
    """x with P(chi2(k) >= x) = p (bisection on chi2_sf in log x)."""
    low, high = -30.0, math.log(1e4 * (k + 10.0))
    for _ in range(200):
        middle = (low + high) / 2
        if chi2_sf(math.exp(middle), k) > p:
            low = middle
        else:
            high = middle
    return math.exp((low + high) / 2)


def scaled_chi2(toys: np.ndarray) -> tuple[float, float] | None:
    """(k, s) of the scaled chi-square s * chi2(k) whose median and 90 % quantile are those of the toys: the ratio of the
    quantiles fixes k, the median s. Quantiles, not moments: a few toys of a failed fit far in the tail (statistics of
    1000 against a median of 17) ruin the variance. None for fewer than 10 finite toys or no spread."""
    finite = toys[np.isfinite(toys)]
    if len(finite) < 10:
        return None
    median, upper = (float(q) for q in np.quantile(finite, [QUANTILE_LOW, QUANTILE_HIGH]))
    if median <= 0.0 or upper <= median:
        return None
    ratio = upper / median

    def model(k: float) -> float:
        return chi2_isf(1.0 - QUANTILE_HIGH, k) / chi2_isf(1.0 - QUANTILE_LOW, k)

    low, high = math.log(0.05), math.log(5000.0)  # the ratio falls with k
    if not model(math.exp(high)) < ratio < model(math.exp(low)):
        return None
    for _ in range(80):
        middle = (low + high) / 2
        if model(math.exp(middle)) > ratio:
            low = middle
        else:
            high = middle
    k = math.exp((low + high) / 2)
    return k, median / chi2_isf(1.0 - QUANTILE_LOW, k)


def p_values(toys: np.ndarray, observed: float) -> dict:
    """p_toys (with its binomial error and the number of toys above), p_fit of the scaled chi-square and the p-value of
    the rule (module docstring) with its method."""
    finite = toys[np.isfinite(toys)]
    above = int(np.sum(finite >= observed))
    n = len(finite)
    p_toys = above / n if n else None
    fit = scaled_chi2(finite)
    p_fit = chi2_sf(observed / fit[1], fit[0]) if fit else None
    if p_toys is not None and (above >= MIN_TOYS_ABOVE or p_fit is None):
        method, p = "toys", p_toys
    else:
        method, p = "chi2 tail", p_fit
    return {"p": p, "method": method, "p_toys": p_toys, "p_toys_error": math.sqrt(max(p_toys * (1 - p_toys), 1.0 / n) / n) if n else None, "toys_above": above,
            "p_fit": p_fit, "chi2_k": fit[0] if fit else None, "chi2_scale": fit[1] if fit else None}


def card_result(card_dir: Path) -> dict:
    """The GoF of a card: checks.gof_result (statistic, toys, p-value of the toys), p_values, the observed statistic
    with the alternative options (alt_job; None without) and the minimizer options it ran again with (None: those of
    the run)."""
    directory = Path(card_dir) / "gof"
    result = checks.gof_result(directory)
    alternative = directory / f"higgsCombine.alt.GoodnessOfFit.mH{checks.MASS}.root"
    minimizer = json.loads((directory / MINIMIZER).read_text())["minimizer"] if (directory / MINIMIZER).exists() else None
    return {**result, **p_values(checks.gof_toys(directory), result["statistic"]),
            "alt_statistic": float(checks._limit_values(alternative)[0]) if alternative.exists() else None, "minimizer": minimizer}


def ks_uniform(values) -> tuple[float, float]:
    """The Kolmogorov-Smirnov distance of the values to the uniform distribution on [0, 1] and its asymptotic p-value
    (the Kolmogorov distribution with the correction (sqrt(n) + 0.12 + 0.11 / sqrt(n)) D)."""
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n == 0:
        return 0.0, 1.0
    steps = np.arange(1, n + 1) / n
    distance = float(max(np.max(steps - x), np.max(x - (steps - 1.0 / n))))
    root = math.sqrt(n)
    lam = (root + 0.12 + 0.11 / root) * distance
    if lam < 0.2:
        return distance, 1.0
    p = 2.0 * sum((-1) ** (j - 1) * math.exp(-2.0 * j * j * lam * lam) for j in range(1, 101))
    return distance, min(1.0, max(0.0, p))


# -- failure rule ----------------------------------------------------------------------------------------------------


def failure_rule(entries: list[dict], alpha: float = 0.05, min_eras: int = 2, n_tests: int | None = None) -> dict:
    """Spec 7.1 on the tests `entries` (era, channel, region, variable, p): N (`n_tests`, else the number of tests, e.g.
    the N of a larger family that a variant of some of its tests belongs to), the threshold alpha / N, per test
    `bonferroni` (p below it) and per (variable, channel, region) the eras with p < alpha; per variable `failed` with
    its reasons; the uniformity of all p-values."""
    tested = [e for e in entries if e.get("p") is not None]
    n = n_tests or len(tested)
    threshold = alpha / n if n else None
    tests, groups = [], {}
    for entry in tested:
        tests.append({**entry, "bonferroni": entry["p"] < threshold, "below_alpha": entry["p"] < alpha})
        groups.setdefault((entry["variable"], entry["channel"], entry["region"]), []).append(entry)
    variables: dict[str, dict] = {}
    for test in tests:
        record = variables.setdefault(test["variable"], {"failed": False, "reasons": [], "tests": 0, "min_p": 1.0})
        record["tests"] += 1
        record["min_p"] = min(record["min_p"], test["p"])
        if test["bonferroni"]:
            record["failed"] = True
            record["reasons"].append(f"p = {test['p']:.2g} < {threshold:.2g} in {test['channel']} {test['era']} {test['region']}")
    for (variable, channel, region), members in sorted(groups.items()):
        low = sorted(e["era"] for e in members if e["p"] < alpha)
        if len(low) >= min_eras:
            variables[variable]["failed"] = True
            variables[variable]["reasons"].append(f"p < {alpha:g} in {len(low)} eras of {channel} {region} ({', '.join(low)})")
    distance, p_uniform = ks_uniform([t["p"] for t in tests])
    return {"alpha": alpha, "min_eras": min_eras, "tests": len(tested), "n": n, "threshold": threshold, "ks_distance": distance, "ks_p_value": p_uniform,
            "below_alpha": sum(t["below_alpha"] for t in tests), "expected_below_alpha": alpha * len(tested), "variables": variables, "entries": tests}


def calibration(entries: list[dict], low: float = 0.02) -> dict:
    """p_fit against p_toys for the tests whose toys resolve the p-value (p_toys >= low): the mean and the largest pull
    (p_fit - p_toys) / error of p_toys, the largest absolute difference."""
    pairs = [(e["p_fit"], e["p_toys"], e["p_toys_error"]) for e in entries if e.get("p_fit") is not None and e.get("p_toys") is not None and e["p_toys"] >= low]
    if not pairs:
        return {"tests": 0}
    pulls = np.array([(fit - toys) / error for fit, toys, error in pairs if error > 0])
    differences = np.array([fit - toys for fit, toys, _ in pairs])
    return {"tests": len(pairs), "p_toys_min": low, "mean_pull": float(np.mean(pulls)) if len(pulls) else None, "rms_pull": float(np.sqrt(np.mean(pulls**2))) if len(pulls) else None,
            "max_abs_pull": float(np.max(np.abs(pulls))) if len(pulls) else None, "max_abs_difference": float(np.max(np.abs(differences)))}


# -- flatness within ranges and shape compatibility -------------------------------------------------------------------


def flatness(data: Histogram, prediction: Histogram, ranges: list[tuple[float, float]], binning: GofBinning = GofBinning(10, 20.0),
             shifts: dict[str, tuple[Histogram, Histogram]] | None = None) -> list[dict]:
    """Whether data / prediction is flat within each range [low, high] of the fine bins (both edges fine edges, e.g. the
    working-point bins of a b-tag score, NN spec NV2): the fine bins of the range merged to equal prediction
    (equal_prediction_edges), the prediction scaled to the data of the range (its normalisation is not tested), then
    chi2 = sum (d - p)^2 / (p + var_p) with n - 1 degrees of freedom, and the slope of d / p against the prediction
    quantile x (0 at the low, 1 at the high edge of the range) with its error (weighted least squares, the variances of
    the ratio (p + var_p) / p^2), z = slope / error. With `shifts` (name -> (Up, Down) of the prediction) also chi2_shifts
    and p_shifts: the covariance plus d d^T per shift, d the half difference of its Up and Down, each scaled to the data
    of the range (shape only). A range without data or prediction gives no p-value."""
    edges = np.asarray(data.edges)
    results = []
    for low, high in ranges:
        first, last = (int(np.argmin(np.abs(edges - bound))) for bound in (low, high))
        if not (np.isclose(edges[first], low) and np.isclose(edges[last], high)) or last <= first:
            raise ValueError(f"the range [{low}, {high}] is not bounded by fine edges")
        values, variances, observed = prediction.values[first:last], prediction.variances[first:last], data.values[first:last]
        record = {"low": float(low), "high": float(high), "data": float(observed.sum()), "prediction": float(values.sum())}
        if record["data"] <= 0.0 or record["prediction"] <= 0.0:
            results.append({**record, "bins": 0, "p": None})
            continue
        merged = equal_prediction_edges(edges[first:last + 1], values, binning)
        indices = [int(np.argmin(np.abs(edges - edge))) - first for edge in merged]
        d = np.add.reduceat(observed, indices[:-1])
        scale = record["data"] / record["prediction"]
        p = np.add.reduceat(values, indices[:-1]) * scale
        var = np.add.reduceat(variances, indices[:-1]) * scale**2
        n = len(d)
        chi2 = float(np.sum((d - p) ** 2 / np.where(p + var > 0, p + var, 1.0)))
        x = (np.cumsum(p) - p / 2) / p.sum()
        ratio, weight = d / p, p**2 / np.where(p + var > 0, p + var, 1.0)
        slope, error = None, None
        if n > 2:
            sw, sx, sxx = weight.sum(), np.sum(weight * x), np.sum(weight * x * x)
            determinant = sw * sxx - sx * sx
            slope = float((sw * np.sum(weight * x * ratio) - sx * np.sum(weight * ratio)) / determinant)
            error = float(math.sqrt(sw / determinant))
        result = {**record, "bins": n, "edges": [float(e) for e in merged], "scale": scale, "chi2": chi2, "ndf": n - 1,
                  "p": chi2_sf(chi2, n - 1) if n > 1 else None, "slope": slope, "slope_error": error,
                  "z": slope / error if slope is not None and error else None, "ratio": ratio.tolist(), "ratio_error": (1.0 / np.sqrt(weight)).tolist(), "x": x.tolist()}
        if shifts:
            covariance = np.diag(np.where(p + var > 0, p + var, 1.0))
            for up, down in shifts.values():
                varied = [np.add.reduceat(h.values[first:last], indices[:-1]) for h in (up, down)]
                varied = [v * record["data"] / v.sum() if v.sum() > 0 else p for v in varied]
                delta = (varied[0] - varied[1]) / 2
                covariance = covariance + np.outer(delta, delta)
            residual = d - p
            result["chi2_shifts"] = float(residual @ np.linalg.solve(covariance, residual))
            result["p_shifts"] = chi2_sf(result["chi2_shifts"], n - 1) if n > 1 else None
        results.append(result)
    return results


def _normalised(h: Histogram) -> tuple[np.ndarray, np.ndarray]:
    """The shape of a histogram (unit sum) and its statistical covariance (J diag(var) J^T, J = (1 - s 1^T) / N)."""
    total = h.values.sum()
    shape = h.values / total
    jacobian = (np.eye(len(shape)) - shape[:, None]) / total
    return shape, jacobian @ np.diag(h.variances) @ jacobian.T


def shape_compatibility(first: Histogram, second: Histogram, shifts: dict[str, tuple[Histogram, Histogram]] | None = None) -> dict:
    """Whether two histograms have the same shape within their statistics and the shifts of the second (name ->
    (Up, Down), e.g. the fake-factor uncertainties of AR x FF): chi2 = D^T C^-1 D of the difference D of the unit-sum
    shapes, C the statistical covariances of both shapes plus d d^T per shift (d the half difference of the Up and Down
    shapes), the last bin left out (the shapes sum to 1), n - 1 degrees of freedom."""
    shape_1, covariance_1 = _normalised(first)
    shape_2, covariance_2 = _normalised(second)
    covariance = covariance_1 + covariance_2
    for up, down in (shifts or {}).values():
        delta = (up.values / up.values.sum() - down.values / down.values.sum()) / 2
        covariance = covariance + np.outer(delta, delta)
    difference = (shape_1 - shape_2)[:-1]
    reduced = covariance[:-1, :-1]
    chi2 = float(difference @ np.linalg.solve(reduced, difference)) if len(difference) else 0.0
    ndf = len(difference)
    return {"chi2": chi2, "ndf": ndf, "p": chi2_sf(chi2, ndf) if ndf else None, "shifts": len(shifts or {}),
            "ratio": (shape_1 / np.where(shape_2 > 0, shape_2, np.nan)).tolist()}


def merge_low_statistics(edges, histograms: list[Histogram], min_effective: float) -> list[float]:
    """A subset of `edges` (also edges of the histograms) whose bins hold at least `min_effective` effective events
    (sum^2 / sum of squares) in each histogram, merged from the left; the last bin joins its neighbour."""
    edges = [float(e) for e in edges]
    rebinned = [h.rebin(edges) for h in histograms]

    def effective(start: int, stop: int) -> float:
        counts = []
        for h in rebinned:
            total, square = h.values[start:stop].sum(), h.variances[start:stop].sum()
            counts.append(total * total / square if square > 0 else 0.0)
        return min(counts)

    indices, start = [0], 0
    for i in range(1, len(edges) - 1):
        if effective(start, i) >= min_effective:
            indices.append(i)
            start = i
    indices.append(len(edges) - 1)
    while len(indices) > 2 and effective(indices[-2], indices[-1]) < min_effective:
        del indices[-2]
    return [edges[i] for i in indices]


# -- runs and summaries ----------------------------------------------------------------------------------------------


def run_cards(hset: HistogramSet, analysis: Analysis, targets: list[CardTarget], output: Path, binning: GofBinning = GofBinning(), exclude: tuple[str, ...] = (), plot: bool = True) -> list[dict]:
    """The cards of the targets of one analysis (one era) and their records."""
    records = []
    for target in targets:
        record = write_card(hset, analysis, target, output, binning, exclude, plot)
        if record is not None:
            records.append(record)
    logger.info(f"{analysis.era}: {len(records)} of {len(targets)} cards in {output}")
    return records


def _run_jobs(records: list[dict], combine: CombineConfig | None, toys: int, output: Path) -> None:
    jobs = [gof_job(Path(record["card"]).parent, combine, toys, card_seed(_key(record))) for record in records]
    logger.info(f"{len(jobs)} saturated GoFs with {toys} toys each, {(combine.parallel if combine else 1)} at a time")
    try:
        cmssw.run_parallel(jobs, combine, Path(output))
    except Exception as error:  # the results of the other cards still count; the failed ones are reported below
        logger.error(f"GoF jobs failed: {type(error).__name__}: {error}")


def _results(records: list[dict]) -> list[dict]:
    results = []
    for record in records:
        try:
            results.append({**record, **card_result(Path(record["card"]).parent)})
        except (OSError, KeyError, IndexError, ValueError) as error:
            logger.error(f"{_key(record)}: no GoF result ({type(error).__name__}: {error})")
    return results


def run_gof(records: list[dict], combine: CombineConfig | None, toys: int, output: Path, skip_combine: bool = False, tail_toys: int = 0,
            tail_below: float = TAIL_BELOW, alt_minimizer: str = "") -> list[dict]:
    """The GoF of every card, `combine.parallel` at a time, and their results; a card whose GoF failed is logged and
    left out. With `alt_minimizer` the observed statistic of every card also with these options added, and the cards
    whose alternative statistic is lower run again with them (module docstring); with `tail_toys` the cards below
    `tail_below` then get their tail toys (run_tail_toys). `skip_combine` skips the GoFs of the run, not these two."""
    if not skip_combine:
        _run_jobs(records, combine, toys, output)
    results = _results(records)
    if alt_minimizer and combine is not None:
        results = run_alternative(results, records, combine, toys, output, alt_minimizer)
    if tail_toys:
        results = run_tail_toys(results, combine, output, tail_toys, tail_below)
    return results


def run_alternative(results: list[dict], records: list[dict], combine: CombineConfig, toys: int, output: Path, options: str) -> list[dict]:
    """The observed statistic of every card with `options` added to the minimizer options; the cards where it is lower
    by more than ALT_TOLERANCE run their GoF again with them (MINIMIZER)."""
    alternative = combine.model_copy(update={"minimizer": f"{combine.minimizer} {options}".strip()})
    try:
        cmssw.run_parallel([alt_job(Path(r["card"]).parent, alternative) for r in results], combine, Path(output))
    except Exception as error:  # the cards without an alternative statistic keep theirs
        logger.error(f"alternative observed fits failed: {type(error).__name__}: {error}")
    results = _results(records) if records else results
    lower = {_key(r) for r in results if r["alt_statistic"] is not None and r["alt_statistic"] < r["statistic"] - ALT_TOLERANCE}
    if not lower:
        return results
    logger.info(f"{len(lower)} cards with a lower observed statistic with {alternative.minimizer} run again with it: {', '.join(sorted(lower))}")
    again = [record for record in records if _key(record) in lower]
    _run_jobs(again, alternative, toys, output)
    for record in again:
        (Path(record["card"]).parent / "gof" / MINIMIZER).write_text(json.dumps({"minimizer": alternative.minimizer}))
    refreshed = {_key(r): r for r in _results(again)}
    return [refreshed.get(_key(r), r) for r in results]


def _tail_seeds(directory: Path, seed: int) -> list[int]:
    """The seeds of the tail toy files of a GoF directory."""
    names = directory.glob(f"higgsCombine.toys.GoodnessOfFit.mH{checks.MASS}.*.root")
    return sorted(s for s in (int(path.name.split(".")[-2]) for path in names) if s >= seed + TAIL_SEED_OFFSET)


def tail_jobs(card_dir: Path, combine: CombineConfig | None, missing: int, seed: int) -> list[list[str]]:
    """The toy jobs of `missing` more toys of a card (TAIL_JOB_TOYS per job), each run in <card>/gof/; their seeds
    continue those of earlier tail toys of the card."""
    directory = (Path(card_dir) / "gof").resolve()
    first = seed + TAIL_SEED_OFFSET + len(_tail_seeds(directory, seed))
    _, jobs = checks.gof_commands(combine, masking.data_masks(card_dir), missing, toys_per_job=TAIL_JOB_TOYS, seed=first)
    return [[f"cd {directory}", *job] for job in jobs]


def run_tail_toys(results: list[dict], combine: CombineConfig | None, output: Path, total: int, below: float = TAIL_BELOW) -> list[dict]:
    """Toy jobs for every card whose p-value is below `below` until it holds `total` toys, then the results of these
    cards from all their toys."""
    selected = [r for r in results if r.get("p") is not None and r["p"] < below and r["toys"] + r["failed_toys"] < total]
    jobs = [job for r in selected for job in tail_jobs(Path(r["card"]).parent, card_combine(Path(r["card"]).parent, combine), total - r["toys"] - r["failed_toys"], card_seed(_key(r)))]
    if not jobs:
        return results
    logger.info(f"{len(selected)} cards below p = {below:g}: {len(jobs)} tail toy jobs ({TAIL_JOB_TOYS} toys each) up to {total} toys per card")
    try:
        cmssw.run_parallel(jobs, combine, Path(output))
    except Exception as error:  # the cards keep the toys they have
        logger.error(f"tail toy jobs failed: {type(error).__name__}: {error}")
    names = {_key(r) for r in selected}
    return [{**r, **card_result(Path(r["card"]).parent)} if _key(r) in names else r for r in results]


def _key(entry: dict) -> str:
    return CardTarget(entry["era"], entry["channel"], entry["region"], entry["variable"]).name


def write_summary(results: list[dict], output: Path, alpha: float = 0.05, min_eras: int = 2, title: str = "") -> dict:
    """<output>/input_gof.json and .md and pvalues.pdf: the results merged with those of earlier runs into the same
    output (by card), the failure rule and the calibration of the fitted tail over all of them."""
    output = Path(output)
    path = output / f"{SUMMARY}.json"
    previous = {_key(entry): entry for entry in json.loads(path.read_text())["results"]} if path.exists() else {}
    merged = {**previous, **{_key(entry): entry for entry in results}}
    entries = sorted(merged.values(), key=_key)
    rule = failure_rule(entries, alpha, min_eras)
    summary = {"results": entries, "rule": {key: value for key, value in rule.items() if key != "entries"}, "calibration": calibration(entries)}
    output.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1))
    (output / f"{SUMMARY}.md").write_text("\n".join(summary_lines(summary, rule, title)) + "\n")
    if rule["tests"]:
        p_value_plot([e["p"] for e in rule["entries"]], output / "pvalues.pdf", f"{rule['tests']} tests, KS p = {rule['ks_p_value']:.3g}")
    logger.info(f"{len(entries)} tests in {path}: {sum(v['failed'] for v in rule['variables'].values())} of {len(rule['variables'])} variables fail, KS p = {rule['ks_p_value']:.3g}")
    return summary


def _p_cell(entry: dict | None, rule: dict) -> str:
    if entry is None or entry.get("p") is None:
        return "-"
    mark = " **!!**" if entry["p"] < rule["threshold"] else (" *" if entry["p"] < rule["alpha"] else "")
    return f"{entry['p']:.2g}{'ᶠ' if entry['method'] != 'toys' else ''}{mark}"


def summary_lines(summary: dict, rule: dict, title: str = "") -> list[str]:
    """Markdown: the rule, the variables with their verdict, and per region a table variable x (channel, era)."""
    lines = ([f"# {title}", ""] if title else []) + [
        f"{rule['tests']} tests, alpha = {rule['alpha']:g}, alpha / N = {rule['threshold']:.3g}" + (f" (N = {rule['n']})" if rule.get("n", rule["tests"]) != rule["tests"] else "") if rule["tests"] else "no tests",
        f"p < alpha: {rule['below_alpha']} (expected under H0: {rule['expected_below_alpha']:.1f}); uniformity of all p-values: KS D = {rule['ks_distance']:.3f}, p = {rule['ks_p_value']:.3g}",
        "", "p-values: ᶠ from the fitted chi-square tail (fewer than "
        f"{MIN_TOYS_ABOVE} toys above the observed statistic), * p < alpha, **!!** p < alpha / N.", "",
    ]
    calibration_ = summary["calibration"]
    if calibration_.get("tests"):
        lines += [f"Fitted tail against the toys ({calibration_['tests']} tests with p_toys >= {calibration_['p_toys_min']:g}): mean pull {calibration_['mean_pull']:+.2f}, "
                  f"rms {calibration_['rms_pull']:.2f}, largest |pull| {calibration_['max_abs_pull']:.2f}, largest |p_fit - p_toys| {calibration_['max_abs_difference']:.3f}", ""]
    lines += ["| variable | verdict | tests | smallest p | reasons |", "|---|---|---|---|---|"]
    for variable, record in sorted(rule["variables"].items()):
        lines.append(f"| {variable} | {'FAIL' if record['failed'] else 'pass'} | {record['tests']} | {record['min_p']:.2g} | {'; '.join(record['reasons']) or '-'} |")
    by_region: dict[str, list[dict]] = {}
    for entry in rule["entries"]:
        by_region.setdefault(entry["region"], []).append(entry)
    for region, members in sorted(by_region.items()):
        columns = sorted({(e["channel"], e["era"]) for e in members})
        table = {(e["variable"], e["channel"], e["era"]): e for e in members}
        lines += ["", f"## {region}", "", "| variable | " + " | ".join(f"{c} {e}" for c, e in columns) + " |", "|---" * (len(columns) + 1) + "|"]
        for variable in sorted({e["variable"] for e in members}):
            lines.append(f"| {variable} | " + " | ".join(_p_cell(table.get((variable, c, e)), rule) for c, e in columns) + " |")
    return lines


def p_value_plot(p_values: list[float], path: Path, title: str) -> Path:
    """The histogram of the p-values (expected flat) and their quantiles against the uniform ones."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.sort(np.asarray(p_values, dtype=float))
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
    left.hist(values, bins=20, range=(0.0, 1.0), histtype="stepfilled", color="#3f90da", alpha=0.7)
    left.axhline(len(values) / 20, color="black", linestyle="--", label="uniform")
    left.set_xlabel("p-value")
    left.set_ylabel("tests")
    left.legend()
    expected = (np.arange(1, len(values) + 1) - 0.5) / len(values)
    right.plot(-np.log10(expected), -np.log10(np.clip(values, 1e-300, None)), "o", markersize=3, color="black")
    top = max(1.0, float(-np.log10(expected[0])) if len(values) else 1.0)
    right.plot([0, top], [0, top], color="gray", linestyle="--")
    right.set_xlabel(r"expected $-\log_{10} p$")
    right.set_ylabel(r"observed $-\log_{10} p$")
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def targets_of(analysis: Analysis, channels: list[str], region: str, variables: list[str] | None) -> list[CardTarget]:
    """Every (channel, variable) of the region: the control variables of each channel (or `variables` where the channel
    has them)."""
    result = []
    for channel in channels:
        names = [name for name in (variables or analysis.channel(channel).variables) if name in analysis.channel(channel).variables]
        result += [CardTarget(analysis.era, channel, region, name) for name in names]
    return result


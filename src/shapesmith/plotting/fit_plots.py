"""Plots of the fits and statistical checks: prefit and postfit stacks per card bin (from the background-only fit),
nuisance pulls, the GoF test statistic of the toys against the data, and the signal injection (r and pulls per
injected signal strength).

A masked (blinded) bin never shows data: its data_obs is the Asimov prediction, and the plot says "blinded".
FitDiagnostics saves the shapes of every card bin in the binning of the largest one (empty bins appended): the stack
of a card bin shows its own bins (the edges of its data histogram in the card); those of a score on [0, 1] (an NN
category) with equal widths, events per bin, with the edges as tick labels, in the CMS style of the category plots.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402

from shapesmith.model import Analysis  # noqa: E402
from shapesmith.plotting.style import card_bin_axis, card_bin_positions, color, grouped_backgrounds, label  # noqa: E402

logger = logging.getLogger(__name__)
TOTALS = ("total", "total_signal", "total_background", "total_covar", "data")
MAX_PULLS = 60  # nuisances in a pull plot (the largest pulls)


def process_groups(analysis: Analysis) -> dict[str, str]:
    """Background process (or estimator output) -> its plot group, over all channels."""
    groups: dict[str, str] = {}
    for channel in analysis.channels:
        for group, members in grouped_backgrounds(analysis, channel):
            for member in members:
                groups.setdefault(member, group)
    return groups


def _ordered_groups(analysis: Analysis, groups: dict[str, np.ndarray]) -> list[str]:
    order = list(analysis.style.group_order) if analysis.style else sorted(groups)
    return [g for g in order if g in groups] + sorted(g for g in groups if g not in order)


def shown_data(bin_shapes: dict, masked: bool) -> list[np.ndarray] | None:
    """[yields, lower errors, upper errors] of the data of a bin as the plot shows them: None in a masked bin."""
    data = bin_shapes.get("data")
    return None if masked or data is None else [np.asarray(values) for values in data]


def card_binning(bin_shapes: dict, edges, title: str = "") -> tuple[dict, np.ndarray | None]:
    """The shapes of a card bin in its own bins and their edges: FitDiagnostics pads the shapes of a bin with fewer bins
    than the largest one of the card with empty bins, which `edges` (from the card's data histogram) cut off; without
    edges, or if a cut bin is not empty, the shapes stay as they are (and the plot shows the bin index)."""
    if edges is None or "total_background" not in bin_shapes:
        return bin_shapes, None
    edges = np.asarray(edges, dtype=float)
    n, saved = len(edges) - 1, len(bin_shapes["total_background"][0])
    if n == saved:
        return bin_shapes, edges
    padding = [np.asarray(entry[0], dtype=float)[n:] for entry in bin_shapes.values()]  # the yields of every process and the data
    if n < 1 or n > saved or any(np.any(part != 0.0) for part in padding):
        logger.warning(f"{title}: {saved} saved bins against {n} bins of the card, plotted by bin index")
        return bin_shapes, None
    return {process: [list(np.asarray(part)[:n]) for part in entry] for process, entry in bin_shapes.items()}, edges


def fit_stack(bin_shapes: dict, analysis: Analysis, title: str, path: Path, masked: bool, edges: np.ndarray | None = None, stage: str = "postfit", xlabel: str | None = None) -> list[Path]:
    """Stacked backgrounds (grouped as the control plots), the total background uncertainty, the signal and the data
    (none in a masked bin) of one card bin; `edges` from the card's data histogram (card_binning), else the bin
    index. The bins of a score on [0, 1] (an NN category) are drawn with equal widths and their edges as tick labels
    (card_bin_axis); `xlabel` of the variable (default: NN output or bin)."""
    mapping = process_groups(analysis)
    background = bin_shapes.get("total_background")
    if background is None:
        logger.warning(f"{title}: no total_background, no {stage} plot")
        return []
    bin_shapes, edges = card_binning(bin_shapes, edges, title)
    background = bin_shapes["total_background"]
    total, total_error = np.asarray(background[0]), np.asarray(background[1])
    n = len(total)
    scored = edges is not None and edges[0] >= 0.0 and edges[-1] <= 1.0  # the card's own edges of a score on [0, 1]
    card_edges = edges
    edges = card_bin_positions(edges) if scored else np.asarray(edges, dtype=float) if edges is not None else np.arange(n + 1, dtype=float)
    groups: dict[str, np.ndarray] = {}
    for process, entry in bin_shapes.items():
        if process in TOTALS or process == analysis.signal:
            continue
        group = mapping.get(process, process)
        groups[group] = groups.get(group, np.zeros(n)) + np.asarray(entry[0])
    ordered = _ordered_groups(analysis, groups)
    widths = np.diff(edges)
    with plt.style.context(hep.style.CMS):  # the style of the category plots (stack.py), whatever was imported before
        fig, (ax, rax) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06}, sharex=True)
        if ordered:
            hep.histplot([groups[g] for g in ordered], bins=edges, stack=True, histtype="fill", color=[color(analysis, g) for g in ordered], label=[label(analysis, g) for g in ordered], ax=ax)
        ax.bar(edges[:-1], 2 * total_error, bottom=total - total_error, width=widths, align="edge", color="none", hatch="////", edgecolor="gray", linewidth=0, label="Bkg. unc.")
        signal = bin_shapes.get(analysis.signal)
        if signal is not None:
            hep.histplot(np.asarray(signal[0]), bins=edges, histtype="step", color=color(analysis, analysis.signal), linewidth=2, label=(analysis.style.signal_label if analysis.style else analysis.signal) + r" ($\mu$ = 1)", ax=ax)
        data = shown_data(bin_shapes, masked)
        centers = 0.5 * (edges[:-1] + edges[1:])
        if data is not None:
            values, low, high = data
            ax.errorbar(centers, values, yerr=[low, high], fmt="o", color="black", markersize=5, label="Data")
            with np.errstate(divide="ignore", invalid="ignore"):
                rax.errorbar(centers, np.where(total > 0, values / total, np.nan), yerr=[np.where(total > 0, low / total, np.nan), np.where(total > 0, high / total, np.nan)], fmt="o", color="black", markersize=4)
        elif masked:
            rax.text(0.5, 0.5, "blinded", transform=rax.transAxes, ha="center", va="center", fontsize=20, color="gray")
        with np.errstate(divide="ignore", invalid="ignore"):
            band = np.where(total > 0, total_error / total, 0.0)
        rax.bar(edges[:-1], 2 * band, bottom=1 - band, width=widths, align="edge", color="none", hatch="////", edgecolor="gray", linewidth=0)
        rax.axhline(1.0, color="gray", linewidth=1)
        rax.set_ylim(0.5, 1.5)
        rax.set_ylabel("Data / Bkg.")
        if scored:
            card_bin_axis(rax, card_edges, xlabel or "NN output")
        else:
            rax.set_xlabel(xlabel or "bin")
            rax.set_xlim(edges[0], edges[-1])
        ax.set_ylabel("Events / bin" if scored else "Events")
        top = max(float(total.max()) if n else 1.0, float(np.max(data[0])) if data is not None and n else 0.0)
        ax.set_ylim(0, 2.0 * top if top > 0 else 1.0)
        ax.legend(loc="upper right", ncol=2, fontsize=14, frameon=False)
        hep.cms.label(llabel="Private Work", rlabel=analysis.style.lumi_label if analysis.style else "", ax=ax, fontsize=22)
        ax.text(0.04, 0.95, f"{title}\n{stage}" + (", blinded" if masked else ""), transform=ax.transAxes, va="top", ha="left", fontsize=16)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        paths = [path.with_suffix(f".{ext}") for ext in ("pdf", "png")]
        for target in paths:
            fig.savefig(target, bbox_inches="tight")
        plt.close(fig)
    return paths


def pull_plot(nuisances: dict, path: Path, title: str, limit: int = MAX_PULLS) -> Path | None:
    """The postfit pulls (value - mean) / width with the constraints as error bars of the Gaussian-constrained
    nuisances, the `limit` largest |pulls|, the largest on top."""
    entries = sorted(((name, entry["pull"], entry["constraint"]) for name, entry in nuisances.items() if entry.get("pull") is not None), key=lambda item: -abs(item[1]))[:limit]
    if not entries:
        logger.warning(f"{title}: no constrained nuisances, no pull plot")
        return None
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(entries) + 1.8))
    positions = np.arange(len(entries))[::-1]
    errors = [constraint if constraint is not None else 0.0 for _, _, constraint in entries]
    ax.fill_betweenx([-1, len(entries)], -2, 2, color="#f5d000", alpha=0.5, linewidth=0)
    ax.fill_betweenx([-1, len(entries)], -1, 1, color="#00a651", alpha=0.5, linewidth=0)
    ax.errorbar([pull for _, pull, _ in entries], positions, xerr=errors, fmt="o", color="black", markersize=4)
    undefined = [i for i, (_, _, constraint) in enumerate(entries) if constraint is None]
    if undefined:
        ax.plot([entries[i][1] for i in undefined], positions[undefined], "x", color="red", label="undefined error")
        ax.legend(loc="lower right", fontsize=10)
    ax.set_yticks(positions)
    ax.set_yticklabels([name for name, _, _ in entries], fontsize=9)
    ax.set_ylim(-1, len(entries))
    ax.set_xlim(-3, 3)
    ax.axvline(0.0, color="gray", linewidth=1)
    ax.set_xlabel(r"pull $(\hat\theta - \theta_0) / \Delta\theta$, error: $\sigma_\mathrm{post} / \sigma_\mathrm{pre}$", fontsize=12)
    ax.set_title(f"{title}: {len(entries)} of {sum(1 for e in nuisances.values() if e.get('pull') is not None)} constrained nuisances", fontsize=12)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def gof_plot(toys: np.ndarray, observed: float, p_value: float, path: Path, title: str) -> Path:
    """The distribution of the saturated test statistic of the toys and the observed value."""
    fig, ax = plt.subplots(figsize=(7, 5))
    finite = toys[np.isfinite(toys)]
    ax.hist(finite, bins=max(10, min(50, int(math.sqrt(max(len(finite), 1))) * 2)), histtype="stepfilled", color="#3f90da", alpha=0.6, label=f"toys ({len(finite)})")
    ax.axvline(observed, color="black", linewidth=2, label=f"data: {observed:.3g}, p = {p_value:.3f}")
    ax.set_xlabel("saturated test statistic")
    ax.set_ylabel("toys")
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def injection_plot(summary: dict, toys: dict[str, dict[str, np.ndarray]], path: Path, title: str) -> Path:
    """Left: the mean fitted r against the injected signal strength, with the diagonal; right: the pull distribution
    of every injected value with the unit Gaussian."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
    injected = [entry["mu"] for entry in summary.values()]
    means = [entry["r_mean"] for entry in summary.values()]
    errors = [entry["r_mean_error"] for entry in summary.values()]
    left.errorbar(injected, means, yerr=errors, fmt="o", color="black")
    span = [min(injected + means), max(injected + means)]
    left.plot(span, span, "--", color="gray", label="r = μ")
    left.set_xlabel(r"injected $\mu$")
    left.set_ylabel(r"mean fitted $r$")
    left.legend(fontsize=10)
    grid = np.linspace(-4, 4, 200)
    for key, entry in summary.items():
        pulls = toys[key]["pull"]
        if len(pulls):
            right.hist(pulls, bins=np.linspace(-4, 4, 33), histtype="step", density=True, linewidth=1.5, label=f"μ = {entry['mu']:.3g}: {entry['pull_mean']:+.2f} ± {entry['pull_mean_error']:.2f}, σ {entry['pull_width']:.2f}")
    right.plot(grid, np.exp(-0.5 * grid**2) / math.sqrt(2 * math.pi), color="gray", label="N(0, 1)")
    right.set_xlabel(r"pull $(r - \mu) / \sigma_r$")
    right.legend(fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path

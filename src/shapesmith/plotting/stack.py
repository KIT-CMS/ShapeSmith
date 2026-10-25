"""Prefit stack plots with mplhep: background stack, data (or Asimov), scaled signal, ratio panel (Spec §11).

A ScoreBinning category (e.g. an NN category) is shown in the bins of its card (score_bins.resolve_category, from the
nominal templates), its processes clipped as in the card; in the nominal region its blinded bins (the hi part, N14)
show the prediction instead of the data, in the plots and in the yields of the gallery. Its plot draws these bins with
equal widths, events per bin, with the edges as tick labels (with normalize_by_bin_width: dN/dx on the score axis).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402

from shapesmith.histogram import Histogram  # noqa: E402
from shapesmith.histogram import INCLUSIVE, NOMINAL_VARIATION, HistKey, HistogramSet  # noqa: E402
from shapesmith.model import NOMINAL, Analysis  # noqa: E402
from shapesmith.plotting.style import axis_label, card_bin_axis, card_bin_positions, color, grouped_backgrounds, label  # noqa: E402
from shapesmith.score_bins import clip, resolve_category, score_binning  # noqa: E402

logger = logging.getLogger(__name__)
hep.style.use("CMS")


def _nominal(hset: HistogramSet, channel: str, category: str, process: str, variable: str, region: str = NOMINAL) -> Histogram | None:
    key = HistKey(channel, category, process, region, NOMINAL_VARIATION, variable)
    return hset.get(key)


def _group_histogram(hset: HistogramSet, channel: str, category: str, members: list[str], variable: str, region: str = NOMINAL) -> Histogram | None:
    total = None
    for process in members:
        h = _nominal(hset, channel, category, process, variable, region)
        if h is not None:
            total = h.copy() if total is None else total.add(h)
    return total


@dataclass
class Stack:
    """What a stack plot shows: the background groups in style order, each the nominal histograms of its processes
    summed (groups without one left out), the data histogram (None if missing) and the signal (None if missing);
    `blinded_from`: the first bin whose data is the prediction (a blinded ScoreBinning category), None if none."""

    groups: list[tuple[str, Histogram]]
    data: Histogram | None
    signal: Histogram | None = None
    blinded_from: int | None = None

    @property
    def background(self) -> np.ndarray:
        return np.sum([h.values for _, h in self.groups], axis=0)

    @property
    def background_error(self) -> np.ndarray:
        return np.sqrt(np.sum([h.variances for _, h in self.groups], axis=0))


def stack(hset: HistogramSet, analysis: Analysis, channel: str, category: str, variable: str, region: str = NOMINAL) -> Stack:
    """The stack of a plot; also the yields of `shapesmith publish`."""
    if score_binning(analysis.channel(channel), category, variable) is not None:
        return _score_stack(hset, analysis, channel, category, variable, region)
    groups = []
    for group, members in grouped_backgrounds(analysis, channel):
        h = _group_histogram(hset, channel, category, members, variable, region)
        if h is not None:
            groups.append((group, h))
    return Stack(groups, _nominal(hset, channel, category, analysis.channel(channel).data(), variable, region), _nominal(hset, channel, category, analysis.signal, variable, region))


def _score_stack(hset: HistogramSet, analysis: Analysis, channel: str, category: str, variable: str, region: str) -> Stack:
    """The stack of a ScoreBinning category in the bins of its card: in the nominal region the card processes as in the
    card and the prediction in place of the data of the blinded bins; in another region its histograms rebinned (the
    clipped processes clipped)."""
    definition = next(c for c in analysis.channel(channel).categories if c.name == category)
    view = resolve_category(hset, analysis, channel, definition)
    if view is None:
        return Stack([], None)

    def shown(process: str) -> Histogram | None:
        if region == NOMINAL:
            return view.templates.get(process)
        h = _nominal(hset, channel, category, process, variable, region)
        return None if h is None else (clip(h.rebin(view.edges), process, NOMINAL_VARIATION)[0] if process in view.clip else h.rebin(view.edges))

    groups = []
    for group, members in grouped_backgrounds(analysis, channel):
        parts = [h for process in members if (h := shown(process)) is not None]
        if parts:
            total = parts[0].copy()
            for h in parts[1:]:
                total.add(h)
            groups.append((group, total))
    data = _nominal(hset, channel, category, analysis.channel(channel).data(), variable, region)
    data = None if data is None else data.rebin(view.edges)
    result = Stack(groups, data, shown(analysis.signal))
    if region == NOMINAL and view.split is not None and data is not None and groups:
        data.values[view.split:] = result.background[view.split:]
        data.variances[view.split:] = np.clip(result.background[view.split:], 0.0, None)
        result.blinded_from = view.split
    return result


def default_signal_scale(hset: HistogramSet, analysis: Analysis, channel: str, category: str, variable: str, region: str = NOMINAL) -> float:
    """1, 2 or 5 x 10^n such that the scaled signal maximum is about 30 % of the background maximum."""
    shown = stack(hset, analysis, channel, category, variable, region)
    signal = shown.signal
    background = max((float(h.values.max()) for _, h in shown.groups), default=0.0)
    if signal is None or signal.values.max() <= 0 or background <= 0:
        return 1.0
    raw = 0.3 * background / float(signal.values.max())
    exponent = np.floor(np.log10(raw))
    mantissa = raw / 10**exponent
    return float(min((1, 2, 5, 10), key=lambda m: abs(m - mantissa)) * 10**exponent)


def plot_stack(hset: HistogramSet, analysis: Analysis, channel: str, category: str, variable: str, output_dir: Path, blind: bool = False, log: bool = False, signal_scale: float | None = None, normalize_by_bin_width: bool = False, region: str = NOMINAL) -> list[Path]:
    if region != NOMINAL and not hset.select(channel=channel, category=category, region=region, variable=variable):
        raise ValueError(f"no histograms for {channel}/{category}/{region}/{variable}; fill them with `shapesmith hist --regions {region}`")
    shown = stack(hset, analysis, channel, category, variable, region)
    groups = shown.groups
    if not groups:
        if region != NOMINAL:
            raise ValueError(f"no background histograms for {channel}/{category}/{region}/{variable}; fill them with `shapesmith hist --regions {region}`")
        logger.warning(f"{channel}/{category}/{variable}: no background histograms, no plot")
        return []
    edges = groups[0][1].edges
    card_bins = not normalize_by_bin_width and score_binning(analysis.channel(channel), category, variable) is not None
    x = card_bin_positions(edges) if card_bins else edges  # the drawn edges
    widths = np.diff(x)
    norm = np.diff(edges) if normalize_by_bin_width else np.ones(len(edges) - 1)
    background = shown.background
    background_error = shown.background_error
    data_hist = shown.data
    if data_hist is None and not blind:
        raise ValueError(f"data histogram is missing for {channel}/{category}/{region}/{variable}; fill data with `shapesmith hist --regions {region}` or use --blind")
    data = background.copy() if blind else data_hist.values
    data_error = np.sqrt(np.abs(data))
    signal_hist = shown.signal
    scale = signal_scale if signal_scale is not None else default_signal_scale(hset, analysis, channel, category, variable, region)
    style = analysis.style

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06}, sharex=True)
    hep.histplot([h.values / norm for _, h in groups], bins=x, stack=True, histtype="fill", color=[color(analysis, g) for g, _ in groups], label=[label(analysis, g) for g, _ in groups], ax=ax)
    ax.bar(x[:-1], 2 * background_error / norm, bottom=(background - background_error) / norm, width=widths, align="edge", color="none", hatch="////", edgecolor="gray", linewidth=0, label="Bkg. stat. unc.")
    if signal_hist is not None:
        signal_label = style.signal_label if style else analysis.signal
        hep.histplot(signal_hist.values * scale / norm, bins=x, histtype="step", color=color(analysis, analysis.signal), linewidth=2, label=f"{scale:g} x {signal_label}" if scale != 1 else signal_label, ax=ax)
    hep.histplot(data / norm, bins=x, yerr=data_error / norm, histtype="errorbar", color="black", label="Asimov" if blind else "Data", ax=ax)
    if shown.blinded_from is not None and not blind:
        for axis in (ax, rax):
            axis.axvspan(x[shown.blinded_from], x[-1], color="gray", alpha=0.15, linewidth=0, label="Blinded: prediction as data" if axis is ax else None)
    ax.set_ylabel("dN/dx" if normalize_by_bin_width else "Events / bin" if card_bins else "Events")
    if log:
        ax.set_yscale("log")
        positive = background[background > 0] / norm[background > 0]
        ax.set_ylim(bottom=max(1e-2, 0.5 * positive.min()) if positive.size else 1e-2)
    else:
        ax.set_ylim(0, 1.8 * max(np.max(data / norm), np.max(background / norm)))
    ax.legend(loc="upper right", ncol=2, fontsize=16, frameon=False)
    hep.cms.label(llabel="Private Work", rlabel=style.lumi_label if style else "", ax=ax, fontsize=22)
    channel_label = style.channel_labels.get(channel, channel) if style else channel
    selection_label = f"{channel_label}, {category}" if region == NOMINAL else f"{channel_label}, {category}, {region}"
    ax.text(0.04, 0.95, selection_label, transform=ax.transAxes, va="top", ha="left", fontsize=20)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(background > 0, data / background, np.nan)
        ratio_error = np.where(background > 0, data_error / background, np.nan)
        band = np.where(background > 0, background_error / background, 0.0)
        signal_ratio = np.where(background > 0, (background + (signal_hist.values if signal_hist is not None else 0.0)) / background, np.nan)
    centers = 0.5 * (x[:-1] + x[1:])
    rax.bar(x[:-1], 2 * band, bottom=1 - band, width=widths, align="edge", color="none", hatch="////", edgecolor="gray", linewidth=0)
    rax.errorbar(centers, ratio, yerr=ratio_error, fmt="o", color="black", markersize=4)
    if signal_hist is not None:
        rax.step(x, np.append(signal_ratio, signal_ratio[-1]), where="post", color=color(analysis, analysis.signal), linewidth=1.5)
    rax.axhline(1.0, color="gray", linewidth=1)
    rax.set_ylim(0.5, 1.5)
    rax.set_ylabel("Data / Bkg.")
    if card_bins:
        card_bin_axis(rax, edges, axis_label(analysis, channel, variable))
    else:
        rax.set_xlabel(axis_label(analysis, channel, variable))
        rax.set_xlim(edges[0], edges[-1])

    output_dir = Path(output_dir) / channel
    if region != NOMINAL:
        output_dir = output_dir / region
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{category}_{variable}.{ext}" for ext in ("pdf", "png")]
    logger.debug(f"{channel}/{category}/{variable}#{region}: {len(groups)} background groups, signal x {scale:g} -> {output_dir}/{category}_{variable}.pdf/png")
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def run_plot(hset: HistogramSet, analysis: Analysis, channels: list[str], control: bool, category: str | None, variables: list[str] | None, output_dir: Path, region: str = NOMINAL, **options) -> list[Path]:
    written = []
    for channel in channels:
        analysis.channel(channel).region(region)
        before = len(written)
        if control:
            for variable in variables or list(analysis.channel(channel).variables):
                written += plot_stack(hset, analysis, channel, INCLUSIVE, variable, output_dir, region=region, **options)
        else:
            for cat in analysis.channel(channel).categories:
                if category is None or cat.name == category:
                    written += plot_stack(hset, analysis, channel, cat.name, cat.variable.name, output_dir, region=region, **options)
        logger.info(f"{channel}: {(len(written) - before) // 2} plots of region {region} in {Path(output_dir) / channel}")
    return written

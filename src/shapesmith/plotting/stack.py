"""Prefit stack plots with mplhep: background stack, data (or Asimov), scaled signal, ratio panel (Spec §11)."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402

from shapesmith.histogram import Histogram  # noqa: E402
from shapesmith.histograms import INCLUSIVE, NOMINAL_REGION, NOMINAL_VARIATION, HistKey, HistogramSet  # noqa: E402
from shapesmith.model import Analysis  # noqa: E402
from shapesmith.plotting.style import axis_label, color, grouped_backgrounds, label  # noqa: E402

logger = logging.getLogger(__name__)
hep.style.use("CMS")


def _nominal(hset: HistogramSet, channel: str, category: str, process: str, variable: str, region: str = NOMINAL_REGION) -> Histogram | None:
    key = HistKey(channel, category, process, region, NOMINAL_VARIATION, variable)
    return hset.get(key) if hset.has(key) else None


def _group_histogram(hset: HistogramSet, channel: str, category: str, members: list[str], variable: str, region: str = NOMINAL_REGION) -> Histogram | None:
    total = None
    for process in members:
        h = _nominal(hset, channel, category, process, variable, region)
        if h is not None:
            total = h.copy() if total is None else total.add(h)
    return total


def default_signal_scale(hset: HistogramSet, analysis: Analysis, channel: str, category: str, variable: str, region: str = NOMINAL_REGION) -> float:
    """1, 2 or 5 x 10^n such that the scaled signal maximum is about 30 % of the background maximum."""
    signal = _nominal(hset, channel, category, analysis.signal, variable, region)
    background = 0.0
    for _, members in grouped_backgrounds(analysis):
        h = _group_histogram(hset, channel, category, members, variable, region)
        if h is not None:
            background = max(background, float(h.values.max()))
    if signal is None or signal.values.max() <= 0 or background <= 0:
        return 1.0
    raw = 0.3 * background / float(signal.values.max())
    exponent = np.floor(np.log10(raw))
    mantissa = raw / 10**exponent
    return float(min((1, 2, 5, 10), key=lambda m: abs(m - mantissa)) * 10**exponent)


def plot_stack(hset: HistogramSet, analysis: Analysis, channel: str, category: str, variable: str, output_dir: Path, blind: bool = False, log: bool = False, signal_scale: float | None = None, normalize_by_bin_width: bool = False, region: str = NOMINAL_REGION) -> list[Path]:
    if region != NOMINAL_REGION and not hset.keys(channel=channel, category=category, region=region, variable=variable):
        raise ValueError(f"no histograms for {channel}/{category}/{region}/{variable}; fill them with `shapesmith hist --regions {region}`")
    groups = []
    for group, members in grouped_backgrounds(analysis):
        h = _group_histogram(hset, channel, category, members, variable, region)
        if h is not None:
            groups.append((group, h))
    if not groups:
        if region != NOMINAL_REGION:
            raise ValueError(f"no background histograms for {channel}/{category}/{region}/{variable}; fill them with `shapesmith hist --regions {region}`")
        logger.warning(f"{channel}/{category}/{variable}: no background histograms, no plot")
        return []
    edges = groups[0][1].edges
    widths = np.diff(edges)
    norm = widths if normalize_by_bin_width else np.ones_like(widths)
    background = np.sum([h.values for _, h in groups], axis=0)
    background_error = np.sqrt(np.sum([h.variances for _, h in groups], axis=0))
    data_hist = _nominal(hset, channel, category, "data", variable, region)
    if data_hist is None and not blind:
        raise ValueError(f"data histogram is missing for {channel}/{category}/{region}/{variable}; fill data with `shapesmith hist --regions {region}` or use --blind")
    data = background.copy() if blind else data_hist.values
    data_error = np.sqrt(np.abs(data))
    signal_hist = _nominal(hset, channel, category, analysis.signal, variable, region)
    scale = signal_scale if signal_scale is not None else default_signal_scale(hset, analysis, channel, category, variable, region)
    style = analysis.style

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06}, sharex=True)
    hep.histplot([h.values / norm for _, h in groups], bins=edges, stack=True, histtype="fill", color=[color(analysis, g) for g, _ in groups], label=[label(analysis, g) for g, _ in groups], ax=ax)
    ax.bar(edges[:-1], 2 * background_error / norm, bottom=(background - background_error) / norm, width=widths, align="edge", color="none", hatch="////", edgecolor="gray", linewidth=0, label="Bkg. stat. unc.")
    if signal_hist is not None:
        signal_label = style.signal_label if style else analysis.signal
        hep.histplot(signal_hist.values * scale / norm, bins=edges, histtype="step", color=color(analysis, analysis.signal), linewidth=2, label=f"{scale:g} x {signal_label}" if scale != 1 else signal_label, ax=ax)
    hep.histplot(data / norm, bins=edges, yerr=data_error / norm, histtype="errorbar", color="black", label="Asimov" if blind else "Data", ax=ax)
    ax.set_ylabel("dN/dx" if normalize_by_bin_width else "Events")
    if log:
        ax.set_yscale("log")
        positive = background[background > 0] / norm[background > 0]
        ax.set_ylim(bottom=max(1e-2, 0.5 * positive.min()) if positive.size else 1e-2)
    else:
        ax.set_ylim(0, 1.8 * max(np.max(data / norm), np.max(background / norm)))
    ax.legend(loc="upper right", ncol=2, fontsize=16, frameon=False)
    hep.cms.label(llabel="Private Work", rlabel=style.lumi_label if style else "", ax=ax, fontsize=22)
    channel_label = style.channel_labels.get(channel, channel) if style else channel
    selection_label = f"{channel_label}, {category}" if region == NOMINAL_REGION else f"{channel_label}, {category}, {region}"
    ax.text(0.04, 0.95, selection_label, transform=ax.transAxes, va="top", ha="left", fontsize=20)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(background > 0, data / background, np.nan)
        ratio_error = np.where(background > 0, data_error / background, np.nan)
        band = np.where(background > 0, background_error / background, 0.0)
        signal_ratio = np.where(background > 0, (background + (signal_hist.values if signal_hist is not None else 0.0)) / background, np.nan)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rax.bar(edges[:-1], 2 * band, bottom=1 - band, width=widths, align="edge", color="none", hatch="////", edgecolor="gray", linewidth=0)
    rax.errorbar(centers, ratio, yerr=ratio_error, fmt="o", color="black", markersize=4)
    if signal_hist is not None:
        rax.step(edges, np.append(signal_ratio, signal_ratio[-1]), where="post", color=color(analysis, analysis.signal), linewidth=1.5)
    rax.axhline(1.0, color="gray", linewidth=1)
    rax.set_ylim(0.5, 1.5)
    rax.set_ylabel("Data / Bkg.")
    rax.set_xlabel(axis_label(analysis, channel, variable))
    rax.set_xlim(edges[0], edges[-1])

    output_dir = Path(output_dir) / channel
    if region != NOMINAL_REGION:
        output_dir = output_dir / region
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{category}_{variable}.{ext}" for ext in ("pdf", "png")]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def run_plot(hset: HistogramSet, analysis: Analysis, channels: list[str], control: bool, category: str | None, variables: list[str] | None, output_dir: Path, region: str = NOMINAL_REGION, **options) -> list[Path]:
    written = []
    for channel in channels:
        analysis.channel(channel).region(region)
        if control:
            for variable in variables or list(analysis.control_variables):
                written += plot_stack(hset, analysis, channel, INCLUSIVE, variable, output_dir, region=region, **options)
        else:
            for cat in analysis.categories:
                if category is None or cat.name == category:
                    written += plot_stack(hset, analysis, channel, cat.name, cat.variable.name, output_dir, region=region, **options)
    return written

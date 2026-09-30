"""combine-style shape files: one writer, and the synced layout <channel>_<category>/{data_obs, <process>, <process>_<variation>}."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable

import uproot

from shapesmith.histogram import NOMINAL_VARIATION, HistKey, Histogram, HistogramSet, is_part
from shapesmith.model import NOMINAL, Analysis

logger = logging.getLogger(__name__)


def write_shapes(path: Path, entries: Iterable[tuple[str, str, Histogram]]) -> Path:
    """Write (folder, name, histogram) entries as TH1D `folder/name` into a new ROOT file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with uproot.recreate(path) as f:
        for folder, name, h in entries:
            f[f"{folder}/{name}"] = h.to_root(name)
            count += 1
    logger.debug(f"{count} histograms written to {path}")
    return path


def shape_name(process: str, variation: str) -> str:
    return process if variation == NOMINAL_VARIATION else f"{process}_{variation}"


def _synced_name(category: str, process: str, variation: str) -> str:
    return shape_name(process, variation)


def synced_entries(hset: HistogramSet, analysis: Analysis, channel_name: str, name: Callable[[str, str, str], str] = _synced_name) -> list[tuple[str, str, Histogram]]:
    """data_obs and every nominal-region histogram of the signal and the backgrounds but the parts of summed
    variations, per category, each named `name(category, process, variation)`."""
    channel = analysis.channel(channel_name)
    entries = []
    for category in channel.categories:
        variable = category.variable.name
        folder = f"{channel_name}_{category.name}"
        data_key = HistKey(channel_name, category.name, channel.data(), NOMINAL, NOMINAL_VARIATION, variable)
        if data_key in hset:
            entries.append((folder, "data_obs", hset[data_key]))
        for process in (*channel.backgrounds(), analysis.signal):
            for key in hset.select(channel=channel_name, category=category.name, process=process, region=NOMINAL, variable=variable):
                if is_part(key.variation):
                    continue
                entries.append((folder, name(category.name, process, key.variation), hset[key]))
    return entries


def run_sync(hset: HistogramSet, analysis: Analysis, channels: list[str], output_dir: Path) -> list[Path]:
    return [write_shapes(Path(output_dir) / "synced" / f"htt_{channel}.inputs-Run{analysis.era}.root", synced_entries(hset, analysis, channel)) for channel in channels]

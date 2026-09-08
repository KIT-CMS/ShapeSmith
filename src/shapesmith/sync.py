"""combine-style shape files per channel: <channel>_<category>/{data_obs, <process>, <process>_<variation>} (Spec §10.1)."""
from __future__ import annotations

from pathlib import Path

import uproot

from shapesmith.histograms import NOMINAL_REGION, NOMINAL_VARIATION, HistKey, HistogramSet
from shapesmith.model import Analysis


def synced_processes(analysis: Analysis) -> tuple[str, ...]:
    return (*analysis.backgrounds(), analysis.signal)


def run_sync(hset: HistogramSet, analysis: Analysis, channels: list[str], output_dir: Path) -> list[Path]:
    directory = Path(output_dir) / "synced"
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for channel in channels:
        path = directory / f"htt_{channel}.inputs-Run{analysis.era}.root"
        with uproot.recreate(path) as f:
            for category in analysis.categories:
                variable = category.variable.name
                folder = f"{channel}_{category.name}"
                data_key = HistKey(channel, category.name, "data", NOMINAL_REGION, NOMINAL_VARIATION, variable)
                if hset.has(data_key):
                    f[f"{folder}/data_obs"] = hset.get(data_key).to_root("data_obs")
                for process in synced_processes(analysis):
                    for key in hset.keys(channel=channel, category=category.name, process=process, region=NOMINAL_REGION, variable=variable):
                        name = process if key.variation == NOMINAL_VARIATION else f"{process}_{key.variation}"
                        f[f"{folder}/{name}"] = hset.get(key).to_root(name)
        written.append(path)
    return written

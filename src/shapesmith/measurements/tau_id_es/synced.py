"""The shapes in the input format of MorphingTauID2017 (that of the predecessor's convert_to_synced_shapes.py).

One file per channel, `<channel>/htt_<channel>.inputs-sm-Run<era>-TauID_ES.root`, with one directory per category,
`<channel>_<category>`, holding data_obs and every background, nominal as `<process>` and varied as
`<process>_<variation>`. The embedded signal of the mt channel is `EMB_<category>_<mass>` per grid point, the
nominal with mass "0.0", and its variations `EMB_<category>_<mass>_<variation>`.
"""
from __future__ import annotations

from pathlib import Path

from shapesmith.histogram import NOMINAL_VARIATION, TEMPLATE_SEPARATOR, HistKey, Histogram, HistogramSet
from shapesmith.measurements.tau_id_es.grid import CHANNEL, SIGNAL, mass, shift_of
from shapesmith.model import NOMINAL, Analysis
from shapesmith.shapes import shape_name, write_shapes

POSTFIX = "-TauID_ES"


def shapes_path(directory: Path, channel: str, era: str) -> Path:
    return Path(directory) / channel / f"htt_{channel}.inputs-sm-Run{era}{POSTFIX}.root"


def signal_name(category: str, variation: str) -> str:
    """MorphingTauID2017's name of the embedded signal under a variation (a grid point, one on top of it, or none)."""
    systematic, _, template = variation.partition(TEMPLATE_SEPARATOR)
    shift = shift_of(template or systematic)
    if shift is None:
        return shape_name(f"{SIGNAL}_{category}_{mass(0)}", variation)
    return f"{SIGNAL}_{category}_{mass(shift)}" + (f"_{systematic}" if template else "")


def entries(hset: HistogramSet, analysis: Analysis, channel_name: str) -> list[tuple[str, str, Histogram]]:
    channel = analysis.channel(channel_name)
    result = []
    for category in channel.categories:
        folder, variable = f"{channel_name}_{category.name}", category.variable.name
        result.append((folder, "data_obs", hset[HistKey(channel_name, category.name, channel.data(), NOMINAL, NOMINAL_VARIATION, variable)]))
        for process in channel.backgrounds():
            for key in hset.select(channel=channel_name, category=category.name, process=process, region=NOMINAL, variable=variable):
                name = signal_name(category.name, key.variation) if channel_name == CHANNEL and process == SIGNAL else shape_name(process, key.variation)
                result.append((folder, name, hset[key]))
    return result


def write_synced(hset: HistogramSet, analysis: Analysis, directory: Path) -> Path:
    """The shapes file of every channel of the analysis under `directory`; returns `directory`."""
    for channel in analysis.channels:
        write_shapes(shapes_path(directory, channel, analysis.era), entries(hset, analysis, channel))
    return Path(directory)

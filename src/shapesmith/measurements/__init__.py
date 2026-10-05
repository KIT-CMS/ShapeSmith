"""Measurements: plugins on the same event query and histogram tools as the analysis stages.

An analysis sets `Analysis.measurement` to an object with a `name` and a `run(context)` method;
`shapesmith measure` runs it with its output directory `<output_dir>/<name>/<era>/`. The measurement reads the skims
itself (`context.events`), possibly in several passes, and writes its results there. A measurement with a
`gallery(output, channels)` method returning a `shapesmith.web.gallery.Content` can be published with
`shapesmith publish`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from shapesmith.config import RunConfig
from shapesmith.events import Events, Query, load
from shapesmith.model import Analysis
from shapesmith.provenance import record, write_versions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MeasureContext:
    config: RunConfig
    analysis: Analysis
    channels: list[str]
    output: Path
    suggest_binning: bool = False  # print proposed bin edges instead of measuring
    merge: bool = False  # combine the results of earlier runs (e.g. one per switch combination) into one payload

    def events(self, query: Query, columns: Iterable[str] = ()) -> Events:
        return load(self.config, self.analysis, query, columns)

    def provenance(self, **extra) -> dict:
        """What a payload records about its origin: the versions record with the analysis name, the era and `extra`."""
        result = record(self.config)
        result["analysis"]["name"] = self.analysis.name
        return {**result, "era": self.analysis.era, **extra}


class Measurement(Protocol):
    name: str

    def run(self, context: MeasureContext) -> None: ...


def run_measure(config: RunConfig, analysis: Analysis, channels: list[str], suggest_binning: bool = False, merge: bool = False) -> Path:
    if analysis.measurement is None:
        raise ValueError(f"analysis {analysis.name} defines no measurement")
    output = Path(config.output_dir) / analysis.measurement.name / analysis.era
    mode = "suggest-binning" if suggest_binning else "merge" if merge else "measure"
    logger.info(f"measurement {analysis.measurement.name} ({mode}) of channels {', '.join(channels)} -> {output}")
    if not suggest_binning:
        write_versions(config, analysis.name, output)
    analysis.measurement.run(MeasureContext(config, analysis, channels, output, suggest_binning, merge))
    return output

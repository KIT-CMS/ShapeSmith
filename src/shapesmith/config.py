"""Run configuration (YAML -> pydantic) and loading of the analysis object."""
from __future__ import annotations

import importlib
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from shapesmith.model import Analysis, SampleKind
from shapesmith.validate import validate

logger = logging.getLogger(__name__)


class FriendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base: str
    applies_to: tuple[SampleKind, ...] = ("mc", "data", "embedding")


class NtupleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    server: str = ""  # e.g. root://cmsdcache-kit-disk.gridka.de ; empty = local file system
    base: str  # .../CROWNRun
    friends: list[FriendConfig] = Field(default_factory=list)


class CombineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cmssw_dir: str
    scram_arch: str = "el9_amd64_gcc12"


VARIANT_PATTERN = r"[a-z0-9_]+"


class WebConfig(BaseModel):
    """Where `shapesmith publish` puts the plots: the gallery directory and the variant (one column of the gallery)."""

    model_config = ConfigDict(extra="forbid")
    dir: Path | None = None
    variant: str | None = Field(None, pattern=rf"^{VARIANT_PATTERN}$")
    label: str | None = None  # of the variant; default: its key with spaces for underscores
    description: str | None = None  # of the variant
    title: str | None = None  # of the gallery
    about: str | None = None  # text of the gallery's "About this gallery" section (collapsed)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analysis: str  # "package.module:function" -> build(config) -> Analysis
    era: str
    channels: list[str]
    switches: dict[str, Any] = Field(default_factory=dict)
    ntuples: NtupleConfig
    skim_dir: Path
    output_dir: Path
    ml_dir: Path | None = None
    sample_database: Path | None = None  # KingMaker datasets.json used for the normalisation
    workers: int = 4
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"  # of ShapeSmith and the analysis (logs.configure)
    combine: CombineConfig | None = None
    web: WebConfig | None = None

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_case(cls, value: Any) -> Any:
        return value.upper() if isinstance(value, str) else value


RELATIVE_PATH_FIELDS = ("skim_dir", "output_dir", "ml_dir", "sample_database", "web.dir")


def apply_overrides(raw: dict, overrides: Iterable[str]) -> dict:
    """Apply `a.b=value` overrides (values parsed as YAML: `workers=4`, `switches.jet_fakes=ff`, `channels=[mt]`)."""
    for item in overrides:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError(f"override must look like key=value, got {item!r}")
        target = raw
        *parents, leaf = key.split(".")
        for part in parents:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError(f"cannot override {key}: {part} is not a mapping")
        target[leaf] = yaml.safe_load(value)
    return raw


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> RunConfig:
    """Load the run YAML (+ `key=value` overrides); relative paths are relative to the YAML's directory, not to the working directory."""
    path = Path(path).absolute()
    with open(path) as handle:
        raw = apply_overrides(yaml.safe_load(handle), overrides)
    for field in RELATIVE_PATH_FIELDS:
        *parents, leaf = field.split(".")
        section = raw
        for part in parents:
            section = section.get(part) if isinstance(section, dict) else None
        if isinstance(section, dict) and section.get(leaf) is not None and not Path(section[leaf]).is_absolute():
            section[leaf] = os.path.normpath(path.parent / section[leaf])
    return RunConfig.model_validate(raw)


def summary(analysis: Analysis) -> list[str]:
    """One line for the analysis and one per channel: samples per kind, processes, regions, categories, variables,
    variations and estimators."""
    lines = [f"analysis {analysis.name}: era {analysis.era}, {analysis.lumi_pb:g} pb^-1, signal {analysis.signal}, channels {', '.join(analysis.channels)}"]
    for name, channel in analysis.channels.items():
        kinds = ", ".join(f"{kind} {count}" for kind, count in sorted(Counter(s.kind for s in channel.samples).items()))
        lines.append(
            f"{name}: {len(channel.samples)} samples ({kinds}), {len(channel.processes)} processes, {len(channel.regions)} regions, "
            f"{len(channel.categories)} categories, {len(channel.variables)} control variables, {len(channel.variations)} variations, "
            f"estimators {', '.join(type(e).__name__ for e in channel.estimators) or 'none'}"
        )
    return lines


def load_analysis(config: RunConfig) -> Analysis:
    """Import `module:function` from the config, build the analysis and validate it."""
    module_name, _, function_name = config.analysis.partition(":")
    if not function_name:
        raise ValueError(f"analysis must be 'module:function', got {config.analysis!r}")
    build = getattr(importlib.import_module(module_name), function_name)
    analysis = build(config)
    validate(analysis)
    for line in summary(analysis):
        logger.info(line)
    return analysis

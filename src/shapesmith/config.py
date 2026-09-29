"""Run configuration (YAML -> pydantic) and loading of the analysis object."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Iterable

import yaml
from pydantic import BaseModel, ConfigDict, Field

from shapesmith.model import Analysis, SampleKind
from shapesmith.validate import validate


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
    combine: CombineConfig | None = None


RELATIVE_PATH_FIELDS = ("skim_dir", "output_dir", "ml_dir", "sample_database")


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
        if raw.get(field) is not None and not Path(raw[field]).is_absolute():
            raw[field] = os.path.normpath(path.parent / raw[field])
    return RunConfig.model_validate(raw)


def load_analysis(config: RunConfig) -> Analysis:
    """Import `module:function` from the config, build the analysis and validate it."""
    module_name, _, function_name = config.analysis.partition(":")
    if not function_name:
        raise ValueError(f"analysis must be 'module:function', got {config.analysis!r}")
    build = getattr(importlib.import_module(module_name), function_name)
    analysis = build(config)
    validate(analysis)
    return analysis

"""Run configuration (YAML -> pydantic) and loading of the analysis object (Spec §13)."""
from __future__ import annotations

import importlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml

from shapesmith import __version__
from pydantic import BaseModel, ConfigDict, Field

from shapesmith.model import Analysis

SampleKind = Literal["data", "mc", "embedding"]


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
    module = importlib.import_module(module_name)
    build = getattr(module, function_name)
    analysis = build(config)
    analysis.validate()
    return analysis


def git_hash(path: Path) -> str | None:
    """HEAD commit of the git checkout containing `path`, or None if there is none (or git is missing)."""
    path = Path(path)
    directory = path.parent if path.is_file() else path
    try:
        return subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_versions(config: RunConfig, directory: Path) -> Path:
    """Record what produced the contents of `directory`: versions, git hashes and the full run configuration."""
    module = importlib.import_module(config.analysis.partition(":")[0])
    record = {
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shapesmith": {"version": __version__, "git": git_hash(Path(__file__))},
        "analysis": {"module": config.analysis, "git": git_hash(Path(module.__file__))},
        "sample_database": {"path": str(config.sample_database), "git": git_hash(config.sample_database)} if config.sample_database else None,
        "config": json.loads(config.model_dump_json()),
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "versions.json"
    path.write_text(json.dumps(record, indent=2))
    return path


EXAMPLE = """\
analysis: my_analysis.analysis:build     # module:function returning an Analysis
era: "2018"
channels: [et, mt, tt]
switches: {jet_fakes: mc, embedding: false}
ntuples:
  server: root://cmsdcache-kit-disk.gridka.de
  base: /store/user/USER/CROWN/ntuples/TAG/CROWNRun
  friends:
    - {base: /store/user/USER/CROWN/ntuples/TAG/CROWNFriends/nn_v1, applies_to: [mc, data, embedding]}
skim_dir: /ceph/USER/shapesmith/TAG
output_dir: output/TAG_v1
ml_dir: /ceph/USER/shapesmith_ml/v1
sample_database: ../../KingMaker_sample_database/nanoAOD_v15/datasets.json   # relative paths: relative to this file
workers: 16
combine: {cmssw_dir: /work/USER/CMSSW_14_1_9, scram_arch: el9_amd64_gcc12}
"""


def write_example_config(path: str | Path) -> None:
    Path(path).write_text(EXAMPLE)

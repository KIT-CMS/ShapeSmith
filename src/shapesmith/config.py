"""Run configuration (YAML -> pydantic) and loading of the analysis object."""
from __future__ import annotations

import importlib
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, field_validator, model_serializer

from shapesmith.model import Analysis, SampleKind
from shapesmith.validate import validate

logger = logging.getLogger(__name__)


class FriendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base: str
    applies_to: tuple[SampleKind, ...] = ("mc", "data", "embedding")
    channels: tuple[str, ...] | None = None  # None: every channel


class NtupleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    server: str = ""  # e.g. root://cmsdcache-kit-disk.gridka.de ; empty = local file system
    base: str  # .../CROWNRun
    # the main n-tuples of a sample kind from another base, e.g. the MC and embedding of a re-production beside the data
    # of the first one ({mc: .../TAG_B/CROWNRun, embedding: ...}); the kinds not listed read `base`
    bases: dict[SampleKind, str] = Field(default_factory=dict)
    friends: list[FriendConfig] = Field(default_factory=list)

    def base_of(self, kind: str) -> str:
        """The base of the main n-tuples of a sample kind: its entry in `bases`, else `base`."""
        return self.bases.get(kind, self.base)

    @model_serializer(mode="wrap")
    def _without_empty_bases(self, handler: SerializerFunctionWrapHandler) -> dict:
        # without per-kind bases the configuration dumps as before, so its resolved form and hash (versions/) stay
        data = handler(self)
        if not self.bases:
            data.pop("bases", None)
        return data


# the options `combine.minimizer` may set: minimizer and robust-fit settings only, never parameters, data or toys
MINIMIZER_OPTION_PREFIXES = ("--cmin", "--X-rt", "--robust", "--setRobustFit", "--stepSize", "--setCrossingTolerance")
STATS_DEFAULTS = {"r_range": (-40.0, 40.0), "minimizer": "", "precise": True, "timeout": 21600.0, "parallel": 8, "xs_theory_nuisances": None}


class CombineConfig(BaseModel):
    """CMSSW with combine and CombineHarvester, and the options of the fits and limits (limits.py, background_fit.py,
    checks.py); STATS_DEFAULTS are the defaults of the latter."""

    model_config = ConfigDict(extra="forbid")
    cmssw_dir: str
    scram_arch: str = "el9_amd64_gcc12"
    r_range: tuple[float, float] = STATS_DEFAULTS["r_range"]  # --setParameterRanges r=low,high of every fit and limit
    # options of every combine call (fits, limits, toys, impacts), replacing the built-in ones of the same name, e.g.
    # "--cminDefaultMinimizerStrategy 1 --X-rtd MINIMIZER_analytic"; only MINIMIZER_OPTION_PREFIXES
    minimizer: str = STATS_DEFAULTS["minimizer"]
    # AsymptoticLimits converged (combine_options.PRECISE_LIMIT): combine's default accuracy puts the median on a 1/32
    # grid and hides effects below ~0.3 % (R1); false: combine's defaults
    precise: bool = STATS_DEFAULTS["precise"]
    timeout: float | None = STATS_DEFAULTS["timeout"]  # seconds per CMSSW step (a hung fit or xrootd read is killed); null: none
    parallel: int = Field(STATS_DEFAULTS["parallel"], ge=1)  # combine processes at a time (toy jobs, impact fits)
    # the theory nuisances of the signal cross section, frozen at 0 in the cross-section limit; null: those of the
    # analysis (Analysis.signal_cross_section)
    xs_theory_nuisances: tuple[str, ...] | None = STATS_DEFAULTS["xs_theory_nuisances"]

    @field_validator("r_range")
    @classmethod
    def _ordered(cls, value: tuple[float, float]) -> tuple[float, float]:
        if value[0] >= value[1]:
            raise ValueError(f"r_range must be (low, high) with low < high, got {value}")
        return value

    @field_validator("minimizer")
    @classmethod
    def _minimizer_options_only(cls, value: str) -> str:
        flags = [token.partition("=")[0] for token in value.split() if token.startswith("--")]
        forbidden = [flag for flag in flags if not flag.startswith(MINIMIZER_OPTION_PREFIXES)]
        if forbidden or any(token.startswith("-") and not token.startswith("--") and token[1:2].isalpha() for token in value.split()):
            raise ValueError(f"combine.minimizer takes only minimizer options ({', '.join(p + '...' for p in MINIMIZER_OPTION_PREFIXES)}), got {', '.join(forbidden) or value}")
        return value

    @model_serializer(mode="wrap")
    def _without_default_stats(self, handler: SerializerFunctionWrapHandler) -> dict:
        # with the default fit options the configuration dumps as before, so its resolved form and hash (versions/) stay
        data = handler(self)
        for name, default in STATS_DEFAULTS.items():
            if getattr(self, name) == default:
                data.pop(name, None)
        return data


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
    # the site that holds the gallery as one of its eras: `dir` is a direct subdirectory of it, whose name is the key in
    # the era switch of the site page (<site>/index.html), e.g. site: .../smrun2_binned_v7b and dir: .../smrun2_binned_v7b/{era}
    site: Path | None = None

    @model_serializer(mode="wrap")
    def _without_empty_site(self, handler: SerializerFunctionWrapHandler) -> dict:
        # a gallery without a site dumps as before, so the resolved configuration and its hash (versions/) stay
        data = handler(self)
        if self.site is None:
            data.pop("site", None)
        return data


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

    @field_validator("era", mode="before")
    @classmethod
    def _era_as_text(cls, value: Any) -> Any:
        # `era: 2018` or `-s era=2017` parse as YAML integers
        return str(value) if isinstance(value, int) and not isinstance(value, bool) else value


RELATIVE_PATH_FIELDS = ("skim_dir", "output_dir", "ml_dir", "sample_database", "web.dir", "web.site")
ERA_PLACEHOLDER = "{era}"


def fill_era(raw: dict) -> dict:
    """Every `{era}` in a string of the configuration (paths, switches, texts; not in `era` itself) becomes the era of
    the run, so that one run YAML serves every era: `-s era=2017`."""
    era = raw.get("era")
    if era is None:
        return raw
    era = str(era)

    def fill(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace(ERA_PLACEHOLDER, era)
        if isinstance(value, dict):
            return {key: fill(item) for key, item in value.items()}
        if isinstance(value, list):
            return [fill(item) for item in value]
        return value

    return {key: value if key == "era" else fill(value) for key, value in raw.items()}


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
    """Load the run YAML (+ `key=value` overrides), then fill `{era}` with the era of the run (fill_era); relative
    paths are relative to the YAML's directory, not to the working directory."""
    path = Path(path).absolute()
    with open(path) as handle:
        raw = fill_era(apply_overrides(yaml.safe_load(handle), overrides))
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
    controls = f" (control {', '.join(analysis.control_channels)})" if analysis.control_channels else ""
    lines = [f"analysis {analysis.name}: era {analysis.era}, {analysis.lumi_pb:g} pb^-1, signal {analysis.signal}, channels {', '.join(analysis.channels)}{controls}"]
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

"""Command-line interface: one sub-command per analysis step."""
from __future__ import annotations

import functools
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from shapesmith import __version__, logs
from shapesmith.config import VARIANT_PATTERN, RunConfig, WebConfig, load_analysis, load_config
from shapesmith.ntuples import describe
from shapesmith.provenance import record, write_versions

app = typer.Typer(help=f"ShapeSmith {__version__}: CROWN ntuples -> skims -> histograms -> datacards, fits, plots, ML training data, measurements.", no_args_is_help=True)
logger = logging.getLogger("shapesmith.cli")  # also under `python -m shapesmith.cli`, where __name__ is __main__
_command = {"name": "shapesmith"}  # the running sub-command, set by _logged (log file name and first log line)

ConfigOption = typer.Option(..., "--config", "-c", help="run configuration YAML")
ChannelsOption = typer.Option(None, "--channels", help="comma separated subset of the configured channels")
WorkersOption = typer.Option(None, "--workers", help="override `workers` of the configuration (1 = inline, best for debugging)")
SetOption = typer.Option(None, "--set", "-s", help="override a configuration entry, e.g. -s switches.jet_fakes=ff -s workers=4 -s log_level=DEBUG (repeatable, values parsed as YAML)")
VariantOption = typer.Option(None, "--variant", help="the datacards of a variant, [a-z0-9_]+ (e.g. stat_only, s0, ff_w): <output_dir>/datacards/<variant>/<final state>/ (default: <output_dir>/datacards/<final state>/)")
UnblindOption = typer.Option(False, "--unblind", help="fit the data WITHOUT the masks of blinding.json: unblinding, logged loudly (default: the blinded channels are masked)")
FinalStatesOption = typer.Option(None, "--final-states", help="comma separated; default: each channel and 'all'")
CombinationOption = typer.Option(None, "--combination", help="the directory of a combination (`shapesmith combine -o DIR`) instead of the final states of the run")
SkipCombineOption = typer.Option(False, "--skip-combine", help="only summarise existing combine outputs")


def _logged(command):
    """Log the end of a command: its duration, or its error (with the traceback in the log file); then close the log."""

    @functools.wraps(command)
    def wrapper(*args, **kwargs):
        _command["name"] = command.__name__.replace("_", "-")
        start = time.monotonic()
        try:
            result = command(*args, **kwargs)
        except Exception as error:
            if logs.active():
                logger.error(f"failed after {logs.duration(time.monotonic() - start)}: {type(error).__name__}: {error}", exc_info=True)
            raise
        else:
            if logs.active():
                logger.info(f"done in {logs.duration(time.monotonic() - start)}")
            return result
        finally:
            logs.reset()

    return wrapper


def _git(entry: dict | None) -> str:
    return f" (git {entry['git'][:10]})" if entry and entry.get("git") else ""


def _log_start(config: RunConfig, config_path: Path, overrides: list[str], selected: list[str], log_file: Path | None) -> None:
    """What runs: command, configuration, overrides, analysis, versions and git hashes (provenance.record)."""
    versions = record(config)
    logger.info(f"ShapeSmith {__version__} {_command['name']}: {Path(config_path).absolute()}" + (f", overrides {' '.join(overrides)}" if overrides else ""))
    logger.info(f"analysis {config.analysis}{_git(versions['analysis'])}, era {config.era}, channels {','.join(selected)}, {config.workers} workers, log level {config.log_level}")
    database = f", sample database {config.sample_database}{_git(versions['sample_database'])}" if config.sample_database else ""
    logger.info(f"shapesmith{_git(versions['shapesmith'])}{database}, ntuples {describe(config.ntuples)}")
    if log_file is not None:
        logger.info(f"log file {log_file}")
    logger.debug(f"resolved configuration: {json.dumps(versions['config'], sort_keys=True)}")


def _setup(config_path: Path, channels: Optional[str], workers: Optional[int] = None, overrides: Optional[list[str]] = None, log_file: bool = True):
    """The configuration (+ overrides), the logging of the command (console, and <output_dir>/logs/ unless `log_file`
    is False) and the validated analysis with the selected channels."""
    config = load_config(config_path, overrides or [])
    if workers is not None:
        config = config.model_copy(update={"workers": workers})
    path = logs.log_path(config.output_dir, _command["name"]) if log_file else None
    logs.configure(config.log_level, path, [config.analysis.partition(":")[0].split(".")[0]])
    selected = channels.split(",") if channels else list(config.channels)
    _log_start(config, config_path, overrides or [], selected, path)
    unknown = set(selected) - set(config.channels)
    if unknown:
        raise typer.BadParameter(f"channels {sorted(unknown)} are not in the configuration")
    analysis = load_analysis(config)
    return config, analysis, selected


def _shapes_path(config: RunConfig, control: bool) -> Path:
    return config.output_dir / ("control_shapes.root" if control else "shapes.root")


def _final_states(channels: list[str], control_channels: tuple[str, ...] = ()) -> dict[str, list[str]]:
    """Each selected channel and, with more than one, all of them ('all'); the selected control channels join every
    final state and are none themselves."""
    controls = [channel for channel in channels if channel in control_channels]
    signal_channels = [channel for channel in channels if channel not in control_channels]
    if not signal_channels:
        raise typer.BadParameter(f"no final state: {', '.join(channels)} are control channels")
    states = {channel: [channel, *controls] for channel in signal_channels}
    if len(signal_channels) > 1:
        states["all"] = list(channels)
    return states


def _split(value: Optional[str]) -> Optional[list[str]]:
    return value.split(",") if value else None


def _variant(variant: Optional[str], channels) -> Optional[str]:
    """A valid variant name: [a-z0-9_]+ and no final state (a channel or 'all'), whose directory it would shadow."""
    if variant is None:
        return None
    if not re.fullmatch(VARIANT_PATTERN, variant) or variant in {*channels, "all"}:
        raise typer.BadParameter(f"variant {variant!r} must match {VARIANT_PATTERN} and differ from the final states {', '.join([*channels, 'all'])}")
    return variant


def _datacard_dir(config: RunConfig, variant: Optional[str], channels) -> Path:
    from shapesmith.combination import datacard_dir

    return datacard_dir(config, _variant(variant, channels))


@app.command()
@_logged
def validate(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption):
    """Load and validate the analysis; report processes, selections and the columns every channel needs."""
    from shapesmith.skim import exported_by, needed_columns

    cfg, analysis, selected = _setup(config, channels, overrides=overrides, log_file=False)
    console = Console()
    console.print(f"[bold]{analysis.name}[/bold]  era {analysis.era}, {analysis.lumi_pb:g} pb^-1, switches {dict(cfg.switches)}")
    for name in selected:
        channel = analysis.channel(name)
        kinds = sorted({s.kind for s in channel.samples})
        console.print(f"[bold]{name}[/bold]: samples {len(channel.samples)} ({', '.join(f'{k} {sum(s.kind == k for s in channel.samples)}' for k in kinds)})")
        table = Table(show_edge=False)
        for column in ("process", "role", "group", "plot group", "samples"):
            table.add_column(column)
        for process in channel.processes:
            table.add_row(process.name, process.role, process.group, process.plot_group, str(len(channel.samples_of(process.group))))
        console.print(table)
        console.print(f"  cuts {list(channel.cuts)}, regions {[r.name for r in channel.regions]}")
        console.print(f"  categories {[c.name for c in channel.categories] or 'none'}, control variables {len(channel.variables)}, variations {len(channel.variations)}")
        console.print(f"  estimators {[type(e).__name__ for e in channel.estimators] or 'none'}, backgrounds {list(channel.backgrounds())}")
        columns = set().union(*(needed_columns(channel, s, exported_by(analysis, channel)) for s in channel.samples))
        typer.echo(f"{name}: {len(columns)} columns required")
    typer.echo(f"analysis {analysis.name} is valid")


@app.command()
@_logged
def skim(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, force: bool = typer.Option(False, "--force", help="skim again, ignoring stored skims"), samples: Optional[str] = typer.Option(None, "--samples", help="comma separated sample groups or nick prefixes (default: all)"), workers: Optional[int] = WorkersOption):
    """Stage 1: ntuples (+ friends) -> Parquet skims."""
    from shapesmith.skim import run_skim

    cfg, analysis, selected = _setup(config, channels, workers=workers, overrides=overrides)
    write_versions(cfg, analysis.name, cfg.skim_dir)
    results = run_skim(cfg, analysis, selected, force=force, samples=_split(samples))
    logger.info(f"{sum(1 for r in results if not r.skipped)} files skimmed, {sum(1 for r in results if r.skipped)} reused, {sum(r.n_out for r in results)} events kept")


@app.command()
@_logged
def hist(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, control: bool = typer.Option(False, "--control", help="control variables instead of NN categories"), variables: Optional[str] = typer.Option(None, help="comma separated control variables"), regions: Optional[str] = typer.Option(None, "--regions", help="comma separated regions, or 'all' (default: nominal plus estimator regions)"), skip_systematics: bool = typer.Option(False, "--skip-systematics", help="no weight or column variations"), processes: Optional[str] = typer.Option(None, help="comma separated process names"), workers: Optional[int] = WorkersOption):
    """Stage 2: skims -> histograms (control_shapes.root or shapes.root); other histograms in the file are kept."""
    from shapesmith.fill import run_hist

    cfg, analysis, selected = _setup(config, channels, workers=workers, overrides=overrides)
    write_versions(cfg, analysis.name, cfg.output_dir)
    output = _shapes_path(cfg, control)
    run_hist(cfg, analysis, selected, control, _split(variables), not skip_systematics, _split(processes), output, _split(regions))


@app.command()
@_logged
def estimate(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, control: bool = typer.Option(False, "--control")):
    """Stage 3: add the estimated processes and variations of every channel to the histogram file."""
    from shapesmith.estimates import run_estimates
    from shapesmith.histogram import HistogramSet

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
    path = _shapes_path(cfg, control)
    hset = HistogramSet.load(path)
    added = run_estimates(hset, analysis, selected)
    hset.save(path)
    logger.info(f"{len(added)} histograms added to {path}")


@app.command()
@_logged
def measure(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, suggest_binning: bool = typer.Option(False, "--suggest-binning", help="log proposed bin edges instead of measuring"), merge: bool = typer.Option(False, "--merge", help="combine the results of the earlier runs (e.g. one per working point) into one payload"), refit: bool = typer.Option(False, "--refit", help="repeat the fits from the inputs an earlier run stored (e.g. synced/ shapes), without histogramming"), workers: Optional[int] = WorkersOption):
    """Run the analysis's measurement; results in <output_dir>/<measurement>/<era>/."""
    from shapesmith.measurements import run_measure

    cfg, analysis, selected = _setup(config, channels, workers=workers, overrides=overrides)
    logger.info(f"measurement output: {run_measure(cfg, analysis, selected, suggest_binning, merge, refit)}")


@app.command()
@_logged
def sync(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption):
    """Write combine-style shape files per channel."""
    from shapesmith.histogram import HistogramSet
    from shapesmith.shapes import run_sync

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
    for path in run_sync(HistogramSet.load(_shapes_path(cfg, False)), analysis, selected, cfg.output_dir):
        logger.info(f"wrote {path}")


@app.command()
@_logged
def datacards(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, systematics: bool = typer.Option(True, "--systematics/--no-systematics"), min_background: Optional[float] = typer.Option(1.0, help="merge bins below this background yield; negative disables"), variant: Optional[str] = VariantOption):
    """Write text datacards and their shapes file per final state (each channel and the combination)."""
    from shapesmith.datacards import run_datacards
    from shapesmith.histogram import HistogramSet

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
    threshold = None if min_background is None or min_background < 0 else min_background
    for name, path in run_datacards(HistogramSet.load(_shapes_path(cfg, False)), analysis, _final_states(selected, analysis.control_channels), _datacard_dir(cfg, variant, analysis.channels), systematics, threshold).items():
        logger.info(f"datacard {name}: {path}")


@app.command()
@_logged
def fit(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, final_states: Optional[str] = typer.Option(None, help="comma separated; default: each channel and 'all' (with --background-only: the combination of the selected channels)"), skip_combine: bool = typer.Option(False, "--skip-combine", help="only collect existing combine outputs"), background_only: bool = typer.Option(False, "--background-only", help="a background-only fit to the data (FitDiagnostics, r frozen to 0) instead of the limits"), variant: Optional[str] = VariantOption, unblind: bool = UnblindOption):
    """Run combine (prefit Asimov expected limits on mu and the signal cross section, significance, best fit), or a background-only fit to the data with the blinded channels masked, and summarise the results."""
    from shapesmith.background_fit import run_background_fit
    from shapesmith.limits import run_limits

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
    states = list(_final_states(selected, analysis.control_channels))
    datacard_dir = _datacard_dir(cfg, variant, analysis.channels)
    if unblind and not background_only:
        raise typer.BadParameter("--unblind applies to the fit to data (--background-only); the limits use the Asimov dataset")
    if background_only:
        results = run_background_fit(cfg, analysis, datacard_dir, _split(final_states) or states[-1:], skip_combine, unblind)  # default: 'all' or the only final state
        typer.echo((datacard_dir / "background_fit.md").read_text())
    else:
        results = run_limits(cfg, datacard_dir, _split(final_states) or states, skip_combine, analysis.signal_cross_section)
        typer.echo((datacard_dir / "limits.md").read_text())
    logger.info(f"{len(results)} final states summarised")


@app.command("combine")
@_logged
def combine_runs(configs: list[Path] = typer.Option(..., "--config", "-c", help="the run YAML of every era (repeat -c)"), output: Path = typer.Option(..., "--output", "-o", help="directory of the combination (combined card, fits, summaries)"), final_state: str = typer.Option("all", "--final-state", help="the final state of each run's datacards: <output_dir>/datacards/<final state>/"), overrides: Optional[list[str]] = SetOption, eras: Optional[list[str]] = typer.Option(None, "--era", help="load every run YAML once per era (-s era=<era>, its {era} filled; repeat --era)"), background_only: bool = typer.Option(False, "--background-only", help="a background-only fit to the data instead of the limits"), skip_combine: bool = typer.Option(False, "--skip-combine", help="only summarise existing combine outputs"), validate: bool = typer.Option(False, "--validate", help="also run CombineHarvester's ValidateDatacards.py on the combined card"), variant: Optional[str] = VariantOption, unblind: bool = UnblindOption):
    """Combine the datacards of several runs (one per era, spec 6.9) with combineCards.py, then the limits or the background-only fit (the blinded channels of every card masked); overrides apply to every run."""
    from shapesmith.combination import Run, run_card, run_combination

    runs = []
    for index, (path, run_overrides) in enumerate((path, [*(overrides or []), *([f"era={era}"] if era else [])]) for path in configs for era in (eras or [None])):
        cfg = load_config(path, run_overrides)
        if index == 0:
            logs.configure(cfg.log_level, logs.log_path(output, _command["name"]), [cfg.analysis.partition(":")[0].split(".")[0]])
        _log_start(cfg, path, run_overrides, list(cfg.channels), None)
        analysis = load_analysis(cfg)
        runs.append(Run(cfg, analysis, run_card(cfg, final_state, _variant(variant, analysis.channels))))
    if unblind and not background_only:
        raise typer.BadParameter("--unblind applies to the fit to data (--background-only); the limits use the Asimov dataset")
    result = run_combination(runs, output, background_only, skip_combine, validate, unblind)
    typer.echo((output / ("background_fit.md" if background_only else "limits.md")).read_text())
    logger.info(f"combination of {len(runs)} runs: {json.dumps(result)[:200]}")


def _targets(cfg: RunConfig, analysis, selected: list[str], variant: Optional[str], final_states: Optional[str], combination: Optional[Path]):
    """The cards of a statistical check and the directory of its summary: the combination in `combination`, else the
    final states (default: each channel and 'all') of the run's datacards (of the variant)."""
    from shapesmith.checks import Target

    if combination is not None:
        if final_states:
            raise typer.BadParameter("--final-states and --combination exclude each other")
        targets, summary_dir = [Target("combination", combination.absolute())], combination.absolute()
    else:
        summary_dir = _datacard_dir(cfg, variant, analysis.channels)
        targets = [Target(name, summary_dir / name) for name in _split(final_states) or list(_final_states(selected, analysis.control_channels))]
    missing = [str(target.card) for target in targets if not target.card.exists()]
    if missing:
        raise typer.BadParameter("datacards missing (run `shapesmith datacards`, or `shapesmith combine` for a combination):\n" + "\n".join(missing))
    return targets, summary_dir


def _stored(summary_dir: Path, check: str, name: str) -> Optional[dict]:
    """The result of a card in an earlier summary <summary_dir>/<check>.json (e.g. limits, aposteriori), or None."""
    path = summary_dir / f"{check}.json"
    return json.loads(path.read_text()).get(name) if path.exists() else None


@app.command()
@_logged
def gof(
    config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption,
    variant: Optional[str] = VariantOption, final_states: Optional[str] = FinalStatesOption, combination: Optional[Path] = CombinationOption,
    toys: int = typer.Option(500, "--toys", help="frequentist toys (jobs of 25, combine.parallel at a time)"),
    skip_combine: bool = SkipCombineOption, unblind: bool = UnblindOption,
):
    """Saturated goodness of fit of the background-only model to the data against toys (spec 7.2), per final state
    (channel x era) or of a combination; the blinded channels are masked. Summary <datacards>/gof.md."""
    from shapesmith.checks import run_gof, write_summary

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
    targets, summary_dir = _targets(cfg, analysis, selected, variant, final_states, combination)
    write_summary("gof", {t.name: run_gof(t, cfg.combine, toys, unblind, skip_combine) for t in targets}, summary_dir, "Saturated GoF, background only")
    typer.echo((summary_dir / "gof.md").read_text())


@app.command()
@_logged
def impacts(
    config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption,
    variant: Optional[str] = VariantOption, final_states: Optional[str] = FinalStatesOption, combination: Optional[Path] = CombinationOption,
    asimov: bool = typer.Option(False, "--asimov", help="on the Asimov dataset with r = 1 (expected impacts, no masks) instead of the data"),
    skip_combine: bool = SkipCombineOption, unblind: bool = UnblindOption,
):
    """Impacts and pulls of the nuisances (combineTool.py -M Impacts, plotImpacts.py), on the data with the blinded
    channels masked or on the Asimov dataset. Plot <card>/impacts[_asimov]/impacts.pdf, summary <datacards>/impacts[_asimov].md."""
    from shapesmith.checks import run_impacts, write_summary

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
    targets, summary_dir = _targets(cfg, analysis, selected, variant, final_states, combination)
    check = "impacts_asimov" if asimov else "impacts"
    write_summary(check, {t.name: run_impacts(t, cfg.combine, asimov, unblind, skip_combine) for t in targets}, summary_dir)
    typer.echo((summary_dir / f"{check}.md").read_text())


@app.command("limit-aposteriori")
@_logged
def limit_aposteriori(
    config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption,
    variant: Optional[str] = VariantOption, final_states: Optional[str] = FinalStatesOption, combination: Optional[Path] = CombinationOption,
    skip_combine: bool = SkipCombineOption, unblind: bool = UnblindOption,
):
    """The a-posteriori expected limit (spec 7.2): background-only fit to the data with the blinded channels masked,
    its snapshot, then AsymptoticLimits on the snapshot's Asimov dataset (-t -1 --toysFrequentist --bypassFrequentistFit)
    with all channels; on mu and the signal cross section. Summary <datacards>/aposteriori.md/.pdf."""
    from shapesmith.checks import run_aposteriori, write_summary

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
    targets, summary_dir = _targets(cfg, analysis, selected, variant, final_states, combination)
    results = {t.name: run_aposteriori(t, cfg.combine, analysis.signal_cross_section, _stored(summary_dir, "limits", t.name), unblind, skip_combine) for t in targets}
    write_summary("aposteriori", results, summary_dir, "A-posteriori expected limits")
    typer.echo((summary_dir / "aposteriori.md").read_text())


def _injected(values: str, expected: Optional[dict], name: str) -> list[float]:
    """The signal strengths of --mu; `limit` is the median expected limit of the card (a-posteriori, else prefit)."""
    mus = []
    for value in values.split(","):
        if value == "limit":
            if not expected:
                raise typer.BadParameter(f"{name}: --mu limit needs the expected limit; run `shapesmith limit-aposteriori` or `shapesmith fit` first")
            mus.append(round(expected["limit"]["exp_median"], 4))
        else:
            mus.append(float(value))
    return mus


@app.command()
@_logged
def injection(
    config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption,
    variant: Optional[str] = VariantOption, final_states: Optional[str] = FinalStatesOption, combination: Optional[Path] = CombinationOption,
    mu: str = typer.Option("0,1,limit", "--mu", help="comma separated injected signal strengths; 'limit': the median expected limit of the card"),
    toys: int = typer.Option(200, "--toys", help="toys per injected value (jobs of 25, combine.parallel at a time)"),
    a_posteriori: bool = typer.Option(False, "--a-posteriori", help="frequentist toys around the masked background-only fit to the data instead of toys of the prefit model"),
    skip_combine: bool = SkipCombineOption, unblind: bool = UnblindOption,
):
    """Signal injection and bias (spec 7.2): toys with r = mu, fitted; r should follow mu linearly and the pulls be
    unbiased. Plot <card>/injection/injection.pdf, summary <datacards>/injection.md."""
    from shapesmith.checks import run_injection, write_summary

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
    targets, summary_dir = _targets(cfg, analysis, selected, variant, final_states, combination)
    check = "injection_aposteriori" if a_posteriori else "injection"
    results = {}
    for t in targets:
        expected = _stored(summary_dir, "aposteriori", t.name) or _stored(summary_dir, "limits", t.name)
        results[t.name] = run_injection(t, cfg.combine, _injected(mu, expected, t.name), toys, a_posteriori, unblind, skip_combine)
    write_summary(check, results, summary_dir, "Signal injection")
    typer.echo((summary_dir / f"{check}.md").read_text())


@app.command("limit-hybridnew")
@_logged
def limit_hybridnew(
    config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption,
    variant: Optional[str] = VariantOption, final_states: Optional[str] = FinalStatesOption, combination: Optional[Path] = CombinationOption,
    toys: int = typer.Option(500, "--toys", help="toys per r point and hypothesis (HybridNew -T)"),
    rel_acc: float = typer.Option(0.02, "--rel-acc", help="relative accuracy of the limit (HybridNew --rRelAcc)"),
    abs_acc: float = typer.Option(0.05, "--abs-acc", help="absolute accuracy of the limit (HybridNew --rAbsAcc)"),
    a_posteriori: bool = typer.Option(False, "--a-posteriori", help="on the Asimov dataset of the masked background-only fit (snapshot) instead of the prefit one"),
    skip_combine: bool = SkipCombineOption, unblind: bool = UnblindOption,
):
    """The HybridNew cross-check of the median expected limit (spec 7.2): CLs with toys on the Asimov dataset, compared
    with the asymptotic median (limits.json, or aposteriori.json). Summary <datacards>/hybridnew.md."""
    from shapesmith.checks import run_hybridnew, write_summary

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
    targets, summary_dir = _targets(cfg, analysis, selected, variant, final_states, combination)
    check = "hybridnew_aposteriori" if a_posteriori else "hybridnew"
    results = {t.name: run_hybridnew(t, cfg.combine, toys, rel_acc, abs_acc, _stored(summary_dir, "aposteriori" if a_posteriori else "limits", t.name), a_posteriori, unblind, skip_combine) for t in targets}
    write_summary(check, results, summary_dir, "HybridNew median expected limit")
    typer.echo((summary_dir / f"{check}.md").read_text())


@app.command("limit-overview")
@_logged
def limit_overview(
    configs: list[Path] = typer.Option(..., "--config", "-c", help="the run YAML of every era (repeat -c)"),
    output: Path = typer.Option(..., "--output", "-o", help="directory of the overview"),
    overrides: Optional[list[str]] = SetOption,
    eras: Optional[list[str]] = typer.Option(None, "--era", help="every run YAML once per era (-s era=<era>; repeat)"),
    variant: Optional[str] = VariantOption,
    combinations: Optional[list[Path]] = typer.Option(None, "--combination", help="the directory of a combination (repeatable)"),
    aposteriori: bool = typer.Option(False, "--aposteriori", help="the a-posteriori expected limits (aposteriori.json) instead of the prefit ones (limits.json)"),
):
    """One table and Brazil plot of the expected limits per channel and era and of the combinations, on mu and on the
    signal cross section: <output>/limit_overview.{json,md,pdf}, limit_overview_xs.pdf."""
    from shapesmith.limits import limit_overview as overview

    summary = "aposteriori.json" if aposteriori else "limits.json"
    sources = []
    for index, (path, run_overrides) in enumerate((path, [*(overrides or []), *([f"era={era}"] if era else [])]) for path in configs for era in (eras or [None])):
        cfg = load_config(path, run_overrides)
        if index == 0:
            logs.configure(cfg.log_level, logs.log_path(output, _command["name"]), [cfg.analysis.partition(":")[0].split(".")[0]])
        _log_start(cfg, path, run_overrides, list(cfg.channels), None)
        sources.append((cfg.era, _datacard_dir(cfg, variant, cfg.channels) / summary))
    sources += [("combined" if len(combinations) == 1 else f"combined {directory.name}", directory / summary) for directory in combinations or []]
    missing = [str(path) for _, path in sources if not path.exists()]
    if missing:
        raise typer.BadParameter(f"{summary} missing (run `shapesmith {'limit-aposteriori' if aposteriori else 'fit'}` first):\n" + "\n".join(missing))
    overview(sources, output, "limit_overview_aposteriori" if aposteriori else "limit_overview")
    typer.echo((output / ("limit_overview_aposteriori.md" if aposteriori else "limit_overview.md")).read_text())


@app.command("input-gof")
@_logged
def input_gof(
    configs: list[Path] = typer.Option(..., "--config", "-c", help="the run YAML of every era (repeat -c)"),
    output: Path = typer.Option(..., "--output", "-o", help="directory of the cards and of the summary input_gof.json/.md (results of earlier runs into it are kept)"),
    overrides: Optional[list[str]] = SetOption,
    eras: Optional[list[str]] = typer.Option(None, "--era", help="every run YAML once per era (-s era=<era>; repeat)"),
    channels: Optional[str] = ChannelsOption,
    variables: Optional[str] = typer.Option(None, "--variables", help="comma separated control variables (default: every control variable of a channel)"),
    region: str = typer.Option("nominal", "--region", help="the region of the control histograms (a control region: hist --control --regions <region>)"),
    toys: int = typer.Option(500, "--toys", help="frequentist toys per card, in one job"),
    tail_toys: int = typer.Option(0, "--tail-toys", help="toys per card in total for the cards below --tail-below (0: none); runs combine also with --skip-combine"),
    tail_below: float = typer.Option(1e-3, "--tail-below", help="the p-value below which a card gets the tail toys"),
    alt_minimizer: str = typer.Option("", "--alt-minimizer", help="minimizer options added for a second observed fit of every card, e.g. '--X-rtd MINIMIZER_no_analytic'; a card with a lower statistic there runs again with them (also with --skip-combine)"),
    bins: int = typer.Option(20, "--bins", help="at most this many bins of about equal prediction"),
    min_prediction: float = typer.Option(20.0, "--min-prediction", help="background events per bin at least"),
    exclude: Optional[list[str]] = typer.Option(None, "--exclude", help="regular expression: nuisances left out of the cards (repeatable)"),
    plots: bool = typer.Option(True, "--plots/--no-plots", help="the prefit plot of every card"),
    skip_cards: bool = typer.Option(False, "--skip-cards", help="use the cards of an earlier run"),
    skip_combine: bool = SkipCombineOption,
):
    """The saturated GoF of single control variables (NN spec 7.1, NV1): per era, channel and variable a card of the
    control histograms of one region (control_shapes.root) with the nuisances of the run, bins merged to equal
    prediction, toys; the failure rule over every test in <output>/input_gof.md."""
    from shapesmith.histogram import HistogramSet
    from shapesmith.input_gof import GofBinning, run_cards, run_gof, stored_records, targets_of, write_summary

    binning = GofBinning(bins, min_prediction)
    records, combine = [], None
    for index, (path, run_overrides) in enumerate((path, [*(overrides or []), *([f"era={era}"] if era else [])]) for path in configs for era in (eras or [None])):
        cfg = load_config(path, run_overrides)
        if index == 0:
            logs.configure(cfg.log_level, logs.log_path(output, _command["name"]), [cfg.analysis.partition(":")[0].split(".")[0]])
            combine = cfg.combine
        selected = channels.split(",") if channels else list(cfg.channels)
        _log_start(cfg, path, run_overrides, selected, None)
        analysis = load_analysis(cfg)
        targets = targets_of(analysis, selected, region, _split(variables))
        if skip_cards:
            records += stored_records(output, targets)
            continue
        names = {(t.channel, t.variable) for t in targets}
        hset = HistogramSet.load(_shapes_path(cfg, True), keep=lambda key: key.region == region and (key.channel, key.variable) in names)
        records += run_cards(hset, analysis, targets, output, binning, tuple(exclude or ()), plots)
    results = run_gof(records, combine, toys, output, skip_combine, tail_toys, tail_below, alt_minimizer)
    write_summary(results, output, title=f"Saturated GoF of single variables, region {region}")
    typer.echo((output / "input_gof.md").read_text())


@app.command()
@_logged
def plot(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, control: bool = typer.Option(False, "--control"), category: Optional[str] = typer.Option(None), variables: Optional[str] = typer.Option(None), region: str = typer.Option("nominal", "--region", help="histogram region to plot"), blind: bool = typer.Option(False, "--blind"), log: bool = typer.Option(False, "--log"), signal_scale: Optional[float] = typer.Option(None), normalize_by_bin_width: bool = typer.Option(False, "--normalize-by-bin-width")):
    """Prefit stack plots of control variables (--control) or NN categories."""
    from shapesmith.histogram import HistogramSet
    from shapesmith.plotting.stack import run_plot

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
    files = run_plot(HistogramSet.load(_shapes_path(cfg, control)), analysis, selected, control, category, _split(variables), cfg.output_dir / "plots", region=region, blind=blind, log=log, signal_scale=signal_scale, normalize_by_bin_width=normalize_by_bin_width)
    logger.info(f"{len(files)} files written to {cfg.output_dir / 'plots'}")


@app.command()
@_logged
def publish(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, to: Optional[Path] = typer.Option(None, "--to", help="gallery directory (default: web.dir of the configuration)"), variant: Optional[str] = typer.Option(None, "--variant", help="the gallery column, [a-z0-9_]+ (default: web.variant)"), label: Optional[str] = typer.Option(None, "--label", help="label of the variant"), description: Optional[str] = typer.Option(None, "--description", help="description of the variant"), title: Optional[str] = typer.Option(None, "--title", help="title of the gallery"), about: Optional[str] = typer.Option(None, "--about", help="text of the gallery's About section"), site: Optional[Path] = typer.Option(None, "--site", help="site directory with an era switch that holds the gallery (--to) as one of its subdirectories (default: web.site)")):
    """Copy the plots and their yields (of a measurement analysis: the measurement's plots) into a static web gallery (one column per variant); publishing a variant again replaces its channels."""
    from shapesmith.web import publish as publish_gallery

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
    options = {"dir": to.absolute() if to else None, "variant": variant, "label": label, "description": description, "title": title, "about": about, "site": site.absolute() if site else None}
    web = (cfg.web or WebConfig()).model_copy(update={key: value for key, value in options.items() if value is not None})
    publish_gallery(cfg, analysis, web, selected if channels else None, config, overrides or [])


@app.command("ml-export")
@_logged
def ml_export(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, workers: Optional[int] = WorkersOption):
    """Training data from the skims: <ml_dir>/<channel>/events.feather (one row per event) and <ml_dir>/classes.json."""
    from shapesmith.ml_export import run_ml_export

    cfg, analysis, selected = _setup(config, channels, workers=workers, overrides=overrides)
    if cfg.ml_dir is not None:
        write_versions(cfg, analysis.name, cfg.ml_dir)
    for path in run_ml_export(cfg, analysis, selected):
        logger.info(f"wrote {path}")


@app.command("ml-merge")
@_logged
def ml_merge(
    configs: list[Path] = typer.Option(..., "--config", "-c", help="the run YAML of every era (repeat -c)"),
    output: Path = typer.Option(..., "--output", "-o", help="directory of the merged export"),
    channels: Optional[str] = ChannelsOption,
    overrides: Optional[list[str]] = SetOption,
    eras: Optional[list[str]] = typer.Option(None, "--era", help="every run YAML once per era (-s era=<era>; repeat)"),
):
    """Merge the ML exports of several eras (the ml_dir of every run) into one per channel, with the era column."""
    from shapesmith.ml_export import merge_eras

    directories = []
    runs = [(path, [*(overrides or []), *([f"era={e}"] if e else [])]) for path in configs for e in (eras or [None])]
    for index, (path, run_overrides) in enumerate(runs):
        cfg = load_config(path, run_overrides)
        if index == 0:
            package = cfg.analysis.partition(":")[0].split(".")[0]
            logs.configure(cfg.log_level, logs.log_path(output, _command["name"]), [package])
        if cfg.ml_dir is None:
            raise typer.BadParameter(f"{path} ({' '.join(run_overrides)}): ml_dir is not set")
        logger.info(f"{path} ({' '.join(run_overrides) or 'no overrides'}): era {cfg.era}, {cfg.ml_dir}")
        directories.append(cfg.ml_dir)
    for path in merge_eras(directories, output, _split(channels)):
        logger.info(f"wrote {path}")


@app.command()
def inspect(shapes: Path = typer.Argument(..., help="a shapes ROOT file written by `hist`"), unchanged: bool = typer.Option(False, "--unchanged", help="list the variations that are bitwise equal to their nominal")):
    """List the processes, regions, variations and variables in a histogram file."""
    from shapesmith.histogram import HistogramSet, unchanged_variations

    hset = HistogramSet.load(shapes)
    if unchanged:
        for key in unchanged_variations(hset):
            typer.echo(key.path)
        return
    for field in ("channel", "category", "process", "region", "variation", "variable"):
        values = sorted({getattr(key, field) for key in hset})
        typer.echo(f"{field:10s} ({len(values)}): {', '.join(values[:40])}{' ...' if len(values) > 40 else ''}")
    typer.echo(f"{len(hset)} histograms")


if __name__ == "__main__":
    app()

"""Command-line interface: one sub-command per analysis step."""
from __future__ import annotations

import functools
import json
import logging
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from shapesmith import __version__, logs
from shapesmith.config import RunConfig, WebConfig, load_analysis, load_config
from shapesmith.ntuples import join_url
from shapesmith.provenance import record, write_versions

app = typer.Typer(help=f"ShapeSmith {__version__}: CROWN ntuples -> skims -> histograms -> datacards, fits, plots, ML folds, measurements.", no_args_is_help=True)
logger = logging.getLogger("shapesmith.cli")  # also under `python -m shapesmith.cli`, where __name__ is __main__
_command = {"name": "shapesmith"}  # the running sub-command, set by _logged (log file name and first log line)

ConfigOption = typer.Option(..., "--config", "-c", help="run configuration YAML")
ChannelsOption = typer.Option(None, "--channels", help="comma separated subset of the configured channels")
WorkersOption = typer.Option(None, "--workers", help="override `workers` of the configuration (1 = inline, best for debugging)")
SetOption = typer.Option(None, "--set", "-s", help="override a configuration entry, e.g. -s switches.jet_fakes=ff -s workers=4 -s log_level=DEBUG (repeatable, values parsed as YAML)")


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
    logger.info(f"shapesmith{_git(versions['shapesmith'])}{database}, ntuples {join_url(config.ntuples.server, config.ntuples.base)}")
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


@app.command()
@_logged
def validate(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption):
    """Load and validate the analysis; report processes, selections and the columns every channel needs."""
    from shapesmith.skim import needed_columns

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
        columns = set().union(*(needed_columns(channel, s) for s in channel.samples))
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
def datacards(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, systematics: bool = typer.Option(True, "--systematics/--no-systematics"), min_background: Optional[float] = typer.Option(1.0, help="merge bins below this background yield; negative disables")):
    """Write text datacards and their shapes file per final state (each channel and the combination)."""
    from shapesmith.datacards import run_datacards
    from shapesmith.histogram import HistogramSet

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
    threshold = None if min_background is None or min_background < 0 else min_background
    for name, path in run_datacards(HistogramSet.load(_shapes_path(cfg, False)), analysis, _final_states(selected, analysis.control_channels), cfg.output_dir / "datacards", systematics, threshold).items():
        logger.info(f"datacard {name}: {path}")


@app.command()
@_logged
def fit(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, final_states: Optional[str] = typer.Option(None, help="comma separated; default: each channel and 'all' (with --background-only: the combination of the selected channels)"), skip_combine: bool = typer.Option(False, "--skip-combine", help="only collect existing combine outputs"), background_only: bool = typer.Option(False, "--background-only", help="a background-only fit to the data (FitDiagnostics, r frozen to 0) instead of the limits")):
    """Run combine (expected limits, significance, best fit), or a background-only fit to the data, and summarise the results."""
    from shapesmith.background_fit import run_background_fit
    from shapesmith.limits import run_limits

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
    states = list(_final_states(selected, analysis.control_channels))
    datacard_dir = cfg.output_dir / "datacards"
    if background_only:
        results = run_background_fit(cfg, analysis, datacard_dir, _split(final_states) or states[-1:], skip_combine)  # default: 'all' or the only final state
        typer.echo((datacard_dir / "background_fit.md").read_text())
    else:
        results = run_limits(cfg, datacard_dir, _split(final_states) or states, skip_combine)
        typer.echo((datacard_dir / "limits.md").read_text())
    logger.info(f"{len(results)} final states summarised")


@app.command("combine")
@_logged
def combine_runs(configs: list[Path] = typer.Option(..., "--config", "-c", help="the run YAML of every era (repeat -c)"), output: Path = typer.Option(..., "--output", "-o", help="directory of the combination (combined card, fits, summaries)"), final_state: str = typer.Option("all", "--final-state", help="the final state of each run's datacards: <output_dir>/datacards/<final state>/"), overrides: Optional[list[str]] = SetOption, background_only: bool = typer.Option(False, "--background-only", help="a background-only fit to the data instead of the limits"), skip_combine: bool = typer.Option(False, "--skip-combine", help="only summarise existing combine outputs"), validate: bool = typer.Option(False, "--validate", help="also run CombineHarvester's ValidateDatacards.py on the combined card")):
    """Combine the datacards of several runs (one per era, spec 6.9) with combineCards.py, then the limits or the background-only fit; overrides apply to every run."""
    from shapesmith.combination import Run, run_card, run_combination

    runs = []
    for index, path in enumerate(configs):
        cfg = load_config(path, overrides or [])
        if index == 0:
            logs.configure(cfg.log_level, logs.log_path(output, _command["name"]), [cfg.analysis.partition(":")[0].split(".")[0]])
        _log_start(cfg, path, overrides or [], list(cfg.channels), None)
        runs.append(Run(cfg, load_analysis(cfg), run_card(cfg, final_state)))
    result = run_combination(runs, output, background_only, skip_combine, validate)
    typer.echo((output / ("background_fit.md" if background_only else "limits.md")).read_text())
    logger.info(f"combination of {len(runs)} runs: {json.dumps(result)[:200]}")


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
def publish(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, to: Optional[Path] = typer.Option(None, "--to", help="gallery directory (default: web.dir of the configuration)"), variant: Optional[str] = typer.Option(None, "--variant", help="the gallery column, [a-z0-9_]+ (default: web.variant)"), label: Optional[str] = typer.Option(None, "--label", help="label of the variant"), description: Optional[str] = typer.Option(None, "--description", help="description of the variant"), title: Optional[str] = typer.Option(None, "--title", help="title of the gallery"), about: Optional[str] = typer.Option(None, "--about", help="text of the gallery's About section")):
    """Copy the plots and their yields (of a measurement analysis: the measurement's plots) into a static web gallery (one column per variant); publishing a variant again replaces its channels."""
    from shapesmith.web import publish as publish_gallery

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
    options = {"dir": to.absolute() if to else None, "variant": variant, "label": label, "description": description, "title": title, "about": about}
    web = (cfg.web or WebConfig()).model_copy(update={key: value for key, value in options.items() if value is not None})
    publish_gallery(cfg, analysis, web, selected if channels else None, config, overrides or [])


@app.command("ml-export")
@_logged
def ml_export(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, workers: Optional[int] = WorkersOption):
    """Training folds (Feather) from the skims."""
    from shapesmith.ml_export import run_ml_export

    cfg, analysis, selected = _setup(config, channels, workers=workers, overrides=overrides)
    if cfg.ml_dir is not None:
        write_versions(cfg, analysis.name, cfg.ml_dir)
    for path in run_ml_export(cfg, analysis, selected):
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

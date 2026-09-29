"""Command-line interface: one sub-command per analysis step."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from shapesmith import __version__
from shapesmith.config import RunConfig, load_analysis, load_config
from shapesmith.provenance import write_versions

app = typer.Typer(help=f"ShapeSmith {__version__}: CROWN ntuples -> skims -> histograms -> datacards, fits, plots, ML folds, measurements.", no_args_is_help=True)

ConfigOption = typer.Option(..., "--config", "-c", help="run configuration YAML")
ChannelsOption = typer.Option(None, "--channels", help="comma separated subset of the configured channels")
WorkersOption = typer.Option(None, "--workers", help="override `workers` of the configuration (1 = inline, best for debugging)")
SetOption = typer.Option(None, "--set", "-s", help="override a configuration entry, e.g. -s switches.jet_fakes=ff -s workers=4 (repeatable, values parsed as YAML)")


def _setup(config_path: Path, channels: Optional[str], log_level: str = "INFO", workers: Optional[int] = None, overrides: Optional[list[str]] = None):
    logging.basicConfig(level=log_level, format="%(message)s", handlers=[RichHandler(show_path=False)], force=True)
    config = load_config(config_path, overrides or [])
    if workers is not None:
        config = config.model_copy(update={"workers": workers})
    analysis = load_analysis(config)
    selected = channels.split(",") if channels else list(config.channels)
    unknown = set(selected) - set(config.channels)
    if unknown:
        raise typer.BadParameter(f"channels {sorted(unknown)} are not in the configuration")
    return config, analysis, selected


def _shapes_path(config: RunConfig, control: bool) -> Path:
    return config.output_dir / ("control_shapes.root" if control else "shapes.root")


def _final_states(channels: list[str]) -> dict[str, list[str]]:
    states = {channel: [channel] for channel in channels}
    if len(channels) > 1:
        states["all"] = list(channels)
    return states


def _split(value: Optional[str]) -> Optional[list[str]]:
    return value.split(",") if value else None


@app.command()
def validate(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption):
    """Load and validate the analysis; report processes, selections and the columns every channel needs."""
    from shapesmith.skim import needed_columns

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
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
def skim(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, force: bool = typer.Option(False, "--force", help="skim again, ignoring stored skims"), samples: Optional[str] = typer.Option(None, "--samples", help="comma separated sample groups or nick prefixes (default: all)"), workers: Optional[int] = WorkersOption):
    """Stage 1: ntuples (+ friends) -> Parquet skims."""
    from shapesmith.skim import run_skim

    cfg, analysis, selected = _setup(config, channels, workers=workers, overrides=overrides)
    write_versions(cfg, analysis.name, cfg.skim_dir)
    results = run_skim(cfg, analysis, selected, force=force, samples=_split(samples))
    typer.echo(f"{sum(1 for r in results if not r.skipped)} files skimmed, {sum(1 for r in results if r.skipped)} reused, {sum(r.n_out for r in results)} events kept")


@app.command()
def hist(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, control: bool = typer.Option(False, "--control", help="control variables instead of NN categories"), variables: Optional[str] = typer.Option(None, help="comma separated control variables"), regions: Optional[str] = typer.Option(None, "--regions", help="comma separated regions, or 'all' (default: nominal plus estimator regions)"), skip_systematics: bool = typer.Option(False, "--skip-systematics", help="no weight or column variations"), processes: Optional[str] = typer.Option(None, help="comma separated process names"), workers: Optional[int] = WorkersOption):
    """Stage 2: skims -> histograms (control_shapes.root or shapes.root); other histograms in the file are kept."""
    from shapesmith.fill import run_hist

    cfg, analysis, selected = _setup(config, channels, workers=workers, overrides=overrides)
    write_versions(cfg, analysis.name, cfg.output_dir)
    output = _shapes_path(cfg, control)
    hset = run_hist(cfg, analysis, selected, control, _split(variables), not skip_systematics, _split(processes), output, _split(regions))
    typer.echo(f"{len(hset)} histograms in {output}")


@app.command()
def estimate(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, control: bool = typer.Option(False, "--control")):
    """Stage 3: add the estimated processes and variations of every channel to the histogram file."""
    from shapesmith.estimates import run_estimates
    from shapesmith.histogram import HistogramSet

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
    path = _shapes_path(cfg, control)
    hset = HistogramSet.load(path)
    added = run_estimates(hset, analysis, selected)
    hset.save(path)
    typer.echo(f"{len(added)} histograms added to {path}")


@app.command()
def sync(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption):
    """Write combine-style shape files per channel."""
    from shapesmith.histogram import HistogramSet
    from shapesmith.shapes import run_sync

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
    for path in run_sync(HistogramSet.load(_shapes_path(cfg, False)), analysis, selected, cfg.output_dir):
        typer.echo(f"wrote {path}")


@app.command()
def datacards(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, systematics: bool = typer.Option(True, "--systematics/--no-systematics"), min_background: Optional[float] = typer.Option(1.0, help="merge bins below this background yield; negative disables")):
    """Write text datacards and their shapes file per final state (each channel and the combination)."""
    from shapesmith.datacards import run_datacards
    from shapesmith.histogram import HistogramSet

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
    threshold = None if min_background is None or min_background < 0 else min_background
    for name, path in run_datacards(HistogramSet.load(_shapes_path(cfg, False)), analysis, _final_states(selected), cfg.output_dir / "datacards", systematics, threshold).items():
        typer.echo(f"{name}: {path}")


@app.command()
def fit(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, final_states: Optional[str] = typer.Option(None, help="comma separated; default: each channel and 'all'"), skip_combine: bool = typer.Option(False, "--skip-combine", help="only collect existing combine outputs")):
    """Run combine (expected limits, significance, best fit) and summarise the results."""
    from shapesmith.limits import run_limits

    cfg, _, selected = _setup(config, channels, overrides=overrides)
    states = _split(final_states) or list(_final_states(selected))
    results = run_limits(cfg, cfg.output_dir / "datacards", states, skip_combine)
    typer.echo((cfg.output_dir / "datacards" / "limits.md").read_text())
    typer.echo(f"{len(results)} final states summarised")


@app.command()
def plot(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, control: bool = typer.Option(False, "--control"), category: Optional[str] = typer.Option(None), variables: Optional[str] = typer.Option(None), region: str = typer.Option("nominal", "--region", help="histogram region to plot"), blind: bool = typer.Option(False, "--blind"), log: bool = typer.Option(False, "--log"), signal_scale: Optional[float] = typer.Option(None), normalize_by_bin_width: bool = typer.Option(False, "--normalize-by-bin-width")):
    """Prefit stack plots of control variables (--control) or NN categories."""
    from shapesmith.histogram import HistogramSet
    from shapesmith.plotting.stack import run_plot

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
    files = run_plot(HistogramSet.load(_shapes_path(cfg, control)), analysis, selected, control, category, _split(variables), cfg.output_dir / "plots", region=region, blind=blind, log=log, signal_scale=signal_scale, normalize_by_bin_width=normalize_by_bin_width)
    typer.echo(f"{len(files)} files written to {cfg.output_dir / 'plots'}")


@app.command("ml-export")
def ml_export(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, workers: Optional[int] = WorkersOption):
    """Training folds (Feather) from the skims."""
    from shapesmith.ml_export import run_ml_export

    cfg, analysis, selected = _setup(config, channels, workers=workers, overrides=overrides)
    if cfg.ml_dir is not None:
        write_versions(cfg, analysis.name, cfg.ml_dir)
    for path in run_ml_export(cfg, analysis, selected):
        typer.echo(f"wrote {path}")


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

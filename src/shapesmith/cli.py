"""Command-line interface: one sub-command per analysis step (Spec §13)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from shapesmith import __version__
from shapesmith.config import RunConfig, load_analysis, load_config, write_example_config, write_versions

app = typer.Typer(help=f"ShapeSmith {__version__}: CROWN ntuples -> skims -> histograms -> datacards, fits, plots, ML folds.", no_args_is_help=True)

ConfigOption = typer.Option(..., "--config", "-c", help="run configuration YAML")
ChannelsOption = typer.Option(None, "--channels", help="comma separated subset of the configured channels")
ForceOption = typer.Option(False, "--force", help="recreate existing outputs")
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


@app.command("example-config")
def example_config(path: Path = typer.Argument(..., help="where to write the example YAML")):
    """Write an example run configuration."""
    write_example_config(path)
    typer.echo(f"wrote {path}")


@app.command()
def inventory(database: Path = typer.Argument(..., help="KingMaker datasets.json that still knows the nicks by name"), nicks: Path = typer.Argument(..., help="text file, one produced nick per line"), output: Path = typer.Argument(..., help="inventory to write: nick and DBS path per line")):
    """Write a production inventory: for every produced nick its DBS dataset path, the key that survives database renames."""
    from shapesmith.samples import write_inventory

    names = [line.strip() for line in nicks.read_text().splitlines() if line.strip()]
    entries = write_inventory(output, database, names)
    typer.echo(f"wrote {len(entries)} entries to {output}")


@app.command()
def validate(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption):
    """Load and validate the analysis; report the columns every channel needs."""
    from shapesmith.skim import required_columns

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
    console = Console()
    console.print(f"[bold]{analysis.name}[/bold]  era {analysis.era}, {analysis.lumi_pb:g} pb^-1, switches {dict(cfg.switches)}")
    console.print(f"channels: {', '.join(selected)}   samples: {len(analysis.samples)} ({', '.join(f'{kind} {sum(s.kind == kind for s in analysis.samples)}' for kind in sorted({s.kind for s in analysis.samples}))})")
    table = Table(title="processes", show_edge=False)
    for column in ("key", "name", "kind", "group", "plot group", *[f"samples {c}" for c in selected]):
        table.add_column(column)
    for process in analysis.processes:
        table.add_row(process.key, process.name, process.kind, process.group, process.plot_group, *[str(len(analysis.samples_for(process.group, c))) for c in selected])
    console.print(table)
    for channel in selected:
        ch = analysis.channel(channel)
        console.print(f"{channel}: baseline cuts {list(ch.baseline.cuts)}, weights {list(ch.baseline.weights)}, regions {[r.name for r in ch.regions]}")
    console.print(f"categories: {[c.name for c in analysis.categories] or 'none'}   control variables: {len(analysis.control_variables)}")
    console.print(f"systematics: {len(analysis.weight_variations)} weight variations, {len(analysis.lnn)} lnN   estimator: {analysis.estimator.name + ' -> ' + analysis.estimator.output if analysis.estimator else 'none'}")
    for channel in selected:
        typer.echo(f"{channel}: {len(required_columns(analysis, channel))} columns required")
    typer.echo(f"analysis {analysis.name} is valid")


@app.command()
def skim(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, force: bool = ForceOption, samples: Optional[str] = typer.Option(None, "--samples", help="comma separated sample groups or nick prefixes (default: all)"), workers: Optional[int] = WorkersOption):
    """Stage 1: ntuples (+ friends) -> Parquet skims."""
    from shapesmith.skim import run_skim

    cfg, analysis, selected = _setup(config, channels, workers=workers, overrides=overrides)
    write_versions(cfg, cfg.skim_dir)
    results = run_skim(cfg, analysis, selected, force=force, samples=samples.split(",") if samples else None)
    typer.echo(f"{sum(1 for r in results if r.n_in)} files skimmed, {sum(1 for r in results if not r.n_in)} skipped, {sum(r.n_out for r in results)} events kept")


@app.command()
def hist(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, control: bool = typer.Option(False, "--control", help="control variables instead of NN categories"), variables: Optional[str] = typer.Option(None, help="comma separated control variables"), skip_systematics: bool = typer.Option(False, "--skip-systematics"), processes: Optional[str] = typer.Option(None, help="comma separated process keys"), force: bool = ForceOption, workers: Optional[int] = WorkersOption):
    """Stage 2: skims -> histograms (control_shapes.root or shapes.root)."""
    from shapesmith.histograms import run_hist

    cfg, analysis, selected = _setup(config, channels, workers=workers, overrides=overrides)
    write_versions(cfg, cfg.output_dir)
    hset = run_hist(cfg, analysis, selected, control, variables.split(",") if variables else None, not skip_systematics, processes.split(",") if processes else None, _shapes_path(cfg, control), force)
    typer.echo(f"{len(hset)} histograms in {_shapes_path(cfg, control)}")


@app.command()
def estimate(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, control: bool = typer.Option(False, "--control")):
    """Add the estimated jet-fake process (and embedding variations) to the histogram file."""
    from shapesmith.estimators import run_estimate
    from shapesmith.histograms import HistogramSet

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
    path = _shapes_path(cfg, control)
    hset = HistogramSet.load(path)
    added = run_estimate(hset, analysis, selected)
    hset.save(path)
    typer.echo(f"{len(added)} histograms added to {path}")


@app.command()
def sync(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption):
    """Write combine-style shape files per channel."""
    from shapesmith.histograms import HistogramSet
    from shapesmith.sync import run_sync

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
    for path in run_sync(HistogramSet.load(_shapes_path(cfg, False)), analysis, selected, cfg.output_dir):
        typer.echo(f"wrote {path}")


@app.command()
def datacards(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, systematics: bool = typer.Option(True, "--systematics/--no-systematics"), min_background: Optional[float] = typer.Option(1.0, help="merge bins below this background yield; negative disables")):
    """Write text datacards and their shapes file per final state (each channel and the combination)."""
    from shapesmith.datacards import run_datacards
    from shapesmith.histograms import HistogramSet

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
    threshold = None if min_background is None or min_background < 0 else min_background
    for name, path in run_datacards(HistogramSet.load(_shapes_path(cfg, False)), analysis, _final_states(selected), cfg.output_dir / "datacards", systematics, threshold).items():
        typer.echo(f"{name}: {path}")


@app.command()
def fit(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, final_states: Optional[str] = typer.Option(None, help="comma separated; default: each channel and 'all'"), skip_combine: bool = typer.Option(False, "--skip-combine", help="only collect existing combine outputs")):
    """Run combine (expected limits, significance, best fit) and summarise the results."""
    from shapesmith.fit import run_fit

    cfg, _, selected = _setup(config, channels, overrides=overrides)
    states = final_states.split(",") if final_states else list(_final_states(selected))
    results = run_fit(cfg, cfg.output_dir / "datacards", states, skip_combine)
    typer.echo((cfg.output_dir / "datacards" / "limits.md").read_text())
    typer.echo(f"{len(results)} final states summarised")


@app.command()
def plot(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, control: bool = typer.Option(False, "--control"), category: Optional[str] = typer.Option(None), variables: Optional[str] = typer.Option(None), blind: bool = typer.Option(False, "--blind"), log: bool = typer.Option(False, "--log"), signal_scale: Optional[float] = typer.Option(None), normalize_by_bin_width: bool = typer.Option(False, "--normalize-by-bin-width")):
    """Prefit stack plots of control variables (--control) or NN categories."""
    from shapesmith.histograms import HistogramSet
    from shapesmith.plotting.stack import run_plot

    cfg, analysis, selected = _setup(config, channels, overrides=overrides)
    files = run_plot(HistogramSet.load(_shapes_path(cfg, control)), analysis, selected, control, category, variables.split(",") if variables else None, cfg.output_dir / "plots", blind=blind, log=log, signal_scale=signal_scale, normalize_by_bin_width=normalize_by_bin_width)
    typer.echo(f"{len(files)} files written to {cfg.output_dir / 'plots'}")


@app.command("ml-export")
def ml_export(config: Path = ConfigOption, channels: Optional[str] = ChannelsOption, overrides: Optional[list[str]] = SetOption, workers: Optional[int] = WorkersOption):
    """Training folds (Feather) from the skims."""
    from shapesmith.ml_export import run_ml_export

    cfg, analysis, selected = _setup(config, channels, workers=workers, overrides=overrides)
    if cfg.ml_dir is not None:
        write_versions(cfg, cfg.ml_dir)
    for path in run_ml_export(cfg, analysis, selected):
        typer.echo(f"wrote {path}")


@app.command()
def inspect(shapes: Path = typer.Argument(..., help="a shapes ROOT file written by `hist`")):
    """List the processes, regions, variations and variables in a histogram file."""
    from shapesmith.histograms import HistogramSet

    hset = HistogramSet.load(shapes)
    for field in ("channel", "category", "process", "region", "variation", "variable"):
        values = sorted({getattr(key, field) for key in hset.keys()})
        typer.echo(f"{field:10s} ({len(values)}): {', '.join(values[:40])}{' ...' if len(values) > 40 else ''}")
    typer.echo(f"{len(hset)} histograms")


if __name__ == "__main__":
    app()

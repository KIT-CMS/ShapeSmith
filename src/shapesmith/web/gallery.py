"""`shapesmith publish`: the plots of a run and their yields in a static web gallery.

A gallery directory holds the viewer (index.html, viewer.js, viewer.css from static/), manifest.json (and the same as
manifest.js, so that the page also works from file://) and the plots in data/<variant>/<channel>/<region>/<category>/
<variable>.png|.pdf. A variant is one column of the gallery, e.g. one run configuration; publishing a variant for some
channels replaces exactly those channels of it and leaves the rest of the gallery as it is.

A run publishes its control or category plots, or, for an analysis with a measurement, what the measurement's
`gallery(output, channels)` returns (its plots on the same four axes, notes per plot and facts per channel). A gallery
holds one kind: "control" or the name of the measurement.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

from shapesmith import __version__
from shapesmith.config import VARIANT_PATTERN, RunConfig, WebConfig
from shapesmith.histogram import INCLUSIVE, NOMINAL_VARIATION, HistogramSet
from shapesmith.model import NOMINAL, Analysis
from shapesmith.plotting.stack import stack
from shapesmith.plotting.style import axis_label, label, plain_text
from shapesmith.provenance import record

logger = logging.getLogger(__name__)

SCHEMA = 1
GENERATOR = "shapesmith"
STAMP = "__SHAPESMITH_STAMP__"  # in index.html, replaced by the stamp of the manifest (cache busting)
EXTENSIONS = ("png", "pdf")
YIELD_VARIABLE = "yield"  # a control variable of this name (one bin counting every event) gives the yields of its plots
FILE_MODE, DIR_MODE = 0o644, 0o755
AXES = ("channels", "regions", "categories", "variables")

Cell = tuple[str, str, str]  # (region, category, variable)
CONTROL = "control"


@dataclass
class Content:
    """What one publish adds to a gallery: per channel the files of each (region, category, variable) cell, the
    labels and the order of the axes and optionally the yields of the cells, a note per plot and facts per channel."""

    kind: str  # CONTROL or the name of the measurement; a gallery holds one kind
    plots: dict[str, dict[Cell, dict[str, Path]]]  # channel -> cell -> {extension: path}
    labels: dict[str, dict[str, str]]  # axis -> key -> plain-text label
    order: dict[str, list[str]]  # axis -> keys in their order
    axes: dict[str, str] = field(default_factory=dict)  # axis -> its title in the viewer (default: Channel, Region, ...)
    yields: dict = field(default_factory=dict)  # channel -> region -> category -> {variable, data, prediction, groups}
    notes: dict = field(default_factory=dict)  # channel -> region -> category -> variable -> text shown with the plot
    facts: dict = field(default_factory=dict)  # channel -> [{key, label, value}], e.g. fitted factors


def _static() -> Traversable:
    """The viewer files, shipped as package data."""
    return resources.files("shapesmith.web") / "static"


def _gallery(target: Path) -> dict | None:
    """The manifest of the gallery in `target`, or None for a new (missing or empty) directory. Any other directory
    is refused, so that publishing never writes into a directory that is not a gallery."""
    if not target.exists():
        return None
    if not target.is_dir():
        raise NotADirectoryError(f"{target} is not a directory")
    path = target / "manifest.json"
    if path.is_file():
        try:
            manifest = json.loads(path.read_text())
        except (OSError, ValueError):
            manifest = None
        if isinstance(manifest, dict) and str(manifest.get("generator", "")).startswith(GENERATOR):
            if manifest.get("schema") != SCHEMA:
                raise ValueError(f"{path} has schema {manifest.get('schema')}, this ShapeSmith writes schema {SCHEMA}")
            return manifest
    elif not any(target.iterdir()):
        return None
    raise FileExistsError(f"{target} is not empty and not a ShapeSmith gallery (no manifest.json written by shapesmith); nothing published")


def _plots(analysis: Analysis, channel: str, directory: Path) -> dict[Cell, dict[str, Path]]:
    """The plot files of a channel, (region, category, variable) -> {extension: path}, in analysis order. File names
    are matched against the channel's categories and variables, whose names contain underscores themselves."""
    ch = analysis.channel(channel)
    names = {f"{INCLUSIVE}_{v}": (INCLUSIVE, v) for v in ch.variables}
    names.update({f"{c.name}_{c.variable.name}": (c.name, c.variable.name) for c in ch.categories})
    folders = {NOMINAL: directory, **{r.name: directory / r.name for r in ch.regions}}
    found, used = {}, set()
    for region, folder in folders.items():
        files = {p.name: p for p in folder.iterdir() if p.is_file()} if folder.is_dir() else {}
        for name, (category, variable) in names.items():
            for extension in EXTENSIONS:
                path = files.get(f"{name}.{extension}")
                if path is not None:
                    found.setdefault((region, category, variable), {})[extension] = path
                    used.add(path)
    unknown = [p for p in directory.iterdir() if p not in used and not (p.is_dir() and p.name in folders)]
    unknown += [p for region, folder in folders.items() if region != NOMINAL and folder.is_dir() for p in folder.iterdir() if p not in used]
    if unknown:
        logger.warning(f"{channel}: {len(unknown)} entries of {directory} are no plots of the analysis and are not published, e.g. {sorted(unknown)[0]}")
    return found


def _yields(config: RunConfig, analysis: Analysis, plots: dict[str, dict[Cell, dict[str, Path]]]) -> dict:
    """{channel: {region: {category: yields}}}: the data, the prediction and the stacked groups of one plot per
    region and category (the `yield` variable if there is one), from the histograms the plots are made of."""
    cells = {}
    for channel, files in plots.items():
        for region, category, variable in files:
            cells.setdefault((channel, region, category), []).append(variable)
    chosen = {cell: YIELD_VARIABLE if YIELD_VARIABLE in variables else variables[0] for cell, variables in cells.items()}
    result = {}
    for control in (True, False):
        wanted = [(channel, category, region, variable) for (channel, region, category), variable in chosen.items() if (category == INCLUSIVE) == control]
        path = Path(config.output_dir) / ("control_shapes.root" if control else "shapes.root")
        if not wanted:
            continue
        if not path.is_file():
            logger.warning(f"{path} is missing: no yields for {len(wanted)} plots")
            continue
        keys = set(wanted)
        hset = HistogramSet.load(path, keep=lambda key: key.variation == NOMINAL_VARIATION and (key.channel, key.category, key.region, key.variable) in keys)
        for channel, category, region, variable in wanted:
            shown = stack(hset, analysis, channel, category, variable, region)
            if not shown.groups:
                logger.warning(f"{channel}/{category}/{region}/{variable}: no histograms in {path}, no yields")
                continue
            result.setdefault(channel, {}).setdefault(region, {})[category] = {
                "variable": variable,
                "data": shown.data.sum() if shown.data is not None else None,
                "prediction": float(shown.background.sum()),
                "groups": [{"key": group, "label": plain_text(label(analysis, group)), "yield": h.sum()} for group, h in shown.groups],
            }
    return result


def _mkdir(path: Path) -> None:
    """The directory and its missing parents, world-readable whatever the umask."""
    missing = []
    while not path.exists():
        missing.append(path)
        path = path.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        os.chmod(directory, DIR_MODE)


def _write(path: Path, data: bytes) -> None:
    """Replace the file atomically (temporary file + rename), world-readable; a file with this content is kept."""
    if path.is_file() and not path.is_symlink() and path.read_bytes() == data:
        if stat.S_IMODE(path.stat().st_mode) != FILE_MODE:
            os.chmod(path, FILE_MODE)
        return
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
    try:
        with handle:
            handle.write(data)
        os.chmod(handle.name, FILE_MODE)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _copy(source: Path, destination: Path) -> bool:
    """Copy a plot where the destination is missing or differs in size or modification time; True if copied."""
    original = source.stat()
    if destination.is_file():
        current = destination.stat()
        if current.st_size == original.st_size and int(current.st_mtime) == int(original.st_mtime):
            if stat.S_IMODE(current.st_mode) != FILE_MODE:
                os.chmod(destination, FILE_MODE)
            return False
    _mkdir(destination.parent)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    os.chmod(temporary, FILE_MODE)
    os.utime(temporary, ns=(original.st_atime_ns, original.st_mtime_ns))
    os.replace(temporary, destination)
    return True


def _links(target: Path, root: Path) -> list[Path]:
    """The symbolic links between the gallery directory and `root` and below it: copying, chmod and removing plots
    through them would reach outside the gallery."""
    links = [p for p in (root, *root.parents) if target in p.parents and p.is_symlink()]
    for directory, subdirectories, files in os.walk(root):
        links += [Path(directory) / name for name in subdirectories + files if (Path(directory) / name).is_symlink()]
    return links


def _remove_stale(root: Path, keep: set[Path]) -> int:
    """Delete everything below `root` that is not in `keep` (symbolic links themselves, never their targets) and
    the directories left empty; the number of files deleted."""
    removed = 0
    for directory, subdirectories, files in os.walk(root, topdown=False):
        for name in files:
            path = Path(directory) / name
            if path not in keep:
                path.unlink()
                removed += 1
        for name in subdirectories:
            path = Path(directory) / name
            if path.is_symlink():
                path.unlink()
            elif not any(path.iterdir()):
                path.rmdir()
    return removed


def merge_order(existing: list[str], new: list[str]) -> list[str]:
    """The existing keys in their order, each other key of `new` placed before the next key that follows it there,
    else after the key that precedes it there, else at the end."""
    merged = list(existing)
    for i, key in enumerate(new):
        if key in merged:
            continue
        following = next((k for k in new[i + 1:] if k in merged), None)
        preceding = next((k for k in reversed(new[:i]) if k in merged), None)
        if following is not None:
            merged.insert(merged.index(following), key)
        else:
            merged.insert(merged.index(preceding) + 1 if preceding is not None else len(merged), key)
    return merged


def _used(plots: dict) -> dict[str, set[str]]:
    """The channels, regions, categories and variables that a plots tree of a manifest refers to."""
    used = {axis: set() for axis in AXES}
    for channels in plots.values():
        for channel, regions in channels.items():
            used["channels"].add(channel)
            for region, categories in regions.items():
                used["regions"].add(region)
                for category, variables in categories.items():
                    used["categories"].add(category)
                    used["variables"].update(variables)
    return used


def _labels(analysis: Analysis, plots: dict[str, dict[Cell, dict[str, Path]]]) -> dict[str, dict[str, str]]:
    """Key -> plain-text label of the channels, regions, categories and variables published. A variable whose axis
    label differs between the channels is labelled by its name."""
    style = analysis.style
    labels = {axis: {} for axis in AXES}
    variable_labels = {}
    for channel, files in plots.items():
        labels["channels"][channel] = plain_text(style.channel_labels.get(channel, channel)) if style else channel
        for region, category, variable in files:
            labels["regions"][region] = region
            labels["categories"][category] = category
            variable_labels.setdefault(variable, set()).add(plain_text(axis_label(analysis, channel, variable)))
    labels["variables"] = {v: texts.pop() if len(texts) == 1 else v for v, texts in variable_labels.items()}
    return labels


def _order(analysis: Analysis) -> dict[str, list[str]]:
    """The channels, regions, categories and variables of the analysis in its order."""
    order = {"channels": list(analysis.channels), "regions": [NOMINAL], "categories": [INCLUSIVE], "variables": []}
    for channel in analysis.channels.values():
        order["regions"] += [r.name for r in channel.regions]
        order["categories"] += [c.name for c in channel.categories]
        order["variables"] += [*channel.variables, *(c.variable.name for c in channel.categories)]
    return {axis: list(dict.fromkeys(keys)) for axis, keys in order.items()}


def _control_content(config: RunConfig, analysis: Analysis, channels: list[str] | None) -> Content:
    """The control and category plots in <output_dir>/plots/ of `channels` (default: the configured channels with
    plots) and their yields."""
    directory = Path(config.output_dir) / "plots"
    if channels is None:
        channels = [c for c in config.channels if (directory / c).is_dir()]
    if not channels:
        raise FileNotFoundError(f"no plots in {directory}; make them with `shapesmith plot`")
    plots = {}
    for channel in channels:
        plots[channel] = _plots(analysis, channel, directory / channel) if (directory / channel).is_dir() else {}
        if not plots[channel]:
            raise FileNotFoundError(f"no plots of channel {channel} in {directory / channel}; make them with `shapesmith plot`")
    return Content(CONTROL, plots, _labels(analysis, plots), _order(analysis), yields=_yields(config, analysis, plots))


def _measurement_content(output: Path, analysis: Analysis, channels: list[str] | None) -> Content:
    """What the analysis's measurement publishes from its output directory."""
    gallery = getattr(analysis.measurement, "gallery", None)
    if gallery is None:
        raise ValueError(f"measurement {analysis.measurement.name} has no gallery(): its results cannot be published")
    content = gallery(output, channels)
    if not any(content.plots.values()):
        raise FileNotFoundError(f"no plots of measurement {analysis.measurement.name} in {output}; run `shapesmith measure` first")
    return content


def _manifest(previous: dict | None, name: str, web: WebConfig, source: dict, content: Content, updated: str) -> dict:
    """The new manifest: the previous one with the published channels of the variant replaced."""
    previous = previous or {}
    plots = content.plots
    variants = [dict(v) for v in previous.get("variants", [])]
    variant = next((v for v in variants if v["key"] == web.variant), None)
    if variant is None:
        variant = {"key": web.variant, "label": None, "description": "", "sources": []}
        variants.append(variant)
    variant["label"] = web.label or variant["label"] or web.variant.replace("_", " ")
    variant["description"] = web.description if web.description is not None else variant["description"]
    sources = [{**s, "channels": [c for c in s["channels"] if c not in plots]} for s in variant["sources"]]
    variant["sources"] = [s for s in sources if s["channels"]] + [source]

    tree, numbers = previous.get("plots", {}), previous.get("yields", {})
    extras = {key: previous.get(key, {}) for key in ("notes", "facts")}
    for channel, files in plots.items():
        branch = tree.setdefault(web.variant, {})[channel] = {}
        for (region, category, variable), paths in files.items():
            branch.setdefault(region, {}).setdefault(category, {})[variable] = [e for e in EXTENSIONS if e in paths]
        numbers.setdefault(web.variant, {})[channel] = content.yields.get(channel, {})
        for key, values in extras.items():
            values.setdefault(web.variant, {})[channel] = getattr(content, key).get(channel) or {}

    labels, order = content.labels, content.order
    others = _used({v: {c: regions for c, regions in channels.items() if v != web.variant or c not in plots} for v, channels in tree.items()})
    used = _used(tree)
    manifest = {
        "schema": SCHEMA,
        "generator": f"{GENERATOR} {__version__}",
        "kind": content.kind,
        "title": web.title or previous.get("title") or name,
        "description": web.about if web.about is not None else previous.get("description", ""),
        "updated": updated,
    }
    axes = {**previous.get("axes", {}), **content.axes}
    if axes:
        manifest["axes"] = axes
    for axis in AXES:
        texts = {entry["key"]: entry["label"] for entry in previous.get(axis, [])}
        for key, text in labels[axis].items():
            conflict = axis == "variables" and key in texts and texts[key] != text and key in others[axis]
            texts[key] = key if conflict else text
        keys = merge_order([entry["key"] for entry in previous.get(axis, [])], order[axis])
        keys += sorted(used[axis] - set(keys))
        if axis == "regions":
            keys.sort(key=lambda key: key != NOMINAL)
        manifest[axis] = [{"key": key, "label": texts.get(key, key)} for key in keys if key in used[axis]]
    manifest.update(variants=variants, plots=tree, yields=numbers)
    for key, values in extras.items():  # only galleries with notes or facts carry them
        values = {variant: {c: v for c, v in channels.items() if v} for variant, channels in values.items()}
        if any(values.values()):
            manifest[key] = {variant: channels for variant, channels in values.items() if channels}
    return manifest


def publish(config: RunConfig, analysis: Analysis, web: WebConfig, channels: list[str] | None = None, config_path: Path | None = None, overrides: list[str] | tuple[str, ...] = ()) -> Path:
    """Publish the plots of `channels` (default: the configured channels with plots) and their yields as variant
    `web.variant` of the gallery in `web.dir`; returns the manifest. An analysis with a measurement publishes the
    measurement's gallery content instead. Copies only changed plots, removes stale ones of the published channels of
    the variant only, and refuses a directory that is neither empty nor a gallery, and a gallery of another kind."""
    if web.dir is None or web.variant is None:
        raise ValueError("publish needs a gallery directory and a variant: --to and --variant, or `web: {dir: ..., variant: ...}` in the run configuration")
    if not re.fullmatch(VARIANT_PATTERN, web.variant):
        raise ValueError(f"variant {web.variant!r} must match {VARIANT_PATTERN}")
    target = Path(web.dir).absolute()
    previous = _gallery(target)
    static = _static()
    viewer = [f for f in static.iterdir() if f.is_file()] if static.is_dir() else []
    if "index.html" not in {f.name for f in viewer}:
        raise FileNotFoundError(f"the viewer is missing from the shapesmith package ({static})")
    if analysis.measurement is None:
        base = Path(config.output_dir)
        content = _control_content(config, analysis, channels)
    else:
        base = Path(config.output_dir) / analysis.measurement.name / analysis.era
        content = _measurement_content(base, analysis, channels)
    kind = (previous or {}).get("kind", CONTROL if previous else content.kind)
    if kind != content.kind:
        raise ValueError(f"{target} is a gallery of {kind} plots, not of {content.kind}: publish into a gallery of its own; nothing published")
    plots = content.plots
    links = [link for channel in plots for link in _links(target, target / "data" / web.variant / channel)]
    if links:
        raise FileExistsError(f"{len(links)} symbolic links in the plot directories of the gallery, e.g. {links[0]}: publishing would write or delete through them outside {target}; nothing published")

    now = datetime.now()
    stamp = now.strftime("%Y%m%dT%H%M%S")
    versions = base / "versions" / f"{analysis.name}.json"
    source = {
        "channels": list(plots),
        "config": str(Path(config_path).absolute()) if config_path else None,
        "overrides": list(overrides),
        "output_dir": str(Path(config.output_dir).absolute()),
        "published": now.isoformat(timespec="seconds"),
        "stamp": stamp,
        "provenance": json.loads(versions.read_text()) if versions.is_file() else record(config),
    }
    manifest = _manifest(previous, analysis.name, web, source, content, now.isoformat(timespec="seconds"))

    _mkdir(target)
    os.chmod(target, DIR_MODE)
    stale = {}
    for channel, files in plots.items():
        root = target / "data" / web.variant / channel
        keep = {root / region / category / f"{variable}.{e}": path for (region, category, variable), paths in files.items() for e, path in paths.items()}
        copied = sum(_copy(path, destination) for destination, path in keep.items())
        stale[channel] = (root, set(keep))
        logger.info(f"{channel}: {len(files)} plots ({copied} files copied) -> {root}")
    for file in viewer:
        content = file.read_bytes()
        _write(target / file.name, content.replace(STAMP.encode(), stamp.encode()) if file.name == "index.html" else content)
    text = json.dumps(manifest, indent=1)
    _write(target / "manifest.json", text.encode())
    _write(target / "manifest.js", f"window.SHAPESMITH_GALLERY = {text};\n".encode())
    for channel, (root, keep) in stale.items():
        removed = _remove_stale(root, keep)
        if removed:
            logger.info(f"{channel}: {removed} stale files removed from {root}")
    logger.info(f"variant {web.variant} ({', '.join(plots)}) published to {target}/index.html")
    return target / "manifest.json"

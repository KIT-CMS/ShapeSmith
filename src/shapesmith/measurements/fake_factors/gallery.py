"""The fake-factor measurement in a web gallery (`shapesmith publish`): its plots by process (the region axis of the
gallery), category of the split and quantity (the variable axis), the p-value of every correction as the note of its
plot and the data/MC factors of the MC fake factors as facts of the channel. Walks the measurement model, so the plots
come in table order and no file name is parsed."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from shapesmith.measurements.fake_factors.model import FakeFactorMeasurement, Split
from shapesmith.measurements.fake_factors.plots import plot_path
from shapesmith.web.gallery import Content, merge_order

logger = logging.getLogger(__name__)

AXES = {"regions": "Process", "categories": "Category", "variables": "Quantity"}
DR_SR_STAGE = "for_DRtoSR/"  # the record prefix of the orthogonal fake factors and their non-closures


def category_key(split: Split, category: int) -> str:
    """A path-safe key of a category of the split from its variable and edges, e.g. n_jets_1p5_2p5; the open last
    category has no upper edge (n_jets_2p5_up), so that it has one key whatever the last edge."""
    def number(x: float) -> str:
        return f"{x:g}".replace("-", "m").replace(".", "p")

    last = category == len(split.edges) - 2
    return f"{split.variable}_{number(split.edges[category])}_{'up' if last else number(split.edges[category + 1])}"


def category_label(split: Split, category: int) -> str:
    """"n_jets = 2", "n_jets ≥ 3", ...: half-integer edges bin an integer variable; the last category is open."""
    low, high = split.edges[category], split.edges[category + 1]
    last = category == len(split.edges) - 2
    if low % 1 == 0.5 and high % 1 == 0.5:
        low, high = math.ceil(low), math.floor(high)
        if last:
            return f"{split.variable} ≥ {low}"
        return f"{split.variable} = {low}" if low == high else f"{low} ≤ {split.variable} ≤ {high}"
    return f"{split.variable} ≥ {low:g}" if last else f"{low:g} ≤ {split.variable} < {high:g}"


def _note(entry: dict) -> str:
    """The p-value of a correction against 1 and whether it was set to 1."""
    return f"p = {entry['p_value']:.2g}" + (", set to 1" if entry.get("reset") else "")


def content(measurement: FakeFactorMeasurement, output: Path, channels: list[str] | None) -> Content:
    """The gallery content of the measurement in `output` (<output_dir>/fake_factors/<era>/) for `channels`
    (default: every channel of the measurement with a record there)."""
    if channels is None:
        channels = [c for c in measurement.legs if (output / f"measurement_{c}.json").is_file()]
    result = Content("fake_factors", {}, {"channels": {}, **{axis: {} for axis in AXES}}, {"channels": [], **{axis: [] for axis in AXES}}, dict(AXES))

    for channel in channels:
        path = output / f"measurement_{channel}.json"
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing: measure channel {channel} with `shapesmith measure` first")
        record = json.loads(path.read_text())
        directory = output / "plots" / channel
        files, notes, used = {}, {}, set()
        order = {axis: [] for axis in AXES}

        def add(group: str, quantity: str, quantity_label: str, name: str, split: Split, correction: bool) -> None:
            for entry in record.get(name, []):
                category = entry.get("category", 0)
                plot = plot_path(directory, name, category)
                if not plot.is_file():
                    logger.warning(f"{channel}: {plot} of {name} category {category} is missing, not published")
                    continue
                key = category_key(split, category)
                files[(group, key, quantity)] = {"png": plot}
                used.add(plot)
                result.labels["regions"][group] = group.replace("_subleading", " subleading").replace("_for_DRtoSR", " for DR→SR")
                result.labels["categories"][key] = category_label(split, category)
                result.labels["variables"][quantity] = quantity_label
                for axis, value in (("regions", group), ("categories", key), ("variables", quantity)):
                    order[axis].append(value)
                if correction and entry.get("p_value") is not None:
                    notes.setdefault(group, {}).setdefault(key, {})[quantity] = _note(entry)

        facts = []
        for leg in measurement.legs[channel]:
            for process in leg.processes:
                name = process.name + leg.suffix
                add(name, "fake_factors", "fake factors", f"{name}_fake_factors", process.split, False)
                if process.dr_sr is not None:
                    add(name, "DR_SR", "DR→SR correction", f"{name}_DR_SR", process.split, True)
                for binned in process.non_closures:
                    add(name, f"non_closure_{binned.variable}", f"non-closure in {binned.variable}", f"{name}_non_closure_{binned.variable}", process.split, True)
                if process.dr_sr is not None:
                    group = f"{name}_for_DRtoSR"
                    add(group, "fake_factors", "fake factors", f"{DR_SR_STAGE}{name}_fake_factors", process.split, False)
                    for binned in process.dr_sr.non_closures:
                        add(group, f"non_closure_{binned.variable}", f"non-closure in {binned.variable}", f"{DR_SR_STAGE}{name}_non_closure_{binned.variable}", process.split, True)
                for stage in ("", DR_SR_STAGE):
                    for entry in record.get(f"{stage}{name}_data_scale", []):
                        facts.append({"key": f"{stage}{name}_data_scale".replace("/", "_"), "label": f"{result.labels['regions'].get(name, name)}{' for DR→SR' if stage else ''} data/MC factor", "value": f"{entry['factor']:.3f}"})
            add(f"fractions{leg.suffix}", "process_fractions", "QCD and ttbar fractions", f"process_fractions{leg.suffix}", leg.fractions.split, False)

        unknown = sorted(p for p in directory.iterdir() if p not in used) if directory.is_dir() else []
        if unknown:
            logger.warning(f"{channel}: {len(unknown)} files of {directory} belong to no plot of the measurement and are not published, e.g. {unknown[0]}")
        result.plots[channel] = files
        result.labels["channels"][channel] = channel
        result.order["channels"].append(channel)
        result.notes[channel] = notes
        result.facts[channel] = facts
        for axis, keys in order.items():
            result.order[axis] = merge_order(result.order[axis], list(dict.fromkeys(keys)))
    return result

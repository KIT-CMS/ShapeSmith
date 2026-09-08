"""Native combine datacards: text card + shapes file, optional rebinning (Spec §10.2).

One card per final state lists every bin (channel x category); processes with a non-positive
rate in a bin are left out of that bin. Systematics: lnN from Analysis.lnn, `shape` for every
weight variation present, `* autoMCStats 0`.
"""
from __future__ import annotations

from pathlib import Path

import uproot

from shapesmith.histogram import Histogram
from shapesmith.histograms import NOMINAL_REGION, NOMINAL_VARIATION, HistKey, HistogramSet
from shapesmith.model import Analysis, LnN


def bin_name(analysis: Analysis, channel: str, category_index: int) -> str:
    return f"htt_{channel}_{category_index + 1}_{analysis.era}"


def shape_processes(analysis: Analysis) -> tuple[str, ...]:
    """Processes that carry shape systematics: simulated backgrounds and the signal."""
    excluded = {analysis.estimator.output} if analysis.estimator else set()
    excluded |= {p.name for p in analysis.processes_of_kind("embedding")}
    return (*(p for p in analysis.backgrounds() if p not in excluded), analysis.signal)


def rebin_edges(total_background: Histogram, min_background: float | None) -> list[float]:
    """Merge bins from the right until every bin holds at least `min_background`."""
    edges = list(total_background.edges)
    if min_background is None:
        return edges
    values = list(total_background.values)
    while len(values) > 1 and values[-1] < min_background:
        last = values.pop()
        values[-1] += last
        edges.pop(-2)
    for i in range(len(values) - 2, -1, -1):  # remaining low bins merge into their right neighbour
        if values[i] < min_background and len(values) > 1:
            low = values.pop(i)
            values[i] += low
            edges.pop(i + 1)
    return edges


def _value(lnn: LnN) -> str:
    if isinstance(lnn.value, tuple):
        down, up = lnn.value
        return f"{down:g}/{up:g}"
    return f"{lnn.value:g}"


def lnn_lines(analysis: Analysis, bins: list[tuple[str, str]], processes_per_bin: dict[str, list[str]]) -> list[str]:
    """One `<name> lnN ...` line per concrete nuisance name ($CHANNEL/$ERA expanded), columns = (bin, process) pairs."""
    lines = []
    for lnn in analysis.lnn:
        concrete_names = sorted({lnn.name.replace("$CHANNEL", channel).replace("$ERA", analysis.era) for channel, _ in bins})
        for name in concrete_names:
            columns = []
            for channel, bin_ in bins:
                this_name = lnn.name.replace("$CHANNEL", channel).replace("$ERA", analysis.era)
                for process in processes_per_bin[bin_]:
                    applies = this_name == name and (lnn.channels is None or channel in lnn.channels) and ("*" in lnn.processes or process in lnn.processes)
                    columns.append(_value(lnn) if applies else "-")
            if any(column != "-" for column in columns):
                lines.append(f"{name} lnN " + " ".join(columns))
    return lines


def write_datacard(hset: HistogramSet, analysis: Analysis, channels: list[str], output_dir: Path, systematics: bool = True, min_background: float | None = 1.0) -> Path:
    output_dir = Path(output_dir)
    (output_dir / "common").mkdir(parents=True, exist_ok=True)
    shapes_file = output_dir / "common" / f"htt_input_{analysis.era}.root"
    processes = (analysis.signal, *analysis.backgrounds())
    bins: list[tuple[str, str]] = []
    processes_per_bin: dict[str, list[str]] = {}
    observations: dict[str, float] = {}
    variations_per_bin: dict[str, dict[str, set[str]]] = {}
    with uproot.recreate(shapes_file) as f:
        for channel in channels:
            for index, category in enumerate(analysis.categories):
                bin_ = bin_name(analysis, channel, index)
                variable = category.variable.name

                def key_of(process, variation=NOMINAL_VARIATION):
                    return HistKey(channel, category.name, process, NOMINAL_REGION, variation, variable)

                nominal = {p: hset.get(key_of(p)) for p in processes if hset.has(key_of(p))}
                if not nominal or not hset.has(key_of("data")):
                    continue
                total = None
                for process, h in nominal.items():
                    if process == analysis.signal:
                        continue
                    total = h.copy() if total is None else total.add(h)
                edges = rebin_edges(total, min_background) if total is not None else list(hset.get(key_of("data")).edges)
                data = hset.get(key_of("data")).rebin(edges)
                f[f"{bin_}/data_obs"] = data.to_root("data_obs")
                observations[bin_] = data.sum()
                kept = []
                variations_per_bin[bin_] = {}
                for process, h in nominal.items():
                    rebinned = h.rebin(edges)
                    if rebinned.sum() <= 0.0:
                        continue
                    kept.append(process)
                    f[f"{bin_}/{process}"] = rebinned.to_root(process)
                    variations_per_bin[bin_][process] = set()
                    for key in hset.keys(channel=channel, category=category.name, process=process, region=NOMINAL_REGION, variable=variable):
                        if key.variation != NOMINAL_VARIATION:
                            f[f"{bin_}/{process}_{key.variation}"] = hset.get(key).rebin(edges).to_root(f"{process}_{key.variation}")
                            variations_per_bin[bin_][process].add(key.variation)
                bins.append((channel, bin_))
                processes_per_bin[bin_] = kept
    lines = ["imax * number of bins", "jmax * number of processes minus 1", "kmax * number of nuisance parameters", "-" * 60]
    for _, bin_ in bins:
        lines.append(f"shapes * {bin_} common/htt_input_{analysis.era}.root {bin_}/$PROCESS {bin_}/$PROCESS_$SYSTEMATIC")
    lines.append("-" * 60)
    lines.append("bin " + " ".join(bin_ for _, bin_ in bins))
    lines.append("observation " + " ".join(f"{observations[bin_]:.1f}" for _, bin_ in bins))
    lines.append("-" * 60)
    columns = [(bin_, process) for _, bin_ in bins for process in processes_per_bin[bin_]]
    lines.append("bin " + " ".join(bin_ for bin_, _ in columns))
    lines.append("process " + " ".join(process for _, process in columns))
    lines.append("process " + " ".join("0" if process == analysis.signal else str(processes.index(process)) for _, process in columns))
    lines.append("rate " + " ".join("-1" for _ in columns))
    lines.append("-" * 60)
    if systematics:
        lines += lnn_lines(analysis, bins, processes_per_bin)
        systematic_names = sorted({v[: -len("Up")] for per_bin in variations_per_bin.values() for vs in per_bin.values() for v in vs if v.endswith("Up")})
        for name in systematic_names:
            entries = ["1" if {f"{name}Up", f"{name}Down"} <= variations_per_bin[bin_].get(process, set()) else "-" for bin_, process in columns]
            if any(entry == "1" for entry in entries):
                lines.append(f"{name} shape " + " ".join(entries))
    lines.append("* autoMCStats 0")
    card = output_dir / "combined.txt"
    card.write_text("\n".join(lines) + "\n")
    return card


def run_datacards(hset: HistogramSet, analysis: Analysis, final_states: dict[str, list[str]], output_dir: Path, systematics: bool, min_background: float | None) -> dict[str, Path]:
    return {name: write_datacard(hset, analysis, channels, Path(output_dir) / name, systematics, min_background) for name, channels in final_states.items()}

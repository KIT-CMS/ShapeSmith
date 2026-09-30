"""Native combine datacards: text card + shapes file, optional rebinning.

One card per final state lists every bin (channel x category); processes with a non-positive rate in a bin are left
out of that bin. Systematics: lnN from Analysis.lnn, `shape` for every Up/Down pair of variations present (names
without a direction are templates and never become shape lines), `* autoMCStats 0`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shapesmith.histogram import NOMINAL_VARIATION, HistKey, Histogram, HistogramSet
from shapesmith.model import NOMINAL, Analysis, Category, LnN
from shapesmith.shapes import shape_name, write_shapes


@dataclass
class Bin:
    channel: str
    name: str
    data: Histogram
    processes: dict[str, Histogram]  # the processes with a positive rate, rebinned, in card order
    variations: dict[str, dict[str, Histogram]]  # process -> variation -> rebinned histogram


def bin_name(analysis: Analysis, channel: str, category_index: int) -> str:
    return f"htt_{channel}_{category_index + 1}_{analysis.era}"


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


def collect_bin(hset: HistogramSet, analysis: Analysis, channel_name: str, index: int, category: Category, min_background: float | None) -> Bin | None:
    """The rebinned data, processes and variations of one bin; None without data or processes."""
    channel = analysis.channel(channel_name)
    variable = category.variable.name

    def key_of(process: str, variation: str = NOMINAL_VARIATION) -> HistKey:
        return HistKey(channel_name, category.name, process, NOMINAL, variation, variable)

    nominal = {p: hset[key_of(p)] for p in (analysis.signal, *channel.backgrounds()) if key_of(p) in hset}
    if not nominal or key_of(channel.data()) not in hset:
        return None
    total = None
    for process, h in nominal.items():
        if process != analysis.signal:
            total = h.copy() if total is None else total.add(h)
    edges = rebin_edges(total, min_background) if total is not None else list(hset[key_of(channel.data())].edges)
    result = Bin(channel_name, bin_name(analysis, channel_name, index), hset[key_of(channel.data())].rebin(edges), {}, {})
    for process, h in nominal.items():
        rebinned = h.rebin(edges)
        if rebinned.sum() <= 0.0:
            continue
        result.processes[process] = rebinned
        keys = hset.select(channel=channel_name, category=category.name, process=process, region=NOMINAL, variable=variable)
        result.variations[process] = {key.variation: hset[key].rebin(edges) for key in keys if key.variation != NOMINAL_VARIATION}
    return result


def _value(lnn: LnN) -> str:
    if isinstance(lnn.value, tuple):
        down, up = lnn.value
        return f"{down:g}/{up:g}"
    return f"{lnn.value:g}"


def lnn_lines(analysis: Analysis, bins: list[Bin]) -> list[str]:
    """One `<name> lnN ...` line per concrete nuisance name ($CHANNEL/$ERA expanded), columns = (bin, process) pairs."""
    lines = []
    for lnn in analysis.lnn:
        concrete_names = sorted({lnn.name.replace("$CHANNEL", b.channel).replace("$ERA", analysis.era) for b in bins})
        for name in concrete_names:
            columns = []
            for b in bins:
                this_name = lnn.name.replace("$CHANNEL", b.channel).replace("$ERA", analysis.era)
                for process in b.processes:
                    applies = this_name == name and (lnn.channels is None or b.channel in lnn.channels) and ("*" in lnn.processes or process in lnn.processes)
                    columns.append(_value(lnn) if applies else "-")
            if any(column != "-" for column in columns):
                lines.append(f"{name} lnN " + " ".join(columns))
    return lines


def shape_lines(bins: list[Bin]) -> list[str]:
    columns = [(b, process) for b in bins for process in b.processes]
    names = sorted({v[: -len("Up")] for b in bins for variations in b.variations.values() for v in variations if v.endswith("Up")})
    lines = []
    for name in names:
        entries = ["1" if {f"{name}Up", f"{name}Down"} <= set(b.variations[process]) else "-" for b, process in columns]
        if "1" in entries:
            lines.append(f"{name} shape " + " ".join(entries))
    return lines


def card_text(analysis: Analysis, bins: list[Bin], systematics: bool) -> str:
    backgrounds = list(dict.fromkeys(p for b in bins for p in analysis.channel(b.channel).backgrounds()))
    index = {name: i for i, name in enumerate((analysis.signal, *backgrounds))}
    columns = [(b.name, process) for b in bins for process in b.processes]
    lines = ["imax * number of bins", "jmax * number of processes minus 1", "kmax * number of nuisance parameters", "-" * 60]
    lines += [f"shapes * {b.name} common/htt_input_{analysis.era}.root {b.name}/$PROCESS {b.name}/$PROCESS_$SYSTEMATIC" for b in bins]
    lines.append("-" * 60)
    lines.append("bin " + " ".join(b.name for b in bins))
    lines.append("observation " + " ".join(f"{b.data.sum():.1f}" for b in bins))
    lines.append("-" * 60)
    lines.append("bin " + " ".join(name for name, _ in columns))
    lines.append("process " + " ".join(process for _, process in columns))
    lines.append("process " + " ".join(str(index[process]) for _, process in columns))
    lines.append("rate " + " ".join("-1" for _ in columns))
    lines.append("-" * 60)
    if systematics:
        lines += lnn_lines(analysis, bins) + shape_lines(bins)
    lines.append("* autoMCStats 0")
    return "\n".join(lines) + "\n"


def write_datacard(hset: HistogramSet, analysis: Analysis, channels: list[str], output_dir: Path, systematics: bool = True, min_background: float | None = 1.0) -> Path:
    output_dir = Path(output_dir)
    bins = [
        b
        for channel in channels
        for index, category in enumerate(analysis.channel(channel).categories)
        if (b := collect_bin(hset, analysis, channel, index, category, min_background)) is not None
    ]
    entries = []
    for b in bins:
        entries.append((b.name, "data_obs", b.data))
        for process, h in b.processes.items():
            entries.append((b.name, process, h))
            entries += [(b.name, shape_name(process, variation), varied) for variation, varied in b.variations[process].items()]
    write_shapes(output_dir / "common" / f"htt_input_{analysis.era}.root", entries)
    card = output_dir / "combined.txt"
    card.write_text(card_text(analysis, bins, systematics))
    return card


def run_datacards(hset: HistogramSet, analysis: Analysis, final_states: dict[str, list[str]], output_dir: Path, systematics: bool, min_background: float | None) -> dict[str, Path]:
    return {name: write_datacard(hset, analysis, channels, Path(output_dir) / name, systematics, min_background) for name, channels in final_states.items()}

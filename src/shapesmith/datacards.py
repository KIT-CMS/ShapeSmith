"""Native combine datacards: text card + shapes file, optional rebinning.

One card per final state lists every bin (channel x category); processes with a non-positive rate in a bin are left
out of that bin. Systematics: lnN from Analysis.lnn (entries with the same name make one line; at most one of them may
apply to a column), `shape` for every Up/Down pair of variations present (names without a direction are templates and
never become shape lines; parts of summed variations are left out of the shapes file), except the variations of
Analysis.lnn_variations, which become an lnN of the varied to the nominal yield and stay out of the shapes file,
`* autoMCStats 0`.

No-op nuisances: an Up/Down pair equal to the nominal in every bin of every process of a card (e.g. a shift of a
fake factor taken from MC without subtraction) is left out of the card and the shapes file, and logged.

Symmetrisation (Analysis.symmetrised_variations, ratio r = Analysis.asymmetry_ratio): per bin and process, with the
relative yield effects u = U/N - 1 and d = D/N - 1 of Up and Down, the pair is replaced when it is one-sided
(u * d >= 0, not both 0) or strongly asymmetric (max(|u|, |d|) > r * min(|u|, |d|)). combine interpolates the yield
of such a pair (AsymPow, vertical template morphing) with a term quadratic in the nuisance of the size of the
asymmetry, (ln kappa_up + ln kappa_down) / 2; for a one-sided pair it dominates and the likelihood is not parabolic,
and robustHesse can find negative eigenvalues there. The replacement is symmetric around the nominal, bin by bin
(i), with u_i = U_i - N_i and d_i = D_i - N_i:
    delta_i = (u_i - d_i) / 2                                 if u_i * d_i < 0 (the half distance of Down to Up),
    delta_i = s_i * max(|u_i|, |d_i|), s_i = sign(u_i - d_i)  otherwise (the larger shift; s_i = sign(u_i + d_i) when
                                                              u_i = d_i),
    Up'_i = N_i + delta_i, Down'_i = N_i - delta_i, each clipped at 0;
the variances of Up and Down stay. Every replacement and no-op is listed in <card dir>/template_changes.json.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from shapesmith.histogram import NOMINAL_VARIATION, HistKey, Histogram, HistogramSet, is_part
from shapesmith.model import NOMINAL, Analysis, Category, LnN
from shapesmith.shapes import shape_name, write_shapes

logger = logging.getLogger(__name__)


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
        logger.warning(f"{channel_name}/{category.name}/{variable}: no {'processes' if not nominal else 'data'}, bin {bin_name(analysis, channel_name, index)} left out")
        return None
    total = None
    for process, h in nominal.items():
        if process != analysis.signal:
            total = h.copy() if total is None else total.add(h)
    edges = rebin_edges(total, min_background) if total is not None else list(hset[key_of(channel.data())].edges)
    result = Bin(channel_name, bin_name(analysis, channel_name, index), hset[key_of(channel.data())].rebin(edges), {}, {})
    if len(edges) < len(hset[key_of(channel.data())].edges):
        logger.debug(f"{result.name}: {len(hset[key_of(channel.data())].edges) - 1} -> {len(edges) - 1} bins (background >= {min_background:g} per bin)")
    for process, h in nominal.items():
        rebinned = h.rebin(edges)
        if rebinned.sum() <= 0.0:
            logger.debug(f"{result.name}: {process} left out, rate {rebinned.sum():.3g}")
            continue
        result.processes[process] = rebinned
        keys = hset.select(channel=channel_name, category=category.name, process=process, region=NOMINAL, variable=variable)
        result.variations[process] = {key.variation: hset[key].rebin(edges) for key in keys if key.variation != NOMINAL_VARIATION and not is_part(key.variation)}
    return result


def _value(lnn: LnN) -> str:
    if isinstance(lnn.value, tuple):
        down, up = lnn.value
        return f"{down:g}/{up:g}"
    return f"{lnn.value:g}"


def _concrete(lnn: LnN, analysis: Analysis, b: Bin) -> str:
    return lnn.name.replace("$CHANNEL", b.channel).replace("$ERA", analysis.era)


def lnn_lines(analysis: Analysis, bins: list[Bin]) -> list[str]:
    """One `<name> lnN ...` line per concrete nuisance name ($CHANNEL/$ERA expanded), in the order of first
    appearance, columns = (bin, process) pairs; entries with the same concrete name share the line."""
    columns = [(b, process) for b in bins for process in b.processes]
    values: dict[str, list[str]] = {}
    for lnn in analysis.lnn:
        for name in sorted({_concrete(lnn, analysis, b) for b in bins}):
            for index, (b, process) in enumerate(columns):
                applies = _concrete(lnn, analysis, b) == name and (lnn.channels is None or b.channel in lnn.channels) and ("*" in lnn.processes or process in lnn.processes)
                if not applies:
                    continue
                row = values.setdefault(name, ["-"] * len(columns))
                if row[index] != "-":
                    raise ValueError(f"lnN {name}: two entries apply to {process} in {b.name}")
                row[index] = _value(lnn)
    return [f"{name} lnN " + " ".join(row) for name, row in values.items()]


def _ratio(varied: Histogram, nominal: Histogram) -> float:
    return varied.sum() / nominal.sum()


def variation_lnn_lines(analysis: Analysis, bins: list[Bin]) -> list[str]:
    """The lnN `down/up` (yield ratios to the nominal) of every Analysis.lnn_variations pair, per (bin, process)."""
    lines = []
    for name in analysis.lnn_variations:
        row = []
        for b in bins:
            for process, nominal in b.processes.items():
                varied = b.variations[process]
                pair = (varied.get(f"{name}Down"), varied.get(f"{name}Up"))
                row.append("-" if None in pair else f"{_ratio(pair[0], nominal):.4f}/{_ratio(pair[1], nominal):.4f}")
        if any(column != "-" for column in row):
            lines.append(f"{name} lnN " + " ".join(row))
    return lines


def _shape_variations(analysis: Analysis, variations: dict[str, Histogram]) -> dict[str, Histogram]:
    """The variations that are written as shapes: all but the pairs of Analysis.lnn_variations."""
    as_lnn = {f"{name}{d}" for name in analysis.lnn_variations for d in ("Up", "Down")}
    return {name: h for name, h in variations.items() if name not in as_lnn}


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
        shapes = [Bin(b.channel, b.name, b.data, b.processes, {p: _shape_variations(analysis, v) for p, v in b.variations.items()}) for b in bins]
        lines += lnn_lines(analysis, bins) + variation_lnn_lines(analysis, bins) + shape_lines(shapes)
    lines.append("* autoMCStats 0")
    return "\n".join(lines) + "\n"


def _pairs(bins: list[Bin]) -> list[str]:
    """The names of the Up/Down pairs of the bins, sorted."""
    return sorted({v[: -len("Up")] for b in bins for variations in b.variations.values() for v in variations if v.endswith("Up") and f"{v[: -len('Up')]}Down" in variations})


def no_op_variations(bins: list[Bin]) -> list[str]:
    """The Up/Down pairs equal to the nominal in every bin of every process that carries them."""
    result = []
    for name in _pairs(bins):
        pairs = [(b.processes[p], v[f"{name}Up"], v[f"{name}Down"]) for b in bins for p, v in b.variations.items() if f"{name}Up" in v and f"{name}Down" in v]
        if all(np.array_equal(up.values, nominal.values) and np.array_equal(down.values, nominal.values) for nominal, up, down in pairs):
            result.append(name)
    return result


def drop_variations(bins: list[Bin], names: list[str]) -> None:
    dropped = {f"{name}{d}" for name in names for d in ("Up", "Down")}
    for b in bins:
        for process, variations in b.variations.items():
            b.variations[process] = {name: h for name, h in variations.items() if name not in dropped}


@dataclass(frozen=True)
class Symmetrised:
    """A symmetrised pair of one bin and process: why, and its Down/Up yield ratios to the nominal before and after."""

    nuisance: str
    bin: str
    process: str
    reason: str
    before: tuple[float, float]
    after: tuple[float, float]
    clipped: int  # bins of Up' or Down' set to 0


def symmetrisation_reason(down: float, up: float, ratio: float) -> str | None:
    """Why a pair with the relative yield effects `down` and `up` (varied / nominal - 1) is symmetrised, or None."""
    if down == 0.0 and up == 0.0:
        return None
    if down * up >= 0.0:
        return "one-sided"
    large, small = max(abs(down), abs(up)), min(abs(down), abs(up))
    return f"asymmetry {large / small:.1f}" if large > ratio * small else None


def symmetric_shift(nominal: np.ndarray, up: np.ndarray, down: np.ndarray) -> np.ndarray:
    """delta per bin (module docstring): the half distance of Down to Up where they lie on both sides of the nominal,
    else the larger shift with the sign of Up - Down."""
    u, d = up - nominal, down - nominal
    sign = np.sign(u - d)
    sign = np.where(sign == 0.0, np.sign(u + d), sign)
    return np.where(u * d < 0.0, (u - d) / 2.0, sign * np.maximum(np.abs(u), np.abs(d)))


def symmetrise(analysis: Analysis, bins: list[Bin]) -> list[Symmetrised]:
    """Replace the one-sided or strongly asymmetric pairs of Analysis.symmetrised_variations in the bins."""
    records = []
    for b in bins:
        for process, nominal in b.processes.items():
            variations = b.variations[process]
            for name in analysis.symmetrised_variations:
                up, down = variations.get(f"{name}Up"), variations.get(f"{name}Down")
                if up is None or down is None:
                    continue
                before = (_ratio(down, nominal), _ratio(up, nominal))
                reason = symmetrisation_reason(before[0] - 1.0, before[1] - 1.0, analysis.asymmetry_ratio)
                if reason is None:
                    continue
                shift = symmetric_shift(nominal.values, up.values, down.values)
                new_up, new_down = nominal.values + shift, nominal.values - shift
                clipped = int(np.sum(new_up < 0.0) + np.sum(new_down < 0.0))
                variations[f"{name}Up"] = Histogram(nominal.edges.copy(), np.clip(new_up, 0.0, None), up.variances.copy())
                variations[f"{name}Down"] = Histogram(nominal.edges.copy(), np.clip(new_down, 0.0, None), down.variances.copy())
                after = (_ratio(variations[f"{name}Down"], nominal), _ratio(variations[f"{name}Up"], nominal))
                records.append(Symmetrised(name, b.name, process, reason, before, after, clipped))
    return records


def _record_line(record: Symmetrised) -> str:
    return f"{record.nuisance} {record.bin}/{record.process} ({record.reason}): {record.before[0]:.4f}/{record.before[1]:.4f} -> {record.after[0]:.4f}/{record.after[1]:.4f}" + (f", {record.clipped} bins clipped at 0" if record.clipped else "")


def write_datacard(hset: HistogramSet, analysis: Analysis, channels: list[str], output_dir: Path, systematics: bool = True, min_background: float | None = 1.0) -> Path:
    output_dir = Path(output_dir)
    bins = [
        b
        for channel in channels
        for index, category in enumerate(analysis.channel(channel).categories)
        if (b := collect_bin(hset, analysis, channel, index, category, min_background)) is not None
    ]
    no_ops = no_op_variations(bins)
    drop_variations(bins, no_ops)
    if no_ops:
        logger.info(f"{output_dir.name}: {len(no_ops)} no-op nuisances left out (Up = Down = nominal in every bin of every process): {', '.join(no_ops)}")
    records = symmetrise(analysis, bins)
    if records:
        logger.info(f"{output_dir.name}: {len(records)} templates symmetrised (one-sided or asymmetry > {analysis.asymmetry_ratio:g}), Down/Up yield ratios before -> after:\n" + "\n".join(_record_line(r) for r in records))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "template_changes.json").write_text(json.dumps({"asymmetry_ratio": analysis.asymmetry_ratio, "no_ops": no_ops, "symmetrised": [asdict(r) for r in records]}, indent=1))
    entries = []
    for b in bins:
        entries.append((b.name, "data_obs", b.data))
        for process, h in b.processes.items():
            entries.append((b.name, process, h))
            entries += [(b.name, shape_name(process, variation), varied) for variation, varied in _shape_variations(analysis, b.variations[process]).items()]
    write_shapes(output_dir / "common" / f"htt_input_{analysis.era}.root", entries)
    card = output_dir / "combined.txt"
    text = card_text(analysis, bins, systematics)
    card.write_text(text)
    nuisances = sum(1 for line in text.splitlines() if " lnN " in line or " shape " in line)
    logger.info(f"{output_dir.name}: {len(bins)} bins, {sum(len(b.processes) for b in bins)} process columns, {nuisances} nuisances -> {card}")
    return card


def run_datacards(hset: HistogramSet, analysis: Analysis, final_states: dict[str, list[str]], output_dir: Path, systematics: bool, min_background: float | None) -> dict[str, Path]:
    return {name: write_datacard(hset, analysis, channels, Path(output_dir) / name, systematics, min_background) for name, channels in final_states.items()}

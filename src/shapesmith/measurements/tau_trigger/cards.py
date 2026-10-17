"""The datacard of one (decay mode, pT) bin of the trigger measurement: a pass and a fail bin, m_vis shapes.

The signal (the embedded genuine taus) has one shape in both bins, that of all its events of the (decay mode, pT) bin
(pass + fail), and the rates eff * r * N in pass and (1 - eff) * r * N in fail: the rateParam `eff` (the POI, the
efficiency of the tau leg, in [0, 1]) on the signal in pass, `eff_fail` = (1 - @0) of it in fail, and the common
`r_<signal>` on the signal in both. The backgrounds are split by their own pass/fail flags; their normalisations are
lnN correlated between pass and fail, and the trigger efficiency of the fakes an lnN that moves `fake_efficiency` of
a fake process's pass yield from fail to pass (kappa 1 + f in pass, 1 - f P / F in fail). The signal's Up/Down shape
variations are shape nuisances. MC statistics: autoMCStats 0. No process is a signal of combine (text2workspace
--X-allow-no-signal); the scans redefine `eff` as the POI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from shapesmith.histogram import Histogram
from shapesmith.shapes import write_shapes

POI = "eff"
SHAPES = "shapes.root"
CARD = "card.txt"
MIN_KAPPA = 0.5  # the fail kappa of the fake-efficiency lnN is not set below this


@dataclass
class TriggerBin:
    """The histograms of one (decay mode, pT) bin: data, the signal template (pass + fail) and its shape variations,
    and the backgrounds of pass and fail."""

    name: str
    data: Mapping[str, Histogram]  # "pass", "fail" -> histogram
    signal: Histogram
    backgrounds: Mapping[str, Mapping[str, Histogram]]  # "pass"/"fail" -> process -> histogram
    signal_shapes: Mapping[str, Histogram] = field(default_factory=dict)  # <nuisance>Up/Down -> template variation

    def bin(self, side: str) -> str:
        return f"{self.name}_{side}"


def rate_parameters(name: str, signal: str, start: float = 0.8) -> list[str]:
    """The rateParams of the efficiency and the common signal normalisation."""
    return [
        f"{POI} rateParam {name}_pass {signal} {start:g} [0,1]",
        f"{POI}_fail rateParam {name}_fail {signal} (1-@0) {POI}",
        f"r_{signal} rateParam * {signal} 1 [0,5]",
    ]


def _yields(trigger_bin: TriggerBin, side: str, signal: str) -> dict[str, Histogram]:
    """The processes of one side with a positive rate, the signal first."""
    processes = {signal: trigger_bin.signal}
    processes.update({name: h for name, h in trigger_bin.backgrounds[side].items() if h.sum() > 0})
    return processes


def fake_kappas(trigger_bin: TriggerBin, process: str, fraction: float) -> tuple[float, float]:
    """The kappas in pass and fail of the fake-efficiency lnN of a process: its pass yield P moves by fraction * P."""
    passing = trigger_bin.backgrounds["pass"].get(process)
    failing = trigger_bin.backgrounds["fail"].get(process)
    p = passing.sum() if passing is not None else 0.0
    f = failing.sum() if failing is not None else 0.0
    return 1.0 + fraction, max(MIN_KAPPA, 1.0 - fraction * p / f) if f > 0 else 1.0


def card_text(trigger_bin: TriggerBin, signal: str, normalisations: Mapping[str, tuple[tuple[str, ...], float]], fake_processes: tuple[str, ...], fake_efficiency: float) -> str:
    """The text datacard of a bin; the shapes are in shapes.root under <bin>_<pass|fail>/<process>[_<nuisance>]."""
    if trigger_bin.signal.sum() <= 0:
        raise ValueError(f"{trigger_bin.name}: the signal template is empty")
    sides = ("pass", "fail")
    columns = [(side, process) for side in sides for process in _yields(trigger_bin, side, signal)]
    index = {process: i + 1 for i, process in enumerate(dict.fromkeys(process for _, process in columns))}
    lines = ["imax 2 number of bins", "jmax * number of processes minus 1", "kmax * number of nuisance parameters", "-" * 60]
    lines.append(f"shapes * * {SHAPES} $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC")
    lines.append("-" * 60)
    lines.append("bin " + " ".join(trigger_bin.bin(side) for side in sides))
    lines.append("observation " + " ".join(f"{trigger_bin.data[side].sum():.1f}" for side in sides))
    lines.append("-" * 60)
    lines.append("bin " + " ".join(trigger_bin.bin(side) for side, _ in columns))
    lines.append("process " + " ".join(process for _, process in columns))
    lines.append("process " + " ".join(str(index[process]) for _, process in columns))
    lines.append("rate " + " ".join("-1" for _ in columns))
    lines.append("-" * 60)
    for name, (processes, kappa) in normalisations.items():
        row = [f"{kappa:g}" if process in processes else "-" for _, process in columns]
        if any(value != "-" for value in row):
            lines.append(f"{name} lnN " + " ".join(row))
    if fake_processes:
        kappas = {process: fake_kappas(trigger_bin, process, fake_efficiency) for process in fake_processes}
        row = [f"{kappas[process][sides.index(side)]:.4f}" if process in kappas else "-" for side, process in columns]
        if any(value != "-" for value in row):
            lines.append("CMS_tauTrg_fake_efficiency lnN " + " ".join(row))
    for nuisance in sorted({name[: -len("Up")] for name in trigger_bin.signal_shapes if name.endswith("Up")}):
        if f"{nuisance}Down" in trigger_bin.signal_shapes:
            lines.append(f"{nuisance} shape " + " ".join("1" if process == signal else "-" for _, process in columns))
    lines += rate_parameters(trigger_bin.name, signal)
    lines.append("* autoMCStats 0")
    return "\n".join(lines) + "\n"


def write(trigger_bin: TriggerBin, directory: Path, signal: str, normalisations: Mapping[str, tuple[tuple[str, ...], float]], fake_processes: tuple[str, ...], fake_efficiency: float) -> Path:
    """card.txt and shapes.root of a bin in `directory`; returns the card."""
    directory = Path(directory)
    entries = []
    for side in ("pass", "fail"):
        folder = trigger_bin.bin(side)
        entries.append((folder, "data_obs", trigger_bin.data[side]))
        for process, h in _yields(trigger_bin, side, signal).items():
            entries.append((folder, process, h))
        entries += [(folder, f"{signal}_{name}", h) for name, h in trigger_bin.signal_shapes.items()]
    write_shapes(directory / SHAPES, entries)
    card = directory / CARD
    card.write_text(card_text(trigger_bin, signal, normalisations, fake_processes, fake_efficiency))
    return card

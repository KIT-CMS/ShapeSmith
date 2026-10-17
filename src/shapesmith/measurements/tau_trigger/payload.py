"""The correctionlib payload of the tau-leg trigger efficiency (v7_tt_emb_trigger 12.6), in the schema of the KIT/POG
`tau_trigger` correction so that the CROWN consumer reads it unchanged:

tau_trigger(pt, dm, trigtype, wp, corrtype, syst): trigtype ditau; wp Medium, VVVLoose_not_Medium; corrtype eff_data
(the measurement), eff_mc (the own MC count); dm -1 (all decay modes), 0, 1, 10, 11; syst nom, up, down; pT in the
edges of the bins of the era and class, flow clamp (the POG files raise outside, which CROWN turns into a silent 1).
The nodes are ordered as in the POG file: trigtype, corrtype, wp, dm, syst, pt.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import correctionlib
import correctionlib.schemav2 as cs
import numpy as np

from shapesmith import payloads

NAME = "tau_trigger"
DECAY_MODES = (-1, 0, 1, 10, 11)
SYSTS = ("nom", "up", "down")
CORRTYPES = ("eff_data", "eff_mc")
BOOKED_CORRTYPES = ("eff_data",)  # what the consumer (CROWN, the tt embedding weight of variant A) reads per booked class
TAU_ID = "DeepTau2018v2p5"
PT_MAX = 1000.0  # the upper edge of the open last pT bin
BOOKED_PTS = (40.0, 60.0, 2.0 * PT_MAX)  # the tt threshold, the plateau and beyond the last edge

# a measured bin: (low pT, high pT or None, {"nom": value, "up": value, "down": value})
Values = tuple[float, float | None, Mapping[str, float]]
# wp -> corrtype -> dm -> the bins in pT order
Results = Mapping[str, Mapping[str, Mapping[int, Sequence[Values]]]]


def edges_of(bins: Sequence[Values], label: str) -> list[float]:
    """The pT edges of contiguous bins (the open last bin ends at PT_MAX)."""
    edges = [float(bins[0][0])]
    for i, (low, high, _) in enumerate(bins):
        if float(low) != edges[-1]:
            raise ValueError(f"{label}: the pT bins are not contiguous at {edges[-1]:g} / {low:g}")
        if high is None and i != len(bins) - 1:
            raise ValueError(f"{label}: an open pT bin that is not the last")
        edges.append(PT_MAX if high is None else float(high))
    return edges


def _pt(bins: Sequence[Values], syst: str, label: str) -> cs.Binning:
    return payloads.binning("pt", edges_of(bins, label), [values[syst] for _, _, values in bins], flow="clamp")


def correction(results: Results, trigtype: str = "ditau") -> cs.Correction:
    corrtypes = {}
    for corrtype in CORRTYPES:
        wps = {}
        for wp, by_corrtype in results.items():
            if corrtype not in by_corrtype:
                continue
            by_dm = by_corrtype[corrtype]
            missing = [dm for dm in DECAY_MODES if dm not in by_dm]
            if missing:
                raise ValueError(f"{wp} {corrtype}: no bins for the decay modes {missing}")
            wps[wp] = payloads.category("dm", {dm: payloads.category("syst", {syst: _pt(by_dm[dm], syst, f"{wp} {corrtype} DM{dm}") for syst in SYSTS}) for dm in DECAY_MODES})
        if wps:
            corrtypes[corrtype] = payloads.category("wp", wps)
    data = payloads.category("trigtype", {trigtype: payloads.category("corrtype", corrtypes)})
    inputs = [
        payloads.variable("pt", "real", "Reconstructed tau pT (flow clamp)"),
        payloads.variable("dm", "int", "Reconstructed tau decay mode: -1 (all), 0, 1, 10, 11"),
        payloads.variable("trigtype", "string", f"Trigger type: {trigtype}"),
        payloads.variable("wp", "string", f"{TAU_ID}VSjet class: {', '.join(results)}"),
        payloads.variable("corrtype", "string", f"Correction type: {', '.join(corrtypes)}"),
        payloads.variable("syst", "string", "nom, up, down"),
    ]
    return payloads.correction(NAME, inputs, data, version=1, description="efficiency of the tau leg of the di-tau trigger for tau embedded samples (data, MC)",
                               output=payloads.variable("eff", "real", "efficiency"))


def correction_set(results: Results, provenance: Mapping, trigtype: str = "ditau") -> cs.CorrectionSet:
    return payloads.correction_set([correction(results, trigtype)], provenance)


def payload_name(era: str) -> str:
    return f"ditau_trigger_{TAU_ID}_embedding{era}UL.json.gz"


def check(cset: cs.CorrectionSet, results: Results, trigtype: str = "ditau", booked: Sequence[str] = ()) -> list[str]:
    """Every combination (wp, corrtype, dm, syst) evaluates in and beyond its pT range, to 0 < eps <= 1, with
    down <= nom <= up; and every combination the consumer books (the classes `booked` with BOOKED_CORRTYPES, every
    decay mode and syst) evaluates, measured or not. The problems."""
    evaluator = correctionlib.CorrectionSet.from_string(payloads.dumps(cset))[NAME]
    problems = []
    for wp in booked:
        for corrtype in BOOKED_CORRTYPES:
            failed = []
            for dm in DECAY_MODES:
                for syst in SYSTS:
                    try:
                        for pt in BOOKED_PTS:
                            evaluator.evaluate(pt, dm, trigtype, wp, corrtype, syst)
                    except Exception:  # noqa: BLE001 - any evaluation failure is a payload problem
                        failed.append(f"DM{dm} {syst}")
            if failed:
                problems.append(f"{wp} {corrtype}: booked by the consumer but not evaluable for {', '.join(failed)}")
    for wp, by_corrtype in results.items():
        for corrtype, by_dm in by_corrtype.items():
            for dm, bins in by_dm.items():
                edges = edges_of(bins, f"{wp} {corrtype} DM{dm}")
                points = [edges[0] - 5.0, *(0.5 * (a + b) for a, b in zip(edges[:-1], edges[1:])), edges[-1] + 100.0]
                for pt in points:
                    try:
                        values = {syst: evaluator.evaluate(pt, dm, trigtype, wp, corrtype, syst) for syst in SYSTS}
                    except Exception as error:  # noqa: BLE001 - any evaluation failure is a payload problem
                        problems.append(f"{wp} {corrtype} DM{dm} pt {pt:g}: {error}")
                        continue
                    if not all(0.0 < value <= 1.0 for value in values.values()):
                        problems.append(f"{wp} {corrtype} DM{dm} pt {pt:g}: values {values} outside (0, 1]")
                    if not values["down"] <= values["nom"] <= values["up"]:
                        problems.append(f"{wp} {corrtype} DM{dm} pt {pt:g}: down <= nom <= up violated {values}")
    return problems


def bin_centres(edges: Sequence[float]) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])

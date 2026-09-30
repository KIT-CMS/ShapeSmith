"""The correctionlib models of the fake-factor payloads, in the conventions the CROWN fake-factor friend reads.

| correction | inputs | syst keys (default: the nominal) |
|---|---|---|
| `{p}{s}_fake_factors` | variable, split, syst | `{p}{s}{StatShift,SystMCShift,SystBandHigh,SystBandLow,SystBandAsym}{Up,Down}` |
| `process_fractions{s}` | process, variable, split, syst | `process_fractions{s}{nominal,frac_QCD_up/down,frac_ttbar_J_up/down}`; processes QCD, ttbar |
| `QCD{s}_DR_SR_correction` | variable, split, syst | `QCD{s}_DR_SR_Corr{StatShift,SystMCShift,SystBandAsym}{Up,Down}`, `nominal` |
| `{p}{s}_non_closure_<var>_correction` | variable, split, syst | `{p}{s}_non_closure_<var>_Corr...` and the global `{p}{s}_non_closure_Corr...`, `nominal` |
| `{p}{s}_compound_correction` | the non-closure variables in order, split, syst | the product of the non-closures of {p}{s} |

{p} is QCD or ttbar, {s} the leg suffix. Every correction is binned in the split categories first, then in its
variable (flow clamp), below a `syst` category.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import correctionlib.schemav2 as cs

from shapesmith import payloads
from shapesmith.measurements.fake_factors.fit import BAND_ASYM, MC, STAT, Fitted, Step
from shapesmith.measurements.fake_factors.model import Binned, Split

CORRECTION_VARIATIONS = tuple(f"{name}{direction}" for name in (STAT, MC, BAND_ASYM) for direction in ("Up", "Down"))
FRACTION_PROCESSES = {"QCD": "QCD", "ttbar_J": "ttbar"}  # TauFakeFactors process -> payload category


def _by_category(split: Split, variable: str, steps: Sequence[Step]) -> cs.Binning:
    return payloads.binning(split.variable, split.edges, [payloads.binning(variable, step.edges, step.values) for step in steps])


def _inputs(*names: str) -> list[cs.Variable]:
    return [payloads.variable(name) for name in names] + [payloads.variable("syst", "string")]


def fake_factors(name: str, binned: Binned, split: Split, fitted: Sequence[Fitted], version: int) -> cs.Correction:
    """`name` = {p}{s}; the stored fit per category."""
    keys = fitted[0].stored_variations
    content = {name + key: _by_category(split, binned.variable, [f.stored_variations[key] for f in fitted]) for key in keys}
    data = payloads.category("syst", content, default=_by_category(split, binned.variable, [f.stored for f in fitted]))
    return payloads.correction(f"{name}_fake_factors", _inputs(binned.variable, split.variable), data, version, output=payloads.variable(f"{name}_ff"))


def fractions(name: str, binned: Binned, split: Split, values: Mapping[str, Mapping[str, Sequence]], version: int) -> cs.Correction:
    """`name` = process_fractions{s}; values[variation][process] holds the fraction values per category, variation
    nominal or frac_<process>_up/down."""

    def by_process(variation: str) -> cs.Category:
        return payloads.category("process", {
            label: _by_category(split, binned.variable, [Step(edges, category) for edges, category in zip(binned.edges, values[variation][process])])
            for process, label in FRACTION_PROCESSES.items()
        })

    data = payloads.category("syst", {name + variation: by_process(variation) for variation in values}, default=by_process("nominal"))
    return payloads.correction(name, [payloads.variable("process", "string"), *_inputs(binned.variable, split.variable)], data, version, output=payloads.variable("fraction"))


def correction(process: str, correction_name: str, binned: Binned, split: Split, fitted: Sequence[Fitted], version: int) -> cs.Correction:
    """`process` = {p}{s}, `correction_name` DR_SR or non_closure_<var>; a non-closure repeats its keys under the
    global non_closure name, which the compound correction shifts as one."""
    names = [f"_{correction_name}"] + (["_non_closure"] if correction_name.startswith("non_closure") else [])
    nominal = _by_category(split, binned.variable, [f.stored for f in fitted])
    content = {
        f"{process}{name}_Corr{variation}": _by_category(split, binned.variable, [f.stored_variations[variation] for f in fitted])
        for variation in CORRECTION_VARIATIONS for name in names
    }
    content["nominal"] = nominal
    full_name = f"{process}_{correction_name}_correction"
    data = payloads.category("syst", content, default=nominal)
    return payloads.correction(full_name, _inputs(binned.variable, split.variable), data, version, output=payloads.variable(full_name))


def compound(process: str, non_closures: Sequence[Binned], split: Split) -> cs.CompoundCorrection:
    """The product of the non-closure corrections of {p}{s} = `process`, in table order."""
    name = f"{process}_compound_correction"
    stack = [f"{process}_non_closure_{binned.variable}_correction" for binned in non_closures]
    return payloads.compound(name, _inputs(*(b.variable for b in non_closures), split.variable), stack, output=payloads.variable(name))

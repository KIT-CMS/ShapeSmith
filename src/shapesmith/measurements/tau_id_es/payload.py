"""The correctionlib payload of the measurement, with the node structure of the predecessor's
friends/create_xpog_json_v15.py (the file the embedding configuration of CROWN reads), plus provenance.

- DeepTau2018v2p5VSjet(pt, dm, genmatch, wp, wp_VSe, syst, flag): the scale factor r; flag "pt" in the pT bins of
  the categories, flag "dm" per decay mode for every pT from 20 GeV.
- tau_energy_scale and tau_energy_scale_dm_binned(pt, eta, dm, genmatch, id, wp, wp_VSe, syst): the energy scale
  factor (100 + ES) / 100 of the fitted ES in percent, per decay mode and pT bin or per decay mode, for |eta| < 2.5.
Every other genmatch gives 1.0. DM10 and DM11 both take the DM1011 result; syst up/down are the ends of the 1 sigma
interval.
"""
from __future__ import annotations

from typing import Callable, Mapping

import correctionlib.schemav2 as cs

from shapesmith import payloads
from shapesmith.measurements.tau_id_es.combine import Interval
from shapesmith.measurements.tau_id_es.grid import CATEGORIES

DECAY_MODES = (0, 1, 10, 11)
OTHER_GENMATCH = (0, 1, 2, 3, 4, 6)
WORKING_POINTS = ("VVVLoose", "VVLoose", "VLoose", "Loose", "Medium", "Tight", "VTight", "VVTight")
MAX_PT = 100000.0  # the upper edge of the per-decay-mode bins
TAU_ID = "DeepTau2018v2p5"

# per working-point combination (vsjet, vsele) and category: the intervals of the scale factor and of ES (percent)
Results = Mapping[tuple[str, str], Mapping[str, tuple[Interval, Interval]]]


def es_factor(percent: float) -> float:
    return (100 + percent) / 100


def _syst(interval: Interval, convert: Callable[[float], float]) -> cs.Category:
    return payloads.category("syst", {"nom": convert(interval.best), "up": convert(interval.high), "down": convert(interval.low)})


def _pt_bins(dm: int, results: Mapping[str, tuple[Interval, Interval]], pt_binned: bool) -> list[tuple[tuple[float, float], str]]:
    """The (pT bin, category) of a decay mode: its pT categories, or its inclusive category from 20 GeV."""
    bins = [(pt, name) for name, (modes, pt) in CATEGORIES.items() if dm in modes and (pt is not None) == pt_binned and name in results]
    if not pt_binned:
        return [((20.0, MAX_PT), name) for _, name in bins]
    return sorted(bins)


def _by_decay_mode(results: Mapping[str, tuple[Interval, Interval]], index: int, pt_binned: bool, convert: Callable[[float], float], wrap: Callable = lambda node: node) -> cs.Category:
    content = {}
    for dm in DECAY_MODES:
        bins = _pt_bins(dm, results, pt_binned)
        if bins:
            edges = [bins[0][0][0]] + [pt[1] for pt, _ in bins]
            content[dm] = wrap(payloads.binning("pt", edges, [_syst(results[name][index], convert) for _, name in bins]))
    return payloads.category("dm", content)


def _genuine_taus(results: Results, node: Callable[[Mapping[str, tuple[Interval, Interval]]], cs.Category]) -> cs.Category:
    """1.0 unless genmatch is 5; then per working point, vsEle working point and decay mode."""
    order = {wp: index for index, wp in enumerate(WORKING_POINTS)}
    by_wp: dict[str, dict] = {}
    for vsjet, vsele in sorted(results, key=lambda pair: (order[pair[0]], order[pair[1]])):
        by_wp.setdefault(vsjet, {})[vsele] = node(results[vsjet, vsele])
    wps = payloads.category("wp", {vsjet: payloads.category("wp_VSe", by_vsele) for vsjet, by_vsele in by_wp.items()})
    return payloads.category("genmatch", {**{genmatch: 1.0 for genmatch in OTHER_GENMATCH}, 5: wps})


def _inputs(results: Results, *names: str) -> list[cs.Variable]:
    order = {wp: index for index, wp in enumerate(WORKING_POINTS)}
    vsjet = ",".join(sorted({pair[0] for pair in results}, key=order.get))
    vsele = ",".join(sorted({pair[1] for pair in results}, key=order.get))
    known = {
        "pt": ("real", "Reconstructed tau pT"),
        "eta": ("real", "Reconstructed tau eta"),
        "dm": ("int", "Reconstructed tau decay mode: 0, 1, 10, 11"),
        "genmatch": ("int", "genmatch: 0 or 6 = unmatched or jet, 1 or 3 = electron, 2 or 4 = muon, 5 = real tau"),
        "id": ("string", f"{TAU_ID}:{TAU_ID}"),
        "wp": ("string", f"{TAU_ID}VSjet working point: {vsjet}"),
        "wp_VSe": ("string", f"{TAU_ID}VSe working point: {vsele}"),
        "syst": ("string", "Systematic variations: 'nom', 'up', 'down'"),
        "flag": ("string", "Flag: 'pt' = DM-pT-dependent SFs(20,40,200), 'dm' = DM-dependent SFs with pt>=20"),
    }
    return [payloads.variable(name, *known[name]) for name in names]


def _abs_eta(node: cs.Binning) -> cs.Transform:
    return payloads.transform("eta", "abs(x)", payloads.binning("eta", [0.0, 2.5], [node]))


def scale_factors(results: Results) -> cs.Correction:
    data = payloads.category("flag", {
        "pt": _genuine_taus(results, lambda r: _by_decay_mode(r, 0, True, float)),
        "dm": _genuine_taus(results, lambda r: _by_decay_mode(r, 0, False, float)),
    })
    output = payloads.variable("sf", "real", "DM-pT-dependent or DM-dependent scale factor")
    return payloads.correction(f"{TAU_ID}VSjet", _inputs(results, "pt", "dm", "genmatch", "wp", "wp_VSe", "syst", "flag"), data, version=0,
                               description="dm-pt-dependent tau ID scale factor for tau embedded samples", output=output)


def energy_scale(results: Results, pt_binned: bool) -> cs.Correction:
    data = payloads.category("id", {TAU_ID: _genuine_taus(results, lambda r: _by_decay_mode(r, 1, pt_binned, es_factor, _abs_eta))})
    inputs = _inputs(results, "pt", "eta", "dm", "genmatch", "id", "wp", "wp_VSe", "syst")
    if pt_binned:
        output = payloads.variable("tes", "real", "DM-pT-dependent tau energy scale")
        return payloads.correction("tau_energy_scale", inputs, data, version=0, description="dm-pt-dependent tau ES scale factor for tau embedded samples", output=output)
    output = payloads.variable("tes", "real", "DM-dependent tau energy scale")
    return payloads.correction("tau_energy_scale_dm_binned", inputs, data, version=0, description="dm-dependent tau ES scale factor for tau embedded samples", output=output)


def correction_set(results: Results, provenance: Mapping) -> cs.CorrectionSet:
    return payloads.correction_set([scale_factors(results), energy_scale(results, True), energy_scale(results, False)], provenance)


def payload_name(era: str) -> str:
    return f"{TAU_ID}_id_es_embedding{era}UL.json.gz"

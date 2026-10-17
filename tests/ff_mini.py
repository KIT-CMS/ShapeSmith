"""A synthetic fake-factor measurement: data, a genuine-tau MC process (ZT) and ttbar jet fakes (TTJ) in one
lepton-tau channel, written straight into skims. The regions follow the SM measurement: same sign for the QCD fake
factors, a lepton-isolation sideband for the DR->SR correction, OS with inverted lepton vetoes for the ttbar scale.
build_w adds W+jets (WJ) and its fake factors: the DR mt_1 > 70 and no b tag, QCD from its same-sign version, the
DR->SR correction from W MC to the baseline (mt_1 < 70 and any b tag), three-process fractions. The W columns (mt_1,
n_bjets) and sample come from their own random stream, so that the other columns stay those of build.
"""
import dataclasses

import numpy as np
import pandas as pd

from shapesmith.measurements.fake_factors import Binned, DataScale, DrSr, Equipopulated, FakeFactorMeasurement, Fit, Fractions, Leg, ProcessFF, SameSignQCD, Split
from shapesmith.model import Analysis, Channel, Process, Region, Sample, Selection
from shapesmith.store import skim_path, write_skim

PASS, FAIL = "(id_pass > 0.5)", "(id_pass < 0.5) & (id_loose > 0.5)"
SS, OS = "((q_1 * q_2) > 0)", "((q_1 * q_2) < 0)"
ISOLATED, SIDEBAND = "(iso_1 < 0.15)", "(iso_1 >= 0.15)"
SUBTRACT = ("ZT", "TTJ")


def _events(rng: np.random.Generator, n: int, kind: str) -> pd.DataFrame:
    frame = pd.DataFrame({
        "q_1": rng.choice([-1, 1], n).astype(np.int32), "q_2": rng.choice([-1, 1], n).astype(np.int32),
        "id_pass": (rng.random(n) < (0.7 if kind == "mc" else 0.25)).astype(np.int32), "id_loose": (rng.random(n) < 0.97).astype(np.int32),
        "iso_1": rng.uniform(0, 0.5, n).astype(np.float32), "veto": (rng.random(n) < 0.2).astype(np.int32),
        "n_jets": rng.choice([2, 3, 4], n, p=[0.5, 0.3, 0.2]).astype(np.int32),
        "pt_2": rng.uniform(30, 150, n).astype(np.float32), "pt_tautau": rng.uniform(0, 150, n).astype(np.float32),
        "tau_decaymode_2": rng.choice([0, 1, 10], n).astype(np.int32), "met": rng.exponential(40, n).astype(np.float32),
        "sf": rng.normal(1.0, 0.05, n).astype(np.float32),
    })
    frame["norm_weight"] = 1.0 if kind == "data" else 0.04
    for column, sample_kind in (("is_data", "data"), ("is_mc", "mc"), ("is_embedding", "embedding")):
        frame[column] = kind == sample_kind
    return frame


def write_skims(skim_dir, seed: int = 3) -> None:
    rng, w_rng = np.random.default_rng(seed), np.random.default_rng(seed + 100)
    for nick, kind, n in (("DATA", "data", 30000), ("DY", "mc", 30000), ("TT", "mc", 60000), ("WJ", "mc", 30000)):
        frame = _events(rng if nick != "WJ" else w_rng, n, kind)
        frame["mt_1"] = w_rng.uniform(0, 150, n).astype(np.float32)
        frame["n_bjets"] = w_rng.choice([0, 1, 2], n, p=[0.5, 0.35, 0.15]).astype(np.int32)
        frame["sample_nick"] = pd.array([nick] * n, dtype="string")
        write_skim(frame, skim_path(skim_dir, "mt", nick, "file_0.root"))


def _regions() -> tuple[Region, ...]:
    def region(name: str, **cuts: str) -> Region:
        return Region(name, replace_cuts=cuts)

    qcd = {"charge": SS, "jets": "(n_jets >= 2)"}
    return (
        region("QCD_sr_like", **qcd), region("QCD_ar_like", **qcd, tau_iso=FAIL),
        region("QCD_orthogonal_sr_like", **qcd, lepton_iso=SIDEBAND), region("QCD_orthogonal_ar_like", **qcd, lepton_iso=SIDEBAND, tau_iso=FAIL),
        region("QCD_dr_sr_sr_like", lepton_iso=SIDEBAND), region("QCD_dr_sr_ar_like", lepton_iso=SIDEBAND, tau_iso=FAIL),
        region("ttbar_sr"), region("ttbar_ar", tau_iso=FAIL),
        region("ttbar_scale_sr_like", vetoes="(veto > 0.5)"), region("ttbar_scale_ar_like", vetoes="(veto > 0.5)", tau_iso=FAIL),
        region("ttbar_scale_sr_like_ss", vetoes="(veto > 0.5)", charge=SS), region("ttbar_scale_ar_like_ss", vetoes="(veto > 0.5)", charge=SS, tau_iso=FAIL),
    )


def binned(variable: str, edges: tuple[float, ...], fit: Fit, categories: int = 2, n_bins: int = 3) -> Binned:
    return Binned(variable, (edges,) * categories, (fit,) * categories, Equipopulated((n_bins,) * categories, edges[0], edges[-1]))


def measurement() -> FakeFactorMeasurement:
    split = Split("n_jets", (1.5, 2.5, 5.5))
    smoothed = Fit("smoothed", 30.0)
    non_closures = (binned("tau_decaymode_2", (-0.5, 0.5, 9.5, 10.5), Fit("binwise")), binned("met", (0.0, 30.0, 60.0, 200.0), Fit("smoothed", 40.0, binwise_left=1)))
    dr_sr = DrSr(
        "QCD_orthogonal_sr_like", "QCD_orthogonal_ar_like", "QCD_dr_sr_sr_like", "QCD_dr_sr_ar_like", SUBTRACT,
        binned("pt_tautau", (0.0, 50.0, 100.0, 150.0), Fit("smoothed", 25.0, binwise_right=1)), non_closures[:1],
    )
    qcd = ProcessFF("QCD", "data", SUBTRACT, "QCD_sr_like", "QCD_ar_like", split, binned("pt_2", (30.0, 50.0, 80.0, 150.0), smoothed), non_closures, dr_sr)
    scale = DataScale("ttbar_scale_sr_like", "ttbar_scale_ar_like", "ttbar_scale_sr_like_ss", "ttbar_scale_ar_like_ss", SUBTRACT)
    ttbar = ProcessFF("ttbar", "TTJ", (), "ttbar_sr", "ttbar_ar", split, binned("pt_2", (30.0, 60.0, 150.0), smoothed), non_closures[1:], scale=scale)
    fractions = Fractions("ttbar_ar", SUBTRACT, "TTJ", Split("n_jets", (1.5, 2.5, 5.5)), Binned("pt_2", ((30.0, 60.0, 150.0),) * 2))
    return FakeFactorMeasurement({"mt": (Leg("", qcd, ttbar, fractions),)})


def build(config=None) -> Analysis:
    weights = Selection(weights={"sf": "sf"})
    channel = Channel(
        name="mt",
        samples=(Sample("DATA", "data", "data"), Sample("DY", "DY", "mc"), Sample("TT", "TT", "mc")),
        skim={},
        cuts={"charge": OS, "vetoes": "(veto < 0.5)", "lepton_iso": ISOLATED, "jets": "(n_jets >= 2)", "tau_iso": PASS},
        processes=(Process("data", "data", "data", "data"), Process("ZT", "DY", "background", "Z", weights), Process("TTJ", "TT", "background", "TT", weights)),
        regions=_regions(),
    )
    return Analysis("ff_mini", "2018", 1.0, None, {"mt": channel}, measurement=measurement())


W_SUBTRACT = ("ZT", "TTJ")  # the W fake factors subtract the other MC and QCD
ALL_MC = ("ZT", "TTJ", "WJ")  # the same-sign QCD subtracts every MC
W_DR = {"mt_cut": "(mt_1 > 70)", "btag": "(n_bjets == 0)"}


def _w_regions() -> tuple[Region, ...]:
    def region(name: str, **cuts: str) -> Region:
        return Region(name, replace_cuts=cuts)

    return (
        region("Wjets_sr_like", **W_DR), region("Wjets_ar_like", **W_DR, tau_iso=FAIL),
        region("Wjets_sr_like_ss", **W_DR, charge=SS), region("Wjets_ar_like_ss", **W_DR, charge=SS, tau_iso=FAIL),
        region("Wjets_sr"), region("Wjets_ar", tau_iso=FAIL),
    )


def measurement_w() -> FakeFactorMeasurement:
    """measurement() with the W+jets fake factors and the three-process fractions."""
    (leg,) = measurement().legs["mt"]
    split = Split("n_jets", (1.5, 2.5, 5.5))
    smoothed = Fit("smoothed", 30.0)
    qcd = SameSignQCD({"Wjets_sr_like": "Wjets_sr_like_ss", "Wjets_ar_like": "Wjets_ar_like_ss"}, ALL_MC)
    dr_sr = DrSr("Wjets_sr_like", "Wjets_ar_like", "Wjets_sr", "Wjets_ar", (), binned("mt_1", (0.0, 25.0, 50.0, 70.0), Fit("smoothed", 20.0)), target="WJ")
    wjets = ProcessFF("Wjets", "data", W_SUBTRACT, "Wjets_sr_like", "Wjets_ar_like", split, binned("pt_2", (30.0, 50.0, 80.0, 150.0), smoothed),
                      (binned("met", (0.0, 30.0, 60.0, 200.0), Fit("smoothed", 40.0)),), dr_sr, qcd=qcd)
    qcd_ff = dataclasses.replace(leg.qcd, subtract=ALL_MC, dr_sr=dataclasses.replace(leg.qcd.dr_sr, subtract=ALL_MC))
    ttbar = dataclasses.replace(leg.ttbar, scale=dataclasses.replace(leg.ttbar.scale, subtract=ALL_MC))
    fractions = dataclasses.replace(leg.fractions, subtract=ALL_MC, wjets="WJ")
    return FakeFactorMeasurement({"mt": (Leg("", qcd_ff, ttbar, fractions, wjets),)}, name="fake_factors_w")


def build_w(config=None) -> Analysis:
    weights = Selection(weights={"sf": "sf"})
    channel = build().channel("mt")
    channel = dataclasses.replace(
        channel,
        samples=(*channel.samples, Sample("WJ", "W", "mc")),
        cuts={**channel.cuts, "mt_cut": "(mt_1 < 70)", "btag": "(n_bjets >= 0)"},
        processes=(*channel.processes, Process("WJ", "W", "background", "W", weights)),
        regions=(*channel.regions, *_w_regions()),
    )
    return Analysis("ff_mini_w", "2018", 1.0, None, {"mt": channel}, measurement=measurement_w())

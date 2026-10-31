"""A synthetic tau-leg trigger measurement: data, the embedded genuine taus (EMB, the signal) and DY in one mu tau_h
channel, written straight into skims. The probe passes the monitoring leg with a known efficiency per decay mode
(EFFICIENCY); two decay-mode groups (0 and all) times two pT bins, each with a pass and a fail category whose m_vis
edges come from the data of the bin (EqualData with the bin cut); QCD from same sign, the counting region mt_1 < 30;
the signal's shape nuisances: its energy scale (column variations) and a template shift by the genuine MC taus (as the
ttbar contamination of the embedded sample).
"""
import numpy as np
import pandas as pd

from shapesmith.measurements.tau_trigger import Bin, TauTriggerMeasurement
from shapesmith.model import Analysis, Category, Channel, ColumnVariation, DataMinus, EqualData, Process, Region, Sample, Selection, TemplateShift, Variable
from shapesmith.store import skim_path, write_skim

EFFICIENCY = {0: 0.85, 1: 0.75}
PASS = "(mon > 0.5)"
SS, OS = "((q_1 * q_2) > 0)", "((q_1 * q_2) < 0)"
BINS = (Bin("DM0_PT40_60", (0,), (40.0, 60.0)), Bin("DM0_PT60", (0,), (60.0, None)), Bin("DMall_PT40_60", (-1,), (40.0, 60.0)), Bin("DMall_PT60", (-1,), (60.0, None)))


def _events(rng: np.random.Generator, n: int, kind: str, genuine: float) -> pd.DataFrame:
    dm = rng.choice([0, 1], n).astype(np.int32)
    real = rng.random(n) < genuine
    efficiency = np.where(real, np.where(dm == 0, EFFICIENCY[0], EFFICIENCY[1]), 0.3)
    frame = pd.DataFrame({
        "q_1": rng.choice([-1, 1], n, p=[0.8, 0.2] if kind == "data" else [0.9, 0.1]).astype(np.int32),
        "q_2": np.full(n, 1, dtype=np.int32),
        "pt_2": rng.uniform(40, 120, n).astype(np.float32),
        "tau_decaymode_2": dm,
        "m_vis": np.where(real, rng.normal(65, 10, n), rng.uniform(30, 160, n)).astype(np.float32),
        "mt_1": rng.uniform(0, 65, n).astype(np.float32),
        "mon": (rng.random(n) < efficiency).astype(np.int32),
        "gen_match_2": np.where(real, 5, 6).astype(np.int32),
        "sf": rng.normal(1.0, 0.02, n).astype(np.float32),
    })
    frame["norm_weight"] = {"data": 1.0, "embedding": 0.3, "mc": 0.05}[kind]
    for column, sample_kind in (("is_data", "data"), ("is_mc", "mc"), ("is_embedding", "embedding")):
        frame[column] = kind == sample_kind
    return frame


def write_skims(skim_dir, seed: int = 7, embedding_monitor: bool = True) -> None:
    """embedding_monitor False: the embedded sample has no monitoring-path decisions (mon 0 everywhere, as the 2018
    mu tau embedding)."""
    rng = np.random.default_rng(seed)
    for nick, kind, n, genuine in (("DATA", "data", 40000, 0.7), ("EMB", "embedding", 30000, 1.0), ("DY", "mc", 60000, 0.5)):
        frame = _events(rng, n, kind, genuine)
        if kind == "embedding" and not embedding_monitor:
            frame["mon"] = 0
        frame["sample_nick"] = pd.array([nick] * n, dtype="string")
        write_skim(frame, skim_path(skim_dir, "mt", nick, "file_0.root"))


def bin_cut(b: Bin) -> str:
    dm = "tau_decaymode_2 >= 0" if b.decay_modes == (-1,) else " | ".join(f"(tau_decaymode_2 == {dm})" for dm in b.decay_modes)
    pt = f"(pt_2 >= {b.pt[0]})" + ("" if b.pt[1] is None else f" & (pt_2 < {b.pt[1]})")
    return f"({dm}) & {pt}"


def categories() -> tuple[Category, ...]:
    result = []
    for b in BINS:
        variable = Variable("m_vis", "m_vis", EqualData(5, 30.0, 160.0, cut=bin_cut(b)))
        result += [Category(b.category("pass"), f"{bin_cut(b)} & {PASS}", variable), Category(b.category("fail"), f"{bin_cut(b)} & ~{PASS}", variable)]
    return tuple(result)


def measurement(**options) -> TauTriggerMeasurement:
    return TauTriggerMeasurement(
        "Medium", BINS, "EMB", signal_template="emb_es", signal_shapes=("emb_es_shift", "emb_ttbar"),
        normalisations={"norm_zj": (("ZJ",), 1.2), "norm_qcd": (("QCD",), 1.2)}, fake_processes=("ZJ", "QCD"),
        mc_process="ZTT", monitor_process="EMB", counting_regions=("counting", "counting_same_sign"), qcd_subtract=("EMB", "ZJ"),
        **options,
    )


def build(config=None) -> Analysis:
    weights = Selection(weights={"sf": "sf"})
    es = {name: ColumnVariation(name, derived={"pt_2": f"pt_2 * {scale}", "m_vis": f"m_vis * {scale}"}, applies_to=("embedding",), regions=("nominal",))
          for name, scale in (("emb_es", 0.99), ("emb_es_shiftUp", 1.0), ("emb_es_shiftDown", 0.98))}
    channel = Channel(
        name="mt",
        samples=(Sample("DATA", "data", "data"), Sample("EMB", "EMB", "embedding"), Sample("DY", "DY", "mc")),
        skim={},
        cuts={"charge": OS, "mt_cut": "(mt_1 < 65)"},
        processes=(
            Process("data", "data", "data", "data"),
            Process("EMB", "EMB", "background", "EMB", Selection(cuts={"gen": "(gen_match_2 == 5)"}, weights={"sf": "sf"})),
            Process("ZJ", "DY", "background", "Z", Selection(cuts={"gen": "(gen_match_2 != 5)"}, weights=weights.weights)),
            Process("ZTT", "DY", "auxiliary", "Z", Selection(cuts={"gen": "(gen_match_2 == 5)"}, weights=weights.weights)),
        ),
        regions=(Region("same_sign", replace_cuts={"charge": SS}), Region("counting", replace_cuts={"mt_cut": "(mt_1 < 30)"}),
                 Region("counting_same_sign", replace_cuts={"mt_cut": "(mt_1 < 30)", "charge": SS})),
        categories=categories(),
        variations=tuple(es.values()),
        estimators=(DataMinus("QCD", "same_sign", ("EMB", "ZJ"), clip_negative=True), TemplateShift("emb_ttbar", "EMB", "ZTT", 0.1)),
    )
    return Analysis("trigger_mini", "2018", 1.0, None, {"mt": channel}, measurement=measurement())

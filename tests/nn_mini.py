"""A synthetic NN analysis for the ScoreBinning chain (skim -> hist -> estimate -> datacards, plots): n-tuples of data,
a signal and two backgrounds with an NN friend (predicted_class, predicted_max_value of three classes) and a fake
factor, the jet fakes estimated from the anti-isolated data.

The signal peaks at high P(HH) in the HH category, the backgrounds fall towards it, so that the data-blind binning
merges at the top and the HH category is blinded from some edge on (S/B(mu=1) > 0.5 %) with real data below it.
"""
from pathlib import Path

import numpy as np

from shapesmith.model import Analysis, Category, Channel, DataMinus, LnN, Process, Region, Sample, ScoreBinning, Selection, Variable, WeightVariation
from shapesmith.nn_binning import BLINDING_S_OVER_B, THESIS_HH_EDGES, THESIS_OTHER_EDGES, category_rules
from shapesmith.testing import make_ntuple

CLASSES = ("HH", "ZTT", "TT")
FINE = tuple(float(round(x, 6)) for x in np.linspace(0.0, 1.0, 201))  # 200 fine bins
REQUIRED = {"Top": ("TT",), "tautau": ("ZTT",), "jetFakes": ("jetFakes",)}
# nick -> (sample type, events, probability of each predicted class, Beta(a, b) of P(class) above 1/3)
SAMPLES = {
    "DATA_A": ("data", 12000, (0.10, 0.45, 0.45), (1.2, 3.0)),
    "ZTT_1": ("mc", 6000, (0.08, 0.80, 0.12), (1.2, 3.0)),
    "TT_1": ("mc", 6000, (0.10, 0.10, 0.80), (1.2, 3.0)),
    "SIG_1": ("mc", 3000, (0.90, 0.05, 0.05), (5.0, 1.0)),
}
MC_WEIGHTS = {"pu": "puweight"}
TIGHT = {"DATA_A": 0.5}  # the share of isolated taus (else 0.7): the data have more jet fakes in the anti-isolated region


def make_nn_dataset(root: Path, era: str = "2018", channel: str = "mt", seed: int = 3) -> None:
    """<root>/CROWNRun/<era>/<nick>/<channel>/<nick>_0.root and the friend <root>/CROWNFriends/nn/... for every nick."""
    rng = np.random.default_rng(seed)
    for nick, (kind, n, probabilities, (a, b)) in SAMPLES.items():
        predicted = rng.choice(3, n, p=probabilities).astype(np.int32)
        score = (1.0 / 3.0 + (2.0 / 3.0) * rng.beta(a, b, n)).astype(np.float32)
        if nick == "SIG_1":  # the signal is HH-like only in its own class
            score = np.where(predicted == 0, score, (1.0 / 3.0 + (2.0 / 3.0) * rng.beta(1.2, 3.0, n))).astype(np.float32)
        else:  # backgrounds in the HH category fall towards P(HH) = 1
            score = np.where(predicted == 0, (1.0 / 3.0 + (2.0 / 3.0) * rng.beta(1.0, 4.0, n)), score).astype(np.float32)
        main = {
            "q_1": rng.choice([-1.0, 1.0], n).astype(np.float32),
            "q_2": rng.choice([-1.0, 1.0], n, p=[0.8, 0.2]).astype(np.float32),
            "id_loose": np.ones(n, dtype=np.float32),
            "id_tight": (rng.random(n) < TIGHT.get(nick, 0.7)).astype(np.float32),
            "n_bjets": rng.choice([1, 2, 3], n, p=[0.5, 0.45, 0.05]).astype(np.int32),
            "event": np.arange(n, dtype=np.uint64),
        }
        if kind == "mc":
            main["gen_match"] = np.full(n, 5, dtype=np.int32)
            main["puweight"] = rng.normal(1.0, 0.05, n).astype(np.float32)
            main["puweight_up"] = (main["puweight"] * 1.04).astype(np.float32)
            main["puweight_down"] = (main["puweight"] * 0.96).astype(np.float32)
            main["genWeight"] = rng.choice([-1.0, 1.0], n, p=[0.05, 0.95]).astype(np.float32)
        fake_factor = np.where(rng.random(n) < 0.05, 0.0, rng.uniform(0.2, 0.4, n)).astype(np.float32)  # some events weigh 0
        friend = {"predicted_class": predicted, "predicted_max_value": score, "fake_factor": fake_factor}
        metadata = {"analysis": "nn_mini", "era": era, "sample_type": kind}
        make_ntuple(Path(root) / "CROWNRun" / era / nick / channel / f"{nick}_0.root", main, metadata)
        make_ntuple(Path(root) / "CROWNFriends" / "nn" / era / nick / channel / f"{nick}_0.root", friend)


def nn_categories(method: str = "data_blind", blinding: float | None = BLINDING_S_OVER_B) -> tuple[Category, ...]:
    """One category per class (argmax, n_bjets >= 3 cut out), the HH category blinded."""
    result = []
    for index, name in enumerate(CLASSES):
        signal = index == 0
        if method == "thesis":
            binning = ScoreBinning(FINE, thesis=THESIS_HH_EDGES if signal else THESIS_OTHER_EDGES, blinding=blinding if signal else None, clip=("jetFakes",))
        else:
            binning = ScoreBinning(FINE, rules=category_rules(len(CLASSES), signal, REQUIRED), blinding=blinding if signal else None, clip=("jetFakes",))
        result.append(Category(name, f"(predicted_class == {index}) & (n_bjets < 3)", Variable("NN_score", "predicted_max_value", binning)))
    return tuple(result)


def channel_mt(method: str = "data_blind") -> Channel:
    return Channel(
        name="mt",
        samples=(
            Sample("DATA_A", "data", "data"),
            Sample("ZTT_1", "ZTT", "mc", xsec=40.0, nevents=6000),
            Sample("TT_1", "TT", "mc", xsec=40.0, nevents=6000),
            Sample("SIG_1", "HH", "mc", xsec=0.05, nevents=3000),
        ),
        skim={"iso_loose": "id_loose > 0.5"},
        cuts={"os": "(q_1 * q_2) < 0", "tau_iso": "id_tight > 0.5"},
        processes=(
            Process("data", "data", "data", "data"),
            Process("ZTT", "ZTT", "background", "Z", Selection(cuts={"gen": "gen_match == 5"}, weights=MC_WEIGHTS)),
            Process("TT", "TT", "background", "TT", Selection(weights=MC_WEIGHTS)),
            Process("HH", "HH", "signal", "signal", Selection(weights=MC_WEIGHTS)),
        ),
        regions=(Region("anti_iso", replace_cuts={"tau_iso": "id_tight < 0.5"}, add_weights={"fake_factor": "fake_factor"}),),
        categories=nn_categories(method),
        variables={"score": Variable("score", "predicted_max_value", tuple(float(x) for x in np.linspace(0.0, 1.0, 11)))},
        variations=(WeightVariation("CMS_puUp", {"pu": "puweight_up"}), WeightVariation("CMS_puDown", {"pu": "puweight_down"})),
        estimators=(DataMinus("jetFakes", "anti_iso", ("ZTT", "TT")),),
        keep_columns=("event",),
    )


def build(config=None) -> Analysis:
    method = (config.switches.get("method", "data_blind") if config is not None else "data_blind")
    return Analysis(
        name="nn_mini",
        era="2018",
        lumi_pb=100.0,
        signal="HH",
        channels={"mt": channel_mt(method)},
        lnn=(LnN("lumi", ("*",), 1.025), LnN("ffNorm", ("jetFakes",), 1.1)),
    )

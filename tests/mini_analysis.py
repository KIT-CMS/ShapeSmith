"""A tiny but complete Analysis used by the test suite (matches shapesmith.testing.make_mini_dataset).

`build` is the analysis of the golden references; `build_embedding` adds the embedded sample, the CROWN shifts of
the dataset with variations and the embedding ttbar-contamination template, as the SM analysis does with embedding
and fake factors.
"""
import dataclasses

from shapesmith.model import (
    Analysis, Category, Channel, ColumnVariation, DataMinus, LnN, MLExportConfig, Process, Region, Sample, Selection, Style, TemplateShift, Variable, WeightVariation,
)

SCORE = Variable("score", "score", (0.0, 0.5, 1.0))
M_VIS = Variable("m_vis", "m_vis", (0.0, 50.0, 100.0, 200.0))
MC_WEIGHTS = {"trg": "trg_wgt", "pu": "puweight"}


def _mc(name: str, group: str, role: str, plot_group: str, cuts: dict) -> Process:
    return Process(name, group, role, plot_group, Selection(cuts=cuts, weights=MC_WEIGHTS))


def channel_mt() -> Channel:
    return Channel(
        name="mt",
        samples=(
            Sample("DATA_A", "data", "data"),
            Sample("ZTT_1", "DY", "mc", xsec=10.0, nevents=100, generator_weight=1.0),
            Sample("SIG_1", "HH", "mc", xsec=1.0, nevents=50, generator_weight=1.0),
        ),
        skim={"veto": "veto < 0.5", "iso_loose": "id_loose > 0.5"},
        cuts={"veto": "veto < 0.5", "os": "(q_1 * q_2) < 0", "tau_iso": "id_tight > 0.5"},
        processes=(
            Process("data", "data", "data", "data"),
            _mc("ZTT", "DY", "background", "Z", {"gen": "gen_match == 5"}),
            _mc("ZL", "DY", "background", "Z", {"gen": "gen_match != 5"}),
            _mc("HH", "HH", "signal", "signal", {}),
        ),
        regions=(
            Region("same_sign", replace_cuts={"os": "(q_1 * q_2) > 0"}),
            Region("anti_iso", replace_cuts={"tau_iso": "(id_tight < 0.5) & (id_loose > 0.5)"}, add_weights={"fake_factor": "fake_factor"}),
        ),
        categories=(Category("sig", "cls == 0", SCORE), Category("bkg", "cls == 1", SCORE)),
        variables={"m_vis": M_VIS},
        variations=(WeightVariation("CMS_puUp", {"pu": "puweight_up"}), WeightVariation("CMS_puDown", {"pu": "puweight_down"})),
        estimators=(DataMinus("jetFakes", "anti_iso", ("ZTT", "ZL")),),
        keep_columns=("event",),
    )


def build(config=None) -> Analysis:
    return Analysis(
        name="mini",
        era="2018",
        lumi_pb=10.0,  # MC per-event weight ~1 so that data-driven estimates stay positive in the mini dataset
        signal="HH",
        channels={"mt": channel_mt()},
        lnn=(LnN("lumi", ("*",), 1.025), LnN("zjXsec", ("ZTT", "ZL"), 1.02), LnN("ffNorm_$CHANNEL", ("jetFakes",), 1.1)),
        style=Style(
            colors={"Z": "#3f90da", "jetFakes": "#b9ac70", "HH": "#bd1f01"},
            labels={"Z": r"Z$\rightarrow\ell\ell$", "jetFakes": r"jet$\rightarrow\tau_h$"},
            group_order=("jetFakes", "Z"),
            signal_label=r"HH$\rightarrow$bb$\tau\tau$",
            lumi_label="1 fb$^{-1}$ (2018, 13 TeV)",
            channel_labels={"mt": r"$\mu\tau_h$"},
            axis_labels={"mt": {"m_vis": r"$m_{vis}$ / GeV", "score": "NN output"}},
        ),
        ml=MLExportConfig(
            variables=("m_vis", "score"),
            processes=("ZTT", "ZL", "HH", "jetFakes"),
            label_of={"ZTT": "is_Z", "ZL": "is_Z", "HH": "is_HH", "jetFakes": "is_jetFakes"},
            region_of={"jetFakes": "anti_iso"},
        ),
    )


def embedding_variations() -> tuple[ColumnVariation, ...]:
    tes = tuple(ColumnVariation(f"CMS_tes{d}", f"__tes{d}", applies_to=("mc", "embedding")) for d in ("Up", "Down"))
    ff = tuple(ColumnVariation(f"CMS_ffStat{d}", f"__ffStat{d}", applies_to=("data", "mc", "embedding"), regions=("anti_iso",)) for d in ("Up", "Down"))
    return tes + ff


def build_embedding(config=None) -> Analysis:
    """EMB (genuine taus) replaces ZTT, which stays as the auxiliary template of the contamination variation."""
    analysis = build(config)
    channel = analysis.channel("mt")
    emb = Process("EMB", "EMB", "background", "EMB", Selection(cuts={"gen": "gen_match == 5"}, weights={"emb": "emb_genweight", "trg": "trg_wgt"}))
    processes = tuple(dataclasses.replace(p, role="auxiliary") if p.name == "ZTT" else p for p in channel.processes) + (emb,)
    channel = dataclasses.replace(
        channel,
        samples=channel.samples + (Sample("EMB_A", "EMB", "embedding"),),
        processes=processes,
        variations=channel.variations + embedding_variations(),
        estimators=(DataMinus("jetFakes", "anti_iso", ("EMB", "ZL")), TemplateShift("CMS_htt_emb_ttbar_2018", "EMB", "ZTT", 0.1)),
    )
    ml = dataclasses.replace(analysis.ml, processes=("EMB", "ZL", "HH", "jetFakes"), label_of={"EMB": "is_Z", "ZL": "is_Z", "HH": "is_HH", "jetFakes": "is_jetFakes"})
    return dataclasses.replace(analysis, channels={"mt": channel}, ml=ml)

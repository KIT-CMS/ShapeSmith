"""A tiny but complete Analysis used by the test suite (matches shapesmith.testing.make_mini_dataset)."""
from shapesmith.model import (
    Analysis, Category, Channel, Estimator, LnN, MLExportConfig, Process, Region, Sample, Selection, Style, Variable, WeightVariation,
)

SCORE = Variable("score", "score", (0.0, 0.5, 1.0))
M_VIS = Variable("m_vis", "m_vis", (0.0, 50.0, 100.0, 200.0))


def build(config=None) -> Analysis:
    channel = Channel(
        name="mt",
        skim=Selection(cuts={"veto": "veto < 0.5", "iso_loose": "id_loose > 0.5"}, weights={}),
        baseline=Selection(
            cuts={"veto": "veto < 0.5", "os": "(q_1 * q_2) < 0", "tau_iso": "id_tight > 0.5"},
            weights={"trg": "trg_wgt"},
        ),
        regions=(
            Region("same_sign", replace_cuts={"os": "(q_1 * q_2) > 0"}),
            Region("anti_iso", replace_cuts={"tau_iso": "(id_tight < 0.5) & (id_loose > 0.5)"}, add_weights={"fake_factor": "fake_factor"}),
        ),
        keep_columns=("event",),
    )
    samples = (
        Sample("DATA_A", "data", "data", channels=("mt",)),
        Sample("ZTT_1", "DY", "mc", xsec=10.0, nevents=100, generator_weight=1.0),
        Sample("SIG_1", "HH", "mc", xsec=1.0, nevents=50, generator_weight=1.0),
    )
    processes = (
        Process("data", "data", "data", "data", "data"),
        Process("ztt", "DY", "ZTT", "true_tau", "Z", Selection(cuts={"gen": "gen_match == 5"}, weights={"pu": "puweight"})),
        Process("zl", "DY", "ZL", "lepton_fake", "Z", Selection(cuts={"gen": "gen_match != 5"}, weights={"pu": "puweight"})),
        Process("hh", "HH", "HH", "signal", "signal", Selection(cuts={}, weights={"pu": "puweight"})),
    )
    return Analysis(
        name="mini",
        era="2018",
        lumi_pb=10.0,  # MC per-event weight ~1 so that data-driven estimates stay positive in the mini dataset
        channels={"mt": channel},
        samples=samples,
        processes=processes,
        signal="HH",
        categories=(Category("sig", "cls == 0", SCORE), Category("bkg", "cls == 1", SCORE)),
        control_variables={"m_vis": M_VIS},
        weight_variations=(WeightVariation("CMS_puUp", {"pu": "puweight_up"}), WeightVariation("CMS_puDown", {"pu": "puweight_down"})),
        column_variations=(),
        lnn=(LnN("lumi", ("*",), 1.025), LnN("zjXsec", ("ZTT", "ZL"), 1.02), LnN("ffNorm_$CHANNEL", ("jetFakes",), 1.1)),
        estimator=Estimator("fake_factors", regions={"anti_iso": "anti_iso"}, subtract=("ZTT", "ZL"), output="jetFakes"),
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

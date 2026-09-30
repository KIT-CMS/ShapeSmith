import dataclasses
import json

import correctionlib
import numpy as np
import pytest
import uproot

from shapesmith import payloads
from shapesmith.histogram import HistKey, Histogram, HistogramSet
from shapesmith.measurements import MeasureContext
from shapesmith.measurements.tau_id_es import TauIdEsMeasurement, check, combine, merge, plots
from shapesmith.measurements.tau_id_es.combine import Interval, Scan
from shapesmith.measurements.tau_id_es.grid import CATEGORIES, factor, grid, grid_name, mass, shift_of
from shapesmith.measurements.tau_id_es.payload import correction_set
from shapesmith.measurements.tau_id_es.synced import shapes_path, signal_name, write_synced
from shapesmith.model import Analysis, Category, Channel, ColumnVariation, DataMinus, Process, Sample, Variable

GRID = grid(-200, 200, 2)  # the predecessor's grid: +-20 % in steps of 0.2 %
M_VIS = Variable("m_vis", "m_vis", (30.0, 60.0, 90.0))


def morphing_masses(es_min=-20.0, es_max=20.0, precision=0.2):
    """The mass strings of MorphingTauID2017's loop: float accumulation, 0 skipped, printed with one decimal."""
    masses, value = [], np.float32(es_min)
    while value <= np.float32(es_max):
        if not -0.005 < value < 0.005:
            masses.append(f"{float(value):.1f}")
        value = np.float32(value + np.float32(precision))
    return masses


def test_grid_names_round_trip_and_match_morphing_masses():
    assert len(GRID) == 200 and 0 not in GRID
    assert all(shift_of(grid_name(shift)) == shift for shift in GRID) and shift_of("CMS_x_Up") is None
    assert grid_name(-198) == "es-198" and mass(-198) == "-19.8" and mass(0) == "0.0" and factor(-198) == 0.802
    assert morphing_masses() == [mass(shift) for shift in GRID[:-1]]  # the float loop drops the last point, +20.0


def _channel(name, categories, processes, variations=(), estimators=()):
    groups = {p.group for p in processes}
    samples = tuple(Sample(f"{group}_1", group, "data" if group == "data" else "embedding" if group in ("EMB", "MUEMB") else "mc") for group in groups)
    return Channel(name, samples, {}, {"os": "q_1 * q_2 < 0"}, tuple(processes), categories=tuple(categories), variations=variations, estimators=estimators)


def _analysis(shifts=(-2, 2)):
    data = Process("data", "data", "data", "data")
    mt = _channel("mt", [Category("DM0", "tau_decaymode_2 == 0", M_VIS)],
                  [data, Process("EMB", "EMB", "background", "EMB"), Process("ZL", "DY", "background", "Z")],
                  variations=tuple(ColumnVariation(grid_name(s), derived={"pt_2": f"pt_2 * {factor(s)}"}, applies_to=("embedding",)) for s in shifts),
                  estimators=(DataMinus("QCD", "same_sign", ("EMB", "ZL")),))
    mm = _channel("mm", [Category("control_region", "m_vis > 70", M_VIS)], [data, Process("MUEMB", "MUEMB", "background", "MUEMB")])
    return Analysis("tau_id", "2018", 1.0, "EMB", {"mt": mt, "mm": mm})


def _h(value):
    return Histogram(M_VIS.edges, [value, value], [value, value])


def test_synced_shapes_use_the_morphing_names(tmp_path):
    hset = HistogramSet()
    for channel, category, process, variation in (
        ("mt", "DM0", "data", "Nominal"), ("mt", "DM0", "EMB", "Nominal"), ("mt", "DM0", "EMB", "es-2"), ("mt", "DM0", "EMB", "es+2"),
        ("mt", "DM0", "EMB", "CMS_emb_ttbar_contamination_Run2018Up"), ("mt", "DM0", "EMB", "CMS_emb_ttbar_contamination_Run2018Up@es-2"),
        ("mt", "DM0", "ZL", "Nominal"), ("mt", "DM0", "ZL", "CMS_scale_j_TotalUp"), ("mt", "DM0", "QCD", "Nominal"),
        ("mm", "control_region", "data", "Nominal"), ("mm", "control_region", "MUEMB", "Nominal"),
    ):
        hset[HistKey(channel, category, process, "nominal", variation, "m_vis")] = _h(1.0)
    write_synced(hset, _analysis(), tmp_path)
    with uproot.open(shapes_path(tmp_path, "mt", "2018")) as f:
        assert sorted(f.keys(cycle=False)) == sorted(["mt_DM0"] + [f"mt_DM0/{name}" for name in (
            "data_obs", "EMB_DM0_0.0", "EMB_DM0_-0.2", "EMB_DM0_0.2", "EMB_DM0_0.0_CMS_emb_ttbar_contamination_Run2018Up",
            "EMB_DM0_-0.2_CMS_emb_ttbar_contamination_Run2018Up", "ZL", "ZL_CMS_scale_j_TotalUp", "QCD")])
    assert (tmp_path / "mm" / "htt_mm.inputs-sm-Run2018-TauID_ES.root").exists()
    assert signal_name("DM1011_PT40_200", "es+200") == "EMB_DM1011_PT40_200_20.0"


def test_check_names_what_the_chain_misses():
    check(_analysis(), TauIdEsMeasurement("Tight", "VVLoose", (-2, 2)))
    with pytest.raises(ValueError, match="grid variation es\\+4 is missing"):
        check(_analysis(), TauIdEsMeasurement("Tight", "VVLoose", (-2, 2, 4)))
    with pytest.raises(ValueError, match="channel mm is missing"):
        check(dataclasses.replace(_analysis(), channels={"mt": _analysis().channel("mt")}), TauIdEsMeasurement("Tight", "VVLoose", (-2, 2)))


def test_commands_have_the_predecessor_options(tmp_path):
    assert combine.morphing_command("DM0", tmp_path, "2018", GRID) == (
        f"MorphingTauID2017 --base_path={tmp_path.resolve()} --input_folder_mt=mt --input_folder_mm=mm --real_data=true --classic_bbb=false"
        " --binomial_bbb=false --jetfakes=0 --embedding=1 --verbose=false --postfix=-TauID_ES --use_control_region=true --auto_rebin=false"
        " --rebin_categories=false --manual_rebin_for_yields=false --categories=DM0 --era=2018 --tes_precision=0.2 --es_min=-20.0 --es_max=20.0 --output=DM0")
    assert combine.workspace_command("DM0") == ("combineTool.py -M T2W -i . -o workspace.root -m 125 -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel"
                                                " --PO \"map=^.*/EMB_DM0:r_EMB_DM0[1,0.1,2.9]\"")
    scan = combine.scan_command("DM0", GRID)
    assert "--setParameterRanges r_EMB_DM0=0.11,2.89:ES_DM0=-19.9,19.9" in scan and "--points=289 --algo grid" in scan
    singles = combine.singles_command("DM0", (0.9, -1.5), (0.7, 1.1), (-8.0, 4.0))
    assert "--setParameters ES_DM0=-1.5,r_EMB_DM0=0.9 --setParameterRanges r_EMB_DM0=0.7,1.1:ES_DM0=-8.0,4.0" in singles
    assert "--algo singles" in singles and "--cminFallbackAlgo Minuit2,Migrad,0:0.001,Minuit2,Migrad,0:0.01" in singles
    with pytest.raises(ValueError, match="uniform grid"):
        combine.morphing_command("DM0", tmp_path, "2018", (-4, -2, 2, 6))


def test_rate_parameters_tie_the_z_normalisation_of_both_regions(tmp_path):
    (tmp_path / "htt_mt_7_Run2018.txt").write_text("process W ZL EMB_DM0\n")
    (tmp_path / "htt_mm_100_Run2018.txt").write_text("process W MUEMB\n")
    combine.add_rate_parameters(tmp_path, "DM0")
    assert (tmp_path / "htt_mt_7_Run2018.txt").read_text().splitlines()[1:] == [
        "r_DY_incl_DM0 rateParam * EMB_DM0 1.0 [0.5,1.5]", "r_DY_incl_DM0 rateParam * ZL 1.0 [0.5,1.5]", "r_DY_incl_DM0 rateParam * ZJ 1.0 [0.5,1.5]"]
    assert (tmp_path / "htt_mm_100_Run2018.txt").read_text().splitlines()[1:] == ["r_DY_incl_DM0 rateParam * MUEMB 1.0 [0.5,1.5]"]


def _write_limit(path, **columns):
    with uproot.recreate(path) as f:
        f["limit"] = {name: np.asarray(values, dtype=np.float32) for name, values in columns.items()}


def test_read_singles_and_its_problems(tmp_path):
    _write_limit(tmp_path / "fit.root", r_EMB_DM0=[0.9, 0.85, 0.95, 0.9, 0.9], ES_DM0=[-1.8, -1.8, -1.8, -4.4, -0.1], quantileExpected=[-1, -0.32, 0.32, -0.32, 0.32])
    sf, es = combine.read_singles(tmp_path / "fit.root", "DM0")
    assert (sf.best, sf.low, sf.high) == pytest.approx((0.9, 0.85, 0.95)) and (es.low, es.high) == pytest.approx((-4.4, -0.1))
    assert combine.interval_problems("r", sf, (0.7, 1.1)) == []
    assert combine.interval_problems("r", sf, (0.85, 1.1))[0].startswith("r: interval [0.85, 0.95] at the fit range")
    assert "no interval" in combine.interval_problems("ES", Interval(1.0, 1.0, 2.0), (-5.0, 5.0))[0]


def _scan(best=(0.9, -2.0), sigma=(0.05, 2.0)):
    r, es = np.meshgrid(np.linspace(0.11, 2.89, 17), np.linspace(-19.9, 19.9, 17))
    dnll = 0.5 * (((r - best[0]) / sigma[0]) ** 2 + ((es - best[1]) / sigma[1]) ** 2)
    return Scan(best, r.ravel(), es.ravel(), dnll.ravel())


def test_singles_inputs_from_the_scan():
    start, r_range, es_range, problems = combine.singles_inputs(_scan(sigma=(0.2, 5.0)), GRID)
    assert start == (0.9, -2.0) and problems == []
    assert r_range[0] < 0.9 - 0.2 and r_range[1] > 0.9 + 0.2 and es_range[0] < -7.0 and es_range[1] > 3.0  # the 1 sigma interval inside
    lost = dataclasses.replace(_scan(sigma=(0.2, 5.0)), best=(0.11, -19.9))  # the scan's initial fit ended in a corner
    assert combine.singles_inputs(lost, GRID)[0] == pytest.approx((0.97875, -2.4875))  # the lowest grid point
    assert combine.singles_inputs(_scan(best=(0.9, 18.0), sigma=(0.2, 5.0)), GRID)[3] == ["the ES scan region reaches the scan boundary"]


def _results(offset=0.0):
    return {name: (Interval(0.9 + offset, 0.85 + offset, 0.95 + offset), Interval(-1.0, -3.0, 1.0 + index)) for index, name in enumerate(CATEGORIES)}


def test_payload_structure_and_values():
    results = {("Medium", "VVLoose"): _results(), ("Tight", "Tight"): _results(0.1)}
    cset = correctionlib.CorrectionSet.from_string(payloads.dumps(correction_set(results, {"era": "2018"})))
    sf, es, es_dm = cset["DeepTau2018v2p5VSjet"], cset["tau_energy_scale"], cset["tau_energy_scale_dm_binned"]
    assert sf.evaluate(30.0, 0, 6, "Medium", "VVLoose", "nom", "pt") == 1.0  # genmatch != 5
    assert sf.evaluate(30.0, 0, 5, "Medium", "VVLoose", "nom", "dm") == pytest.approx(0.9)
    assert sf.evaluate(500.0, 1, 5, "Tight", "Tight", "up", "dm") == pytest.approx(1.05)
    assert sf.evaluate(25.0, 10, 5, "Medium", "VVLoose", "down", "pt") == sf.evaluate(25.0, 11, 5, "Medium", "VVLoose", "down", "pt")  # both DM1011
    index = list(CATEGORIES).index
    assert es.evaluate(50.0, -2.1, 11, 5, "DeepTau2018v2p5", "Medium", "VVLoose", "up") == pytest.approx((100 + 1.0 + index("DM1011_PT40_200")) / 100)
    assert es.evaluate(25.0, 0.3, 0, 5, "DeepTau2018v2p5", "Medium", "VVLoose", "nom") == pytest.approx(0.99)
    assert es_dm.evaluate(300.0, 0.3, 1, 5, "DeepTau2018v2p5", "Tight", "Tight", "up") == pytest.approx((100 + 1.0 + index("DM1")) / 100)
    assert es_dm.evaluate(300.0, 0.3, 1, 2, "DeepTau2018v2p5", "Tight", "Tight", "up") == 1.0


def _record(directory, vsjet, vsele, problems=()):
    fits = {name: {"sf": sf.__dict__, "es": es.__dict__, "problems": list(problems)} for name, (sf, es) in _results().items()}
    directory.mkdir(parents=True)
    (directory / "results.json").write_text(json.dumps({"vsjet_wp": vsjet, "vsele_wp": vsele, "categories": fits, "provenance": {"run": f"{vsjet}_{vsele}"}}))


def test_merge_writes_the_payload_of_every_working_point(tmp_path, mini_run):
    config, analysis = mini_run
    _record(tmp_path / "Medium_VVLoose", "Medium", "VVLoose")
    _record(tmp_path / "Tight_VVLoose", "Tight", "VVLoose")
    context = MeasureContext(config, analysis, ["mt"], tmp_path, merge=True)
    path = merge(context)
    assert path == tmp_path / "DeepTau2018v2p5_id_es_embedding2018UL.json.gz"
    back = payloads.read(path)
    assert payloads.provenance(back)["measurements"] == {"Medium_VVLoose": {"run": "Medium_VVLoose"}, "Tight_VVLoose": {"run": "Tight_VVLoose"}}
    assert back.corrections[0].inputs[3].description == "DeepTau2018v2p5VSjet working point: Medium,Tight"
    _record(tmp_path / "Tight_Tight", "Tight", "Tight", problems=["r: interval at the fit range"])
    with pytest.raises(ValueError, match="Tight_Tight/DM0: r: interval at the fit range"):
        merge(context)


def test_plots_are_written(tmp_path):
    scan = _scan(sigma=(0.2, 5.0))
    assert plots.plot_scan(scan, Interval(0.9, 0.7, 1.1), Interval(-2.0, -7.0, 3.0), "DM0", tmp_path / "scan")[0].exists()
    hset = HistogramSet()
    for process, variation, value in (("data", "Nominal", 3.0), ("EMB", "Nominal", 2.0), ("EMB", "es-2", 1.8), ("ZL", "Nominal", 1.0)):
        hset[HistKey("mt", "DM0", process, "nominal", variation, "m_vis")] = _h(value)
    assert plots.plot_control(hset, _analysis(), "DM0", tmp_path / "control")[0].exists()
    with uproot.recreate(tmp_path / "postfit_shapes.root") as f:
        for name, value in (("data_obs", 3.0), ("EMB_DM0", 2.0), ("ZL", 1.0), ("TotalProcs", 3.0)):
            f[f"htt_mt_7_Run2018_postfit/{name}"] = _h(value).to_root(name)
    assert [p.name for p in plots.plot_postfit(tmp_path / "postfit_shapes.root", tmp_path / "postfit")] == ["htt_mt_7_Run2018_postfit.pdf", "htt_mt_7_Run2018_postfit.png"]

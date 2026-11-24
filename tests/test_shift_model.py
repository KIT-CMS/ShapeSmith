"""The parts of the model that a production with many CROWN shifts needs: declared-only shifts (regions=()), the
columns a shift requires shifted (a friend produced without the shift), summed variations with a scale and a source
(partial correlations) and the quadrature sum of templates (Hessian eigenvectors)."""
import dataclasses

import numpy as np
import pytest
import uproot

from shapesmith import model
from shapesmith.config import FriendConfig, NtupleConfig, RunConfig
from shapesmith.datacards import write_datacard
from shapesmith.estimates import run_estimates
from shapesmith.fill import bookings, plan_booking, run_hist, targets
from shapesmith.histogram import INCLUSIVE, HistKey, Histogram, HistogramSet, part_of
from shapesmith.model import ColumnVariation, DataMinus, QuadratureSum, VariationSum, WeightVariation, declared_only
from shapesmith.skim import check_shifts, column_variations, read_variations, region_columns, required_shifted, run_skim
from shapesmith.store import read_manifest, read_skims
from shapesmith.testing import make_mini_dataset
from shapesmith.validate import validate
from tests.mini_analysis import build, build_embedding

TES = ("CMS_tesUp", "CMS_tesDown")
FF = ("CMS_ffStatUp", "CMS_ffStatDown")


@pytest.fixture
def config(tmp_path):
    make_mini_dataset(tmp_path / "data", n_files=2, n_events=300, variations=True)
    return RunConfig(
        analysis="tests.mini_analysis:build_embedding",
        era="2018",
        channels=["mt"],
        ntuples=NtupleConfig(base=str(tmp_path / "data" / "CROWNRun"), friends=[FriendConfig(base=str(tmp_path / "data" / "CROWNFriends" / "nn"))]),
        skim_dir=tmp_path / "skims",
        output_dir=tmp_path / "out",
        workers=1,
    )


def _with_channel(analysis, **changes):
    return dataclasses.replace(analysis, channels={"mt": dataclasses.replace(analysis.channel("mt"), **changes)})


def _replaced(analysis, **changes):
    """The embedding analysis with the fields of its CMS_tes pair changed (e.g. regions=() or required=...)."""
    channel = analysis.channel("mt")
    variations = tuple(dataclasses.replace(v, **changes) if v.name in TES else v for v in channel.variations)
    return _with_channel(analysis, variations=variations)


def _sample(analysis, nick):
    return next(s for s in analysis.channel("mt").samples if s.nick == nick)


# ---- declared-only shifts ----


def test_a_variation_without_regions_is_declared_only():
    assert declared_only(ColumnVariation("a", "__a", regions=()))
    assert not declared_only(ColumnVariation("a", "__a")) and not declared_only(ColumnVariation("a", "__a", regions=("nominal",)))
    analysis = _replaced(build_embedding(), regions=())
    validate(analysis)
    channel, sample = analysis.channel("mt"), _sample(analysis, "EMB_A")
    assert {v.name for v in column_variations(channel, sample)} == set(TES + FF)  # declared
    assert {v.name for v in read_variations(channel, sample)} == set(FF)  # but not read


def test_a_declared_only_shift_is_never_filled():
    analysis = _replaced(build_embedding(), regions=())
    channel = analysis.channel("mt")
    available = {"m_vis", "m_vis__tesUp", "m_vis__tesDown", "fake_factor", "fake_factor__ffStatUp", "fake_factor__ffStatDown"}
    planned = [fill for booking in bookings(channel, True, ["all"]) for fill in plan_booking(channel, booking, targets(channel, True, None), available)]
    filled = {fill.variation.name for fill in planned if fill.variation is not None}
    assert not filled & set(TES) and set(FF) <= filled
    read = [fill for booking in bookings(build_embedding().channel("mt"), True, ["all"]) for fill in plan_booking(build_embedding().channel("mt"), booking, targets(channel, True, None), available)]
    assert set(TES) <= {fill.variation.name for fill in read if fill.variation is not None}  # filled when it has regions


def test_the_skim_checks_a_declared_only_shift_but_reads_nothing_for_it(config):
    analysis = _replaced(build_embedding(), regions=())
    analysis = _with_channel(analysis, skim={**analysis.channel("mt").skim, "mass": "m_vis > 100"})
    run_skim(config, analysis)
    emb = read_skims(config.skim_dir, "mt", ["EMB_A"], None)
    assert "m_vis__tesUp" not in emb.columns and "fake_factor__ffStatUp" in emb.columns
    assert (emb["m_vis"] > 100).all()  # no event kept for passing the cut only under the declared-only shift
    contract = read_manifest(config.skim_dir / "mt" / "EMB_A" / "manifest.json")["contract"]
    assert contract["column_variations"] == {"CMS_ffStatUp": "__ffStatUp", "CMS_ffStatDown": "__ffStatDown"}
    hset = run_hist(config, analysis, ["mt"], True, None, True, None, config.output_dir / "control.root")
    assert not [key for key in hset if key.variation in TES]
    assert hset.select(process="EMB", variation="CMS_ffStatUp")


def test_a_skim_with_the_shift_read_serves_the_declared_only_run(config):
    run_skim(config, build_embedding())
    assert all(r.skipped for r in run_skim(config, _replaced(build_embedding(), regions=())))


def test_a_declared_only_shift_needs_some_branch_with_its_suffix():
    """The column a declared-only shift moves may be one the sample does not read (e.g. a weight no process carries):
    any branch with the suffix counts; without one the production lacks the shift."""
    analysis = build_embedding()
    channel = analysis.channel("mt")
    scale = tuple(ColumnVariation(f"QCDscale{d}", f"__QCDscale{d}", regions=()) for d in ("Up", "Down"))
    sample = _sample(analysis, "ZTT_1")
    channel = dataclasses.replace(channel, variations=tuple(v for v in channel.variations if v.name not in FF) + scale)
    columns = {"m_vis", "score"}
    branches = {"m_vis", "score", "m_vis__tesUp", "m_vis__tesDown", "lhe_scale_weight", "lhe_scale_weight__QCDscaleUp", "lhe_scale_weight__QCDscaleDown"}
    check_shifts(channel, sample, columns, branches)
    with pytest.raises(ValueError, match=r"declared shift QCDscaleDown \(__QCDscaleDown\) has no shifted branch"):
        check_shifts(channel, sample, columns, branches - {"lhe_scale_weight__QCDscaleDown"})
    # read as a variation, the same shift needs a shifted branch of a column the sample reads
    read = tuple(dataclasses.replace(v, regions=None) for v in scale)
    with pytest.raises(ValueError, match=r"declared shift QCDscaleUp \(__QCDscaleUp\) has no shifted branch"):
        check_shifts(dataclasses.replace(channel, variations=channel.variations[:-2] + read), sample, columns, branches)


def test_an_undeclared_shift_still_fails_next_to_declared_only_ones():
    analysis = _replaced(build_embedding(), regions=())
    channel = _with_channel(analysis, variations=tuple(v for v in analysis.channel("mt").variations if v.name not in FF)).channel("mt")
    branches = {"m_vis", "m_vis__tesUp", "m_vis__tesDown", "fake_factor", "fake_factor__ffStatUp", "fake_factor__ffStatDown"}
    with pytest.raises(ValueError, match="undeclared shift __ffStatDown; undeclared shift __ffStatUp"):
        check_shifts(channel, _sample(analysis, "EMB_A"), {"m_vis", "fake_factor"}, branches)


def test_a_declared_only_shift_gives_no_card_line(tmp_path):
    """Filled by mistake under another name it would be a shape line; declared only, the card and the shapes file do
    not know its name."""
    analysis = _replaced(build_embedding(), regions=())
    hset = HistogramSet()
    for category in ("sig", "bkg"):
        for process, values in (("data", [20.0, 10.0]), ("EMB", [10.0, 5.0]), ("ZL", [5.0, 3.0]), ("HH", [0.1, 0.2]), ("jetFakes", [4.0, 2.0])):
            hset[HistKey("mt", category, process, "nominal", "Nominal", "score")] = Histogram([0.0, 0.5, 1.0], values, values)
        for d, factor in (("Up", 1.1), ("Down", 0.9)):
            hset[HistKey("mt", category, "jetFakes", "nominal", f"CMS_ffStat{d}", "score")] = Histogram([0.0, 0.5, 1.0], [4.0 * factor, 2.0 * factor], [4.0, 2.0])
    card = write_datacard(hset, analysis, ["mt"], tmp_path / "mt")
    text = card.read_text()
    assert "CMS_ffStat shape" in " ".join(text.split()) and "CMS_tes" not in text
    with uproot.open(card.parent / "common" / "htt_input_2018.root") as f:
        assert not [name for name in f.keys(recursive=True, cycle=False) if "CMS_tes" in name]


# ---- required columns ----


def test_region_columns_are_those_of_the_regions():
    analysis = build_embedding()
    channel, sample = analysis.channel("mt"), _sample(analysis, "EMB_A")
    nominal = region_columns(channel, sample, ("nominal",))
    assert {"score", "cls", "m_vis", "id_tight", "emb_genweight"} <= nominal and "fake_factor" not in nominal
    assert "fake_factor" in region_columns(channel, sample, ("anti_iso",)) and "fake_factor" in region_columns(channel, sample, None)


@pytest.mark.parametrize("regions,required,problem", [
    (("nominal",), ("score",), "the required column score has no shifted branch"),
    (None, ("score", "cls"), "the required column cls has no shifted branch"),
    (("nominal",), ("fake_factor",), None),  # read in the anti-isolated region only, where the shift is not filled
    (("anti_iso",), ("fake_factor",), "the required column fake_factor has no shifted branch"),
    (None, ("fake_factor",), "the required column fake_factor has no shifted branch"),
    (("nominal",), ("not_read",), None),  # a column no expression reads
])
def test_a_required_column_must_exist_shifted_where_a_region_of_the_shift_reads_it(regions, required, problem):
    analysis = _replaced(build_embedding(), regions=regions, required=required)
    validate(analysis)
    channel, sample = analysis.channel("mt"), _sample(analysis, "EMB_A")
    columns = {"m_vis", "score", "cls", "fake_factor"}
    main = {"m_vis", "m_vis__tesUp", "m_vis__tesDown", "score", "cls", "fake_factor", "fake_factor__ffStatUp", "fake_factor__ffStatDown"}
    shifted = {f"{c}__tes{d}" for c in required for d in ("Up", "Down")}
    check_shifts(channel, sample, columns, main | shifted)  # the friend has the shifted columns
    if problem is None:
        check_shifts(channel, sample, columns, main)
        return
    with pytest.raises(ValueError, match=rf"declared shift CMS_tesUp \(__tesUp\): {problem}"):
        check_shifts(channel, sample, columns, main)


def test_a_friend_without_the_shift_fails_the_skim(config):
    """The friend of the mini dataset has no score__tesUp: with the score required under the shift the skim stops
    instead of filling the nominal score under it."""
    run_skim(config, build_embedding(), samples=["EMB"])  # not required: the nominal score is read under the shift
    with pytest.raises(RuntimeError, match=r"EMB_A_0.root: declared shift CMS_tesUp \(__tesUp\): the required column cls has no shifted branch") as excinfo:
        run_skim(config, _replaced(build_embedding(), required=("score", "cls")), force=True)
    assert "declared shift CMS_tesDown (__tesDown): the required column score has no shifted branch" in str(excinfo.value)
    assert "DATA_A" not in str(excinfo.value)  # the shift does not apply to data


def test_a_skim_without_the_required_shifted_columns_is_not_reused(config):
    """A skim made while the score was not required holds no score__tesUp (the friend has none): it does not serve a
    run that requires it, where the fill would read the nominal score under the shift."""
    run_skim(config, build_embedding())
    required = _replaced(build_embedding(), required=("score",))
    with pytest.raises(ValueError, match=r"mt/EMB_A: required shifted columns missing: score__tesDown, score__tesUp") as excinfo:
        run_skim(config, required)
    assert "DATA_A" not in str(excinfo.value)
    channel, sample = required.channel("mt"), _sample(required, "EMB_A")
    assert required_shifted(channel, sample) == {"CMS_tesUp": ["score__tesUp"], "CMS_tesDown": ["score__tesDown"]}
    assert required_shifted(channel, _sample(required, "DATA_A")) == {}
    # a required column that no region of the shift reads asks for nothing, and the skim is reused
    elsewhere = _replaced(build_embedding(), required=("fake_factor",), regions=("nominal",))
    assert required_shifted(elsewhere.channel("mt"), _sample(elsewhere, "EMB_A")) == {}
    assert all(r.skipped for r in run_skim(config, elsewhere))


def test_required_columns_need_a_suffix():
    analysis = build_embedding()
    derived = ColumnVariation("CMS_xUp", derived={"m_vis": "m_vis * 1.1"}, required=("score",)), ColumnVariation("CMS_xDown", derived={"m_vis": "m_vis * 0.9"})
    with pytest.raises(model.AnalysisError, match="variation CMS_xUp: required columns need a suffix"):
        validate(_with_channel(analysis, variations=analysis.channel("mt").variations + derived))


# ---- one nuisance from differently named shifts ----


def test_column_variations_of_disjoint_groups_may_share_a_name(config):
    """A nuisance whose shift is named differently for some samples (e.g. those borrowed from another era): one
    variation per group of samples under the one name; the same name on a common sample stays a duplicate."""
    analysis = build_embedding()
    channel = analysis.channel("mt")
    others = tuple(v for v in channel.variations if v.name not in TES + FF)
    ff = tuple(v for v in channel.variations if v.name in FF)
    split = tuple(ColumnVariation(f"CMS_tes{d}", f"__tes{d}", applies_to=("mc", "embedding"), groups=groups) for groups in (("DY", "EMB"), ("HH",)) for d in ("Up", "Down"))
    shared = _with_channel(analysis, variations=others + ff + split)
    validate(shared)
    assert [v.groups for v in column_variations(shared.channel("mt"), _sample(shared, "SIG_1")) if v.name == "CMS_tesUp"] == [("HH",)]
    run_skim(config, shared)
    hset = run_hist(config, shared, ["mt"], True, None, True, None, config.output_dir / "control.root")
    assert hset.select(process="HH", variation="CMS_tesUp") and hset.select(process="EMB", variation="CMS_tesUp")
    overlapping = tuple(dataclasses.replace(v, groups=("DY", "HH")) if v.groups == ("HH",) else v for v in split)
    with pytest.raises(model.AnalysisError, match="duplicate variation CMS_tesDown\n.*duplicate variation CMS_tesUp"):
        validate(_with_channel(analysis, variations=others + ff + overlapping))
    for clash in (ColumnVariation("CMS_tesUp", "__tesUp", applies_to=("mc", "embedding")), WeightVariation("CMS_tesUp", {"pu": "puweight"})):  # no groups: every sample
        with pytest.raises(model.AnalysisError, match="duplicate variation CMS_tesUp"):
            validate(_with_channel(analysis, variations=others + ff + split + (clash,)))
    # other kinds never meet
    data_only = tuple(ColumnVariation(f"CMS_tes{d}", f"__other{d}", applies_to=("data",)) for d in ("Up", "Down"))
    validate(_with_channel(analysis, variations=others + ff + split + data_only))


# ---- summed variations with a scale and a source ----


def _hist(values, variances=None):
    return Histogram([0.0, 1.0, 2.0], values, variances if variances is not None else values)


def _key(process, variation="Nominal", region="nominal"):
    return HistKey("mt", INCLUSIVE, process, region, variation, "x")


def _parts(name, tokens, kinds=("mc", "embedding")):
    return tuple(ColumnVariation(part_of(f"{name}{d}", token), f"__{token}{d}", applies_to=kinds) for token in tokens for d in ("Up", "Down"))


def test_a_scaled_sum_takes_a_fraction_of_the_summed_difference():
    analysis = build_embedding()
    channel = analysis.channel("mt")
    estimators = (DataMinus("jetFakes", "anti_iso", ("EMB", "ZL")), VariationSum("CMS_own", ("dm10", "dm11"), scale=0.866))
    analysis = _with_channel(analysis, variations=channel.variations + _parts("CMS_own", ("dm10", "dm11")), estimators=estimators)
    validate(analysis)
    hset = HistogramSet()
    hset[_key("EMB")] = _hist([10.0, 20.0], [1.0, 2.0])
    hset[_key("EMB", part_of("CMS_ownUp", "dm10"))] = _hist([12.0, 20.0], [5.0, 5.0])
    hset[_key("EMB", part_of("CMS_ownUp", "dm11"))] = _hist([10.0, 26.0])
    hset[_key("EMB", part_of("CMS_ownDown", "dm10"))] = _hist([9.0, 20.0])
    run_estimates(hset, analysis, ["mt"])
    up = hset[_key("EMB", "CMS_ownUp")]
    assert up.values.tolist() == pytest.approx([10 + 0.866 * 2, 20 + 0.866 * 6]) and up.variances.tolist() == [1.0, 2.0]
    assert hset[_key("EMB", "CMS_ownDown")].values.tolist() == pytest.approx([10 - 0.866, 20.0])


def test_two_sums_share_the_parts_of_one_shift_with_their_own_scales():
    """A partial correlation: the shift of the embedded sample enters the nuisance it shares with the simulation with
    0.5 and its own nuisance with 0.866 (0.5^2 + 0.866^2 = 1); the simulation carries the shared nuisance as a plain
    column variation, which the sum leaves alone."""
    analysis = build_embedding()
    channel = analysis.channel("mt")
    shared = tuple(ColumnVariation(f"CMS_shared{d}", f"__mc{d}", applies_to=("mc",)) for d in ("Up", "Down"))
    estimators = (
        *channel.estimators,
        VariationSum("CMS_own", ("dm0",), scale=0.866),
        VariationSum("CMS_shared", ("dm0",), scale=0.5, source="CMS_own"),
    )
    analysis = _with_channel(analysis, variations=channel.variations + _parts("CMS_own", ("dm0",), ("embedding",)) + shared, estimators=estimators)
    validate(analysis)
    hset = HistogramSet()
    hset[_key("EMB")] = _hist([10.0, 20.0])
    hset[_key("EMB", part_of("CMS_ownUp", "dm0"))] = _hist([14.0, 20.0])
    hset[_key("EMB", part_of("CMS_ownDown", "dm0"))] = _hist([8.0, 20.0])
    hset[_key("ZL")] = _hist([5.0, 5.0])
    hset[_key("ZL", "CMS_sharedUp")] = _hist([6.0, 5.0])
    hset[_key("ZL", "CMS_sharedDown")] = _hist([4.5, 5.0])
    run_estimates(hset, analysis, ["mt"])
    own_up, shared_up = hset[_key("EMB", "CMS_ownUp")].values, hset[_key("EMB", "CMS_sharedUp")].values
    assert own_up.tolist() == pytest.approx([10 + 0.866 * 4, 20.0]) and shared_up.tolist() == pytest.approx([12.0, 20.0])
    assert hset[_key("EMB", "CMS_sharedDown")].values.tolist() == pytest.approx([9.0, 20.0])
    # the two parts of the embedded shift add up to the whole in quadrature
    assert np.hypot(own_up - 10.0, shared_up - 10.0)[0] == pytest.approx(4.0, rel=1e-4)
    assert hset[_key("ZL", "CMS_sharedUp")].values.tolist() == [6.0, 5.0]  # the column variation of the simulation, untouched
    assert not hset.select(process="ZL", variation="CMS_ownUp")


def test_validate_checks_scales_sources_and_duplicate_sums():
    analysis = build_embedding()
    channel = analysis.channel("mt")
    variations = channel.variations + _parts("CMS_own", ("dm0",))

    def problems(*estimators):
        return _with_channel(analysis, variations=variations, estimators=(*channel.estimators, *estimators))

    validate(problems(VariationSum("CMS_own", ("dm0",)), VariationSum("CMS_shared", ("dm0",), 0.5, "CMS_own")))
    for estimators, problem in (
        ((VariationSum("CMS_own", ("dm0",), scale=0.0),), "variation sum CMS_own: the scale must be finite and not 0"),
        ((VariationSum("CMS_own", ("dm0",), scale=float("nan")),), "the scale must be finite and not 0"),
        ((VariationSum("CMS_shared", ("dm0",), 0.5, "CMS_other"),), r"variation sum CMS_shared: part CMS_otherUp%dm0 is not a column variation"),
        ((VariationSum("CMS_own", ("dm0",)), VariationSum("CMS_own", ("dm0",), 0.5)), "duplicate summed variation CMS_own"),
    ):
        with pytest.raises(model.AnalysisError, match=problem):
            validate(problems(*estimators))


# ---- quadrature sums ----


def _eigenvectors(n):
    """`n` templates of one weight each, the parts of the quadrature sum pdf (as the eigenvector weights of a PDF set)."""
    return tuple(WeightVariation(part_of("pdf", f"{i}"), {"pu": f"puweight * pdf_weight_{i}"}) for i in range(1, n + 1))


def test_a_quadrature_sum_is_the_root_of_the_squared_differences_per_bin():
    analysis = build()
    channel = analysis.channel("mt")
    analysis = _with_channel(analysis, variations=channel.variations + _eigenvectors(3), estimators=(*channel.estimators, QuadratureSum("pdf", ("1", "2", "3"))))
    validate(analysis)
    hset = HistogramSet()
    hset[_key("HH")] = _hist([10.0, 20.0], [1.0, 2.0])
    hset[_key("HH", part_of("pdf", "1"))] = _hist([13.0, 20.0])  # +3, 0
    hset[_key("HH", part_of("pdf", "2"))] = _hist([6.0, 21.0])  # -4, +1
    hset[_key("HH", part_of("pdf", "3"))] = _hist([10.0, 20.0])  # an eigenvector without effect
    hset[_key("ZTT")] = _hist([5.0, 5.0])  # no part: no variation
    run_estimates(hset, analysis, ["mt"])
    up, down = hset[_key("HH", "pdfUp")], hset[_key("HH", "pdfDown")]
    assert up.values.tolist() == pytest.approx([15.0, 21.0]) and down.values.tolist() == pytest.approx([5.0, 19.0])
    assert up.variances.tolist() == down.variances.tolist() == [1.0, 2.0]
    assert not hset.select(process="ZTT", variation="pdfUp")


def test_a_quadrature_sum_differs_from_the_sum_per_event():
    """Per bin, eigenvectors that move events between bins cancel in the inclusive yield; the envelope of the bins is
    not the envelope of the total."""
    analysis = build()
    channel = analysis.channel("mt")
    analysis = _with_channel(analysis, variations=channel.variations + _eigenvectors(2), estimators=(*channel.estimators, QuadratureSum("pdf", ("1", "2"))))
    hset = HistogramSet()
    hset[_key("HH")] = _hist([10.0, 10.0])
    hset[_key("HH", part_of("pdf", "1"))] = _hist([11.0, 9.0])  # a migration: the total stays
    hset[_key("HH", part_of("pdf", "2"))] = _hist([10.0, 10.0])
    run_estimates(hset, analysis, ["mt"])
    assert hset[_key("HH", "pdfUp")].values.tolist() == [11.0, 11.0] and hset[_key("HH", "pdfDown")].values.tolist() == [9.0, 9.0]


def test_the_down_side_of_a_quadrature_sum_is_clipped_at_zero():
    analysis = build()
    channel = analysis.channel("mt")
    analysis = _with_channel(analysis, variations=channel.variations + _eigenvectors(1), estimators=(*channel.estimators, QuadratureSum("pdf", ("1",))))
    hset = HistogramSet()
    hset[_key("HH")] = _hist([1.0, 10.0])
    hset[_key("HH", part_of("pdf", "1"))] = _hist([3.0, 10.0])
    run_estimates(hset, analysis, ["mt"])
    assert hset[_key("HH", "pdfDown")].values.tolist() == [0.0, 10.0] and hset[_key("HH", "pdfUp")].values.tolist() == [3.0, 10.0]


def test_the_parts_of_a_quadrature_sum_stay_out_of_the_card(tmp_path):
    analysis = build()
    channel = analysis.channel("mt")
    analysis = _with_channel(analysis, variations=channel.variations + _eigenvectors(2), estimators=(*channel.estimators, QuadratureSum("pdf", ("1", "2"))))
    hset = HistogramSet()
    for category in ("sig", "bkg"):
        for process, values in (("data", [20.0, 10.0]), ("ZTT", [10.0, 5.0]), ("ZL", [5.0, 3.0]), ("HH", [1.0, 2.0]), ("jetFakes", [4.0, 2.0])):
            hset[HistKey("mt", category, process, "nominal", "Nominal", "score")] = Histogram([0.0, 0.5, 1.0], values, values)
        hset[HistKey("mt", category, "HH", "nominal", part_of("pdf", "1"), "score")] = Histogram([0.0, 0.5, 1.0], [1.1, 2.0], [1.0, 2.0])
        hset[HistKey("mt", category, "HH", "nominal", part_of("pdf", "2"), "score")] = Histogram([0.0, 0.5, 1.0], [1.0, 1.8], [1.0, 2.0])
    run_estimates(hset, analysis, ["mt"])
    card = write_datacard(hset, analysis, ["mt"], tmp_path / "mt", min_background=None)
    text = " ".join(card.read_text().split())
    assert "pdf shape" in text and "%" not in text
    with uproot.open(card.parent / "common" / "htt_input_2018.root") as f:
        names = f.keys(recursive=True, cycle=False)
        assert "htt_mt_1_2018/HH_pdfUp" in names and "htt_mt_1_2018/HH_pdfDown" in names and not [name for name in names if "%" in name]
        assert f["htt_mt_1_2018/HH_pdfUp"].values().tolist() == pytest.approx([1.1, 2.2])


def test_validate_checks_quadrature_sums():
    analysis = build()
    channel = analysis.channel("mt")
    variations = channel.variations + _eigenvectors(2)
    validate(_with_channel(analysis, variations=variations, estimators=(*channel.estimators, QuadratureSum("pdf", ("1", "2")))))
    for estimator, problem in (
        (QuadratureSum("pdfUp", ("1",)), "quadrature sum pdfUp: the name carries a direction"),
        (QuadratureSum("pdf", ()), "quadrature sum pdf: no parts"),
        (QuadratureSum("pdf", ("1", "3")), r"quadrature sum pdf: part pdf%3 is not a variation"),
    ):
        with pytest.raises(model.AnalysisError, match=problem):
            validate(_with_channel(analysis, variations=variations, estimators=(*channel.estimators, estimator)))
    # its Up/Down pair can be written as an lnN of the yield ratios, like that of any estimator
    with_lnn = _with_channel(analysis, variations=variations, estimators=(*channel.estimators, QuadratureSum("pdf", ("1", "2"))))
    validate(dataclasses.replace(with_lnn, lnn_variations=("pdf",)))

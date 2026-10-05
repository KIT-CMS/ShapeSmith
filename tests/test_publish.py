import dataclasses
import json
import os
import re
import shutil
import stat

import numpy as np
import pytest
from typer.testing import CliRunner

from shapesmith.cli import app
from shapesmith.config import NtupleConfig, RunConfig, WebConfig, load_config
from shapesmith.histogram import INCLUSIVE, HistKey, Histogram, HistogramSet
from shapesmith.measurements import run_measure
from shapesmith.measurements.fake_factors import gallery as ff_gallery
from shapesmith.measurements.fake_factors.model import Split
from shapesmith.model import Variable
from shapesmith.plotting import stack as stack_module
from shapesmith.plotting.stack import run_plot
from shapesmith.plotting.style import plain_text
from shapesmith.web import gallery, publish
from tests import ff_mini
from tests.mini_analysis import build
from tests.test_cli import _run, _write_config

runner = CliRunner()
INDEX = '<link rel="stylesheet" href="viewer.css?v=__SHAPESMITH_STAMP__">\n<script src="manifest.js?v=__SHAPESMITH_STAMP__"></script>\n<script src="viewer.js?v=__SHAPESMITH_STAMP__"></script>\n'


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    """The mini dataset taken through the chain: control plots in the nominal and the same_sign region, NN category plots."""
    tmp_path = tmp_path_factory.mktemp("mini")
    path = _write_config(tmp_path)
    _run("skim", "-c", str(path))
    _run("hist", "-c", str(path), "--control", "--regions", "all")
    _run("estimate", "-c", str(path), "--control")
    _run("plot", "-c", str(path), "--control")
    _run("plot", "-c", str(path), "--control", "--region", "same_sign")
    _run("hist", "-c", str(path))
    _run("estimate", "-c", str(path))
    _run("plot", "-c", str(path))
    config = load_config(path)
    return path, config, build(config)


@pytest.fixture(autouse=True)
def viewer(tmp_path, monkeypatch):
    """Stand-ins for the viewer files of the package."""
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text(INDEX)
    (static / "viewer.js").write_text("// viewer\n")
    (static / "viewer.css").write_text("/* style */\n")
    monkeypatch.setattr(gallery, "_static", lambda: static)
    return static


def _copy_output(config, tmp_path, name="out"):
    """A private copy of the run output (tests that change plots must not touch the module fixture)."""
    shutil.copytree(config.output_dir, tmp_path / name, ignore=shutil.ignore_patterns("logs"))
    return config.model_copy(update={"output_dir": tmp_path / name})


def _manifest(directory):
    return json.loads((directory / "manifest.json").read_text())


def test_publish_writes_gallery_manifest_and_world_readable_files(mini, tmp_path):
    path, config, analysis = mini
    target = tmp_path / "web" / "gallery"
    umask = os.umask(0o077)
    try:
        manifest_path = publish(config, analysis, WebConfig(dir=target, variant="emb_ff", description="embedding, fake factors"), config_path=path, overrides=["switches.embedding=false"])
    finally:
        os.umask(umask)

    manifest = _manifest(target)
    assert manifest_path == target / "manifest.json"
    assert manifest["schema"] == 1 and manifest["generator"].startswith("shapesmith ") and manifest["title"] == "mini" and manifest["description"] == ""
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", manifest["updated"])
    assert manifest["channels"] == [{"key": "mt", "label": "μτh"}]
    assert manifest["regions"] == [{"key": "nominal", "label": "nominal"}, {"key": "same_sign", "label": "same_sign"}]
    assert [c["key"] for c in manifest["categories"]] == ["inclusive", "sig", "bkg"]
    assert manifest["variables"] == [{"key": "m_vis", "label": "mvis / GeV"}, {"key": "score", "label": "NN output"}]
    [variant] = manifest["variants"]
    assert (variant["key"], variant["label"], variant["description"]) == ("emb_ff", "emb ff", "embedding, fake factors")
    [source] = variant["sources"]
    assert source["channels"] == ["mt"] and source["config"] == str(path) and source["overrides"] == ["switches.embedding=false"]
    assert source["output_dir"] == str(config.output_dir) and source["published"] == manifest["updated"]
    assert source["stamp"] == re.sub(r"[-:]", "", manifest["updated"])
    assert source["provenance"] == json.loads((config.output_dir / "versions" / "mini.json").read_text())
    assert manifest["plots"] == {"emb_ff": {"mt": {
        "nominal": {"inclusive": {"m_vis": ["png", "pdf"]}, "sig": {"score": ["png", "pdf"]}, "bkg": {"score": ["png", "pdf"]}},
        "same_sign": {"inclusive": {"m_vis": ["png", "pdf"]}},
    }}}
    yields = manifest["yields"]["emb_ff"]["mt"]
    assert set(yields) == {"nominal", "same_sign"} and set(yields["nominal"]) == {"inclusive", "sig", "bkg"}
    assert [g["key"] for g in yields["nominal"]["inclusive"]["groups"]] == ["jetFakes", "Z"]
    assert [g["label"] for g in yields["nominal"]["inclusive"]["groups"]] == ["jet→τh", "Z→ℓℓ"]
    assert [g["key"] for g in yields["same_sign"]["inclusive"]["groups"]] == ["Z"]  # jetFakes exists in the nominal region only

    plots = config.output_dir / "plots" / "mt"
    for published, original in (("nominal/inclusive/m_vis.png", "inclusive_m_vis.png"), ("same_sign/inclusive/m_vis.pdf", "same_sign/inclusive_m_vis.pdf"), ("nominal/sig/score.png", "sig_score.png")):
        assert (target / "data" / "emb_ff" / "mt" / published).read_bytes() == (plots / original).read_bytes()
    for directory, subdirectories, files in os.walk(target):
        assert stat.S_IMODE(os.stat(directory).st_mode) == 0o755, directory
        for name in files:
            assert stat.S_IMODE(os.stat(os.path.join(directory, name)).st_mode) == 0o644, name
    assert stat.S_IMODE((tmp_path / "web").stat().st_mode) == 0o755  # created parents too
    assert not [p for p in target.rglob(".*")]  # no temporary files left

    script = (target / "manifest.js").read_text()
    assert script == f"window.SHAPESMITH_GALLERY = {(target / 'manifest.json').read_text()};\n"
    index = (target / "index.html").read_text()
    assert "__SHAPESMITH_STAMP__" not in index and index.count(f"?v={source['stamp']}") == 3
    assert (target / "viewer.js").read_text() == "// viewer\n" and (target / "viewer.css").read_text() == "/* style */\n"


def test_publish_refuses_a_directory_that_is_not_a_gallery(mini, tmp_path):
    _, config, analysis = mini
    foreign = tmp_path / "public_html"
    foreign.mkdir()
    (foreign / "index.html").write_text("my home page")
    with pytest.raises(FileExistsError, match="not a ShapeSmith gallery"):
        publish(config, analysis, WebConfig(dir=foreign, variant="v"))
    assert [p.name for p in foreign.iterdir()] == ["index.html"]

    (foreign / "manifest.json").write_text(json.dumps({"schema": 1, "generator": "something else"}))
    with pytest.raises(FileExistsError):
        publish(config, analysis, WebConfig(dir=foreign, variant="v"))
    assert sorted(p.name for p in foreign.iterdir()) == ["index.html", "manifest.json"]

    empty = tmp_path / "empty"
    empty.mkdir()
    publish(config, analysis, WebConfig(dir=empty, variant="v"))
    assert _manifest(empty)["variants"][0]["key"] == "v"

    with pytest.raises(ValueError, match="variant"):
        publish(config, analysis, WebConfig(dir=empty).model_copy(update={"variant": "../v"}))
    with pytest.raises(ValueError, match="--to and --variant"):
        publish(config, analysis, WebConfig(variant="v"))


def test_publish_refuses_symbolic_links_in_the_plot_directories(mini, tmp_path):
    _, config, analysis = mini
    target = tmp_path / "gallery"
    publish(config, analysis, WebConfig(dir=target, variant="v"))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "inclusive").mkdir()
    (outside / "inclusive" / "keep.png").write_bytes(b"not part of the gallery")
    nominal = target / "data" / "v" / "mt" / "nominal"
    shutil.rmtree(nominal)
    nominal.symlink_to(outside)
    before = _manifest(target)
    with pytest.raises(FileExistsError, match="symbolic links"):
        publish(config, analysis, WebConfig(dir=target, variant="v"))
    assert [p.name for p in (outside / "inclusive").iterdir()] == ["keep.png"] and _manifest(target) == before

    nominal.unlink()
    shutil.move(target / "data" / "v", tmp_path / "moved")
    (target / "data" / "v").symlink_to(tmp_path / "moved")
    with pytest.raises(FileExistsError, match="symbolic links"):
        publish(config, analysis, WebConfig(dir=target, variant="v"))
    publish(config, analysis, WebConfig(dir=target, variant="other"))  # other variants are not affected


def test_republish_replaces_the_channel_and_keeps_other_variants(mini, tmp_path):
    _, config, analysis = mini
    config = _copy_output(config, tmp_path)
    target = tmp_path / "gallery"
    publish(config, analysis, WebConfig(dir=target, variant="first", label="First"))
    publish(config, analysis, WebConfig(dir=target, variant="second"))
    first = target / "data" / "first" / "mt"
    unchanged, touched = first / "nominal" / "inclusive" / "m_vis.png", first / "same_sign" / "inclusive" / "m_vis.png"
    inodes = (unchanged.stat().st_ino, touched.stat().st_ino, (target / "viewer.js").stat().st_ino)
    (first / "nominal" / "inclusive" / "old_variable.png").write_bytes(b"stale")
    for path in (config.output_dir / "plots" / "mt").glob("*_score.*"):
        path.unlink()
    os.utime(config.output_dir / "plots" / "mt" / "same_sign" / "inclusive_m_vis.png", (1e9, 1e9))

    publish(config, analysis, WebConfig(dir=target, variant="first", description="again"))

    manifest = _manifest(target)
    assert [(v["key"], v["label"], v["description"], len(v["sources"])) for v in manifest["variants"]] == [("first", "First", "again", 1), ("second", "second", "", 1)]
    assert set(manifest["plots"]["first"]["mt"]["nominal"]) == {"inclusive"} and set(manifest["yields"]["first"]["mt"]["nominal"]) == {"inclusive"}
    assert set(manifest["plots"]["second"]["mt"]["nominal"]) == {"inclusive", "sig", "bkg"}
    assert [c["key"] for c in manifest["categories"]] == ["inclusive", "sig", "bkg"]  # still shown by the second variant
    assert not (first / "nominal" / "sig").exists() and not (first / "nominal" / "inclusive" / "old_variable.png").exists()
    assert (target / "data" / "second" / "mt" / "nominal" / "sig" / "score.png").exists()
    assert unchanged.stat().st_ino == inodes[0] and touched.stat().st_ino != inodes[1]  # only changed plots are copied
    assert (target / "viewer.js").stat().st_ino == inodes[2]  # nor unchanged viewer files
    assert touched.stat().st_mtime == 1e9

    publish(config, analysis, WebConfig(dir=target, variant="second"))
    assert [c["key"] for c in _manifest(target)["categories"]] == ["inclusive"]


def _two_channel_analysis():
    """The mini analysis with a copy of mt as a second channel et."""
    analysis = build()
    mt = analysis.channel("mt")
    style = dataclasses.replace(analysis.style, channel_labels={"mt": r"$\mu\tau_h$", "et": r"e$\tau_h$"}, axis_labels={"mt": analysis.style.axis_labels["mt"], "et": {"m_vis": r"$m_{e\tau}$ / GeV"}})
    return dataclasses.replace(analysis, channels={"mt": mt, "et": dataclasses.replace(mt, name="et")}, style=style)


def test_channels_of_a_variant_from_two_configurations(mini, tmp_path):
    path, config, _ = mini
    analysis = _two_channel_analysis()
    other = tmp_path / "out_et"
    hset = HistogramSet({dataclasses.replace(key, channel="et"): h for key, h in HistogramSet.load(config.output_dir / "control_shapes.root").items()})
    hset.save(other / "control_shapes.root")
    run_plot(hset, analysis, ["et"], True, None, None, other / "plots")
    config_et = config.model_copy(update={"output_dir": other, "channels": ["et"]})
    target = tmp_path / "gallery"

    publish(config, analysis, WebConfig(dir=target, variant="combined"), channels=["mt"], config_path=path)
    publish(config_et, analysis, WebConfig(dir=target, variant="combined", title="Two channels"))

    manifest = _manifest(target)
    assert manifest["title"] == "Two channels"
    assert manifest["channels"] == [{"key": "mt", "label": "μτh"}, {"key": "et", "label": "eτh"}]
    assert {v["key"]: v["label"] for v in manifest["variables"]} == {"m_vis": "m_vis", "score": "NN output"}  # the channels label m_vis differently
    [variant] = manifest["variants"]
    assert [(s["channels"], s["output_dir"]) for s in variant["sources"]] == [(["mt"], str(config.output_dir)), (["et"], str(other))]
    assert variant["sources"][1]["provenance"]["config"]["channels"] == ["et"]  # no versions file: the record of this run
    assert set(manifest["plots"]["combined"]) == {"mt", "et"} and set(manifest["yields"]["combined"]) == {"mt", "et"}
    assert manifest["yields"]["combined"]["et"]["nominal"]["inclusive"] == manifest["yields"]["combined"]["mt"]["nominal"]["inclusive"]

    publish(config_et, analysis, WebConfig(dir=target, variant="combined"))
    sources = _manifest(target)["variants"][0]["sources"]
    assert [s["channels"] for s in sources] == [["mt"], ["et"]] and sources[1]["stamp"] >= variant["sources"][1]["stamp"]


@pytest.fixture
def plotted(monkeypatch):
    """The arrays the stack plots draw: (stacked values, their labels, data) per plot."""
    calls, original = [], stack_module.hep.histplot

    def spy(values, *args, **kwargs):
        if kwargs.get("stack"):
            calls.append({"stack": [np.asarray(v) for v in values], "labels": kwargs["label"]})
        elif kwargs.get("histtype") == "errorbar":
            calls[-1]["data"] = np.asarray(values)
        return original(values, *args, **kwargs)

    monkeypatch.setattr(stack_module.hep, "histplot", spy)
    return calls


def _assert_yields_are_the_stack(entry, drawn):
    assert [g["label"] for g in entry["groups"]] == [plain_text(text) for text in drawn["labels"]]
    assert [g["yield"] for g in entry["groups"]] == [float(values.sum()) for values in drawn["stack"]]
    assert entry["prediction"] == float(np.sum(drawn["stack"], axis=0).sum())
    assert entry["data"] == float(drawn["data"].sum())


def test_published_yields_are_the_sums_of_the_plotted_stacks(mini, tmp_path, plotted):
    _, config, analysis = mini
    config = _copy_output(config, tmp_path)
    shutil.rmtree(config.output_dir / "plots")
    control = HistogramSet.load(config.output_dir / "control_shapes.root")
    drawn = {}
    for region in ("nominal", "same_sign"):
        run_plot(control, analysis, ["mt"], True, None, None, config.output_dir / "plots", region=region)
        drawn[region, INCLUSIVE] = plotted[-1]
    run_plot(HistogramSet.load(config.output_dir / "shapes.root"), analysis, ["mt"], False, None, None, config.output_dir / "plots")
    drawn["nominal", "sig"], drawn["nominal", "bkg"] = plotted[-2:]

    publish(config, analysis, WebConfig(dir=tmp_path / "gallery", variant="v"))

    yields = _manifest(tmp_path / "gallery")["yields"]["v"]["mt"]
    for (region, category), arrays in drawn.items():
        entry = yields[region][category]
        assert entry["variable"] == ("m_vis" if category == INCLUSIVE else "score")
        _assert_yields_are_the_stack(entry, arrays)
    assert yields["nominal"]["inclusive"]["prediction"] > 0 and yields["nominal"]["inclusive"]["data"] > 0


def test_yields_come_from_the_yield_variable(tmp_path, plotted):
    analysis = build()
    mt = analysis.channel("mt")
    analysis = dataclasses.replace(analysis, channels={"mt": dataclasses.replace(mt, variables={**mt.variables, "yield": Variable("yield", "1.0", (0.5, 1.5))})})
    hset = HistogramSet()
    for process, values in (("data", [40, 25, 10]), ("ZTT", [20, 12, 5]), ("ZL", [10, 8, 2]), ("jetFakes", [8, 4, 2]), ("HH", [0.05, 0.1, 0.02])):
        hset[HistKey("mt", INCLUSIVE, process, "nominal", "Nominal", "m_vis")] = Histogram([0.0, 50.0, 100.0, 200.0], values, values)
        hset[HistKey("mt", INCLUSIVE, process, "nominal", "Nominal", "yield")] = Histogram([0.5, 1.5], [sum(values) + 1.0], [sum(values)])  # with events outside m_vis
        hset[HistKey("mt", INCLUSIVE, process, "nominal", "CMS_puUp", "yield")] = Histogram([0.5, 1.5], [1e6], [1e6])
    output = tmp_path / "out"
    hset.save(output / "control_shapes.root")
    run_plot(hset, analysis, ["mt"], True, None, ["m_vis", "yield"], output / "plots")
    config = RunConfig(analysis="tests.mini_analysis:build", era="2018", channels=["mt"], ntuples=NtupleConfig(base=str(tmp_path / "none")), skim_dir=tmp_path / "skims", output_dir=output)

    publish(config, analysis, WebConfig(dir=tmp_path / "gallery", variant="v"))

    entry = _manifest(tmp_path / "gallery")["yields"]["v"]["mt"]["nominal"]["inclusive"]
    assert entry["variable"] == "yield" and entry["data"] == 76.0 and entry["prediction"] == 74.0
    _assert_yields_are_the_stack(entry, plotted[-1])


def test_cli_publish_with_the_web_section_and_options(mini, tmp_path):
    path, config, _ = mini
    run = tmp_path / "run.yaml"
    run.write_text(path.read_text() + "web: {dir: site/gallery, variant: from_yaml, label: From YAML, title: Mini gallery}\n")
    _run("publish", "-c", str(run))
    target = tmp_path / "site" / "gallery"  # relative to the YAML
    manifest = _manifest(target)
    assert manifest["title"] == "Mini gallery" and [(v["key"], v["label"]) for v in manifest["variants"]] == [("from_yaml", "From YAML")]
    assert manifest["variants"][0]["sources"][0]["config"] == str(run)

    _run("publish", "-c", str(run), "--variant", "cli", "--label", "From CLI", "--description", "text", "--channels", "mt", "-s", "workers=1", "--about", "all runs")
    manifest = _manifest(target)
    assert [(v["key"], v["label"]) for v in manifest["variants"]] == [("from_yaml", "From YAML"), ("cli", "From CLI")]
    assert manifest["variants"][1]["sources"][0]["overrides"] == ["workers=1"] and manifest["description"] == "all runs"

    _run("publish", "-c", str(run), "--to", str(tmp_path / "elsewhere"))
    assert _manifest(tmp_path / "elsewhere")["variants"][0]["key"] == "from_yaml"

    for arguments in (["-c", str(path)], ["-c", str(path), "--to", str(tmp_path / "x")], ["-c", str(run), "--variant", "Not-Valid"]):
        result = runner.invoke(app, ["publish", *arguments])
        assert result.exit_code != 0
    assert not (tmp_path / "x").exists()


def test_web_section_of_the_run_configuration(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("analysis: tests.mini_analysis:build\nera: '2018'\nchannels: [mt]\nntuples: {base: /data}\nskim_dir: skims\noutput_dir: out\nweb: {dir: ../gallery, variant: emb_ff}\n")
    web = load_config(path).web
    assert web.dir == tmp_path.parent / "gallery" and web.variant == "emb_ff" and web.label is None
    assert load_config(path, ["web.dir=/srv/web"]).web.dir.as_posix() == "/srv/web"
    with pytest.raises(ValueError, match="variant"):
        load_config(path, ["web.variant=emb-ff"])


def test_merge_order():
    assert gallery.merge_order(["et", "mt", "tt"], ["em", "mm", "ee"]) == ["et", "mt", "tt", "em", "mm", "ee"]
    assert gallery.merge_order(["mt"], ["et", "mt", "tt"]) == ["et", "mt", "tt"]
    assert gallery.merge_order(["nominal", "b"], ["nominal", "a", "b", "c"]) == ["nominal", "a", "b", "c"]
    assert gallery.merge_order(["x", "b"], ["a", "b"]) == ["x", "a", "b"]


def test_plain_text():
    assert plain_text(r"$\mu\tau_h$") == "μτh"
    assert plain_text(r"e$\tau_h$") == "eτh"
    assert plain_text(r"Z$\rightarrow\ell\ell$ / $\tau\tau$") == "Z→ℓℓ / ττ"
    assert plain_text(r"$m_{vis}$ / GeV") == "mvis / GeV"
    assert plain_text(r"$p_T^{miss}$ / GeV") == "pTmiss / GeV"
    assert plain_text(r"$\Delta R(\tau\tau) + \Delta R(bb)$") == "ΔR(ττ) + ΔR(bb)"
    assert plain_text(r"t$\bar{\mathrm{t}}$") == "tt\u0304"
    assert plain_text(r"$\bar{\nu}$, $\sum E_T$") == "ν\u0304, ΣET"
    assert plain_text(r"59.8 fb$^{-1}$ (2018, 13 TeV)") == "59.8 fb⁻¹ (2018, 13 TeV)"
    assert plain_text("m_vis") == "m_vis" and plain_text(r"costs \$5") == "costs $5"


@pytest.mark.skipif(not gallery.resources.files("shapesmith.web").joinpath("static", "index.html").is_file(), reason="viewer not in this checkout")
def test_package_viewer_has_the_stamp_placeholder():
    index = gallery.resources.files("shapesmith.web").joinpath("static", "index.html").read_text()
    assert gallery.STAMP in index


@pytest.fixture(scope="module")
def ff_measured(tmp_path_factory):
    """The fake-factor measurement of tests/ff_mini.py, measured once."""
    root = tmp_path_factory.mktemp("ff_gallery")
    ff_mini.write_skims(root / "skims")
    config = RunConfig(analysis="tests.ff_mini:build", era="2018", channels=["mt"], ntuples=NtupleConfig(base=str(root)), skim_dir=root / "skims", output_dir=root / "out", workers=1)
    analysis = ff_mini.build(config)
    output = run_measure(config, analysis, ["mt"])
    return config, analysis, output, json.loads((output / "measurement_mt.json").read_text())


def test_a_measurement_publishes_its_gallery(ff_measured, tmp_path):
    config, analysis, output, record = ff_measured
    manifest = json.loads(publish(config, analysis, WebConfig(dir=tmp_path / "ff", variant="v1")).read_text())
    assert manifest["kind"] == "fake_factors" and manifest["axes"] == {"regions": "Process", "categories": "Category", "variables": "Quantity"}
    assert [r["key"] for r in manifest["regions"]] == ["QCD", "QCD_for_DRtoSR", "ttbar", "fractions"]
    assert {c["key"]: c["label"] for c in manifest["categories"]} == {"n_jets_1p5_2p5": "n_jets = 2", "n_jets_2p5_up": "n_jets ≥ 3"}
    assert manifest["yields"] == {"v1": {"mt": {}}}
    plots = manifest["plots"]["v1"]["mt"]
    assert set(plots["QCD"]["n_jets_2p5_up"]) == {"fake_factors", "DR_SR", *(n.removeprefix("QCD_") for n in record if n.startswith("QCD_non_closure"))}
    published = [(region, category, variable) for region, categories in plots.items() for category, variables in categories.items() for variable in variables]
    assert len(published) == len(list((output / "plots" / "mt").iterdir()))  # every plot of the measurement, once
    for region, category, variable in published:
        assert (tmp_path / "ff" / "data" / "v1" / "mt" / region / category / f"{variable}.png").is_file()
    notes = manifest["notes"]["v1"]["mt"]
    assert "fake_factors" not in notes["QCD"]["n_jets_1p5_2p5"]  # notes only for the corrections
    entry = record["QCD_DR_SR"][0]
    assert notes["QCD"]["n_jets_1p5_2p5"]["DR_SR"] == f"p = {entry['p_value']:.2g}" + (", set to 1" if entry["reset"] else "")
    assert manifest["facts"]["v1"]["mt"] == [{"key": "ttbar_data_scale", "label": "ttbar data/MC factor", "value": f"{record['ttbar_data_scale'][0]['factor']:.3f}"}]


def test_a_gallery_holds_one_kind(ff_measured, mini, tmp_path):
    config, analysis, _, _ = ff_measured
    _, control_config, control_analysis = mini
    publish(config, analysis, WebConfig(dir=tmp_path / "ff", variant="v1"))
    before = (tmp_path / "ff" / "manifest.json").read_text()
    with pytest.raises(ValueError, match="gallery of fake_factors plots, not of control"):
        publish(control_config, control_analysis, WebConfig(dir=tmp_path / "ff", variant="control"))
    assert (tmp_path / "ff" / "manifest.json").read_text() == before and not (tmp_path / "ff" / "data" / "control").exists()
    publish(control_config, control_analysis, WebConfig(dir=tmp_path / "control", variant="control"))
    with pytest.raises(ValueError, match="gallery of control plots, not of fake_factors"):
        publish(config, analysis, WebConfig(dir=tmp_path / "control", variant="v1"))


def test_fake_factor_category_keys_and_labels():
    n_jets = Split("n_jets", (-0.5, 1.5, 2.5, 22.5))
    assert [ff_gallery.category_key(n_jets, i) for i in n_jets.categories] == ["n_jets_m0p5_1p5", "n_jets_1p5_2p5", "n_jets_2p5_up"]
    assert [ff_gallery.category_label(n_jets, i) for i in n_jets.categories] == ["0 ≤ n_jets ≤ 1", "n_jets = 2", "n_jets ≥ 3"]
    assert ff_gallery.category_label(Split("n_jets", (1.5, 22.5)), 0) == "n_jets ≥ 2"
    pt = Split("pt_2", (20.0, 40.0, 200.0))
    assert [ff_gallery.category_label(pt, i) for i in pt.categories] == ["20 ≤ pt_2 < 40", "pt_2 ≥ 40"]

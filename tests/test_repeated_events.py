"""Repeated events (Analysis.event_key): every data and embedding sample is read without the skim rows that repeat the
event of an earlier row of the sample, in file order (store.read_skims, events.read_events), in hist, load and the ML
export; the records in repeated_events.json; the skim keeps the key columns."""
import dataclasses
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from shapesmith.events import REPEATED, UNIQUE_KINDS, Query, load, read_events, repeated_record, unique_nicks, write_repeated
from shapesmith.fill import run_hist
from shapesmith.histogram import INCLUSIVE, HistKey
from shapesmith.ml_export import EVENTS, merge_eras, run_ml_export
from shapesmith.model import Sample
from shapesmith.skim import needed_columns
from shapesmith.store import SkimMissingError, read_skims, repeated_events, repeated_rows
from tests.mini_analysis import build

KEY = ("run", "lumi", "event")


def _write(skim_dir, nick, name, keys, x):
    run, lumi, event = (np.array(column) for column in zip(*keys))
    table = pa.table({"run": run.astype(np.uint32), "lumi": lumi.astype(np.uint32), "event": event.astype(np.uint64), "x": np.asarray(x, dtype=np.float64)})
    path = skim_dir / "mt" / nick / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


@pytest.fixture
def hand_made(tmp_path):
    """DATA_0 holds (1, 1, 1), (1, 1, 2), (1, 1, 3); DATA_1 repeats (1, 1, 2) and (1, 1, 1) and holds (1, 2, 1) twice;
    MC has the same rows as DATA."""
    for nick in ("DATA", "MC"):
        _write(tmp_path, nick, f"{nick}_0.parquet", [(1, 1, 1), (1, 1, 2), (1, 1, 3)], [0, 1, 2])
        _write(tmp_path, nick, f"{nick}_1.parquet", [(1, 1, 2), (1, 2, 1), (1, 1, 1), (1, 2, 1)], [10, 11, 12, 13])
    return tmp_path


def test_a_unique_sample_keeps_the_first_row_of_every_event_in_file_order(hand_made):
    assert read_skims(hand_made, "mt", ["DATA"], None, KEY, {"DATA"})["x"].tolist() == [0, 1, 2, 11]
    assert read_skims(hand_made, "mt", ["DATA", "MC"], ["x"], KEY, {"DATA"})["x"].tolist() == [0, 1, 2, 11, 0, 1, 2, 10, 11, 12, 13]
    assert list(read_skims(hand_made, "mt", ["DATA"], ["x", "event"], KEY, {"DATA"}).columns) == ["x", "event"]  # no other key column
    assert read_skims(hand_made, "mt", ["DATA"], None)["x"].tolist() == [0, 1, 2, 10, 11, 12, 13]  # without a key: every row
    assert read_skims(hand_made, "mt", ["DATA"], None, (), {"DATA"})["x"].tolist() == [0, 1, 2, 10, 11, 12, 13]
    assert read_skims(hand_made, "mt", ["DATA"], None, ("event",), {"DATA"})["x"].tolist() == [0, 1, 2]  # (1, 2, 1) has event 1


def test_the_record_of_a_sample(hand_made):
    assert repeated_events(hand_made, "mt", "DATA", KEY) == {"rows": 7, "repeated": 3, "files": {"DATA_1.parquet": 3}}
    assert repeated_events(hand_made, "mt", "DATA", ("run", "lumi")) == {"rows": 7, "repeated": 5, "files": {"DATA_0.parquet": 2, "DATA_1.parquet": 3}}
    table = pa.table({"a": [1, 2, 1, 1], "b": [1, 1, 1, 2]})
    assert repeated_rows(table, ("a", "b")).tolist() == [False, False, True, False]


def test_repeated_rows_equal_a_comparison_of_every_row():
    """Many event numbers repeat in other runs and lumi sections (the candidates), some rows repeat the whole key."""
    rng = np.random.default_rng(1)
    n = 5000
    table = pa.table({"run": rng.integers(1, 4, n), "lumi": rng.integers(1, 6, n), "event": rng.integers(1, 300, n).astype(np.uint64)})
    expected = table.to_pandas().duplicated(keep="first").to_numpy()
    assert 0 < expected.sum() < n - 300 and (repeated_rows(table, KEY) == expected).all()
    assert not repeated_rows(table.slice(0, 0), KEY).size


def test_a_unique_sample_needs_the_key_columns(hand_made):
    (hand_made / "mt" / "OLD").mkdir()
    pq.write_table(pa.table({"event": np.arange(3, dtype=np.uint64), "x": np.zeros(3)}), hand_made / "mt" / "OLD" / "OLD_0.parquet")
    assert len(read_skims(hand_made, "mt", ["OLD"], ["x"])) == 3
    for columns in (["x"], None):
        with pytest.raises(SkimMissingError, match=r"lack columns \['lumi', 'run'\]"):
            read_skims(hand_made, "mt", ["OLD"], columns, KEY, {"OLD"})
    with pytest.raises(SkimMissingError, match="lack columns"):
        repeated_events(hand_made, "mt", "OLD", KEY)


def test_data_and_embedding_are_unique_with_an_event_key():
    analysis = build()
    channel = dataclasses.replace(analysis.channel("mt"), samples=(*analysis.channel("mt").samples, Sample("EMB_A", "EMB", "embedding")))
    nicks = [s.nick for s in channel.samples]
    assert unique_nicks(analysis, channel, nicks) == set()
    keyed = dataclasses.replace(analysis, event_key=KEY)
    assert unique_nicks(keyed, channel, nicks) == {"DATA_A", "EMB_A"} and UNIQUE_KINDS == ("data", "embedding")
    assert unique_nicks(keyed, channel, ["ZTT_1", "EMB_A"]) == {"EMB_A"}
    for sample in channel.samples:
        columns = needed_columns(channel, sample, None, KEY)
        assert set(KEY) <= columns if sample.kind in UNIQUE_KINDS else not {"run", "lumi"} & columns


def _repeat(config, nick, rows=50):
    """A third skim file of the nick with the first `rows` rows of its first file (after it in file order)."""
    directory = config.skim_dir / "mt" / nick
    first = pq.read_table(directory / f"{nick}_0.parquet")
    pq.write_table(first.slice(0, rows), directory / f"{nick}_2.parquet")
    return first.slice(0, rows)


def test_hist_load_and_the_export_count_a_repeated_data_event_once(mini_run, tmp_path):
    """50 events of the data and of the simulation repeated in a third skim file: with the event key the data are
    filled as without the repeats (the simulation keeps them), load and the ML export drop them, and both record them."""
    config, analysis = mini_run
    config = config.model_copy(update={"workers": 1})
    keyed = dataclasses.replace(analysis, event_key=("event",))
    clean = run_hist(config, analysis, ["mt"], True, None, False, None, tmp_path / "clean" / "control_shapes.root")
    run_ml_export(config.model_copy(update={"ml_dir": tmp_path / "ml_clean"}), analysis, ["mt"])
    _repeat(config, "DATA_A")
    _repeat(config, "ZTT_1")
    plain = run_hist(config, analysis, ["mt"], True, None, False, None, tmp_path / "plain" / "control_shapes.root")
    unique = run_hist(config, keyed, ["mt"], True, None, False, None, tmp_path / "unique" / "control_shapes.root")
    for region in ("nominal", "anti_iso"):
        key = HistKey("mt", INCLUSIVE, "data", region, "Nominal", "m_vis")
        np.testing.assert_array_equal(unique[key].values, clean[key].values)
        assert plain[key].sum() > clean[key].sum()
    ztt = HistKey("mt", INCLUSIVE, "ZTT", "nominal", "Nominal", "m_vis")
    np.testing.assert_array_equal(unique[ztt].values, plain[ztt].values)  # simulation: every row
    assert unique[ztt].sum() > clean[ztt].sum()
    assert not (tmp_path / "plain" / REPEATED).exists()
    record = json.loads((tmp_path / "unique" / REPEATED).read_text())
    rows = sum(pq.ParquetFile(f).metadata.num_rows for f in (config.skim_dir / "mt" / "DATA_A").glob("*.parquet"))
    assert record == {"event_key": ["event"], "kinds": ["data", "embedding"], "eras": {"2018": {"mt": {
        "checked": ["DATA_A"], "repeated": {"DATA_A": {"rows": rows, "repeated": 50, "files": {"DATA_A_2.parquet": 50}}}}}}}
    events = load(config, keyed, Query("mt", "data", "anti_iso"), ["event"]).frame
    assert not events["event"].duplicated().any() and len(load(config, analysis, Query("mt", "data", "anti_iso"), ["event"]).frame) >= len(events)
    assert read_events(config, keyed, keyed.channel("mt"), ["DATA_A"], ["event"])["event"].tolist() == read_skims(config.skim_dir, "mt", ["DATA_A"], ["event"])["event"].tolist()[:-50]
    # the export: the jet fakes (anti-isolated data) as without the repeats, recorded per era and merged
    keyed_config = config.model_copy(update={"ml_dir": tmp_path / "ml" / "2018"})
    written = run_ml_export(keyed_config, keyed, ["mt"])
    assert tmp_path / "ml" / "2018" / REPEATED in written
    exported, before = pd.read_feather(tmp_path / "ml" / "2018" / "mt" / EVENTS), pd.read_feather(tmp_path / "ml_clean" / "mt" / EVENTS)
    jet_fakes = lambda frame: frame[(frame["process"] == "jetFakes")].reset_index(drop=True)  # noqa: E731
    pd.testing.assert_frame_equal(jet_fakes(exported).drop(columns="row"), jet_fakes(before).drop(columns="row"))
    assert len(exported[exported["sample_nick"] == "ZTT_1"]) > len(before[before["sample_nick"] == "ZTT_1"])
    other = config.model_copy(update={"ml_dir": tmp_path / "ml" / "2017"})
    run_ml_export(other, dataclasses.replace(keyed, era="2017"), ["mt"])
    merged = merge_eras([tmp_path / "ml" / "2017", tmp_path / "ml" / "2018"], tmp_path / "ml" / "run2")
    assert merged[-1] == tmp_path / "ml" / "run2" / REPEATED
    assert set(json.loads(merged[-1].read_text())["eras"]) == {"2017", "2018"}


def test_the_record_merges_channels_and_eras(mini_run, tmp_path):
    config, analysis = mini_run
    keyed = dataclasses.replace(analysis, event_key=("event",))
    assert repeated_record(config, analysis, ["mt"]) == {}
    assert repeated_record(config, keyed, ["mt"]) == {"mt": {"checked": ["DATA_A"], "repeated": {}}}
    assert write_repeated(tmp_path / REPEATED, analysis, {}) is None and not (tmp_path / REPEATED).exists()
    write_repeated(tmp_path / REPEATED, keyed, {"mt": {"checked": ["DATA_A"], "repeated": {}}})
    write_repeated(tmp_path / REPEATED, keyed, {"et": {"checked": [], "repeated": {}}})
    write_repeated(tmp_path / REPEATED, dataclasses.replace(keyed, era="2017"), {"mt": {"checked": [], "repeated": {}}})
    stored = json.loads((tmp_path / REPEATED).read_text())
    assert set(stored["eras"]) == {"2017", "2018"} and set(stored["eras"]["2018"]) == {"mt", "et"}
    assert stored["event_key"] == ["event"] and stored["kinds"] == ["data", "embedding"]

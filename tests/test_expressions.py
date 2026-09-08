import numpy as np
import pandas as pd
import pytest

from shapesmith import expressions as ex
from shapesmith.model import Region, Selection, WeightVariation


@pytest.fixture
def frame():
    return pd.DataFrame({"pt_1": [10.0, 30.0, 50.0], "q_1": [1, -1, 1], "q_2": [-1, 1, 1], "w": [0.5, 2.0, 1.0], "eta_1": [-2.5, 0.1, 2.0]})


def test_columns_in_ignores_functions_keywords_and_numbers():
    assert ex.columns_in("(abs(eta_1) < 2.1) & (pt_1 > 25) and not (q_1 * q_2 > 0) | sqrt(w) > 1e3") == {"eta_1", "pt_1", "q_1", "q_2", "w"}
    assert ex.columns_in("1.0") == set()


def test_evaluate_mask_weight(frame):
    assert ex.evaluate(frame, "pt_1 > 20").tolist() == [False, True, True]
    assert ex.evaluate(frame, "1.0").tolist() == [1.0, 1.0, 1.0]
    assert ex.mask(frame, {"a": "pt_1 > 20", "b": "(q_1 * q_2) < 0"}).tolist() == [False, True, False]
    assert ex.mask(frame, {}).tolist() == [True, True, True]
    assert ex.weight(frame, {"w": "w", "two": "2.0"}).tolist() == [1.0, 4.0, 2.0]
    assert ex.weight(frame, {}).dtype == np.float64 and ex.weight(frame, {}).tolist() == [1.0, 1.0, 1.0]


def test_bool_product_falls_back_to_the_python_engine(frame):
    assert ex.evaluate(frame, "(pt_1 > 20) * (q_1 > 0) * w").tolist() == [0.0, 0.0, 1.0]


def test_missing_column_is_reported(frame):
    with pytest.raises(ex.ExpressionError, match="missing columns: nope"):
        ex.evaluate(frame, "nope > 1")


def test_apply_region_and_variation():
    base = Selection(cuts={"os": "(q_1 * q_2) < 0", "iso": "id > 0.5"}, weights={"pu": "puweight"})
    ss = ex.apply_region(base, Region("same_sign", replace_cuts={"os": "(q_1 * q_2) > 0"}, add_weights={"ff": "fake_factor"}))
    assert ss.cuts["os"] == "(q_1 * q_2) > 0" and ss.cuts["iso"] == "id > 0.5"
    assert list(ss.weights) == ["pu", "ff"] and ss.weights["ff"] == "fake_factor"
    up = ex.apply_weight_variation(base, WeightVariation("puUp", {"pu": "puweight_up"}))
    assert up.weights["pu"] == "puweight_up" and up.cuts == base.cuts
    with pytest.raises(KeyError):
        ex.apply_region(base, Region("bad", replace_cuts={"nope": "1"}))
    with pytest.raises(KeyError):
        ex.apply_weight_variation(base, WeightVariation("bad", {"nope": "1"}))


def test_apply_column_variation_only_renames_available_shifted_columns():
    available = {"pt_1", "pt_1__jesUp", "met"}
    assert ex.apply_column_variation("(pt_1 > 25) & (met > 30) & (pt_10 > 1)", "__jesUp", available) == "(pt_1__jesUp > 25) & (met > 30) & (pt_10 > 1)"


def test_selection_columns():
    sel = Selection(cuts={"a": "pt_1 > 25"}, weights={"w": "puweight * trg_wgt"})
    assert ex.selection_columns(sel) == {"pt_1", "puweight", "trg_wgt"}

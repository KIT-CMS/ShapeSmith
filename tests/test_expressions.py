import numpy as np
import pandas as pd
import pytest

from shapesmith import expressions as ex
from shapesmith.model import ColumnVariation


@pytest.fixture
def frame():
    return pd.DataFrame({"pt_1": [10.0, 30.0, 50.0], "q_1": [1, -1, 1], "q_2": [-1, 1, 1], "w": [0.5, 2.0, 1.0], "eta_1": [-2.5, 0.1, 2.0]})


def test_columns_in_ignores_functions_keywords_and_numbers():
    assert ex.columns_in("(abs(eta_1) < 2.1) & (pt_1 > 25) and not (q_1 * q_2 > 0) | sqrt(w) > 1e3") == {"eta_1", "pt_1", "q_1", "q_2", "w"}
    assert ex.columns_in("1.0") == set()


def test_evaluate_mask_weight(frame):
    assert ex.evaluate(frame, "pt_1 > 20").tolist() == [False, True, True]
    assert ex.evaluate(frame, "1.0").tolist() == [1.0, 1.0, 1.0]
    assert ex.mask(frame, ["pt_1 > 20", "(q_1 * q_2) < 0"]).tolist() == [False, True, False]
    assert ex.mask(frame, []).tolist() == [True, True, True]
    assert ex.product(frame, ["w", "2.0"]).tolist() == [1.0, 4.0, 2.0]
    assert ex.product(frame, []).dtype == np.float64 and ex.product(frame, []).tolist() == [1.0, 1.0, 1.0]


def test_bool_product_falls_back_to_the_python_engine(frame):
    assert ex.evaluate(frame, "(pt_1 > 20) * (q_1 > 0) * w").tolist() == [0.0, 0.0, 1.0]


def test_missing_column_is_reported(frame):
    with pytest.raises(ex.ExpressionError, match="missing columns: nope"):
        ex.evaluate(frame, "nope > 1")


def test_shift_renames_only_columns_with_a_shifted_branch():
    available = {"pt_1", "pt_1__jesUp", "met"}
    jes = ColumnVariation("jesUp", "__jesUp")
    assert ex.shift("(pt_1 > 25) & (met > 30) & (pt_10 > 1)", jes, available) == "(pt_1__jesUp > 25) & (met > 30) & (pt_10 > 1)"
    assert ex.shift("abs(pt_1) > 1", jes, available) == "abs(pt_1__jesUp) > 1"


def test_shift_replaces_derived_columns_by_their_expression():
    grid = ColumnVariation("emb1p002", derived={"pt_2": "pt_2 * 1.002", "m_vis": "m_vis * sqrt(1.002)"})
    assert ex.shift("(pt_2 > 20) & (m_vis < 100) & (pt_1 > 25)", grid, set()) == "((pt_2 * 1.002) > 20) & ((m_vis * sqrt(1.002)) < 100) & (pt_1 > 25)"


def test_columns_of():
    assert ex.columns_of(["pt_1 > 25", "puweight * trg_wgt"]) == {"pt_1", "puweight", "trg_wgt"}

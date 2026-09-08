import numpy as np
import pytest
import uproot

from shapesmith.histogram import Histogram


def test_fill_add_scale_sum():
    h = Histogram.fill([0.0, 0.5, 1.0], [0.1, 0.2, 0.7], weights=[1.0, 2.0, 3.0])
    assert h.values.tolist() == [3.0, 3.0] and h.variances.tolist() == [5.0, 9.0]
    other = Histogram([0.0, 0.5, 1.0], [1.0, 1.0], [0.5, 0.5])
    assert h.copy().add(other, -2.0).values.tolist() == [1.0, 1.0]
    assert h.copy().add(other, -2.0).variances.tolist() == [7.0, 11.0]
    assert h.copy().scale(2.0).variances.tolist() == [20.0, 36.0] and h.sum() == 6.0
    with pytest.raises(ValueError):
        h.add(Histogram([0.0, 1.0], [1.0], [1.0]))


def test_rebin():
    h = Histogram([0.0, 0.25, 0.5, 0.75, 1.0], [1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    r = h.rebin([0.0, 0.5, 1.0])
    assert r.values.tolist() == [3.0, 7.0] and np.allclose(r.variances, [0.3, 0.7]) and r.edges.tolist() == [0.0, 0.5, 1.0]
    with pytest.raises(ValueError):
        h.rebin([0.0, 0.3, 1.0])


def test_invalid_edges_rejected():
    with pytest.raises(ValueError):
        Histogram([0.0, 1.0, 1.0], [1.0, 1.0], [1.0, 1.0])


def test_root_roundtrip_keeps_edges_and_errors(tmp_path):
    h = Histogram([0.0, 0.2, 0.5, 1.0], [1.0, 2.0, 3.0], [0.5, 1.0, 1.5])
    with uproot.recreate(tmp_path / "h.root") as f:
        f["dir/name"] = h.to_root("name")
    with uproot.open(tmp_path / "h.root") as f:
        th = f["dir/name"]
        assert th.classname == "TH1D"
        assert th.axis().edges().tolist() == [0.0, 0.2, 0.5, 1.0]
        back = Histogram.from_root(th)
    assert back.values.tolist() == [1.0, 2.0, 3.0] and back.variances.tolist() == [0.5, 1.0, 1.5] and back.edges.tolist() == [0.0, 0.2, 0.5, 1.0]

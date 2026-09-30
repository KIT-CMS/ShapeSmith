"""The numpy smoothing against ROOT TGraphSmooth and TauFakeFactors (tests/reference/make_smoothing_reference.py)."""
import json
from pathlib import Path

import numpy as np
import pytest

from shapesmith.measurements.smoothing import binwise, smooth, smooth_kern

REFERENCE = json.loads((Path(__file__).parent / "reference" / "smoothing_root.json").read_text())
BINWISE_BINS = {  # TauFakeFactors option -> the number of binwise bins on the left and on the right
    "smoothed": (0, 0), "binwise#[0]+smoothed": (1, 0), "binwise#[-1]+smoothed": (0, 1), "binwise#[0,]#[-1,]+smoothed": (1, 1),
}


def _equal(a, b) -> bool:
    """Bitwise: the sums run in ROOT's order with libm's exp."""
    return np.array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("case", REFERENCE["kernel"], ids=lambda c: f"n{len(c['x'])}_bw{c['bandwidth']}")
def test_kernel_matches_root(case):
    assert _equal(smooth_kern(np.array(case["x"]), np.array(case["y"]), np.array(case["x_out"]), case["bandwidth"]), case["y_out"])


@pytest.mark.parametrize("case", REFERENCE["smooth"], ids=lambda c: f"{c['option']}_bw{c['bandwidth']}_n{len(c['y'])}")
def test_smoothing_matches_tau_fake_factors(case):
    if case["option"] == "binwise":
        curve = binwise(case["y"], case["errors"], case["bin_edges"])
    else:
        left, right = BINWISE_BINS[case["option"]]
        curve = smooth(case["x"], case["y"], case["errors"], case["bin_edges"], case["bandwidth"], left, right)
    for name in ("edges", "nominal", "up", "down"):
        assert _equal(getattr(curve, name), case[name]), name


def test_the_kernel_sums_the_last_point_left_of_the_window_as_root_does():
    x, y = np.array([0.0, 10.0]), np.array([1.0, 2.0])
    assert smooth_kern(x, y, np.array([0.0, 2.0, 5.0]), 1.0).tolist() == [1.0, 1.0, 1.0]  # a textbook smoother gives 1, 0, 0
    assert smooth_kern(x, y, np.array([5.0]), 1.0).tolist() == [0.0]  # nothing within reach of the first output point


def test_the_kernel_smooths_all_samples_at_once():
    x, samples = np.array([1.0, 2.0, 4.0]), np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    both = smooth_kern(x, samples, np.array([1.5, 3.0]), 2.0)
    assert np.allclose(both[:, 1], smooth_kern(x, samples[:, 1], np.array([1.5, 3.0]), 2.0))
    with pytest.raises(ValueError, match="sorted"):
        smooth_kern(x, samples, np.array([3.0, 1.5]), 2.0)


def test_binwise_keeps_the_bins_and_clips_at_zero():
    curve = binwise([0.2, -0.1], [0.3, 0.05], [0.0, 1.0, 2.0])
    assert curve.edges.tolist() == [0.0, 1.0, 2.0] and curve.nominal.tolist() == [0.2, 0.0]
    assert curve.up.tolist() == pytest.approx([0.5, 0.0]) and curve.down.tolist() == [0.0, 0.0]

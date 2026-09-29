"""The numpy smoothing against ROOT TGraphSmooth and TauFakeFactors (tests/reference/make_smoothing_reference.py)."""
import json
from pathlib import Path

import numpy as np
import pytest

from shapesmith.measurements.smoothing import hybrid_slices, smooth, smooth_kern

REFERENCE = json.loads((Path(__file__).parent / "reference" / "smoothing_root.json").read_text())


def _close(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.all(np.abs(a - b) <= 1e-12 * np.maximum(np.abs(b), 1e-300)))


@pytest.mark.parametrize("case", REFERENCE["kernel"], ids=lambda c: f"n{len(c['x'])}_bw{c['bandwidth']}")
def test_kernel_matches_root(case):
    assert _close(smooth_kern(np.array(case["x"]), np.array(case["y"]), np.array(case["x_out"]), case["bandwidth"]), case["y_out"])


@pytest.mark.parametrize("case", REFERENCE["smooth"], ids=lambda c: f"{c['option']}_bw{c['bandwidth']}_n{len(c['y'])}")
def test_smoothing_matches_tau_fake_factors(case):
    curve = smooth(case["x"], case["y"], case["errors"], case["bin_edges"], case["option"], case["bandwidth"])
    for name in ("edges", "nominal", "up", "down"):
        assert _close(getattr(curve, name), case[name]), name


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


def test_hybrid_slices():
    assert hybrid_slices("smoothed") == (0, -1, slice(0, 0), slice(-1, -1), slice(None, None))
    assert hybrid_slices("binwise#[0,1]+smoothed") == (2, -1, slice(None, 2), slice(-1, -1), slice(None, None))
    assert hybrid_slices("binwise#[-1]+smoothed") == (0, -2, slice(0, 0), slice(-1, None), slice(None, -1))
    assert hybrid_slices("binwise#[0,]#[-1,]+smoothed") == (1, -2, slice(None, 1), slice(-1, None), slice(None, -1))
    with pytest.raises(ValueError, match="one side"):
        hybrid_slices("binwise#[0,-1]+smoothed")


def test_binwise_keeps_the_bins_and_clips_at_zero():
    curve = smooth([0.5, 1.5], [0.2, -0.1], [0.3, 0.05], [0.0, 1.0, 2.0], "binwise", 1.0)
    assert curve.edges.tolist() == [0.0, 1.0, 2.0] and curve.nominal.tolist() == [0.2, 0.0]
    assert curve.up.tolist() == pytest.approx([0.5, 0.0]) and curve.down.tolist() == [0.0, 0.0]

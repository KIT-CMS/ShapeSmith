"""ABCD method: QCD(A) = (data - MC)(B) * (data - MC)(C).sum() / (data - MC)(D).sum().

A = signal region (OS, isolated), B = OS anti-isolated, C = SS isolated, D = SS anti-isolated.
"""
from __future__ import annotations

import logging

from shapesmith.estimators import categories_and_variables, data_minus
from shapesmith.histogram import Histogram
from shapesmith.histograms import NOMINAL_REGION, NOMINAL_VARIATION, HistKey, HistogramSet
from shapesmith.model import Analysis

logger = logging.getLogger(__name__)


def clip_negative_bins(h: Histogram) -> Histogram:
    """Set negative bins to zero and rescale to the original integral (zero everything if the integral is <= 0)."""
    total = h.sum()
    positive = float(h.values[h.values > 0].sum())
    h.values[h.values < 0] = 0.0
    if total <= 0.0:
        logger.warning(f"integral {total:.3f} <= 0 after subtraction, histogram set to zero")
        h.values[:] = 0.0
        h.variances[:] = 0.0
    elif positive > 0.0 and positive != total:
        h.scale(total / positive)
    return h


def estimate(hset: HistogramSet, analysis: Analysis, channel: str) -> list[HistKey]:
    estimator = analysis.estimator
    regions = estimator.regions
    added = []
    for category, variable in categories_and_variables(hset, channel, regions["B"]):
        b = data_minus(hset, channel, category, variable, regions["B"], estimator.subtract)
        c = data_minus(hset, channel, category, variable, regions["C"], estimator.subtract)
        d = data_minus(hset, channel, category, variable, regions["D"], estimator.subtract)
        if b is None or c is None or d is None:
            logger.warning(f"{channel}/{category}/{variable}: ABCD control region missing, skipped")
            continue
        yield_c, yield_d = c.sum(), d.sum()
        if yield_c <= 0.0 or yield_d <= 0.0:
            logger.warning(f"{channel}/{category}/{variable}: C={yield_c:.2f}, D={yield_d:.2f}, QCD set to zero")
            factor = 0.0
        else:
            factor = yield_c / yield_d
        key = HistKey(channel, category, estimator.output, NOMINAL_REGION, NOMINAL_VARIATION, variable)
        hset.add(key, clip_negative_bins(b.scale(factor)))
        added.append(key)
    return added

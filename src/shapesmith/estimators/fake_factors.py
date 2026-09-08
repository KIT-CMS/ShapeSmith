"""Fake-factor method: jetFakes = FF-weighted anti-isolated data minus genuine-tau and lepton-fake MC."""
from __future__ import annotations

from shapesmith.estimators import categories_and_variables, data_minus
from shapesmith.histograms import NOMINAL_REGION, NOMINAL_VARIATION, HistKey, HistogramSet
from shapesmith.model import Analysis


def estimate(hset: HistogramSet, analysis: Analysis, channel: str) -> list[HistKey]:
    estimator = analysis.estimator
    region = estimator.regions["anti_iso"]
    added = []
    for category, variable in categories_and_variables(hset, channel, region):
        h = data_minus(hset, channel, category, variable, region, estimator.subtract)
        if h is None:
            continue
        key = HistKey(channel, category, estimator.output, NOMINAL_REGION, NOMINAL_VARIATION, variable)
        hset.add(key, h)
        added.append(key)
    return added

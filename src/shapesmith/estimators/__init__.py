"""Derived processes computed from histograms of the estimation regions (Spec §9)."""
from __future__ import annotations

import logging

from shapesmith.histogram import Histogram
from shapesmith.histograms import HistKey, HistogramSet
from shapesmith.model import Analysis

logger = logging.getLogger(__name__)


def data_minus(hset: HistogramSet, channel: str, category: str, variable: str, region: str, subtract: tuple[str, ...]) -> Histogram | None:
    """data[region] - sum of the given processes[region]; None if the data histogram is missing."""
    data_key = HistKey(channel, category, "data", region, "Nominal", variable)
    if not hset.has(data_key):
        return None
    result = hset.get(data_key).copy()
    for process in subtract:
        key = HistKey(channel, category, process, region, "Nominal", variable)
        if not hset.has(key):
            logger.warning(f"{key.path} missing, not subtracted")
            continue
        result.add(hset.get(key), -1.0)
    return result


def categories_and_variables(hset: HistogramSet, channel: str, region: str) -> list[tuple[str, str]]:
    return sorted({(k.category, k.variable) for k in hset.keys(channel=channel, process="data", region=region)})


def run_estimate(hset: HistogramSet, analysis: Analysis, channels: list[str]) -> list[HistKey]:
    from shapesmith.estimators import abcd, embedding, fake_factors  # local import: the modules import this package

    added: list[HistKey] = []
    for channel in channels:
        if analysis.estimator is not None:
            estimate = {"fake_factors": fake_factors.estimate, "abcd": abcd.estimate}[analysis.estimator.name]
            added += estimate(hset, analysis, channel)
        if analysis.processes_of_kind("embedding"):
            added += embedding.ttbar_contamination(hset, analysis, channel)
    return added

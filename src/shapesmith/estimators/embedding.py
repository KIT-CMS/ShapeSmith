"""ttbar contamination variation of the embedded sample: EMB -/+ 10 % of the genuine-tau ttbar histogram."""
from __future__ import annotations

import logging

from shapesmith.histograms import NOMINAL_REGION, NOMINAL_VARIATION, HistKey, HistogramSet
from shapesmith.model import Analysis

logger = logging.getLogger(__name__)
TTBAR_GENUINE = "TTT"
FRACTION = 0.1


def ttbar_contamination(hset: HistogramSet, analysis: Analysis, channel: str) -> list[HistKey]:
    added = []
    for emb in analysis.processes_of_kind("embedding"):
        for key in hset.keys(channel=channel, process=emb.name, region=NOMINAL_REGION, variation=NOMINAL_VARIATION):
            ttbar_key = HistKey(channel, key.category, TTBAR_GENUINE, NOMINAL_REGION, NOMINAL_VARIATION, key.variable)
            if not hset.has(ttbar_key):
                logger.warning(f"{ttbar_key.path} missing, no embedding ttbar variation")
                continue
            for direction, sign in (("Down", -1.0), ("Up", +1.0)):
                varied = hset.get(key).copy().add(hset.get(ttbar_key), sign * FRACTION)
                new_key = HistKey(channel, key.category, emb.name, NOMINAL_REGION, f"CMS_htt_emb_ttbar_{analysis.era}{direction}", key.variable)
                hset.add(new_key, varied)
                added.append(new_key)
    return added

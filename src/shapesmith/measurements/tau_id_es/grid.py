"""The tau energy-scale grid and the naming conventions of MorphingTauID2017.

A grid point is an energy-scale shift in integer units of 0.1 %: shift -198 scales the tau four-momentum by 0.802.
Its column variation is named `es-198`, and MorphingTauID2017 reads its shape under the mass string "-19.8" (the
shift in percent with one decimal). Integers keep the three names in exact correspondence.
"""
from __future__ import annotations

import re

SIGNAL = "EMB"  # the embedded signal of the mt channel, one shape per grid point
CHANNEL = "mt"
CONTROL_CHANNEL = "mm"  # the Z->mumu control region, one category
CONTROL_CATEGORY = "control_region"

# MorphingTauID2017's categories: decay modes and tau pT bin in GeV (None: every tau above 20 GeV)
CATEGORIES = {
    "DM0": ((0,), None),
    "DM1": ((1,), None),
    "DM1011": ((10, 11), None),
    "DM0_PT20_40": ((0,), (20.0, 40.0)),
    "DM1_PT20_40": ((1,), (20.0, 40.0)),
    "DM1011_PT20_40": ((10, 11), (20.0, 40.0)),
    "DM0_PT40_200": ((0,), (40.0, 200.0)),
    "DM1_PT40_200": ((1,), (40.0, 200.0)),
    "DM1011_PT40_200": ((10, 11), (40.0, 200.0)),
}

_NAME = re.compile(r"es([+-]\d+)")


def grid(low: int, high: int, step: int) -> tuple[int, ...]:
    """The shifts from `low` to `high` (inclusive) in steps of `step`, without the nominal 0."""
    return tuple(shift for shift in range(low, high + 1, step) if shift != 0)


def grid_name(shift: int) -> str:
    return f"es{shift:+d}"


def shift_of(name: str) -> int | None:
    """The shift of a grid variation name, None for any other name."""
    match = _NAME.fullmatch(name)
    return int(match.group(1)) if match else None


def factor(shift: int) -> float:
    """The scale factor of the tau four-momentum."""
    return (1000 + shift) / 1000


def mass(shift: int) -> str:
    """MorphingTauID2017's mass string of a shift: the percentage with one decimal."""
    return f"{shift / 10:.1f}"

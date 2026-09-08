"""Evaluation of cut and weight expressions (pandas.eval syntax, numexpr engine), Spec §6."""
from __future__ import annotations

import re
from typing import Mapping

import logging

import numpy as np
import pandas as pd

from shapesmith.model import Region, Selection, WeightVariation

FUNCTIONS = frozenset(
    "abs sqrt exp log log10 sin cos tan arctan2 arcsin arccos arctan sinh cosh tanh expm1 log1p".split()
)
KEYWORDS = frozenset({"and", "or", "not", "True", "False", "inf", "nan"})
_IDENTIFIER = re.compile(r"(?<![\w.])[A-Za-z_][A-Za-z0-9_]*")


logger = logging.getLogger(__name__)


class ExpressionError(ValueError):
    pass


def columns_in(expr: str) -> set[str]:
    """Column names referenced by an expression (functions, keywords and numbers excluded)."""
    return {token for token in _IDENTIFIER.findall(expr) if token not in FUNCTIONS and token not in KEYWORDS}


def evaluate(frame: pd.DataFrame, expr: str) -> np.ndarray:
    """Evaluate `expr` on `frame`; scalars are broadcast to one value per row."""
    missing = sorted(columns_in(expr) - set(frame.columns))
    if missing:
        raise ExpressionError(f"expression {expr!r}: missing columns: {', '.join(missing)}")
    try:
        result = frame.eval(expr, engine="numexpr")
    except NotImplementedError as error:  # numexpr lacks an opcode (e.g. bool * bool): the python engine is slower but complete
        logger.debug(f"numexpr cannot evaluate {expr!r} ({error}); using the python engine")
        try:
            result = frame.eval(expr, engine="python")
        except Exception as fallback_error:
            raise ExpressionError(f"expression {expr!r} failed: {fallback_error}") from fallback_error
    except Exception as error:  # numexpr/pandas raise a mix of ValueError, TypeError, SyntaxError
        raise ExpressionError(f"expression {expr!r} failed: {error}") from error
    if np.ndim(result) == 0:
        return np.full(len(frame), result)
    return np.asarray(result)


def mask(frame: pd.DataFrame, cuts: Mapping[str, str]) -> np.ndarray:
    """Logical AND of all cuts (all True for an empty mapping)."""
    result = np.ones(len(frame), dtype=bool)
    for expr in cuts.values():
        result &= evaluate(frame, expr).astype(bool)
    return result


def weight(frame: pd.DataFrame, weights: Mapping[str, str]) -> np.ndarray:
    """Product of all weights as float64 (all 1.0 for an empty mapping)."""
    result = np.ones(len(frame), dtype=np.float64)
    for expr in weights.values():
        result *= evaluate(frame, expr).astype(np.float64)
    return result


def apply_region(selection: Selection, region: Region) -> Selection:
    """Replace the named cuts of `selection` and append the region weights."""
    cuts = dict(selection.cuts)
    for name, expr in region.replace_cuts.items():
        if name not in cuts:
            raise KeyError(f"region {region.name}: cut {name!r} not in selection {sorted(cuts)}")
        cuts[name] = expr
    weights = dict(selection.weights)
    weights.update(region.add_weights)
    return Selection(cuts=cuts, weights=weights)


def apply_weight_variation(selection: Selection, variation: WeightVariation) -> Selection:
    """Replace the named weights of `selection` (weights not present are an error)."""
    weights = dict(selection.weights)
    for name, expr in variation.replace_weights.items():
        if name not in weights:
            raise KeyError(f"variation {variation.name}: weight {name!r} not in selection {sorted(weights)}")
        weights[name] = expr
    return Selection(cuts=dict(selection.cuts), weights=weights)


def apply_column_variation(expr: str, suffix: str, available: set[str]) -> str:
    """Rename every column `c` in `expr` to `c + suffix` if that shifted column exists."""

    def rename(match: re.Match) -> str:
        name = match.group(0)
        shifted = name + suffix
        return shifted if name in available and shifted in available else name

    return _IDENTIFIER.sub(rename, expr)


def selection_columns(selection: Selection) -> set[str]:
    columns: set[str] = set()
    for expr in list(selection.cuts.values()) + list(selection.weights.values()):
        columns |= columns_in(expr)
    return columns

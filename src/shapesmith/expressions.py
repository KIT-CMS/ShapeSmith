"""Evaluation of cut and weight expressions (pandas.eval syntax, numexpr engine) and their column variations."""
from __future__ import annotations

import logging
import re
from typing import Iterable

import numpy as np
import pandas as pd

from shapesmith.model import ColumnVariation

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


def columns_of(exprs: Iterable[str]) -> set[str]:
    columns: set[str] = set()
    for expr in set(exprs):  # the selections of many regions and variations repeat the same expressions
        columns |= columns_in(expr)
    return columns


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


def mask(frame: pd.DataFrame, cuts: Iterable[str]) -> np.ndarray:
    """Logical AND of all cuts (all True for none)."""
    result = np.ones(len(frame), dtype=bool)
    for expr in cuts:
        result &= evaluate(frame, expr).astype(bool)
    return result


def product(frame: pd.DataFrame, weights: Iterable[str]) -> np.ndarray:
    """Product of all weights as float64, in the given order (all 1.0 for none)."""
    result = np.ones(len(frame), dtype=np.float64)
    for expr in weights:
        result *= evaluate(frame, expr).astype(np.float64)
    return result


def shift(expr: str, variation: ColumnVariation, available: set[str]) -> str:
    """`expr` under a column variation: a derived column becomes its expression, a column `c` whose shifted branch
    `c + suffix` is in `available` is read shifted, and every other column stays nominal."""

    def rename(match: re.Match) -> str:
        name = match.group(0)
        if name in variation.derived:
            return f"({variation.derived[name]})"
        shifted = name + variation.suffix
        return shifted if variation.suffix and shifted in available else name

    return _IDENTIFIER.sub(rename, expr)

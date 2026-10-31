"""The options of the combine commands of fits and limits, from `combine` of the run configuration (CombineConfig):
the range of r, the converged AsymptoticLimits (`precise`) and the minimizer options, which replace built-in options
of the same name.

PRECISE_LIMIT (R1, 2026-10-25): at combine's default accuracy AsymptoticLimits puts the median on a 1/32 grid of its
bisection and hides effects below ~0.3 % (thesis card 10.90625 at the defaults, 10.9048 converged); with these options
the limits converge, as paired comparisons of models need. They apply to AsymptoticLimits only: HybridNew reads
--rAbsAcc and --rRelAcc as the accuracy of its toy-based search.
"""
from __future__ import annotations

import shlex

from shapesmith.config import CombineConfig

PRECISE_LIMIT = "--rAbsAcc 0.00001 --rRelAcc 0.00001 --cminDefaultMinimizerTolerance 0.0001"
DEFAULT = CombineConfig(cmssw_dir="")  # the options of a run without `combine` settings beyond cmssw_dir


def _starts_option(token: str) -> bool:
    """`--name`, `--name=value` or `-x`; not a negative number (`-1`, `-40,40`)."""
    return token.startswith("--") or (token.startswith("-") and token[1:2].isalpha())


def _flag(token: str) -> str:
    return token.partition("=")[0]


def split_options(text: str) -> list[list[str]]:
    """The tokens of a command grouped per option: [[program, ...], [flag, values...], ...]."""
    groups: list[list[str]] = []
    for token in shlex.split(text):
        if _starts_option(token) or not groups:
            groups.append([token])
        else:
            groups[-1].append(token)
    return groups


def merge(command: str, *overrides: str) -> str:
    """The command with the options of `overrides` appended; an option of an override replaces every occurrence of
    the same flag before it (a flag repeated within one string, e.g. --X-rtd, stays repeated)."""
    groups = split_options(command)
    for override in overrides:
        added = split_options(override)
        flags = {_flag(group[0]) for group in added}
        groups = [group for i, group in enumerate(groups) if i == 0 or _flag(group[0]) not in flags] + added
    return " ".join(shlex.join(group) for group in groups)


def r_range(combine: CombineConfig | None) -> str:
    low, high = (combine or DEFAULT).r_range
    return f"--setParameterRanges r={low:g},{high:g}"


def command(base: str, combine: CombineConfig | None, precise_limit: bool = False, with_range: bool = True) -> str:
    """`base` (method, workspace, name, ...) with the options of the run, merged: the range of r, PRECISE_LIMIT for an
    AsymptoticLimits call (`precise_limit`) of a run with `precise`, the minimizer options last."""
    combine = combine or DEFAULT
    overrides = [r_range(combine)] if with_range else []
    if precise_limit and combine.precise:
        overrides.append(PRECISE_LIMIT)
    if combine.minimizer:
        overrides.append(combine.minimizer)
    return merge(base, *overrides)


def parameters(assignments: list[str]) -> str:
    """--setParameters of the assignments (e.g. r=0, mask_<bin>=1), empty without any."""
    return f"--setParameters {','.join(assignments)}" if assignments else ""


def frozen(names: list[str]) -> str:
    return f"--freezeParameters {','.join(names)}" if names else ""

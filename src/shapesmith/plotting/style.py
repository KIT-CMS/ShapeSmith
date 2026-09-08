"""Plot styling taken from Analysis.style (colours, labels, group order), Spec §11."""
from __future__ import annotations

from shapesmith.model import Analysis


def grouped_backgrounds(analysis: Analysis) -> list[tuple[str, list[str]]]:
    """[(group, [process names]), ...] in Style.group_order; the estimator output forms its own group."""
    groups: dict[str, list[str]] = {}
    for process in analysis.processes:
        if process.kind in ("data", "signal"):
            continue
        groups.setdefault(process.plot_group, []).append(process.name)
    if analysis.estimator is not None:
        groups.setdefault(analysis.estimator.output, []).append(analysis.estimator.output)
    order = list(analysis.style.group_order) if analysis.style else sorted(groups)
    ordered = [(g, groups[g]) for g in order if g in groups]
    ordered += [(g, members) for g, members in groups.items() if g not in order]
    return ordered


def axis_label(analysis: Analysis, channel: str, variable: str) -> str:
    if analysis.style is None:
        return variable
    return analysis.style.axis_labels.get(channel, {}).get(variable, variable)


def color(analysis: Analysis, group: str) -> str:
    return analysis.style.colors.get(group, "#999999") if analysis.style else "#999999"


def label(analysis: Analysis, group: str) -> str:
    return analysis.style.labels.get(group, group) if analysis.style else group

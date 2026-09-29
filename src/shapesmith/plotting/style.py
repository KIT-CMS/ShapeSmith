"""Plot styling taken from Analysis.style (colours, labels, group order), Spec §11."""
from __future__ import annotations

from shapesmith.model import Analysis


def grouped_backgrounds(analysis: Analysis, channel: str) -> list[tuple[str, list[str]]]:
    """[(group, [process names]), ...] of the channel's backgrounds in Style.group_order; an estimator output forms its own group."""
    ch = analysis.channel(channel)
    estimated = set(ch.backgrounds()) - {p.name for p in ch.processes}
    groups: dict[str, list[str]] = {}
    for name in ch.backgrounds():
        groups.setdefault(name if name in estimated else ch.process(name).plot_group, []).append(name)
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

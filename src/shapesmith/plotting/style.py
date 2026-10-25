"""Plot styling taken from Analysis.style (colours, labels, group order), Spec §11."""
from __future__ import annotations

import re

import numpy as np

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


def edge_label(edge: float) -> str:
    """A bin edge as a tick label: three decimals at most (the fine bins of an NN score are 0.001 wide), without
    trailing zeros."""
    text = f"{edge:.3f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def card_bin_positions(edges) -> np.ndarray:
    """The drawn edges of the bins of a card shown with equal widths: 0, 1, ..., n (card_bin_axis)."""
    return np.arange(len(edges), dtype=np.float64)


def card_bin_axis(axis, edges, label: str) -> None:
    """The x axis of a plot drawn on card_bin_positions (the bins of a card with equal widths, e.g. of an NN score
    with a wide last bin): one tick per edge labelled with its value, no minor ticks, the label says so."""
    positions = card_bin_positions(edges)
    axis.set_xticks(positions, [edge_label(e) for e in edges], rotation=90 if len(edges) > 7 else 0, fontsize=16)
    axis.tick_params(axis="x", which="minor", bottom=False, top=False)
    axis.tick_params(axis="x", which="major", pad=10)  # clear of the lowest label of the y axis
    axis.set_xlim(positions[0], positions[-1])
    axis.set_xlabel(f"{label} (card bins)")


def color(analysis: Analysis, group: str) -> str:
    return analysis.style.colors.get(group, "#999999") if analysis.style else "#999999"


def label(analysis: Analysis, group: str) -> str:
    return analysis.style.labels.get(group, group) if analysis.style else group


_SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "Gamma": "Γ", "delta": "δ", "Delta": "Δ", "epsilon": "ε", "zeta": "ζ", "eta": "η",
    "theta": "θ", "Theta": "Θ", "kappa": "κ", "lambda": "λ", "Lambda": "Λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ",
    "sigma": "σ", "Sigma": "Σ", "tau": "τ", "phi": "φ", "varphi": "φ", "Phi": "Φ", "chi": "χ", "psi": "ψ", "Psi": "Ψ", "omega": "ω",
    "Omega": "Ω", "ell": "ℓ", "sum": "Σ", "rightarrow": "→", "to": "→", "leftarrow": "←", "leftrightarrow": "↔", "times": "×",
    "pm": "±", "cdot": "·", "circ": "°", "geq": "≥", "leq": "≤", "neq": "≠", "approx": "≈", "infty": "∞",
}
_SUPERSCRIPT_CHARACTERS = "0123456789+-=()"
_SUPERSCRIPTS = str.maketrans(_SUPERSCRIPT_CHARACTERS, "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")
_FONTS = re.compile(r"\\(?:mathrm|mathit|mathbf|mathsf|text|textrm|textit|textbf)\{([^{}]*)\}")
_BAR = re.compile(r"\\(?:bar|overline)\{([^{}]*)\}")
_SCRIPT = re.compile(r"([_^])(?:\{([^{}]*)\}|(.))")


def _script(match: re.Match) -> str:
    content = match.group(2) if match.group(2) is not None else match.group(3)
    if match.group(1) == "^" and all(c in _SUPERSCRIPT_CHARACTERS for c in content):
        return content.translate(_SUPERSCRIPTS)
    return content  # subscripts and other superscripts inline: $\tau_h$ -> τh


def _math(text: str) -> str:
    while _FONTS.search(text):
        text = _FONTS.sub(r"\1", text)
    text = _BAR.sub(lambda m: m.group(1) + "\u0304", text)  # combining macron over the (last) character
    text = re.sub(r"\\([A-Za-z]+)\s*", lambda m: _SYMBOLS.get(m.group(1), m.group(1)), text)
    text = _SCRIPT.sub(_script, text)
    text = re.sub(r"\\[,;:! ]", " ", text)
    return text.replace("{", "").replace("}", "")


def plain_text(text: str) -> str:
    """A mathtext label as plain (Unicode) text, e.g. for a web page: r"$\\mu\\tau_h$" -> "μτh", r"$m_{vis}$ / GeV" -> "mvis / GeV"."""
    parts = re.split(r"(?<!\\)\$", text)
    return "".join(_math(part) if i % 2 else part.replace(r"\$", "$") for i, part in enumerate(parts))

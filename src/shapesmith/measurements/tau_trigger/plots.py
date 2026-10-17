"""Plots of the trigger measurement: the efficiency of the tau leg against pT per decay mode, from the fit (data), the
counting cross-check, MC and the monitoring flags of the embedded taus, with a reference payload (POG) if given."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402

hep.style.use("CMS")
SERIES = (("fit", "data (fit)", "black", "o"), ("counting", "data (counting)", "#5790fc", "s"), ("mc", "MC", "#e42536", "^"), ("emb_monitor", "embedding, monitoring flags", "#f89c20", "v"))
PLOT_PT_MAX = 200.0  # the open last bin drawn up to here


def _value(entry: Mapping | None, key: str) -> tuple[float, float, float] | None:
    """(value, error down, error up) of a series entry."""
    if not entry:
        return None
    if key == "fit":
        return entry["best"], entry["best"] - entry["low"], entry["high"] - entry["best"]
    return entry["value"], entry["error"], entry["error"]


def plot_efficiencies(bins: list[dict], title: str, path: Path, reference: Callable[[float], float] | None = None) -> list[Path]:
    """`bins`: the bin records of one decay mode in pT order (keys pt, fit, counting, mc, emb_monitor)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    lows = np.array([b["pt"][0] for b in bins], dtype=float)
    highs = np.array([PLOT_PT_MAX if b["pt"][1] is None else min(b["pt"][1], PLOT_PT_MAX) for b in bins], dtype=float)
    centres = 0.5 * (lows + highs)
    for offset, (key, label, colour, marker) in zip((0.0, 0.08, -0.08, 0.16), SERIES):
        points = [(i, v) for i, b in enumerate(bins) if (v := _value(b.get(key), key)) is not None]
        if not points:
            continue
        x = np.array([centres[i] + offset * (highs[i] - lows[i]) for i, _ in points])
        values = np.array([v for _, v in points])
        ax.errorbar(x, values[:, 0], yerr=[values[:, 1], values[:, 2]], fmt=marker, color=colour, label=label, markersize=6)
    if reference is not None:
        grid = np.linspace(lows[0], PLOT_PT_MAX, 200)
        ax.plot(grid, [reference(pt) for pt in grid], color="gray", linestyle="--", label="reference (POG)")
    ax.set_xlim(lows[0], PLOT_PT_MAX)
    ax.set_ylim(0.0, 1.15)
    ax.set_xlabel(r"$p_T^{\tau_h}$ [GeV]")
    ax.set_ylabel("efficiency of the tau leg")
    ax.text(0.04, 0.95, title, transform=ax.transAxes, va="top", fontsize=16)
    ax.legend(loc="lower right", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    paths = [path.with_suffix(ext) for ext in (".pdf", ".png")]
    for p in paths:
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return paths


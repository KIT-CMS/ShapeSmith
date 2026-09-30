"""Plots of the measurement: the 2D likelihood scan per category, the postfit shapes of the singles fit, and prefit
control plots of m_vis with a family of energy-scale grid points."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402
import uproot  # noqa: E402

from shapesmith.histogram import NOMINAL_VARIATION, HistKey, HistogramSet  # noqa: E402
from shapesmith.measurements.tau_id_es.combine import Interval, Scan  # noqa: E402
from shapesmith.measurements.tau_id_es.grid import CHANNEL, SIGNAL, grid_name  # noqa: E402
from shapesmith.model import NOMINAL, Analysis  # noqa: E402

hep.style.use("CMS")
ES_FAMILY = (-200, -100, 100, 200)  # grid points drawn in the control plots, if present
LEVELS = {2.30: "68 % CL", 5.99: "95 % CL"}  # 2 delta NLL of two parameters


def _save(fig, path: Path) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    paths = [path.with_suffix(ext) for ext in (".pdf", ".png")]
    for p in paths:
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_scan(scan: Scan, sf: Interval, es: Interval, category: str, path: Path) -> list[Path]:
    """2 delta NLL of the scan grid with the 68 % and 95 % contours and the singles result."""
    finite = np.isfinite(scan.dnll)
    r, e, z = scan.r[finite], scan.es[finite], 2 * (scan.dnll[finite] - scan.dnll[finite].min())
    fig, ax = plt.subplots(figsize=(10, 9))
    filled = ax.tricontourf(r, e, np.minimum(z, 30.0), levels=30, cmap="viridis")
    lines = ax.tricontour(r, e, z, levels=sorted(LEVELS), colors=["white", "orange"])
    ax.clabel(lines, fmt={level: label for level, label in LEVELS.items()}, fontsize=14)
    ax.errorbar([sf.best], [es.best], xerr=[[sf.best - sf.low], [sf.high - sf.best]], yerr=[[es.best - es.low], [es.high - es.best]], fmt="o", color="red", label="singles fit")
    ax.set_xlabel(f"$r_{{EMB}}$ ({category})")
    ax.set_ylabel("tau ES shift [%]")
    ax.legend(loc="upper right")
    fig.colorbar(filled, ax=ax, label=r"$2\,\Delta$NLL")
    return _save(fig, path)


def plot_postfit(shapes: Path, output_dir: Path) -> list[Path]:
    """Data and the stacked postfit processes of every bin of the singles fit, with the data/prediction ratio."""
    written = []
    with uproot.open(shapes) as f:
        for directory in (name for name in f.keys(cycle=False, recursive=False) if name.endswith("_postfit")):
            processes = [name for name in f[directory].keys(cycle=False) if not name.startswith(("Total", "data_obs"))]
            edges = f[f"{directory}/data_obs"].axis().edges()
            stack = [f[f"{directory}/{name}"].values() for name in processes]
            data = f[f"{directory}/data_obs"]
            total = f[f"{directory}/TotalProcs"]
            fig, (ax, rax) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06}, sharex=True)
            hep.histplot(stack, bins=edges, stack=True, histtype="fill", label=processes, ax=ax)
            hep.histplot(data.values(), bins=edges, yerr=np.sqrt(data.variances()), histtype="errorbar", color="black", label="Data", ax=ax)
            ax.set_ylabel("Events")
            ax.legend(ncol=2, fontsize=14, frameon=False)
            ax.text(0.04, 0.95, directory, transform=ax.transAxes, va="top", fontsize=16)
            with np.errstate(divide="ignore", invalid="ignore"):
                rax.errorbar(0.5 * (edges[1:] + edges[:-1]), data.values() / total.values(), yerr=np.sqrt(data.variances()) / total.values(), fmt="o", color="black")
            rax.axhline(1.0, color="gray")
            rax.set_ylim(0.8, 1.2)
            rax.set_ylabel("Data / fit")
            rax.set_xlabel(r"$m_{vis}$ [GeV]")
            written += _save(fig, Path(output_dir) / directory)
    return written


def plot_control(hset: HistogramSet, analysis: Analysis, category: str, path: Path) -> list[Path]:
    """Prefit m_vis of one category: data, the stacked backgrounds with the nominal embedded signal, and the total
    prediction with the embedded signal at a few grid points."""
    channel = analysis.channel(CHANNEL)
    variable = next(c.variable for c in channel.categories if c.name == category)

    def get(process: str, variation: str = NOMINAL_VARIATION):
        return hset.get(HistKey(CHANNEL, category, process, NOMINAL, variation, variable.name))

    backgrounds = [(name, h) for name in channel.backgrounds() if (h := get(name)) is not None]
    edges, widths = np.asarray(variable.edges), np.diff(variable.edges)
    total = np.sum([h.values for _, h in backgrounds], axis=0)
    signal = get(SIGNAL)
    fig, ax = plt.subplots(figsize=(10, 8))
    hep.histplot([h.values / widths for _, h in backgrounds], bins=edges, stack=True, histtype="fill", label=[name for name, _ in backgrounds], ax=ax)
    for shift in ES_FAMILY:
        shifted = get(SIGNAL, grid_name(shift))
        if shifted is not None:
            hep.histplot((total - signal.values + shifted.values) / widths, bins=edges, histtype="step", linewidth=2, label=f"ES {shift / 10:+.1f} %", ax=ax)
    data = get(channel.data())
    hep.histplot(data.values / widths, bins=edges, yerr=np.sqrt(data.variances) / widths, histtype="errorbar", color="black", label="Data", ax=ax)
    ax.set_ylabel("dN / dm$_{vis}$ [1/GeV]")
    ax.set_xlabel(r"$m_{vis}$ [GeV]")
    ax.legend(ncol=2, fontsize=14, frameon=False)
    ax.text(0.04, 0.95, f"{CHANNEL}, {category}", transform=ax.transAxes, va="top", fontsize=16)
    return _save(fig, path)

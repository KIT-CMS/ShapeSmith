"""Plots of the fake-factor measurement from its record: every fitted ratio with its curve and bands per category,
the SR-like yield against the (weighted) AR-like yield below it (the closure), and the fractions."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402

hep.style.use("CMS")


def plot_record(record: dict[str, list[dict]], directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, entries in record.items():
        for entry in entries:
            path = directory / f"{name.replace('/', '_')}_category{entry.get('category', 0)}.png"
            if "curve" in entry:
                paths.append(_plot_fit(name, entry, path))
            elif "fractions" in entry:
                paths.append(_plot_fractions(name, entry, path))
    return paths


def _plot_fit(name: str, entry: dict, path: Path) -> Path:
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={"height_ratios": (2, 1)})
    curve = entry["curve"]
    edges = np.asarray(curve["edges"])
    for band, colour in (("SystBandAsym", "#f89c20"), ("StatShift", "#5790fc")):
        top.stairs(np.asarray(curve[f"{band}Up"]), edges, baseline=np.asarray(curve[f"{band}Down"]), fill=True, alpha=0.4, color=colour, label=band)
    top.stairs(np.asarray(curve["nominal"]), edges, baseline=None, color="black", label="fit")
    top.errorbar(entry["x"], entry["y"], yerr=entry["errors"], fmt="o", color="black", label="measured")
    top.set_ylabel("ratio")
    top.set_title(f"{name} {entry['split']}" + (" (reset to 1)" if entry["reset"] else ""), fontsize=14)
    top.legend(fontsize=12)
    bin_edges = entry["edges"]
    bottom.stairs(entry["numerator"], bin_edges, color="black", label="SR-like")
    bottom.stairs(entry["denominator"], bin_edges, color="#e42536", label="AR-like (weighted)")
    bottom.set_ylabel("yield")
    bottom.set_xlabel(entry["variable"])
    bottom.legend(fontsize=12)
    figure.savefig(path)
    plt.close(figure)
    return path


def _plot_fractions(name: str, entry: dict, path: Path) -> Path:
    figure, ax = plt.subplots(figsize=(10, 8))
    bottom = np.zeros(len(entry["edges"]) - 1)
    for (process, values), colour in zip(entry["fractions"].items(), ("#5790fc", "#e42536")):
        ax.stairs(bottom + np.asarray(values), entry["edges"], baseline=bottom, fill=True, color=colour, label=process)
        bottom = bottom + np.asarray(values)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("fraction")
    ax.set_title(f"{name} category {entry['category']}", fontsize=14)
    ax.legend(fontsize=12)
    figure.savefig(path)
    plt.close(figure)
    return path

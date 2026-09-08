"""A minimal weighted 1D histogram on numpy arrays with ROOT TH1D conversion through uproot.

boost-histogram/hist are deliberately not used: in LCG_108 (boost_histogram 1.3.2, numpy 2.1) the axis
`edges` property returns constant values, and uproot writes exactly those corrupted edges. numpy plus
uproot's low-level TH1 constructor gives correct variable bins and Sumw2 for combine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from uproot.writing.identify import to_TAxis, to_TH1x


@dataclass
class Histogram:
    edges: np.ndarray
    values: np.ndarray
    variances: np.ndarray

    def __post_init__(self):
        self.edges = np.asarray(self.edges, dtype=np.float64)
        self.values = np.asarray(self.values, dtype=np.float64)
        self.variances = np.asarray(self.variances, dtype=np.float64)
        if len(self.values) != len(self.edges) - 1 or len(self.variances) != len(self.values):
            raise ValueError("values and variances need len(edges) - 1 entries")
        if not np.all(np.diff(self.edges) > 0):
            raise ValueError(f"edges must be strictly increasing, got {self.edges.tolist()}")

    @classmethod
    def empty(cls, edges) -> "Histogram":
        n = len(edges) - 1
        return cls(edges, np.zeros(n), np.zeros(n))

    @classmethod
    def fill(cls, edges, x, weights=None) -> "Histogram":
        x = np.asarray(x, dtype=np.float64)
        w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=np.float64)
        values, _ = np.histogram(x, bins=edges, weights=w)
        variances, _ = np.histogram(x, bins=edges, weights=w * w)
        return cls(edges, values, variances)

    def copy(self) -> "Histogram":
        return Histogram(self.edges.copy(), self.values.copy(), self.variances.copy())

    def add(self, other: "Histogram", scale: float = 1.0) -> "Histogram":
        """self += scale * other (variances add with scale squared); returns self."""
        if len(other.edges) != len(self.edges) or not np.allclose(other.edges, self.edges):
            raise ValueError(f"binning mismatch: {self.edges.tolist()} vs {other.edges.tolist()}")
        self.values += scale * other.values
        self.variances += scale * scale * other.variances
        return self

    def scale(self, factor: float) -> "Histogram":
        self.values *= factor
        self.variances *= factor * factor
        return self

    def sum(self) -> float:
        return float(self.values.sum())

    def rebin(self, edges) -> "Histogram":
        edges = np.asarray(edges, dtype=np.float64)
        indices = [int(np.argmin(np.abs(self.edges - edge))) for edge in edges]
        if not np.allclose(self.edges[indices], edges):
            raise ValueError(f"new edges {edges.tolist()} are not a subset of {self.edges.tolist()}")
        return Histogram(edges, np.add.reduceat(self.values, indices[:-1]), np.add.reduceat(self.variances, indices[:-1]))

    def to_root(self, name: str):
        """TH1D with variable bins and Sumw2, writable with uproot: `file[path] = histogram.to_root(name)`."""
        data = np.concatenate([[0.0], self.values, [0.0]])  # underflow, bins, overflow
        sumw2 = np.concatenate([[0.0], self.variances, [0.0]])
        axis = to_TAxis(fName="xaxis", fTitle="", fNbins=len(self.values), fXmin=float(self.edges[0]), fXmax=float(self.edges[-1]), fXbins=self.edges)
        return to_TH1x(fName=name, fTitle=name, data=data, fEntries=float(self.values.sum()), fTsumw=float(self.values.sum()), fTsumw2=float(self.variances.sum()), fTsumwx=0.0, fTsumwx2=0.0, fSumw2=sumw2, fXaxis=axis)

    @classmethod
    def from_root(cls, th) -> "Histogram":
        return cls(th.axis().edges(), th.values(), th.variances())

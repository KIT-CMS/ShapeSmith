"""A minimal weighted 1D histogram on numpy arrays with ROOT TH1D conversion through uproot, and the keyed set
of histograms the stages exchange (one ROOT file plus a JSON index).

boost-histogram/hist are deliberately not used: in LCG_108 (boost_histogram 1.3.2, numpy 2.1) the axis
`edges` property returns constant values, and uproot writes exactly those corrupted edges. numpy plus
uproot's low-level TH1 constructor gives correct variable bins and Sumw2 for combine.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import uproot
from uproot.writing.identify import to_TAxis, to_TH1x

NOMINAL_VARIATION = "Nominal"
INCLUSIVE = "inclusive"  # the category of control-variable histograms


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


@dataclass(frozen=True, order=True)
class HistKey:
    channel: str
    category: str
    process: str
    region: str
    variation: str
    variable: str

    @property
    def directory(self) -> str:
        return f"{self.channel}_{self.category}"

    @property
    def object_name(self) -> str:
        return f"{self.process}#{self.region}#{self.variation}#{self.variable}"

    @property
    def path(self) -> str:
        return f"{self.directory}/{self.object_name}"

    @classmethod
    def parse(cls, path: str) -> "HistKey":
        directory, name = path.split("/", 1)
        channel, category = directory.split("_", 1)
        process, region, variation, variable = name.split("#")
        return cls(channel, category, process, region, variation, variable)


class HistogramSet(dict):
    """HistKey -> Histogram, saved as one ROOT file (`<channel>_<category>/<process>#<region>#<variation>#<variable>`)
    plus a JSON index next to it."""

    def select(self, **fields) -> list[HistKey]:
        """The keys whose fields equal the given values, sorted."""
        return sorted(key for key in self if all(getattr(key, name) == value for name, value in fields.items()))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        items = sorted(self.items())
        with uproot.recreate(path) as f:
            for key, h in items:
                f[key.path] = h.to_root(key.object_name)
        index = [{**asdict(key), "sum": h.sum(), "bins": len(h.values)} for key, h in items]
        path.with_suffix(".json").write_text(json.dumps(index, indent=1))

    @classmethod
    def load(cls, path: Path) -> "HistogramSet":
        hset = cls()
        with uproot.open(path) as f:
            for name, classname in f.classnames(recursive=True, cycle=False).items():
                if classname.startswith("TH1"):
                    hset[HistKey.parse(name)] = Histogram.from_root(f[name])
        return hset


def unchanged_variations(hset: HistogramSet) -> list[HistKey]:
    """The variation histograms that are bitwise equal to their nominal (a declared variation that has no effect)."""
    result = []
    for key in hset.select():
        if key.variation == NOMINAL_VARIATION:
            continue
        nominal = hset.get(HistKey(key.channel, key.category, key.process, key.region, NOMINAL_VARIATION, key.variable))
        if nominal is not None and np.array_equal(hset[key].values, nominal.values) and np.array_equal(hset[key].variances, nominal.variances):
            result.append(key)
    return result

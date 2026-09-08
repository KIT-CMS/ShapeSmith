"""ShapeSmith: from CROWN ntuples to histograms, estimates, datacards, fits and ML folds without ROOT."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("shapesmith")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

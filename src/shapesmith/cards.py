"""What a text datacard declares: its bins, its nuisances and the shape file of a bin's data."""
from __future__ import annotations

from pathlib import Path

NUISANCE_KINDS = {"lnN", "shape", "shapeN", "shape?", "lnU", "gmN", "param"}


def card_nuisances(card: Path) -> list[str]:
    """The nuisance names of a text datacard (lnN, shape and the like; rateParams and autoMCStats excluded)."""
    names = []
    for line in Path(card).read_text().splitlines():
        tokens = line.split()
        if len(tokens) > 1 and tokens[1] in NUISANCE_KINDS and not line.startswith(("bin ", "process ", "rate ", "observation ")):
            names.append(tokens[0])
    return names


def card_bins(card: Path) -> list[str]:
    """The bins (channels) of a text datacard: its first `bin` line, the one of the observations."""
    for line in Path(card).read_text().splitlines():
        tokens = line.split()
        if tokens and tokens[0] == "bin":
            return tokens[1:]
    return []


def data_histogram(card: Path, card_bin: str) -> tuple[Path, str] | None:
    """(shape file, histogram path) of the data of a bin from the card's `shapes` lines (a line of the bin before one
    of `*`), None without one; a relative shape file is relative to the card's directory."""
    entries = {}
    for line in Path(card).read_text().splitlines():
        tokens = line.split()
        if len(tokens) >= 5 and tokens[0] == "shapes" and tokens[1] in ("*", "data_obs") and tokens[2] in (card_bin, "*"):
            entries.setdefault((tokens[2] != card_bin, tokens[1] != "data_obs"), (tokens[3], tokens[4]))
    if not entries:
        return None
    file, pattern = entries[min(entries)]
    path = Path(file) if Path(file).is_absolute() else Path(card).parent / file
    return path, pattern.replace("$PROCESS", "data_obs").replace("$CHANNEL", card_bin)

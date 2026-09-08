"""Read columns from a CROWN ntuple and its friend trees with uproot (Spec §7 steps 2-3)."""
from __future__ import annotations

import json

import pandas as pd
import uproot

from shapesmith.io.discovery import NtupleFile

TREE = "ntuple"


class MissingColumnsError(ValueError):
    def __init__(self, missing: list[str], path: str):
        self.missing = sorted(missing)
        self.path = path
        super().__init__(f"{path}: missing columns: {', '.join(self.missing)}")

    def __reduce__(self):  # picklable with its two arguments, so that worker processes can hand it back
        return (MissingColumnsError, (self.missing, self.path))


def available_columns(ntuple: NtupleFile) -> dict[str, str]:
    """column name -> file that provides it (the main file wins over friends)."""
    result: dict[str, str] = {}
    for path in (*ntuple.friends, ntuple.path):  # main file last so that it overrides
        with uproot.open(path) as handle:
            for name in handle[TREE].keys():
                result[name] = path
    return result


def read_ntuple(ntuple: NtupleFile, columns: set[str], optional: set[str] = frozenset()) -> tuple[pd.DataFrame, dict]:
    """The requested columns of one ntuple (+ friends) and the CROWN metadata of the main file, opening every file once.

    Remote opens cost seconds, so columns are listed and read from the same handle. Columns in `optional` may be
    absent (they are left out); any other missing column raises MissingColumnsError.
    """
    paths = (*ntuple.friends, ntuple.path)  # main file last so that it overrides friends
    handles = [uproot.open(path) for path in paths]
    try:
        sources: dict[str, int] = {}
        for index, handle in enumerate(handles):
            for name in handle[TREE].keys():
                sources[name] = index
        missing = set(columns) - set(sources)
        if missing - set(optional):
            raise MissingColumnsError(sorted(missing - set(optional)), ntuple.path)
        by_handle: dict[int, list[str]] = {}
        for column in sorted(set(columns) & set(sources)):
            by_handle.setdefault(sources[column], []).append(column)
        frames = [pd.DataFrame(handles[index][TREE].arrays(names, library="np")) for index, names in by_handle.items()]
        main = handles[-1]
        metadata = json.loads(str(main["metadata"])) if "metadata" in main else {}
    finally:
        for handle in handles:
            handle.close()
    lengths = {len(frame) for frame in frames}
    if len(lengths) > 1:
        raise ValueError(f"{ntuple.path}: friend trees have different numbers of entries {sorted(lengths)}")
    return (pd.concat(frames, axis=1) if frames else pd.DataFrame()), metadata


def read_columns(ntuple: NtupleFile, columns: set[str]) -> pd.DataFrame:
    """The requested columns of one ntuple (+ friends) as a DataFrame; dtypes as stored."""
    return read_ntuple(ntuple, columns)[0]


def read_metadata(path: str) -> dict:
    """The CROWN `metadata` JSON block (empty dict when absent)."""
    with uproot.open(path) as handle:
        if "metadata" not in handle:
            return {}
        return json.loads(str(handle["metadata"]))


def num_entries(path: str) -> int:
    with uproot.open(path) as handle:
        return handle[TREE].num_entries

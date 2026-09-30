"""Locate CROWN ntuples and their friend trees, and read columns from them with uproot."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import pandas as pd
import uproot

from shapesmith.config import NtupleConfig

TREE = "ntuple"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NtupleFile:
    path: str               # full path or xrootd URL of the main ntuple
    friends: tuple[str, ...]  # full paths/URLs of the friend files, same order as the config
    nick: str
    channel: str
    basename: str


class MissingColumnsError(ValueError):
    def __init__(self, missing: list[str], path: str):
        self.missing = sorted(missing)
        self.path = path
        super().__init__(f"{path}: missing columns: {', '.join(self.missing)}")

    def __reduce__(self):  # picklable with its two arguments, so that worker processes can hand it back
        return (MissingColumnsError, (self.missing, self.path))


def join_url(server: str, path: str) -> str:
    """'root://server' + '/store/...' -> 'root://server//store/...'; local paths pass through."""
    if not server:
        return path
    return f"{server.rstrip('/')}/{path}"


def list_root_files(server: str, directory: str) -> list[str]:
    """Sorted basenames of the .root files in `directory` (xrootd when `server` is set)."""
    if server:
        from XRootD import client  # optional dependency, only needed for remote access
        from XRootD.client.flags import DirListFlags

        status, listing = client.FileSystem(server).dirlist(directory, DirListFlags.STAT)
        if not status.ok or listing is None:
            raise FileNotFoundError(f"cannot list {join_url(server, directory)}: {status.message}")
        names = [entry.name for entry in listing]
    else:
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"cannot list {directory}")
        names = os.listdir(directory)
    return sorted(name for name in names if name.endswith(".root"))


def friend_bases(ntuples: NtupleConfig, kind: str) -> list[str]:
    """The friend bases that apply to a sample kind, in configured order (later friends override earlier ones)."""
    return [friend.base for friend in ntuples.friends if kind in friend.applies_to]


def discover(ntuples: NtupleConfig, era: str, nick: str, channel: str, kind: str) -> list[NtupleFile]:
    """Main files of one sample and channel with their friend files; missing friends are an error."""
    main_dir = f"{ntuples.base.rstrip('/')}/{era}/{nick}/{channel}"
    basenames = list_root_files(ntuples.server, main_dir)
    friend_dirs = [f"{base.rstrip('/')}/{era}/{nick}/{channel}" for base in friend_bases(ntuples, kind)]
    friend_listings = [set(list_root_files(ntuples.server, directory)) for directory in friend_dirs]
    missing = [f"{directory}/{name}" for directory, names in zip(friend_dirs, friend_listings) for name in basenames if name not in names]
    if missing:
        raise FileNotFoundError("missing friend files:\n" + "\n".join(missing))
    logger.debug(f"{channel}/{nick}: {len(basenames)} files in {join_url(ntuples.server, main_dir)}" + (f", {len(friend_dirs)} friends each" if friend_dirs else ""))
    return [
        NtupleFile(
            path=join_url(ntuples.server, f"{main_dir}/{name}"),
            friends=tuple(join_url(ntuples.server, f"{directory}/{name}") for directory in friend_dirs),
            nick=nick,
            channel=channel,
            basename=name,
        )
        for name in basenames
    ]


def read_ntuple(ntuple: NtupleFile, columns: set[str], optional: set[str] = frozenset()) -> tuple[pd.DataFrame, dict, set[str]]:
    """The requested columns of one ntuple (+ friends), the CROWN metadata of the main file and the names of all
    branches, opening every file once (remote opens cost seconds).

    Columns in `optional` may be absent (they are left out); any other missing column raises MissingColumnsError.
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
    return (pd.concat(frames, axis=1) if frames else pd.DataFrame()), metadata, set(sources)

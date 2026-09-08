"""Locate CROWN ntuples and their friend trees (Spec §7 step 1)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from shapesmith.config import NtupleConfig


@dataclass(frozen=True)
class NtupleFile:
    path: str               # full path or xrootd URL of the main ntuple
    friends: tuple[str, ...]  # full paths/URLs of the friend files, same order as the config
    nick: str
    channel: str
    basename: str


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


def discover(ntuples: NtupleConfig, era: str, nick: str, channel: str, kind: str) -> list[NtupleFile]:
    """Main files of one sample and channel with their friend files; missing friends are an error."""
    main_dir = f"{ntuples.base.rstrip('/')}/{era}/{nick}/{channel}"
    basenames = list_root_files(ntuples.server, main_dir)
    friend_dirs = [f"{f.base.rstrip('/')}/{era}/{nick}/{channel}" for f in ntuples.friends if kind in f.applies_to]
    friend_listings = [set(list_root_files(ntuples.server, directory)) for directory in friend_dirs]
    missing = [f"{directory}/{name}" for directory, names in zip(friend_dirs, friend_listings) for name in basenames if name not in names]
    if missing:
        raise FileNotFoundError("missing friend files:\n" + "\n".join(missing))
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

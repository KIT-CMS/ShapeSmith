"""Synthetic ntuples for tests and examples.

The mini dataset mimics the CROWN layout:
    <root>/CROWNRun/<era>/<nick>/<channel>/<nick>_<i>.root          main ntuples
    <root>/CROWNFriends/nn/<era>/<nick>/<channel>/<nick>_<i>.root   friend trees (score, cls, fake_factor)
With `variations`, it also has an embedded sample and CROWN shifts (`<column>__<shift>`) in main and friend trees.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import uproot

MINI_NICKS = ["DATA_A", "ZTT_1", "SIG_1"]
EMBEDDING_NICK = "EMB_A"


def make_ntuple(path: Path, columns: dict[str, np.ndarray], metadata: dict | None = None, tree: str = "ntuple") -> Path:
    """Write a flat TTree (and an optional JSON `metadata` TObjString) with uproot."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as f:
        f[tree] = columns
        if metadata is not None:
            f["metadata"] = json.dumps(metadata)
    return path


def _main_columns(rng: np.random.Generator, n: int, offset: int, is_data: bool) -> dict[str, np.ndarray]:
    columns = {
        "veto": (rng.random(n) < 0.1).astype(np.float32),
        "q_1": rng.choice([-1.0, 1.0], n).astype(np.float32),
        "q_2": rng.choice([-1.0, 1.0], n).astype(np.float32),
        "id_loose": (rng.random(n) < 0.95).astype(np.float32),
        "id_tight": (rng.random(n) < 0.6).astype(np.float32),
        "trg_wgt": rng.normal(1.0, 0.02, n).astype(np.float32),
        "m_vis": rng.uniform(0.0, 200.0, n).astype(np.float32),
        "event": (np.arange(n) + offset).astype(np.uint64),
    }
    if not is_data:
        columns["gen_match"] = rng.choice([5, 6], n, p=[0.8, 0.2]).astype(np.int32)
        columns["puweight"] = rng.normal(1.0, 0.1, n).astype(np.float32)
        columns["puweight_up"] = (columns["puweight"] * 1.05).astype(np.float32)
        columns["puweight_down"] = (columns["puweight"] * 0.95).astype(np.float32)
        columns["genWeight"] = rng.choice([-1.0, 1.0], n, p=[0.1, 0.9]).astype(np.float32)
    return columns


def _friend_columns(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    return {
        "score": rng.random(n).astype(np.float32),
        "cls": rng.choice([0, 1], n).astype(np.int32),
        "fake_factor": rng.uniform(0.1, 0.3, n).astype(np.float32),
    }


def _shifts(columns: dict[str, np.ndarray], column: str, name: str, factors: tuple[float, float]) -> dict[str, np.ndarray]:
    """CROWN-style shifted copies `<column>__<name>Up/Down` of one column, scaled by the given factors."""
    return {f"{column}__{name}{direction}": (columns[column] * factor).astype(columns[column].dtype) for direction, factor in zip(("Up", "Down"), factors)}


def make_mini_dataset(root: Path, era: str = "2018", channel: str = "mt", n_files: int = 2, n_events: int = 1000, seed: int = 1, variations: bool = False) -> dict:
    """Create main ntuples and friend trees for three nicks (data, DY, signal) below `root`.

    With `variations`, an embedded nick EMB_A (with `emb_genweight`) is added, the simulated and embedded main trees
    carry the shift `m_vis__tesUp/Down` and every friend tree `fake_factor__ffStatUp/Down`. The columns of the three
    nicks without variations stay exactly those of the plain dataset."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    nicks = MINI_NICKS + ([EMBEDDING_NICK] if variations else [])
    for nick in nicks:
        sample_type = {"DATA_A": "data", EMBEDDING_NICK: "embedding"}.get(nick, "mc")
        for i in range(n_files):
            name = f"{nick}_{i}.root"
            main = _main_columns(rng, n_events, i * n_events, sample_type == "data")
            friend = _friend_columns(rng, n_events)
            if variations:
                if sample_type == "embedding":
                    main["emb_genweight"] = np.ones(n_events, dtype=np.float32)
                if sample_type != "data":
                    main.update(_shifts(main, "m_vis", "tes", (1.03, 0.97)))
                friend.update(_shifts(friend, "fake_factor", "ffStat", (1.2, 0.8)))
            metadata = {"analysis": "mini", "era": era, "sample_type": sample_type}
            make_ntuple(root / "CROWNRun" / era / nick / channel / name, main, metadata)
            make_ntuple(root / "CROWNFriends" / "nn" / era / nick / channel / name, friend)
    return {"ntuples": root / "CROWNRun", "friends": root / "CROWNFriends" / "nn", "nicks": nicks}

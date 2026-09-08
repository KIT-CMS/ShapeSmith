"""Synthetic ntuples for tests and examples.

The mini dataset mimics the CROWN layout:
    <root>/CROWNRun/<era>/<nick>/<channel>/<nick>_<i>.root          main ntuples
    <root>/CROWNFriends/nn/<era>/<nick>/<channel>/<nick>_<i>.root   friend trees (score, cls, fake_factor)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import uproot

MINI_NICKS = ["DATA_A", "ZTT_1", "SIG_1"]


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


def make_mini_dataset(root: Path, era: str = "2018", channel: str = "mt", n_files: int = 2, n_events: int = 1000, seed: int = 1) -> dict:
    """Create main ntuples and friend trees for three nicks (data, DY, signal) below `root`."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    for nick in MINI_NICKS:
        for i in range(n_files):
            name = f"{nick}_{i}.root"
            metadata = {"analysis": "mini", "era": era, "sample_type": "data" if nick == "DATA_A" else "mc"}
            make_ntuple(root / "CROWNRun" / era / nick / channel / name, _main_columns(rng, n_events, i * n_events, nick == "DATA_A"), metadata)
            make_ntuple(root / "CROWNFriends" / "nn" / era / nick / channel / name, _friend_columns(rng, n_events))
    return {"ntuples": root / "CROWNRun", "friends": root / "CROWNFriends" / "nn", "nicks": list(MINI_NICKS)}

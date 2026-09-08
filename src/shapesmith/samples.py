"""Production inventories and the normalisation lookup in the KingMaker sample database (datasets.json).

The database renames nicks now and then (2026-08: the 2018 v15 nicks gained the campaign suffix
`_mc2018_realistic_v1-v2`) while CROWN output directories keep the nick used at production time. The DBS
dataset path is stable, so an inventory lists `nick dbs` pairs and the lookup goes by nick first, by DBS second.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

FIELDS = ("sample_type", "xsec", "nevents", "generator_weight")


@dataclass(frozen=True)
class InventoryEntry:
    nick: str  # directory name in the CROWN output
    dbs: str  # DBS dataset path, stable across database versions


def kind_of(sample_type: str) -> str:
    if sample_type == "data":
        return "data"
    if "embedding" in sample_type:
        return "embedding"
    return "mc"


def read_inventory(path: Path) -> list[InventoryEntry]:
    """`<nick> <dbs>` per line; blank lines and `#` comments are ignored."""
    entries = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{path}: expected '<nick> <dbs>', got {line!r}")
        entries.append(InventoryEntry(*parts))
    return entries


def write_inventory(path: Path, database: Path, nicks: list[str]) -> list[InventoryEntry]:
    """Create an inventory for nicks that the given database version still knows by name."""
    db = json.loads(Path(database).read_text())
    missing = [nick for nick in nicks if nick not in db]
    if missing:
        raise KeyError(f"nicks not in {database}: {', '.join(missing)}")
    entries = [InventoryEntry(nick, db[nick]["dbs"]) for nick in nicks]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(f"{entry.nick} {entry.dbs}\n" for entry in entries))
    return entries


def normalisation(database: Path, inventory: list[InventoryEntry]) -> dict[str, dict]:
    """nick -> {kind, sample_type, xsec, nevents, generator_weight, database_nick} for every inventory entry."""
    db = json.loads(Path(database).read_text())
    by_dbs: dict[str, list[str]] = {}
    for key, entry in db.items():
        by_dbs.setdefault(entry["dbs"], []).append(key)
    result, unresolved = {}, []
    for item in inventory:
        if item.nick in db:
            key = item.nick
        else:
            candidates = by_dbs.get(item.dbs, [])
            values = {tuple(db[c][field] for field in FIELDS) for c in candidates}
            if len(values) != 1:
                unresolved.append(f"{item.nick} ({len(candidates)} entries for {item.dbs})")
                continue
            key = candidates[0]
        entry = db[key]
        result[item.nick] = {"kind": kind_of(entry["sample_type"]), "database_nick": key, **{field: entry[field] for field in FIELDS}}
    if unresolved:
        raise KeyError(f"not resolvable in {database}: " + "; ".join(unresolved))
    return result

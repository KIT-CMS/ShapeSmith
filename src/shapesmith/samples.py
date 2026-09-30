"""Sample lists and the normalisation lookup in the KingMaker sample database (datasets.json).

A sample list is an unchanged copy of the KingMaker sample list a production ran with: one nick per line. The
normalisation of every nick comes from the configured database, looked up by nick. The database renames nicks now
and then, so an older production may need the database checkout it was produced with.
"""
from __future__ import annotations

import json
from pathlib import Path

from shapesmith.model import AnalysisError

FIELDS = ("xsec", "nevents", "generator_weight")


def kind_of(sample_type: str) -> str:
    """The sample kind of a database `sample_type`: data, embedding, or mc for every simulated type."""
    return sample_type if sample_type in ("data", "embedding") else "mc"


def read_sample_list(path: Path) -> tuple[str, ...]:
    """The nicks of a sample list in file order; blank lines are skipped, every other line is exactly one nick."""
    nicks: list[str] = []
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) != 1:
            raise ValueError(
                f"{path}:{number}: expected one nick per line (an unchanged KingMaker sample list), got {line!r};"
                " the former DBS column is no longer supported"
            )
        if tokens[0] in nicks:
            raise ValueError(f"{path}:{number}: duplicate nick {tokens[0]}")
        nicks.append(tokens[0])
    return tuple(nicks)


def normalisation(database: Path, nicks: tuple[str, ...] | list[str]) -> dict[str, dict]:
    """nick -> {kind, xsec, nevents, generator_weight} from the database; every missing nick is reported at once."""
    db = json.loads(Path(database).read_text())
    missing = [nick for nick in nicks if nick not in db]
    if missing:
        raise AnalysisError(
            f"{len(missing)} nicks are not in {database} (the database may have renamed them, see the git log of the"
            " sample database):\n" + "\n".join(missing)
        )
    return {nick: {"kind": kind_of(db[nick]["sample_type"]), **{field: db[nick][field] for field in FIELDS}} for nick in nicks}
